"""
Main script to evaluate Gin Rummy: PPO Agent vs Random Agent

Usage:
python main.py --player1 ppo --player2 random --model ./artifacts/models/ppo_gin_rummy/ppo_gin_rummy_final --tournament --num-games 10000
"""
from src.gin_rummy_api import GinRummyEnvAPI
from agents.random_agent import RandomAgent
from agents.ppo_agent import PPOAgent
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from datetime import datetime


def play_game(env, agents_dic, agent_names, verbose=True):
    """
    Play a single game of Gin Rummy between two agents.
    """
    env.reset(seed=None)
    
    step_count = 0
    rewards = {'player_0': 0, 'player_1': 0}
    game_info = {
        'gin_winner': None,
        'knock_winner': None,
        'winner': None,
        'point_differential': 0
    }
    
    for agent in env.agent_iter():
        obs, reward, term, trunc, info = env.get_current_state()
        
        # Check for gin or knock based on reward
        if reward == 1.5 or reward == 1:  # Gin
            game_info['gin_winner'] = agent
            reward = 1.0  # Normalize to 1
        elif reward == 0.5:  # Knock
            game_info['knock_winner'] = agent
            reward = 0.5  # Normalize to 0.5

        rewards[agent] += reward
        
        if term or trunc:
            action = None
            
            # Check for gin or knock in info
            if 'gin' in info and info['gin']:
                game_info['gin_winner'] = agent
            if 'knock' in info and info['knock']:
                game_info['knock_winner'] = agent
            
            # Determine winner
            if rewards['player_0'] > rewards['player_1']:
                game_info['winner'] = 'player_0'
                game_info['point_differential'] = rewards['player_0'] - rewards['player_1']
            elif rewards['player_1'] > rewards['player_0']:
                game_info['winner'] = 'player_1'
                game_info['point_differential'] = rewards['player_1'] - rewards['player_0']
            
            if verbose:
                print(f"Game ended after {step_count} steps")
                print(f"Player 0: {rewards['player_0']:.2f}, Player 1: {rewards['player_1']:.2f}")
                if game_info['gin_winner']:
                    print(f"GIN by {agent_names[game_info['gin_winner']]}")
                elif game_info['knock_winner']:
                    print(f"KNOCK by {agent_names[game_info['knock_winner']]}")
        else:
            action = agents_dic[agent].do_action()
        
        env.step(action)
        step_count += 1
    
    game_info['rewards'] = rewards
    
    return game_info


def run_tournament(env, agent1, agent2, agent1_name, agent2_name, num_games=100):
    """
    Run a tournament between two agents.
    """
    stats = {
        agent1_name: {
            'wins': 0, 'losses': 0, 'draws': 0,
            'gin_count': 0, 'knock_count': 0,
            'total_reward': 0, 'point_differentials': [],
            'weighted_score': 0
        },
        agent2_name: {
            'wins': 0, 'losses': 0, 'draws': 0,
            'gin_count': 0, 'knock_count': 0,
            'total_reward': 0, 'point_differentials': [],
            'weighted_score': 0
        }
    }
    
    for game_num in range(num_games):
        # Alternate positions
        if game_num % 2 == 0:
            agent1.set_player('player_0')
            agent2.set_player('player_1')
            agents_dic = {'player_0': agent1, 'player_1': agent2}
            agent_names = {'player_0': agent1_name, 'player_1': agent2_name}
            agent1_pos, agent2_pos = 'player_0', 'player_1'
        else:
            agent1.set_player('player_1')
            agent2.set_player('player_0')
            agents_dic = {'player_0': agent2, 'player_1': agent1}
            agent_names = {'player_0': agent2_name, 'player_1': agent1_name}
            agent1_pos, agent2_pos = 'player_1', 'player_0'
        
        result = play_game(env, agents_dic, agent_names, verbose=False)
        
        # Update rewards
        stats[agent1_name]['total_reward'] += result['rewards'][agent1_pos]
        stats[agent2_name]['total_reward'] += result['rewards'][agent2_pos]
        
        # Update win/loss
        winner = result['winner']
        if winner:
            winner_name = agent_names[winner]
            loser_name = agent1_name if winner_name == agent2_name else agent2_name
            
            stats[winner_name]['wins'] += 1
            stats[loser_name]['losses'] += 1
            
            # Point differentials
            if winner_name == agent1_name:
                stats[agent1_name]['point_differentials'].append(result['point_differential'])
                stats[agent2_name]['point_differentials'].append(-result['point_differential'])
            else:
                stats[agent2_name]['point_differentials'].append(result['point_differential'])
                stats[agent1_name]['point_differentials'].append(-result['point_differential'])
            
            # Gin/Knock tracking
            if result['gin_winner'] == winner:
                stats[winner_name]['gin_count'] += 1
                stats[winner_name]['weighted_score'] += 1.0
            elif result['knock_winner'] == winner:
                stats[winner_name]['knock_count'] += 1
                stats[winner_name]['weighted_score'] += 0.5
            else:
                stats[winner_name]['weighted_score'] += 0.5
        else:
            stats[agent1_name]['draws'] += 1
            stats[agent2_name]['draws'] += 1
            stats[agent1_name]['point_differentials'].append(0)
            stats[agent2_name]['point_differentials'].append(0)
        
        if (game_num + 1) % 1000 == 0:
            print(f"Completed {game_num + 1}/{num_games} games...")
    
    return stats


def print_results(stats, num_games):
    """Print tournament results."""
    print(f"\n{'='*70}")
    print(f"TOURNAMENT RESULTS ({num_games} games)")
    print(f"{'='*70}\n")
    
    for name, results in stats.items():
        win_rate = results['wins'] / num_games * 100
        avg_reward = results['total_reward'] / num_games
        avg_weighted_score = results['weighted_score'] / num_games
        
        loss_diffs = [pd for pd in results['point_differentials'] if pd < 0]
        avg_loss_margin = abs(np.mean(loss_diffs)) if loss_diffs else 0
        
        print(f"{name}:")
        print(f"  Wins/Losses/Draws: {results['wins']}/{results['losses']}/{results['draws']}")
        print(f"  Win Rate: {win_rate:.2f}%")
        print(f"  Gin Count: {results['gin_count']} ({results['gin_count']/num_games*100:.2f}%)")
        print(f"  Knock Count: {results['knock_count']} ({results['knock_count']/num_games*100:.2f}%)")
        print(f"  Weighted Score: {results['weighted_score']:.2f} (avg: {avg_weighted_score:.3f})")
        print(f"  Average Reward: {avg_reward:.3f}")
        print(f"  Average Loss Margin: {avg_loss_margin:.3f}")
        print()
    
    print(f"{'='*70}\n")


def plot_results(stats, num_games, output_dir='./plots'):
    """
    Create comprehensive plots of tournament results using Seaborn.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Set scientific style
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    palette = sns.color_palette("viridis")
    
    agent_names = list(stats.keys())
    
    # --- Data Preparation ---
    # Convert stats to DataFrame for easier plotting with Seaborn
    data_summary = []
    for name in agent_names:
        data_summary.append({
            'Agent': name,
            'Result': 'Wins',
            'Count': stats[name]['wins']
        })
        data_summary.append({
            'Agent': name,
            'Result': 'Losses',
            'Count': stats[name]['losses']
        })
        data_summary.append({
            'Agent': name,
            'Result': 'Draws',
            'Count': stats[name]['draws']
        })
    df_summary = pd.DataFrame(data_summary)

    # --- Figure 1: Win/Loss/Draw Summary ---
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df_summary, x='Agent', y='Count', hue='Result', palette="muted")
    ax.set_title(f'Tournament Results ({num_games} Games)', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Agent', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    sns.despine(left=True)
    plt.legend(title='Outcome')
    plt.savefig(f'{output_dir}/win_loss_draw_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # --- Figure 2: Win Type Breakdown ---
    data_types = []
    for name in agent_names:
        regular = stats[name]['wins'] - stats[name]['gin_count'] - stats[name]['knock_count']
        data_types.append({'Agent': name, 'Type': 'Gin', 'Count': stats[name]['gin_count']})
        data_types.append({'Agent': name, 'Type': 'Knock', 'Count': stats[name]['knock_count']})
        data_types.append({'Agent': name, 'Type': 'Regular', 'Count': regular})
    df_types = pd.DataFrame(data_types)
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df_types, x='Agent', y='Count', hue='Type', palette="deep")
    ax.set_title('Win Type Breakdown', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(f'{output_dir}/win_types_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # --- Figure 3: Point Differential Distribution ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for idx, name in enumerate(agent_names):
        diffs = stats[name]['point_differentials']
        sns.histplot(diffs, kde=True, ax=axes[idx], color=palette[idx % len(palette)], bins=30)
        axes[idx].axvline(np.mean(diffs), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(diffs):.2f}')
        axes[idx].set_title(f'{name} - Point Differential', fontsize=14, fontweight='bold')
        axes[idx].set_xlabel('Point Diff')
        axes[idx].legend()
    plt.suptitle('Score Differential Distribution', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/point_differential_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # --- Figure 4: Performance Metrics ---
    metrics_data = []
    for name in agent_names:
        metrics_data.append({'Agent': name, 'Metric': 'Win Rate (%)', 'Value': stats[name]['wins'] / num_games * 100})
        metrics_data.append({'Agent': name, 'Metric': 'Avg Reward', 'Value': stats[name]['total_reward'] / num_games})
        metrics_data.append({'Agent': name, 'Metric': 'Weighted Score', 'Value': stats[name]['weighted_score']})
        
        loss_diffs = [pd for pd in stats[name]['point_differentials'] if pd < 0]
        avg_loss = abs(np.mean(loss_diffs)) if loss_diffs else 0
        metrics_data.append({'Agent': name, 'Metric': 'Avg Loss Margin', 'Value': avg_loss})
    
    df_metrics = pd.DataFrame(metrics_data)
    
    g = sns.FacetGrid(df_metrics, col="Metric", col_wrap=2, height=4, aspect=1.5, sharey=False)
    g.map_dataframe(sns.barplot, x="Agent", y="Value", palette="viridis")
    g.set_titles("{col_name}", fontweight='bold', size=14)
    
    # Add labels
    for ax in g.axes.flat:
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f', padding=3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/performance_metrics_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Plots saved to '{output_dir}/' directory (Seaborn Style)")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Gin Rummy agents')
    parser.add_argument('--player1', type=str, default='random', choices=['random', 'ppo'])
    parser.add_argument('--player2', type=str, default='random', choices=['random', 'ppo'])
    parser.add_argument('--model', type=str, default='./artifacts/models/ppo_gin_rummy/best_model')
    parser.add_argument('--render', type=str, default='ansi', choices=['ansi', 'human'])
    parser.add_argument('--tournament', action='store_true')
    parser.add_argument('--num-games', type=int, default=100)
    parser.add_argument('--plot-dir', type=str, default='./plots', help='Directory to save plots')
    parser.add_argument('--player1-name', type=str, default=None, help='Custom name for Player 1')
    parser.add_argument('--player2-name', type=str, default=None, help='Custom name for Player 2')
    
    args = parser.parse_args()
    
    env = GinRummyEnvAPI(render_mode=args.render)
    
    if args.player1 == 'random':
        agent1 = RandomAgent(env)
        agent1_name = args.player1_name if args.player1_name else 'Random Agent'
    else:
        agent1 = PPOAgent(model_path=args.model, env=env)
        agent1_name = args.player1_name if args.player1_name else 'PPO Agent'
    
    if args.player2 == 'random':
        agent2 = RandomAgent(env)
        agent2_name = args.player2_name if args.player2_name else 'Random Opponent'
    else:
        agent2 = PPOAgent(model_path=args.model, env=env)
        agent2_name = args.player2_name if args.player2_name else 'PPO Opponent'
    
    if args.tournament:
        print(f"\nStarting Tournament: {agent1_name} vs {agent2_name}")
        print(f"Playing {args.num_games} games...\n")
        
        stats = run_tournament(env, agent1, agent2, agent1_name, agent2_name, args.num_games)
        print_results(stats, args.num_games)
        
        # Generate and save plots
        plot_results(stats, args.num_games, output_dir=args.plot_dir)
    else:
        agent1.set_player('player_0')
        agent2.set_player('player_1')
        agents_dic = {'player_0': agent1, 'player_1': agent2}
        agent_names = {'player_0': agent1_name, 'player_1': agent2_name}
        
        result = play_game(env, agents_dic, agent_names, verbose=True)
    
    env.close()


if __name__ == '__main__':
    main()