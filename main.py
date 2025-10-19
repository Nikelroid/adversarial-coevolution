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
    Create comprehensive plots of tournament results.
    
    Args:
        stats: Statistics dictionary from tournament
        num_games: Total number of games
        output_dir: Directory to save plots
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    agent_names = list(stats.keys())
    colors = ['#3498db', '#e74c3c']
    
    # Figure 1: Win/Loss/Draw Summary
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    x = np.arange(len(agent_names))
    width = 0.25
    
    wins = [stats[name]['wins'] for name in agent_names]
    losses = [stats[name]['losses'] for name in agent_names]
    draws = [stats[name]['draws'] for name in agent_names]
    
    ax1.bar(x - width, wins, width, label='Wins', color='#2ecc71')
    ax1.bar(x, losses, width, label='Losses', color='#e74c3c')
    ax1.bar(x + width, draws, width, label='Draws', color='#95a5a6')
    
    ax1.set_xlabel('Agent', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax1.set_title(f'Win/Loss/Draw Summary ({num_games} games)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(agent_names)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/win_loss_draw_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Gin vs Knock vs Regular Wins
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    gin_counts = [stats[name]['gin_count'] for name in agent_names]
    knock_counts = [stats[name]['knock_count'] for name in agent_names]
    regular_wins = [stats[name]['wins'] - stats[name]['gin_count'] - stats[name]['knock_count'] 
                    for name in agent_names]
    
    ax2.bar(x - width, gin_counts, width, label='Gin Wins', color='#f39c12')
    ax2.bar(x, knock_counts, width, label='Knock Wins', color='#9b59b6')
    ax2.bar(x + width, regular_wins, width, label='Regular Wins', color='#3498db')
    
    ax2.set_xlabel('Agent', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax2.set_title('Win Type Breakdown', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(agent_names)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/win_types_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 3: Point Differential Distribution
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, name in enumerate(agent_names):
        diffs = stats[name]['point_differentials']
        axes3[idx].hist(diffs, bins=30, color=colors[idx], alpha=0.7, edgecolor='black')
        axes3[idx].axvline(np.mean(diffs), color='red', linestyle='--', linewidth=2, 
                          label=f'Mean: {np.mean(diffs):.2f}')
        axes3[idx].set_xlabel('Point Differential', fontsize=11, fontweight='bold')
        axes3[idx].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes3[idx].set_title(f'{name} - Point Differential Distribution', fontsize=12, fontweight='bold')
        axes3[idx].legend()
        axes3[idx].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/point_differential_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 4: Performance Metrics Comparison
    fig4, axes4 = plt.subplots(2, 2, figsize=(14, 10))
    
    # Win Rate
    win_rates = [stats[name]['wins'] / num_games * 100 for name in agent_names]
    axes4[0, 0].bar(agent_names, win_rates, color=colors)
    axes4[0, 0].set_ylabel('Win Rate (%)', fontweight='bold')
    axes4[0, 0].set_title('Win Rate Comparison', fontweight='bold')
    axes4[0, 0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(win_rates):
        axes4[0, 0].text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')
    
    # Average Reward
    avg_rewards = [stats[name]['total_reward'] / num_games for name in agent_names]
    axes4[0, 1].bar(agent_names, avg_rewards, color=colors)
    axes4[0, 1].set_ylabel('Average Reward', fontweight='bold')
    axes4[0, 1].set_title('Average Reward Comparison', fontweight='bold')
    axes4[0, 1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(avg_rewards):
        axes4[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
    
    # Weighted Score
    weighted_scores = [stats[name]['weighted_score'] for name in agent_names]
    axes4[1, 0].bar(agent_names, weighted_scores, color=colors)
    axes4[1, 0].set_ylabel('Weighted Score', fontweight='bold')
    axes4[1, 0].set_title('Weighted Score (Gin=1.0, Knock=0.5)', fontweight='bold')
    axes4[1, 0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(weighted_scores):
        axes4[1, 0].text(i, v + 5, f'{v:.1f}', ha='center', fontweight='bold')
    
    # Average Loss Margin
    avg_loss_margins = []
    for name in agent_names:
        loss_diffs = [pd for pd in stats[name]['point_differentials'] if pd < 0]
        avg_loss_margins.append(abs(np.mean(loss_diffs)) if loss_diffs else 0)
    
    axes4[1, 1].bar(agent_names, avg_loss_margins, color=colors)
    axes4[1, 1].set_ylabel('Average Loss Margin', fontweight='bold')
    axes4[1, 1].set_title('Average Loss Margin', fontweight='bold')
    axes4[1, 1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(avg_loss_margins):
        axes4[1, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/performance_metrics_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 5: Pie Chart - Win Distribution
    fig5, axes5 = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, name in enumerate(agent_names):
        sizes = [stats[name]['gin_count'], stats[name]['knock_count'], 
                 stats[name]['wins'] - stats[name]['gin_count'] - stats[name]['knock_count']]
        labels = ['Gin', 'Knock', 'Regular']
        colors_pie = ['#f39c12', '#9b59b6', '#3498db']
        
        # Only plot if there are wins
        if sum(sizes) > 0:
            axes5[idx].pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%',
                          startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
            axes5[idx].set_title(f'{name} - Win Type Distribution', fontsize=12, fontweight='bold')
        else:
            axes5[idx].text(0.5, 0.5, 'No Wins', ha='center', va='center', fontsize=14)
            axes5[idx].set_title(f'{name} - Win Type Distribution', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/win_distribution_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Plots saved to '{output_dir}/' directory")
    print(f"   - win_loss_draw_{timestamp}.png")
    print(f"   - win_types_{timestamp}.png")
    print(f"   - point_differential_{timestamp}.png")
    print(f"   - performance_metrics_{timestamp}.png")
    print(f"   - win_distribution_{timestamp}.png\n")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Gin Rummy agents')
    parser.add_argument('--player1', type=str, default='random', choices=['random', 'ppo'])
    parser.add_argument('--player2', type=str, default='random', choices=['random', 'ppo'])
    parser.add_argument('--model', type=str, default='./artifacts/models/ppo_gin_rummy/best_model')
    parser.add_argument('--render', type=str, default='ansi', choices=['ansi', 'human'])
    parser.add_argument('--tournament', action='store_true')
    parser.add_argument('--num-games', type=int, default=100)
    parser.add_argument('--plot-dir', type=str, default='./plots', help='Directory to save plots')
    
    args = parser.parse_args()
    
    env = GinRummyEnvAPI(render_mode=args.render)
    
    if args.player1 == 'random':
        agent1 = RandomAgent(env)
        agent1_name = 'Random Agent'
    else:
        agent1 = PPOAgent(model_path=args.model, env=env)
        agent1_name = 'PPO Agent'
    
    if args.player2 == 'random':
        agent2 = RandomAgent(env)
        agent2_name = 'Random Opponent'
    else:
        agent2 = PPOAgent(model_path=args.model, env=env)
        agent2_name = 'PPO Opponent'
    
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