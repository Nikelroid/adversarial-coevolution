"""
Main script to evaluate Gin Rummy: PPO Agent vs Random Agent

Usage:
python main.py --player1 ppo --player2 random --model ./artifacts/models/ppo_gin_rummy/ppo_gin_rummy_final --tournament --num-games 10000
"""
from src.gin_rummy_api import GinRummyEnvAPI  # Assuming similar API structure
from agents.random_agent import RandomAgent
from agents.ppo_agent import PPOAgent
import argparse
import random
import numpy as np


def display_game_state(obs, agent_name):
    """
    Display the current game state for Gin Rummy.
    
    Args:
        obs: Observation dict from environment
        agent_name: Name of the current agent
    """
    print(f"\n--- {agent_name}'s Turn ---")
    # Note: Actual display depends on your observation structure
    # This is a placeholder - adjust based on your environment
    print(f"Observation shape: {obs['observation'].shape if hasattr(obs['observation'], 'shape') else 'N/A'}")


def play_game(env, agents_dic, agent_names, verbose=True, show_state=False):
    """
    Play a single game of Gin Rummy between two agents.
    
    Args:
        env: GinRummyEnvAPI instance
        agents_dic: Dictionary mapping player names to agents
        agent_names: Dictionary mapping player names to agent names
        verbose: Whether to print game progress
        show_state: Whether to display the game state
        
    Returns:
        Dictionary with game statistics
    """
    env.reset(seed=None)
    
    step_count = 0
    rewards = {'player_0': 0, 'player_1': 0}
    game_info = {
        'gin_winner': None,
        'knock_winner': None,
        'winner': None,
        'point_differential': 0,
        'total_steps': 0
    }
    
    for agent in env.agent_iter():
        obs, reward, term, trunc, info = env.get_current_state()
        
        # Update rewards
        rewards[agent] += reward
        
        if term or trunc:
            action = None
            
            # Determine game outcome from info if available
            if 'gin' in info:
                if info['gin']:
                    game_info['gin_winner'] = agent
            if 'knock' in info:
                if info['knock']:
                    game_info['knock_winner'] = agent
            
            # Determine winner based on rewards
            if rewards['player_0'] > rewards['player_1']:
                game_info['winner'] = 'player_0'
                game_info['point_differential'] = rewards['player_0'] - rewards['player_1']
            elif rewards['player_1'] > rewards['player_0']:
                game_info['winner'] = 'player_1'
                game_info['point_differential'] = rewards['player_1'] - rewards['player_0']
            else:
                game_info['winner'] = None
                game_info['point_differential'] = 0
            
            if verbose:
                print(f"\n{'='*60}")
                print(f"Game ended after {step_count} steps!")
                print(f"Player 0 ({agent_names['player_0']}) total reward: {rewards['player_0']:.2f}")
                print(f"Player 1 ({agent_names['player_1']}) total reward: {rewards['player_1']:.2f}")
                
                if game_info['gin_winner']:
                    print(f"🎯 GIN! Winner: {agent_names[game_info['gin_winner']]}")
                elif game_info['knock_winner']:
                    print(f"👊 KNOCK! Winner: {agent_names[game_info['knock_winner']]}")
                elif game_info['winner']:
                    print(f"🏆 Winner: {agent_names[game_info['winner']]}")
                else:
                    print("🤝 Draw!")
                print(f"Point differential: {game_info['point_differential']:.2f}")
                print(f"{'='*60}\n")
        else:
            if show_state and verbose:
                display_game_state(obs, agent_names[agent])
            
            action = agents_dic[agent].do_action()
            
            if verbose and step_count % 10 == 0:
                print(f"Step {step_count}: {agent_names[agent]} takes action {action}")
        
        env.step(action)
        step_count += 1
    
    game_info['total_steps'] = step_count
    game_info['rewards'] = rewards
    
    return game_info


def run_tournament(env, agent1, agent2, agent1_name, agent2_name, num_games=100):
    """
    Run a tournament between two agents with comprehensive statistics.
    
    Args:
        env: GinRummyEnvAPI instance
        agent1, agent2: Agent instances
        agent1_name, agent2_name: Agent names
        num_games: Number of games to play
        
    Returns:
        Detailed tournament statistics
    """
    stats = {
        agent1_name: {
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'gin_count': 0,
            'knock_count': 0,
            'total_reward': 0,
            'reward_as_player0': 0,
            'reward_as_player1': 0,
            'games_as_player0': 0,
            'games_as_player1': 0,
            'point_differentials': [],
            'weighted_score': 0,  # Gin=1, Knock=0.5
        },
        agent2_name: {
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'gin_count': 0,
            'knock_count': 0,
            'total_reward': 0,
            'reward_as_player0': 0,
            'reward_as_player1': 0,
            'games_as_player0': 0,
            'games_as_player1': 0,
            'point_differentials': [],
            'weighted_score': 0,
        }
    }
    
    total_steps = 0
    
    for game_num in range(num_games):
        # Alternate who goes first
        if game_num % 2 == 0:
            agent1.set_player('player_0')
            agent2.set_player('player_1')
            agents_dic = {'player_0': agent1, 'player_1': agent2}
            agent_names = {'player_0': agent1_name, 'player_1': agent2_name}
            agent1_position = 'player_0'
            agent2_position = 'player_1'
        else:
            agent1.set_player('player_1')
            agent2.set_player('player_0')
            agents_dic = {'player_0': agent2, 'player_1': agent1}
            agent_names = {'player_0': agent2_name, 'player_1': agent1_name}
            agent1_position = 'player_1'
            agent2_position = 'player_0'
        
        result = play_game(env, agents_dic, agent_names, verbose=False, show_state=False)
        
        # Track position-specific stats
        if agent1_position == 'player_0':
            stats[agent1_name]['games_as_player0'] += 1
            stats[agent2_name]['games_as_player1'] += 1
            stats[agent1_name]['reward_as_player0'] += result['rewards']['player_0']
            stats[agent2_name]['reward_as_player1'] += result['rewards']['player_1']
        else:
            stats[agent1_name]['games_as_player1'] += 1
            stats[agent2_name]['games_as_player0'] += 1
            stats[agent1_name]['reward_as_player1'] += result['rewards']['player_1']
            stats[agent2_name]['reward_as_player0'] += result['rewards']['player_0']
        
        # Track total rewards
        stats[agent1_name]['total_reward'] += result['rewards'][agent1_position]
        stats[agent2_name]['total_reward'] += result['rewards'][agent2_position]
        
        # Determine winner and update stats
        winner = result['winner']
        if winner:
            winner_name = agent_names[winner]
            loser_name = agent1_name if winner_name == agent2_name else agent2_name
            
            stats[winner_name]['wins'] += 1
            stats[loser_name]['losses'] += 1
            
            # Track point differential
            if winner_name == agent1_name:
                stats[agent1_name]['point_differentials'].append(result['point_differential'])
                stats[agent2_name]['point_differentials'].append(-result['point_differential'])
            else:
                stats[agent2_name]['point_differentials'].append(result['point_differential'])
                stats[agent1_name]['point_differentials'].append(-result['point_differential'])
            
            # Check for gin or knock
            if result['gin_winner'] == winner:
                stats[winner_name]['gin_count'] += 1
                stats[winner_name]['weighted_score'] += 1.0  # Gin = 1 point
            elif result['knock_winner'] == winner:
                stats[winner_name]['knock_count'] += 1
                stats[winner_name]['weighted_score'] += 0.5  # Knock = 0.5 points
            else:
                # Regular win (neither gin nor knock explicitly marked)
                stats[winner_name]['weighted_score'] += 0.5
        else:
            stats[agent1_name]['draws'] += 1
            stats[agent2_name]['draws'] += 1
            stats[agent1_name]['point_differentials'].append(0)
            stats[agent2_name]['point_differentials'].append(0)
        
        total_steps += result['total_steps']
        
        if (game_num + 1) % 1000 == 0:
            print(f"Completed {game_num + 1}/{num_games} games...")
    
    # Calculate averages
    stats['meta'] = {
        'total_games': num_games,
        'total_steps': total_steps,
        'avg_steps_per_game': total_steps / num_games
    }
    
    return stats


def print_tournament_results(stats, num_games):
    """
    Print comprehensive tournament results.
    
    Args:
        stats: Statistics dictionary from tournament
        num_games: Total number of games played
    """
    print(f"\n{'='*80}")
    print(f"🏆 TOURNAMENT RESULTS ({num_games} games)")
    print(f"{'='*80}")
    print(f"Average steps per game: {stats['meta']['avg_steps_per_game']:.1f}")
    print(f"{'='*80}\n")
    
    agent_names = [k for k in stats.keys() if k != 'meta']
    
    for name in agent_names:
        results = stats[name]
        win_rate = results['wins'] / num_games * 100
        loss_rate = results['losses'] / num_games * 100
        draw_rate = results['draws'] / num_games * 100
        
        avg_reward = results['total_reward'] / num_games
        avg_reward_p0 = results['reward_as_player0'] / max(results['games_as_player0'], 1)
        avg_reward_p1 = results['reward_as_player1'] / max(results['games_as_player1'], 1)
        
        avg_point_diff = np.mean(results['point_differentials']) if results['point_differentials'] else 0
        avg_weighted_score = results['weighted_score'] / num_games
        
        # Calculate average loss margin (point differential when losing)
        loss_differentials = [pd for i, pd in enumerate(results['point_differentials']) if pd < 0]
        avg_loss_margin = abs(np.mean(loss_differentials)) if loss_differentials else 0
        
        print(f"📊 {name}")
        print(f"{'-'*80}")
        print(f"  Win/Loss/Draw:     {results['wins']} / {results['losses']} / {results['draws']}")
        print(f"  Win Rate:          {win_rate:.2f}%")
        print(f"  Loss Rate:         {loss_rate:.2f}%")
        print(f"  Draw Rate:         {draw_rate:.2f}%")
        print(f"")
        print(f"  Gin Count:         {results['gin_count']} ({results['gin_count']/num_games*100:.2f}%)")
        print(f"  Knock Count:       {results['knock_count']} ({results['knock_count']/num_games*100:.2f}%)")
        print(f"  Regular Wins:      {results['wins'] - results['gin_count'] - results['knock_count']}")
        print(f"")
        print(f"  Weighted Score:    {results['weighted_score']:.2f} (avg: {avg_weighted_score:.3f})")
        print(f"    (Gin=1.0, Knock=0.5)")
        print(f"")
        print(f"  Average Reward:    {avg_reward:.3f}")
        print(f"    As Player 0:     {avg_reward_p0:.3f} ({results['games_as_player0']} games)")
        print(f"    As Player 1:     {avg_reward_p1:.3f} ({results['games_as_player1']} games)")
        print(f"")
        print(f"  Avg Point Diff:    {avg_point_diff:+.3f}")
        print(f"  Avg Loss Margin:   {avg_loss_margin:.3f}")
        print(f"")
    
    print(f"{'='*80}\n")
    
    # Head-to-head comparison
    if len(agent_names) == 2:
        agent1, agent2 = agent_names
        print(f"HEAD-TO-HEAD COMPARISON")
        print(f"{'-'*80}")
        print(f"  {agent1} vs {agent2}")
        print(f"    Score: {stats[agent1]['wins']} - {stats[agent2]['wins']} (Draws: {stats[agent1]['draws']})")
        print(f"    Gin: {stats[agent1]['gin_count']} - {stats[agent2]['gin_count']}")
        print(f"    Knock: {stats[agent1]['knock_count']} - {stats[agent2]['knock_count']}")
        print(f"    Weighted: {stats[agent1]['weighted_score']:.2f} - {stats[agent2]['weighted_score']:.2f}")
        print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Gin Rummy agents')
    parser.add_argument('--player1', type=str, default='random', 
                       choices=['random', 'ppo', 'human'], help='Type of player 1')
    parser.add_argument('--player2', type=str, default='random',
                       choices=['random', 'ppo'], help='Type of player 2')
    parser.add_argument('--model', type=str, default='./artifacts/models/ppo_gin_rummy/best_model',
                       help='Path to PPO model (if using PPO agent)')
    parser.add_argument('--render', type=str, default='ansi', 
                       choices=['ansi', 'human'], help='Render mode')
    parser.add_argument('--tournament', action='store_true',
                       help='Run a tournament with multiple games')
    parser.add_argument('--num-games', type=int, default=100,
                       help='Number of games for tournament mode')
    
    args = parser.parse_args()
    
    # Create environment
    env = GinRummyEnvAPI(render_mode=args.render)
    
    # Create agents
    if args.player1 == 'random':
        agent1 = RandomAgent(env)
        agent1_name = 'Random Agent'
    elif args.player1 == 'ppo':
        agent1 = PPOAgent(model_path=args.model, env=env)
        agent1_name = 'PPO Agent'
    else:  # human
        print("Human player not yet implemented. Using random instead.")
        agent1 = RandomAgent(env)
        agent1_name = 'Random Agent'
    
    if args.player2 == 'random':
        agent2 = RandomAgent(env)
        agent2_name = 'Random Opponent'
    else:  # ppo
        agent2 = PPOAgent(model_path=args.model, env=env)
        agent2_name = 'PPO Opponent'
    
    if args.tournament:
        # Run tournament
        print(f"\n🎮 Starting Gin Rummy Tournament")
        print(f"   {agent1_name} vs {agent2_name}")
        print(f"   Playing {args.num_games} games (alternating positions)...\n")
        
        stats = run_tournament(env, agent1, agent2, agent1_name, agent2_name, args.num_games)
        
        # Display results
        print_tournament_results(stats, args.num_games)
    else:
        # Play single game
        if random.random() < 0.5:
            print(f"Starting: {agent1_name} as Player 0, {agent2_name} as Player 1\n")
            agent1.set_player('player_0')
            agent2.set_player('player_1')
            agents_dic = {'player_0': agent1, 'player_1': agent2}
            agent_names = {'player_0': agent1_name, 'player_1': agent2_name}
        else:
            print(f"Starting: {agent2_name} as Player 0, {agent1_name} as Player 1\n")
            agent1.set_player('player_1')
            agent2.set_player('player_0')
            agents_dic = {'player_0': agent2, 'player_1': agent1}
            agent_names = {'player_0': agent2_name, 'player_1': agent1_name}
        
        result = play_game(env, agents_dic, agent_names, verbose=True, show_state=True)
    
    env.close()


if __name__ == '__main__':
    main()