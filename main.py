"""
Main script to play Tic Tac Toe: PPO Agent vs Random Agent

Usage:
python main.py --player1 ppo --player2 random --model ./models/ppo_tictactoe/ppo_tictactoe_final
"""
from src.tictoctoe_api import TicTacToeEnvAPI
from agents.random_agent import RandomAgent
from agents.ppo_agent import PPOAgent
import argparse
import random


def display_board(obs):
    """
    Display the tic tac toe board from observation.
    
    Args:
        obs: Observation dict from environment
    """
    # Extract board state from observation
    board = obs['observation']
    
    # For tic tac toe, the observation is typically:
    # First 9 values: player 1 positions (X)
    # Next 9 values: player 2 positions (O)
    
    display = []
    for i in range(9):
        if board[i] == 1:  # Player 1 (X)
            display.append('X')
        elif board[i + 9] == 1:  # Player 2 (O)
            display.append('O')
        else:
            display.append(str(i))
    
    print("\n Current Board:")
    print(f" {display[0]} | {display[1]} | {display[2]} ")
    print("---|---|---")
    print(f" {display[3]} | {display[4]} | {display[5]} ")
    print("---|---|---")
    print(f" {display[6]} | {display[7]} | {display[8]} ")
    print()


def play_game(env, agents_dic, agent_names, verbose=True, show_board=True):
    """
    Play a single game between two agents.
    
    Args:
        env: TicTacToeEnvAPI instance
        agents_dic: Dictionary mapping player names to agents
        agent_names: Dictionary mapping player names to agent names
        verbose: Whether to print game progress
        show_board: Whether to display the board
        
    Returns:
        Dictionary with game statistics
    """
    env.reset(seed=None)  # Random seed for variety
    
    step_count = 0
    rewards = {'player_1': 0, 'player_2': 0}
    
    for agent in env.agent_iter():
        obs, reward, term, trunc, info = env.get_current_state()
        
        # Update rewards
        rewards[agent] += reward
        
        if term or trunc:
            action = None
            if verbose:
                print(f"\n{'='*50}")
                print(f"Game ended!")
                print(f"Player 1 (X - {agent_names['player_1']}) total reward: {rewards['player_1']:.0f}")
                print(f"Player 2 (O - {agent_names['player_2']}) total reward: {rewards['player_2']:.0f}")
                
                if rewards['player_1'] > rewards['player_2']:
                    print(f"🏆 Winner: {agent_names['player_1']} (X)")
                elif rewards['player_2'] > rewards['player_1']:
                    print(f"🏆 Winner: {agent_names['player_2']} (O)")
                else:
                    print("🤝 Draw!")
                print(f"{'='*50}\n")
        else:
            if show_board and verbose:
                display_board(obs)
                
            action = agents_dic[agent].do_action()
            
            if verbose:
                symbol = 'X' if agent == 'player_1' else 'O'
                print(f"Step {step_count}: {agent_names[agent]} ({symbol}) plays position {action}")
        
        env.step(action)
        
        step_count += 1
    
    # Determine winner
    winner = None
    if rewards['player_1'] > rewards['player_2']:
        winner = 'player_1'
    elif rewards['player_2'] > rewards['player_1']:
        winner = 'player_2'
    
    return {
        'steps': step_count,
        'rewards': rewards,
        'winner': winner,
        'winner_name': agent_names.get(winner, 'Draw')
    }


def run_tournament(env, agent1, agent2, agent1_name, agent2_name, num_games=100):
    """
    Run a tournament between two agents.
    
    Args:
        env: TicTacToeEnvAPI instance
        agent1, agent2: Agent instances
        agent1_name, agent2_name: Agent names
        num_games: Number of games to play
        
    Returns:
        Tournament statistics
    """
    stats = {
        agent1_name: {'wins': 0, 'losses': 0, 'draws': 0},
        agent2_name: {'wins': 0, 'losses': 0, 'draws': 0}
    }
    
    for game_num in range(num_games):
        # Alternate who goes first
        if game_num % 2 == 0:
            agent1.set_player('player_1')
            agent2.set_player('player_2')
            agents_dic = {'player_1': agent1, 'player_2': agent2}
            agent_names = {'player_1': agent1_name, 'player_2': agent2_name}
        else:
            agent1.set_player('player_2')
            agent2.set_player('player_1')
            agents_dic = {'player_1': agent2, 'player_2': agent1}
            agent_names = {'player_1': agent2_name, 'player_2': agent1_name}
        
        result = play_game(env, agents_dic, agent_names, verbose=False, show_board=False)
        
        if result['winner_name'] == agent1_name:
            stats[agent1_name]['wins'] += 1
            stats[agent2_name]['losses'] += 1
        elif result['winner_name'] == agent2_name:
            stats[agent2_name]['wins'] += 1
            stats[agent1_name]['losses'] += 1
        else:
            stats[agent1_name]['draws'] += 1
            stats[agent2_name]['draws'] += 1
        
        if (game_num + 1) % 20 == 0:
            print(f"Completed {game_num + 1}/{num_games} games...")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Play Tic Tac Toe with different agents')
    parser.add_argument('--player1', type=str, default='random', 
                       choices=['random', 'ppo', 'human'], help='Type of player 1')
    parser.add_argument('--player2', type=str, default='random',
                       choices=['random', 'ppo'], help='Type of player 2')
    parser.add_argument('--model', type=str, default='./models/ppo_tictactoe/ppo_tictactoe_final',
                       help='Path to PPO model (if using PPO agent)')
    parser.add_argument('--render', type=str, default='ansi', 
                       choices=['ansi', 'human'], help='Render mode')
    parser.add_argument('--tournament', action='store_true',
                       help='Run a tournament with multiple games')
    parser.add_argument('--num-games', type=int, default=100,
                       help='Number of games for tournament mode')
    
    args = parser.parse_args()
    
    # Create environment
    env = TicTacToeEnvAPI(render_mode=args.render)
    
    # Create agents
    if args.player1 == 'random':
        agent1 = RandomAgent(env)
        agent1_name = 'Random Agent 1'
    elif args.player1 == 'ppo':
        agent1 = PPOAgent(model_path=args.model, env=env)
        agent1_name = 'PPO Agent 1'
    else:  # human
        print("Human player not yet implemented. Using random instead.")
        agent1 = RandomAgent(env)
        agent1_name = 'Random Agent 1'
    
    if args.player2 == 'random':
        agent2 = RandomAgent(env)
        agent2_name = 'Random Agent 2'
    else:  # ppo
        agent2 = PPOAgent(model_path=args.model, env=env)
        agent2_name = 'PPO Agent 2'
    
    if args.tournament:
        # Run tournament
        print(f"\n🏆 Starting Tournament: {agent1_name} vs {agent2_name}")
        print(f"Playing {args.num_games} games (alternating first player)...\n")
        
        stats = run_tournament(env, agent1, agent2, agent1_name, agent2_name, args.num_games)
        
        # Display results
        print(f"\n{'='*60}")
        print(f"Tournament Results ({args.num_games} games)")
        print(f"{'='*60}")
        for name, results in stats.items():
            win_rate = results['wins'] / args.num_games * 100
            print(f"\n{name}:")
            print(f"  Wins:   {results['wins']} ({win_rate:.1f}%)")
            print(f"  Losses: {results['losses']}")
            print(f"  Draws:  {results['draws']}")
        print(f"{'='*60}\n")
    else:
        # Play single game with random starting position
        if random.random() < 0.5:
            print(f"Starting positions: {agent1_name} plays X, {agent2_name} plays O\n")
            agent1.set_player('player_1')
            agent2.set_player('player_2')
            agents_dic = {'player_1': agent1, 'player_2': agent2}
            agent_names = {'player_1': agent1_name, 'player_2': agent2_name}
        else:
            print(f"Starting positions: {agent2_name} plays X, {agent1_name} plays O\n")
            agent1.set_player('player_2')
            agent2.set_player('player_1')
            agents_dic = {'player_1': agent2, 'player_2': agent1}
            agent_names = {'player_1': agent2_name, 'player_2': agent1_name}
        
        result = play_game(env, agents_dic, agent_names, verbose=True, show_board=True)
    
    env.close()


if __name__ == '__main__':
    main()