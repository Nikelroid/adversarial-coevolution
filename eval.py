"""
Evaluation script for trained PPO Gin Rummy agent.

Usage:
    python evaluate_model.py --model-path ./artifacts/models/ppo_gin_rummy/best_model.zip --episodes 1000
"""

import argparse
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from gym_wrapper import GinRummySB3Wrapper
from agents.random_agent import RandomAgent
from tqdm import tqdm
import json
import os
from datetime import datetime


def evaluate_model(model_path, num_episodes=1000, save_results=True, verbose=True):
    """
    Evaluate a trained PPO model.
    
    Args:
        model_path: Path to the trained model (.zip file)
        num_episodes: Number of episodes to evaluate
        save_results: Whether to save results to files
        verbose: Whether to print progress
        
    Returns:
        Dictionary with evaluation results
    """
    
    print(f"\n{'='*80}")
    print(f"EVALUATING MODEL: {model_path}")
    print(f"Number of episodes: {num_episodes}")
    print(f"{'='*80}\n")
    
    # Load the trained model
    print("Loading model...")
    model = PPO.load(model_path)
    print(f"✓ Model loaded successfully")
    
    # Create evaluation environment
    print("Creating environment...")
    env = GinRummySB3Wrapper(opponent_policy=RandomAgent, randomize_position=True)
    print(f"✓ Environment created\n")
    
    # Storage for results
    episode_rewards = []
    episode_lengths = []
    episode_outcomes = []  # win/loss/draw
    action_distributions = []
    
    # Run evaluation
    print(f"Starting evaluation for {num_episodes} episodes...")
    for episode in tqdm(range(num_episodes), desc="Evaluating"):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        episode_actions = []
        
        while not done:
            # Get action from model (deterministic for evaluation)
            action, _ = model.predict(obs, deterministic=True)
            episode_actions.append(int(action))
            
            # Take step
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            if done or truncated:
                break
        
        # Store episode data
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Determine outcome
        if episode_reward > 0:
            outcome = 'win'
        elif episode_reward < 0:
            outcome = 'loss'
        else:
            outcome = 'draw'
        episode_outcomes.append(outcome)
        
        # Store action distribution for this episode
        action_distributions.extend(episode_actions)
        
        # Print periodic updates
        if verbose and (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"\nEpisode {episode + 1}/{num_episodes} - Last 100 avg reward: {avg_reward:.3f}")
    
    env.close()
    
    # Calculate statistics
    results = {
        'model_path': model_path,
        'num_episodes': num_episodes,
        'evaluation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        
        # Reward statistics
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'min_reward': float(np.min(episode_rewards)),
        'max_reward': float(np.max(episode_rewards)),
        'median_reward': float(np.median(episode_rewards)),
        
        # Episode length statistics
        'mean_length': float(np.mean(episode_lengths)),
        'std_length': float(np.std(episode_lengths)),
        'min_length': int(np.min(episode_lengths)),
        'max_length': int(np.max(episode_lengths)),
        
        # Win/Loss statistics
        'wins': episode_outcomes.count('win'),
        'losses': episode_outcomes.count('loss'),
        'draws': episode_outcomes.count('draw'),
        'win_rate': episode_outcomes.count('win') / num_episodes * 100,
        'loss_rate': episode_outcomes.count('loss') / num_episodes * 100,
        'draw_rate': episode_outcomes.count('draw') / num_episodes * 100,
        
        # Action statistics
        'total_actions': len(action_distributions),
        'unique_actions': len(set(action_distributions)),
        'most_common_action': int(pd.Series(action_distributions).mode()[0]),
    }
    
    # Print results
    print(f"\n{'='*80}")
    print("EVALUATION RESULTS")
    print(f"{'='*80}")
    print(f"\n📊 REWARD STATISTICS:")
    print(f"  Mean Reward:     {results['mean_reward']:>10.3f}")
    print(f"  Std Deviation:   {results['std_reward']:>10.3f}")
    print(f"  Min Reward:      {results['min_reward']:>10.3f}")
    print(f"  Max Reward:      {results['max_reward']:>10.3f}")
    print(f"  Median Reward:   {results['median_reward']:>10.3f}")
    
    print(f"\n📏 EPISODE LENGTH STATISTICS:")
    print(f"  Mean Length:     {results['mean_length']:>10.1f} steps")
    print(f"  Std Deviation:   {results['std_length']:>10.1f} steps")
    print(f"  Min Length:      {results['min_length']:>10d} steps")
    print(f"  Max Length:      {results['max_length']:>10d} steps")
    
    print(f"\n🏆 WIN/LOSS STATISTICS:")
    print(f"  Wins:            {results['wins']:>10d} ({results['win_rate']:>6.2f}%)")
    print(f"  Losses:          {results['losses']:>10d} ({results['loss_rate']:>6.2f}%)")
    print(f"  Draws:           {results['draws']:>10d} ({results['draw_rate']:>6.2f}%)")
    
    print(f"\n🎯 ACTION STATISTICS:")
    print(f"  Total Actions:   {results['total_actions']:>10d}")
    print(f"  Unique Actions:  {results['unique_actions']:>10d}")
    print(f"  Most Common:     {results['most_common_action']:>10d}")
    print(f"\n{'='*80}\n")
    
    # Save results if requested
    if save_results:
        # Create results directory
        results_dir = "evaluation_results"
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save summary JSON
        json_path = os.path.join(results_dir, f"eval_summary_{timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"✓ Saved summary to: {json_path}")
        
        # Save detailed episode data
        episodes_df = pd.DataFrame({
            'episode': range(1, num_episodes + 1),
            'reward': episode_rewards,
            'length': episode_lengths,
            'outcome': episode_outcomes
        })
        csv_path = os.path.join(results_dir, f"eval_episodes_{timestamp}.csv")
        episodes_df.to_csv(csv_path, index=False)
        print(f"✓ Saved episode data to: {csv_path}")
        
        # Save action distribution
        actions_df = pd.DataFrame({'action': action_distributions})
        actions_csv = os.path.join(results_dir, f"eval_actions_{timestamp}.csv")
        actions_df.to_csv(actions_csv, index=False)
        print(f"✓ Saved action data to: {actions_csv}")
        
        # Create summary plot
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Model Evaluation Results - {num_episodes} Episodes', fontsize=16)
            
            # Reward distribution
            axes[0, 0].hist(episode_rewards, bins=50, edgecolor='black', alpha=0.7)
            axes[0, 0].axvline(results['mean_reward'], color='red', linestyle='--', 
                              label=f"Mean: {results['mean_reward']:.2f}")
            axes[0, 0].set_xlabel('Episode Reward')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Reward Distribution')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Rewards over time
            window = 50
            rolling_mean = pd.Series(episode_rewards).rolling(window=window).mean()
            axes[0, 1].plot(episode_rewards, alpha=0.3, label='Episode Reward')
            axes[0, 1].plot(rolling_mean, linewidth=2, label=f'{window}-Episode MA')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Reward')
            axes[0, 1].set_title('Rewards Over Time')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Episode length distribution
            axes[1, 0].hist(episode_lengths, bins=50, edgecolor='black', alpha=0.7, color='green')
            axes[1, 0].axvline(results['mean_length'], color='red', linestyle='--',
                              label=f"Mean: {results['mean_length']:.1f}")
            axes[1, 0].set_xlabel('Episode Length (steps)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Episode Length Distribution')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Win/Loss pie chart
            outcomes_counts = [results['wins'], results['losses'], results['draws']]
            labels = [f"Wins ({results['win_rate']:.1f}%)", 
                     f"Losses ({results['loss_rate']:.1f}%)",
                     f"Draws ({results['draw_rate']:.1f}%)"]
            colors = ['#2ecc71', '#e74c3c', '#95a5a6']
            axes[1, 1].pie(outcomes_counts, labels=labels, autopct='%1.1f%%', 
                          startangle=90, colors=colors)
            axes[1, 1].set_title('Win/Loss Distribution')
            
            plt.tight_layout()
            plot_path = os.path.join(results_dir, f"eval_plots_{timestamp}.png")
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✓ Saved plots to: {plot_path}")
            
        except ImportError:
            print("⚠ Matplotlib not available, skipping plots")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained PPO Gin Rummy model')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to the trained model (.zip file)')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of episodes to evaluate (default: 1000)')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save results to files')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    # Run evaluation
    results = evaluate_model(
        model_path=args.model_path,
        num_episodes=args.episodes,
        save_results=not args.no_save,
        verbose=not args.quiet
    )
    
    return results


if __name__ == '__main__':
    main()