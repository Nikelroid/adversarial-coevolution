"""
Training script for PPO agent on Tic Tac Toe environment with action masking.

Requirements:
pip install stable-baselines3[extra] pettingzoo gymnasium wandb

Usage:
python train_ppo.py --train
"""

import os
import numpy as np
import torch as th
import torch.nn as nn
from torch.distributions import Categorical
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import wandb
from typing import Union, Optional, Dict as TypingDict
from gymnasium import spaces

# Import your custom components
from gym_wrapper import TicTacToeSB3Wrapper
from agents.random_agent import RandomAgent

# ============================================
# W&B Configuration - Works on Colab & Local
# ============================================
WANDB_API_KEY = "41fe78a601dfc0909950ad6ec7e6c4fb042d032a"  # Replace with your actual API key
WANDB_PROJECT = "TicTacToe-PPO"

# Login to W&B
wandb.login(key=WANDB_API_KEY)


class DictObservationExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for dict observations with observation and action_mask.
    """
    
    def __init__(self, observation_space: spaces.Dict):
        # We'll only use the observation part, not the mask for features
        obs_shape = observation_space['observation'].shape
        super().__init__(observation_space, features_dim=obs_shape[0])
        
    def forward(self, observations: TypingDict[str, th.Tensor]) -> th.Tensor:
        """
        Extract features from dict observation.
        Only uses the 'observation' part, not the 'action_mask'.
        """
        if isinstance(observations, dict):
            # Extract just the observation tensor
            obs_tensor = observations['observation']
            
            # Ensure it's a tensor on the right device
            if not isinstance(obs_tensor, th.Tensor):
                obs_tensor = th.as_tensor(obs_tensor, device=self.device).float()
            
            return obs_tensor
        else:
            # If somehow we get a tensor, just return it
            return observations


class MaskedTicTacToePolicy(ActorCriticPolicy):
    """
    PPO-compatible masked MLP policy for PettingZoo Tic Tac Toe.
    Works with dict observation: {'observation': ..., 'action_mask': ...}
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_entropy = None

    def _extract_obs_and_mask(self, obs):
        """
        Extract observation vector and action mask from dict observation.
        """
        if isinstance(obs, dict):
            obs_tensor = obs['observation']
            mask_tensor = obs.get('action_mask', None)
        else:
            # Should not happen with our setup, but handle it
            obs_tensor = obs
            mask_tensor = None
        
        # Ensure tensors are on correct device
        if not isinstance(obs_tensor, th.Tensor):
            obs_tensor = th.as_tensor(obs_tensor, device=self.device).float()
        if mask_tensor is not None and not isinstance(mask_tensor, th.Tensor):
            mask_tensor = th.as_tensor(mask_tensor, device=self.device)
            
        return obs_tensor, mask_tensor

    def _apply_action_mask(self, logits, action_mask):
        """
        Apply action mask to logits by setting invalid actions to -inf.
        """
        if action_mask is None:
            return logits
        
        # Ensure mask is boolean tensor on same device
        mask = action_mask.to(dtype=th.bool, device=logits.device)
        
        # Use -inf for invalid actions
        logits = th.where(mask, logits, th.tensor(float('-inf'), device=logits.device, dtype=logits.dtype))
        
        return logits

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass with action masking applied to logits.
        """
        # Extract observation and mask from the dict
        _, action_mask = self._extract_obs_and_mask(obs)
        
        # Get features using the parent's extract_features (which will use our custom extractor)
        features = self.extract_features(obs)
        
        # Get latent representations
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        
        # Get action logits from latent features
        logits = self.action_net(latent_pi)  # This gives you [batch, 9] for tic tac toe

        # Apply the mask to logits
        masked_logits = self._apply_action_mask(logits, action_mask)

        # Create distribution from masked logits
        distribution = Categorical(logits=masked_logits)

        # Track entropy
        entropy = distribution.entropy()
        self.last_entropy = entropy.mean().item()

        # Get values
        values = self.value_net(latent_vf)
        
        # Sample actions
        actions = distribution.sample() if not deterministic else th.argmax(masked_logits, dim=1)
        log_prob = distribution.log_prob(actions)
        
        return actions, values, log_prob


class WandbCallback(BaseCallback):
    """
    Custom callback for logging to Weights & Biases.
    """
    def __init__(self, verbose=0):
        super(WandbCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        if hasattr(self.model.policy, 'last_entropy') and self.model.policy.last_entropy is not None:
            wandb.log({
                "train/policy_entropy": self.model.policy.last_entropy,
                "train/timesteps": self.num_timesteps
            })
        
        # Log episode data when episodes end
        if len(self.model.ep_info_buffer) > 0:
            for info in self.model.ep_info_buffer:
                wandb.log({
                    "train/episode_reward": info['r'],
                    "train/episode_length": info['l'],
                    "train/timesteps": self.num_timesteps,
                })
        
        return True
    
    def _on_rollout_end(self) -> bool:
        # Log training metrics after each rollout
        if hasattr(self.model, 'logger') and self.model.logger is not None:
            wandb.log({
                "train/timesteps": self.num_timesteps,
                "train/fps": self.model.logger.name_to_value.get("time/fps", 0),
            })
        return True


class WandbBestModelCallback(BaseCallback):
    """
    Callback to log when a new best model is found during evaluation.
    """
    def __init__(self, verbose=0):
        super(WandbBestModelCallback, self).__init__(verbose)
        
    def _on_step(self) -> bool:
        wandb.log({"eval/new_best_model": 1, "eval/timesteps": self.num_timesteps})
        return True


def make_env():
    """Create and wrap the environment."""
    env = TicTacToeSB3Wrapper(opponent_policy=RandomAgent, randomize_position=True)
    env = Monitor(env)
    return env


def train_ppo(
    total_timesteps=100_000,  # Reduced for tic tac toe (simpler game)
    save_path='./artifacts/models/ppo_tictactoe',
    log_path='./logs/',
    checkpoint_freq=10_000,
    eval_freq=5_000,
    n_eval_episodes=50,  # More episodes for evaluation since games are quick
    randomize_position=True,
    wandb_project=WANDB_PROJECT,
    wandb_run_name=None,
    wandb_config=None,
):
    """
    Train a PPO agent on Tic Tac Toe with W&B logging.
    
    Args:
        total_timesteps: Total number of training steps
        save_path: Path to save the trained model
        log_path: Path for logs
        checkpoint_freq: Frequency (in timesteps) to save checkpoints
        eval_freq: Frequency to evaluate the model
        n_eval_episodes: Number of episodes for evaluation
        randomize_position: Whether to randomize training agent position each episode
        wandb_run_name: W&B run name (optional)
        wandb_config: Additional config dict for W&B (optional)
    """
    
    # Create directories
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    
    # Initialize Weights & Biases
    config = {
        "algorithm": "PPO",
        "policy": "MaskedTicTacToePolicy",
        "total_timesteps": total_timesteps,
        "learning_rate": 1e-3,  # Higher learning rate for simpler game
        "n_steps": 512,          # Smaller batch size for tic tac toe
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,  # Lower entropy for tic tac toe
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "randomize_position": randomize_position,
    }
    
    if wandb_config:
        config.update(wandb_config)
    
    wandb.init(
        project=WANDB_PROJECT,
        name=wandb_run_name,
        config=config,
        sync_tensorboard=False,
        monitor_gym=True,
    )
    
    # Create training environment
    print("Creating training environment...")
    print(f"Position randomization: {'ENABLED' if randomize_position else 'DISABLED'}")
    train_env = DummyVecEnv([make_env])
    
    # Create evaluation environment
    print("Creating evaluation environment...")
    eval_env = DummyVecEnv([make_env])
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=save_path,
        name_prefix='ppo_tictactoe_checkpoint'
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        log_path=log_path,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
        callback_on_new_best=WandbBestModelCallback()
    )
    
    wandb_callback = WandbCallback()
    
    # Create PPO model
    print("Initializing PPO model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Simpler network for tic tac toe with custom feature extractor
    policy_kwargs = dict(
        features_extractor_class=DictObservationExtractor,
        net_arch=dict(pi=[64, 64], vf=[64, 64]),  # Smaller network
        activation_fn=torch.nn.ReLU,
    )
    
    model = PPO(
        MaskedTicTacToePolicy,
        train_env,
        verbose=1,
        learning_rate=config["learning_rate"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        n_epochs=config["n_epochs"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_range=config["clip_range"],
        ent_coef=config["ent_coef"],
        tensorboard_log=None,
        device=device,
        policy_kwargs=policy_kwargs
    )
    print('____________MODEL CREATED SUCCESSFULLY______________')
    
    # Log model architecture to W&B
    wandb.watch(model.policy, log="all", log_freq=1000)
    
    print(f"Training on device: {model.device}")
    print(f"Total timesteps: {total_timesteps:,}")
    print("Starting training...")
    
    # Train the model
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback, wandb_callback],
            progress_bar=True
        )
        
        # Save final model
        final_path = os.path.join(save_path, 'ppo_tictactoe_final')
        model.save(final_path)
        print(f"\nTraining complete! Model saved to {final_path}")
        
        # Log final model to W&B
        wandb.save(final_path + ".zip")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        interrupted_path = os.path.join(save_path, 'ppo_tictactoe_interrupted')
        model.save(interrupted_path)
        print(f"Model saved to {interrupted_path}")
        wandb.save(interrupted_path + ".zip")
    
    finally:
        train_env.close()
        eval_env.close()
        wandb.finish()
    
    return model


def test_trained_model(model_path, num_episodes=100, log_to_wandb=False):
    """
    Test a trained model.
    
    Args:
        model_path: Path to the trained model
        num_episodes: Number of episodes to test
        log_to_wandb: Whether to log results to W&B
    """
    print(f"\nTesting model: {model_path}")
    
    if log_to_wandb:
        wandb.init(project=WANDB_PROJECT, name="model_test", job_type="evaluation")
    
    # Load model
    model = PPO.load(model_path)
    
    # Create test environment
    env = make_env()
    
    total_rewards = []
    wins = 0
    losses = 0
    draws = 0
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            
            if done or truncated:
                break
        
        total_rewards.append(episode_reward)
        if episode_reward > 0:
            wins += 1
        elif episode_reward < 0:
            losses += 1
        else:
            draws += 1
        
        if episode % 20 == 0:
            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")
        
        if log_to_wandb:
            wandb.log({
                "test/episode_reward": episode_reward,
                "test/episode": episode + 1,
            })
    
    env.close()
    
    avg_reward = sum(total_rewards) / len(total_rewards)
    win_rate = wins / num_episodes * 100
    loss_rate = losses / num_episodes * 100
    draw_rate = draws / num_episodes * 100
    
    print(f"\n=== Test Results ===")
    print(f"Episodes: {num_episodes}")
    print(f"Average Reward: {avg_reward:.3f}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Loss Rate: {loss_rate:.1f}%")
    print(f"Draw Rate: {draw_rate:.1f}%")
    
    if log_to_wandb:
        wandb.log({
            "test/avg_reward": avg_reward,
            "test/win_rate": win_rate,
            "test/loss_rate": loss_rate,
            "test/draw_rate": draw_rate,
        })
        wandb.finish()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train PPO on Tic Tac Toe')
    parser.add_argument('--train', action='store_true', help='Train a new model')
    parser.add_argument('--test', type=str, help='Test a trained model (provide path)')
    parser.add_argument('--timesteps', type=int, default=100_000, help='Training timesteps')
    parser.add_argument('--save-path', type=str, default='./artifacts/models/ppo_tictactoe', 
                       help='Path to save models')
    parser.add_argument('--no-randomize', action='store_true',
                       help='Disable position randomization during training')
    parser.add_argument('--wandb-project', type=str, default='TicTacToe-PPO',
                       help='Weights & Biases project name')
    parser.add_argument('--wandb-run-name', type=str, default=None,
                       help='Weights & Biases run name')
    parser.add_argument('--wandb-key', type=str, default=None,
                       help='Weights & Biases API key')
    
    args = parser.parse_args()
    
    # Setup W&B login if key provided
    if args.wandb_key:
        wandb.login(key=args.wandb_key)
    
    if args.train:
        # Train model
        train_ppo(
            total_timesteps=args.timesteps,
            save_path=args.save_path,
            randomize_position=not args.no_randomize,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name,
        )
        
        # Test the trained model
        final_model = os.path.join(args.save_path, 'ppo_tictactoe_final')
        if os.path.exists(final_model + '.zip'):
            test_trained_model(final_model, num_episodes=100, log_to_wandb=True)
    
    elif args.test:
        # Test existing model
        test_trained_model(args.test, num_episodes=100, log_to_wandb=True)
    
    else:
        print("Please specify --train or --test <model_path>")
        print("\nExamples:")
        print("  python train_ppo.py --train")
        print("  python train_ppo.py --train --timesteps 50000")
        print("  python train_ppo.py --test ./artifacts/models/ppo_tictactoe/ppo_tictactoe_final")