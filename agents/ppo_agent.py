import numpy as np
from typing import Optional
from agents import Agent
import torch


class PPOAgent(Agent):
    """
    PPO Agent wrapper for playing Gin Rummy using a trained model.
    """
    
    def __init__(self, model_path: Optional[str] = None, env=None):
        """
        Initialize PPO Agent.
        
        Args:
            model_path: Path to the trained PPO model
            env: GinRummyEnvAPI instance
        """
        self.env = env
        self.player = None
        self.model = None
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load a trained PPO model."""
        from stable_baselines3 import PPO
        self.model = PPO.load(model_path)
        print(f"Loaded PPO model from {model_path}")
    
    # def get_observation(self):
    #     """Get the current observation for this agent."""
    #     obs, _, _, _, _ = self.env.get_current_state()
    #     return obs

    #TEMP
    def get_observation(self):
        """Get the current observation for this agent."""
        # Handle both GinRummyEnvAPI and raw PettingZoo env
        if hasattr(self.env, 'get_current_state'):
            # Using GinRummyEnvAPI wrapper
            obs, _, _, _, _ = self.env.get_current_state()
        else:
            # Using raw PettingZoo environment
            obs, _, _, _, _ = self.env.last()
        return obs
    
    def do_action(self) -> int:
        """
        Get action from PPO model.
        
        Returns:
            Action index
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        obs = self.get_observation()
        
        # Extract observation array and action mask
        obs_array = obs['observation']
        action_mask = obs['action_mask']
        
        # Get action from model
        action, _ = self.model.predict(obs_array, deterministic=True)
        
        # Ensure action is valid according to mask
        if not action_mask[action]:
            # If predicted action is invalid, sample from valid actions
            valid_actions = np.where(action_mask)[0]
            action = np.random.choice(valid_actions)
        
        return int(action)
    
    def train_step(self, obs, action, reward, next_obs, done):
        """
        Optional: For online training/fine-tuning.
        This is a placeholder - actual implementation depends on your training setup.
        """
        pass