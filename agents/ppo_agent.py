import numpy as np
from typing import Optional
from agents import Agent
import torch
from stable_baselines3 import PPO

class PPOAgent(Agent):
    """
    PPO Agent wrapper for playing Gin Rummy using a trained model.
    Can be initialized with a path (for loading) or a model object (for caching).
    """
    
    # --- MODIFY __init__ ---
    def __init__(self, 
                 env, 
                 model_path: Optional[str] = "./artifacts/models/ppo_gin_rummy/ppo_gin_rummy_final_knock", 
                 model: Optional[PPO] = None 
                 ):
        """
        Initialize PPO Agent.
        
        Args:
            model_path: Path to the trained PPO model (if 'model' is not provided)
            env: GinRummyEnvAPI instance
            model: An already-loaded PPO model object (preferred)
        """
        self.env = env
        self.player = None
        self.model = model  # Assign pre-loaded model first
        
        if self.model is None and model_path is not None:
            # Fallback: load from path if no model was given
            self.load_model(model_path)
        elif self.model is None:
            # Error: we were given neither
            raise ValueError("PPOAgent must be initialized with either a 'model' object or a 'model_path'.")
        # else:
            # print("PPOAgent initialized with pre-loaded model.") # (Optional debug)
    
    def load_model(self, model_path: str):
        """Load a trained PPO model."""
        # from stable_baselines3 import PPO # (Already imported at top)
        self.model = PPO.load(model_path)
        print(f"Loaded PPO model from {model_path}")
    
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
        
        # Get action from model using the *entire* observation dictionary
        action, _ = self.model.predict(obs, deterministic=True)
        
        # Extract action mask *after* prediction to validate the action
        action_mask = obs['action_mask']
        
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