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
        """Pick the highest-probability LEGAL action from the policy's action distribution
        -- never a random fallback.

        We read the policy's per-action probabilities and take the argmax over the legal
        actions only, instead of calling predict() and substituting a RANDOM legal action
        whenever the predicted one happens to be illegal. The random fallback silently makes
        the agent play a random move in those states, which understates a hero's true strength
        in evaluation and weakens pool/self opponents during training. Masked-argmax guarantees
        the agent always plays its best legal move, so eval win-rates are faithful and the
        opponents are genuinely as strong as the model.

        This is robust whether or not the policy masks internally: our masked policies already
        zero illegal logits, and we re-apply the legal mask here so any non-masked model is
        handled too.
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        from agents.action_utils import masked_argmax
        return masked_argmax(self.model, self.get_observation())
    
    def train_step(self, obs, action, reward, next_obs, done):
        """
        Optional: For online training/fine-tuning.
        This is a placeholder - actual implementation depends on your training setup.
        """
        pass