"""
LLM-based agent for playing Gin Rummy using Ollama.
"""
from .agent import Agent
import numpy as np
from llm.api import OllamaAPI
from llm.player_handler import LLMPlayerHandler
from llm.validator import ActionValidator


class LLMAgent(Agent):
    """
    Agent that uses a Large Language Model (via Ollama) to make decisions in Gin Rummy.
    """

    def __init__(self, env, model: str = "llama3.2:1b", prompt_name: str = "default_prompt"):
        """
        Initialize LLM Agent for Gin Rummy.
        
        Args:
            env: Gin Rummy game environment
            model: Ollama model name (default: llama3.2:1b - lightweight and fast)
            prompt_name: Name of prompt to use from config/prompt.txt
        """
        super().__init__(env)  # Call parent class __init__
        self.model = model
        self.prompt_name = prompt_name
        
        # Initialize player handler (coordinates API and validation)
        self.player_handler = LLMPlayerHandler(
            config_path="config/prompt.txt",
            model=model,
            fallback_strategy="random"
        )
        
        print(f"[INFO] LLM Agent initialized for Gin Rummy with model: {model}")
        print(f"[INFO] Using prompt: {prompt_name}")

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
    
    def do_action(self):
        """
        Get action from LLM based on current Gin Rummy observation.
        
        Returns:
            Valid action index
        """
        # Get current observation
        obs = self.get_observation()

        # print(f"valid moves for test{obs}")
        
        # Get action mask
        action_mask = obs.get("action_mask")
        
        if action_mask is None:
            raise ValueError("Observation must contain 'action_mask'")
        
        # Check if there are valid actions
        valid_actions = np.where(action_mask)[0]
        if len(valid_actions) == 0:
            raise ValueError("No valid actions available!")
        
        # Get action from LLM player handler
        try:
            print(f"valid moves for test{obs}")
            action = self.player_handler.get_action(obs, self.prompt_name)
            return action
        except Exception as e:
            print(f"[ERROR] LLM Agent failed to get action: {e}")
            # Fallback to random valid action
            print("[INFO] Using random fallback action")
            return np.random.choice(valid_actions)
    
    def get_statistics(self):
        """
        Get agent statistics.
        
        Returns:
            Dictionary with statistics including action validity rate
        """
        return self.player_handler.get_statistics()
    
    def print_statistics(self):
        """Print agent statistics."""
        print(f"\n=== LLM Agent Statistics (Model: {self.model}) ===")
        print(f"Prompt used: {self.prompt_name}")
        self.player_handler.print_statistics()
    
    def reset_statistics(self):
        """Reset agent statistics."""
        self.player_handler.reset_statistics()