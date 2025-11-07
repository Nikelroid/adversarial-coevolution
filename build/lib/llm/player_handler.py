"""
Player handler for coordinating LLM API and action validation.
"""
import yaml
import os
from typing import Dict, Any
from .api import OllamaAPI
from .validator import ActionValidator


class LLMPlayerHandler:
    """
    Handles LLM player logic, coordinating between API calls and action validation.
    """
    
    def __init__(self, 
                 config_path: str = "config/prompt.txt",
                 model: str = "llama3.2:1b",
                 fallback_strategy: str = "random"):
        """
        Initialize LLM player handler.
        
        Args:
            config_path: Path to prompts configuration file
            model: Ollama model name
            fallback_strategy: Fallback strategy for invalid actions
        """
        self.api = OllamaAPI(model=model)
        self.validator = ActionValidator(fallback_strategy=fallback_strategy)
        self.prompts = self._load_prompts(config_path)
        
        # Check Ollama connection
        if not self.api.check_connection():
            print("[WARNING] Ollama connection failed. Make sure Ollama is running.")
    
    def _load_prompts(self, config_path: str) -> Dict[str, str]:
        """
        Load prompts from YAML configuration file.
        
        Args:
            config_path: Path to prompts YAML file
            
        Returns:
            Dictionary of prompts
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                prompt_text = f.read().strip()
                return {'default_prompt': prompt_text}
        except FileNotFoundError:
            print(f"[WARNING] Prompt file not found at {config_path}. Using default prompt.")
            return {'default_prompt': "You are playing Tic Tac Toe. Choose the best move."}
        except Exception as e:
            print(f"[ERROR] Error loading prompt file: {e}")
            return {'default_prompt': "You are playing Tic Tac Toe. Choose the best move."}
    
    def get_action(self, observation: Dict[str, Any], prompt_name: str = "default_prompt") -> int:
        """
        Get action from LLM based on observation.
        
        Args:
            observation: Game observation dictionary
            prompt_name: Name of prompt to use from config
            
        Returns:
            Valid action index
        """
        # Get action mask
        action_mask = observation.get('action_mask')
        if action_mask is None:
            raise ValueError("Observation must contain 'action_mask'")
        
        # Get valid actions
        
        valid_actions = self.validator.get_valid_actions(action_mask)
        
        if len(valid_actions) == 0:
            raise ValueError("No valid actions available!")
        
        # Get prompt
        prompt = self.prompts.get(prompt_name, self.prompts.get('default_prompt', 
                                                                 "You are playing Gin_Rummy. Choose the best move from."))
        
        # Get action from LLM
        try:
            llm_action = self.api.get_action(prompt, observation, valid_actions)
        except Exception as e:
            print(f"[ERROR] LLM API call failed: {e}")
            llm_action = None
        
        # Validate and correct if needed
        valid_action = self.validator.validate_action(llm_action, action_mask)
        
        return valid_action
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get handler statistics.
        
        Returns:
            Dictionary with statistics
        """
        return self.validator.get_statistics()
    
    def print_statistics(self):
        """Print handler statistics."""
        self.validator.print_statistics()
    
    def reset_statistics(self):
        """Reset handler statistics."""
        self.validator.reset_statistics()