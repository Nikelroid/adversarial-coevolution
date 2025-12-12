from .agent import Agent
import numpy as np
import os
from llm.api import OllamaAPI, DistributedOllamaAPI
from llm.player_handler import LLMPlayerHandler
from llm.validator import ActionValidator
from utils.config import get_config

CONFIG = get_config()
DEFAULT_MODEL = CONFIG.get("models", {}).get("default_evaluator", "llama3.2:1b")
MASTER_URL_CONF = CONFIG.get("distributed", {}).get("master", {}).get("url", "http://localhost:8000")


class LLMAgent(Agent):
    """
    Agent that uses a Large Language Model (via Ollama) to make decisions in Gin Rummy.
    """

    def __init__(self, env, model: str = None, prompt_name: str = "default_prompt", distributed: bool = False, master_url: str = None):
        super().__init__(env)
        if model is None:
            model = DEFAULT_MODEL
        if master_url is None:
            master_url = MASTER_URL_CONF
            
        self.model = model
        self.prompt_name = prompt_name
        self.config_path = "config/prompt.txt"
        
        # Initialize API client
        if distributed:
            self.api_client = DistributedOllamaAPI(master_url=master_url)
        else:
            self.api_client = OllamaAPI(model=model)
            
        # We manually hook the handler to use our custom API client if possible
        # But LLMPlayerHandler creates its own. We should refactor handler or just use API directly here.
        # For simplicity, let's inject validatior and use our API client directly in do_action.
        self.validator = ActionValidator(fallback_strategy="random")
        self.prompts = self._load_prompts(self.config_path)
        
        print(f"[INFO] LLM Agent initialized (Distributed={distributed})")

    def _load_prompts(self, config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return {'default_prompt': f.read().strip()}
        except:
            return {'default_prompt': "You are playing Gin Rummy. Pick the best action."}

    def save_prompt(self, new_prompt):
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                f.write(new_prompt)
            self.prompts['default_prompt'] = new_prompt
            print("[INFO] Saved enhanced prompt to file.")
        except Exception as e:
            print(f"[ERROR] Failed to save prompt: {e}")

    def get_observation(self):
        if hasattr(self.env, 'get_current_state'):
            obs, _, _, _, _ = self.env.get_current_state()
        else:
            obs, _, _, _, _ = self.env.last()
        return obs 
    
    def do_action(self):
        obs = self.get_observation()
        action_mask = obs.get("action_mask")
        
        if action_mask is None: raise ValueError("No action mask")
        
        valid_actions = np.where(action_mask)[0]
        if len(valid_actions) == 0: raise ValueError("No valid actions")
        
        prompt = self.prompts.get(self.prompt_name, self.prompts['default_prompt'])
        
        # Use our API client (Distributed or Local)
        action = self.api_client.get_action(prompt, obs, valid_actions)
        
        # Validation
        final_action = self.validator.validate_action(action, action_mask)
        return final_action

    def on_game_end(self, reward):
        """
        Called by training loop when game ends.
        """
        if isinstance(self.api_client, DistributedOllamaAPI) and reward < 0:
            print(f"[Enhancer] Agent lost (Reward {reward}). Triggering Prompt Enhancement...")
            current_prompt = self.prompts.get(self.prompt_name, "")
            stats = self.validator.get_statistics()
            
            # Request new prompt
            new_prompt = self.api_client.enhance_prompt(current_prompt, stats)
            
            # Save it
            if new_prompt and len(new_prompt) > 10:
                self.save_prompt(new_prompt)
    
    def get_statistics(self):
        return self.validator.get_statistics()
    
    def print_statistics(self):
        self.validator.print_statistics()
    
    def reset_statistics(self):
        self.validator.reset_statistics()