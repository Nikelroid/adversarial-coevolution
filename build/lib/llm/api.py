"""
Ollama API wrapper for LLM agent interactions.
"""
import requests
import json
from typing import Dict, Any, Optional


class OllamaAPI:
    """
    Wrapper for Ollama API to generate actions based on game observations.
    """
    
    def __init__(self, model: str = "llama3.2:1b", base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama API client.
        
        Args:
            model: Name of the Ollama model to use (default: llama3.2:1b - lightweight)
            base_url: Base URL for Ollama API (default: http://localhost:11434)
        """
        self.model = model
        self.base_url = base_url
        self.api_endpoint = f"{base_url}/api/generate"
        
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 100) -> str:
        """
        Generate text using Ollama API.
        
        Args:
            prompt: Input prompt for the model
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        try:
            response = requests.post(self.api_endpoint, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Ollama API request failed: {e}")
            raise
        except Exception as e:
            print(f"[ERROR] Unexpected error in Ollama API: {e}")
            raise
    
    def get_action(self, prompt: str, observation: Dict[str, Any], valid_actions: list) -> Optional[int]:
        """
        Get action from LLM based on observation and valid actions.
        
        Args:
            prompt: Base prompt template
            observation: Current game observation
            valid_actions: List of valid action indices
            
        Returns:
            Selected action index or None if parsing fails
        """
        # Format the full prompt with observation and valid actions
        full_prompt = self._format_prompt(prompt, observation, valid_actions)
        print('fullprompt: ',full_prompt)
        # Get LLM response
        response = self.generate(full_prompt, temperature=0.3, max_tokens=3072)
        print(response)
        # Parse action from response
        action = self._parse_action(response, valid_actions)
        
        return action
    
    def _format_prompt(self, base_prompt: str, observation: Dict[str, Any], valid_actions: list) -> str:
        """
        Format the prompt with observation and valid actions.
        
        Args:
            base_prompt: Base prompt template
            observation: Current game observation
            valid_actions: List of valid action indices
            
        Returns:
            Formatted prompt string
        """
        # Extract observation board if available
        obs_str = ""
        if 'observation' in observation:
            board = observation['observation']
            obs_str = self._board_to_string(board)
        
        # Format valid actions
        valid_actions_str = ", ".join(map(str, valid_actions))
        
        # Combine everything
        full_prompt = f"""{base_prompt}

Current Board State:
{obs_str}

Valid Actions (positions): {valid_actions_str}

Your action (respond with ONLY a single number from the valid actions):"""
        
        return full_prompt
    
    def _board_to_string(self, board) -> str:
        board = board[0]
        """
        Convert Gin Rummy observation to readable string.
        
        Args:
            board: Gin Rummy observation array
            
        Returns:
            String representation of the game state
        """
        try:
            import numpy as np
            if isinstance(board, np.ndarray):
                # Gin Rummy observation is typically a flattened array
                # Try to extract meaningful information
                obs_str = f"Observation vector (length {len(board)}): "
                obs_str += f"[{', '.join(f'{x:.2f}' for x in board[:10])}..."
                if len(board) > 10:
                    obs_str += f"...{', '.join(f'{x:.2f}' for x in board[-5:])}]"
                return obs_str
        except Exception as e:
            print(f"[WARNING] Could not parse observation: {e}")
        
        return str(board)
    
    def _parse_action(self, response: str, valid_actions: list) -> Optional[int]:
        """
        Parse action number from LLM response.
        
        Args:
            response: LLM response text
            valid_actions: List of valid action indices
            
        Returns:
            Parsed action index or None
        """
        # Try to extract a number from the response
        import re
        
        # Find all numbers in the response
        # numbers = re.findall(r'\b\d+\b', response)
        match = re.search(r'\\boxed\{(\d+)\}', response)
        if match:  final_answer = match.group(1)
        
        # Try each number to see if it's a valid action
        return int(final_answer)
        # for num_str in numbers:
        #     try:
        #         action = int(num_str)
        #         if action in valid_actions:
        #             return action
        #     except ValueError:
        #         continue
        
        # If no valid action found, return None
        print(f"[WARNING] Could not parse valid action from response: {response}")
        return None
    
    def check_connection(self) -> bool:
        """
        Check if Ollama server is running and model is available.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            
            if self.model in model_names or any(self.model in name for name in model_names):
                print(f"[INFO] Ollama connection successful. Model '{self.model}' is available.")
                return True
            else:
                print(f"[WARNING] Model '{self.model}' not found. Available models: {model_names}")
                print(f"[INFO] Run 'ollama pull {self.model}' to download the model.")
                return False
        except Exception as e:
            print(f"[ERROR] Cannot connect to Ollama at {self.base_url}: {e}")
            print("[INFO] Make sure Ollama is running (run 'ollama serve' in terminal)")
            return False