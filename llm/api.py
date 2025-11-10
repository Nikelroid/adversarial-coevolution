"""
Ollama API wrapper for LLM agent interactions.
"""
import requests
import json
import re  # ### MODIFIED ###
from typing import Dict, Any, Optional, List, Tuple  # ### MODIFIED ###
import numpy as np  # ### MODIFIED ###
import logging


# ### NEW ###
# Constants for card and action translation
SUITS = ["Spades", "Hearts", "Diamonds", "Clubs"]
RANKS = ["Ace", "2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King"]
# 52 cards, ordered exactly as in the observation space
CARD_NAMES = [f"{rank} of {suit}" for suit in SUITS for rank in RANKS]


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
        logging.basicConfig(filename='app.log',level=logging.INFO, format='%(message)s')
        self.model = model
        self.base_url = base_url
        self.api_endpoint = f"{self.base_url}/api/generate"

        # ### NEW ### Build action translation maps
        self.action_to_string, self.string_to_action = self._build_action_maps()
        # This map will be populated *during* prompt formatting
        self.current_action_map: Dict[str, int] = {}

    # ### NEW ###
    def _build_action_maps(self) -> Tuple[Dict[int, str], Dict[str, int]]:
        """Builds dictionaries to map action IDs to/from human-readable strings."""
        action_to_string = {
            # 0: "score player 0", # Not a decision
            # 1: "score player 1", # Not a decision
            2: "draw from stock",
            3: "pick top card from discard pile",  # This is a base string
            4: "declare dead hand",
            5: "declare Gin",
        }

        # Actions 6-57: Discard a card
        for i in range(52):
            action_id = 6 + i
            action_to_string[action_id] = f"discard {CARD_NAMES[i]}"

        # Actions 58-109: Knock
        for i in range(52):
            action_id = 58 + i
            action_to_string[action_id] = f"knock with {CARD_NAMES[i]}"

        # Create the reverse map
        string_to_action = {s: i for i, s in action_to_string.items()}
        # Add special case for action 3, as its string is dynamic
        string_to_action["pick top card from discard pile"] = 3

        return action_to_string, string_to_action
    
    # ### NEW ###
    def _sort_cards(self, card_name: str) -> Tuple[int, int]:
        """Helper function to sort cards by suit, then rank for the LLM."""
        try:
            rank, suit = card_name.split(" of ")
            suit_index = SUITS.index(suit)
            rank_index = RANKS.index(rank)
            return (suit_index, rank_index)
        except ValueError:
            return (99, 99) # Put invalid cards at the end

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 3072) -> str:
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
            response = requests.post(self.api_endpoint, json=payload, timeout=1000)
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
        logging.info ('='*80)
        logging.info ('')
        logging.info ('                    NEW MOVE')
        logging.info ('')
        logging.info ('='*80)
        logging.info ('')
        logging.info('                     Fullprompt: ')
        logging.info(full_prompt)

        
        # Get LLM response
        # Low temperature to reduce creativity and stick to the requested format
        response = self.generate(full_prompt, temperature=0.7, max_tokens=3072) 
        logging.info ('+'*80)
        logging.info ('')
        logging.info ('                    Response Recieved')
        logging.info ('')
        logging.info ('+'*80)
        logging.info ('')
        logging.info('                     Response: ')
        logging.info(response)
        # Parse action from response
        # ### MODIFIED ### Pass valid_actions for context, though parser uses class map
        action = self._parse_action(response, valid_actions)

        return action

    # ### MODIFIED ###
    def _format_prompt(self, base_prompt: str, observation: Dict[str, Any], valid_actions: list) -> str:
        """
        Format the prompt with HUMAN-READABLE observation and valid actions.
        
        Args:
            base_prompt: Base prompt template
            observation: Current game observation
            valid_actions: List of valid action indices
            
        Returns:
            Formatted prompt string
        """
        # Extract observation board
        board = observation.get('observation')
        if board is None:
            raise ValueError("Observation dictionary missing 'observation' key")

        # Convert observation board to human-readable string
        obs_str, top_discard_name = self._obs_to_string(board)

        # ### NEW ###
        # Convert valid action IDs to human-readable strings
        self.current_action_map.clear()  # Clear map from previous turn
        human_valid_actions = []
        for action_id in valid_actions:
            action_string = self.action_to_string.get(action_id)
            
            if not action_string:
                continue # Skip actions we don't want the LLM to choose (like 'score')

            # Special case for action 3: dynamically add card name
            if action_id == 3:
                action_string = f"pick top card from discard pile ({top_discard_name})"

            human_valid_actions.append(action_string)
            self.current_action_map[action_string] = action_id # Store for parser

        valid_actions_str = "\n".join(f"- {s}" for s in human_valid_actions)

        # Combine everything into the new prompt format
        full_prompt = base_prompt.format(
            game_state=obs_str,
            valid_actions=valid_actions_str
        )

        return full_prompt

    # ### REPLACED ###
    def _obs_to_string(self, board: np.ndarray) -> Tuple[str, str]:
        """
        Convert Gin Rummy 4x52 observation to readable string.
        
        Args:
            board: Gin Rummy observation array (expects 4x52)
            
        Returns:
            Tuple[str, str]: 
            1. String representation of the game state
            2. The name of the top discard card (or "None")
        """
        
        # Handle environments that batch observations (e.g., (1, 4, 52))
        # This matches your original code's `board = board[0]` artifact
        if board.ndim == 3 and board.shape[0] == 1:
            board = board[0]

        # Your observation space: 0: Hand, 1: Top Discard, 2: Other Discard, 3: Unknown
        if board.shape[0] != 4:
            print(f"[WARNING] Expected 4-row observation, got {board.shape}. Parsing may fail.")

        def get_cards(row_index):
            indices = np.where(board[row_index] == 1)[0]
            return [CARD_NAMES[i] for i in indices]

        # Row 0: Player's Hand
        hand_cards = get_cards(0)
        # Sort cards to help LLM see melds
        hand_cards_sorted = sorted(hand_cards, key=self._sort_cards)
        hand_str = "Your Hand: " + (", ".join(hand_cards_sorted) if hand_cards else "Empty")

        # Row 1: Top of Discard Pile
        top_discard_cards = get_cards(1)
        top_discard_name = top_discard_cards[0] if top_discard_cards else "None"
        top_discard_str = f"Top of Discard Pile: {top_discard_name}"

        # Row 2: Other Cards in Discard
        other_discard_cards = get_cards(2)
        other_discard_str = "Other Discarded Cards: " + (", ".join(other_discard_cards) if other_discard_cards else "None")

        # Row 3: Unknown cards
        unknown_count = int(np.sum(board[3]))
        unknown_str = f"Unknown Cards (in stock pile or opponent's hand): {unknown_count}"

        full_obs_str = "\n".join([hand_str, top_discard_str, other_discard_str, unknown_str])
        
        return full_obs_str, top_discard_name

    # ### REPLACED ###
    def _parse_action(self, response: str, valid_actions: list) -> Optional[int]:
        """
        Parse HUMAN-READABLE action string from LLM response.
        
        Args:
            response: LLM response text (which includes thinking + final action)
            valid_actions: List of valid action indices (unused, we use class map)
            
        Returns:
            Parsed action index or None
        """
        
        # The prompt asks the LLM to put the action on the last line.
        lines = response.strip().split('\n')
        last_line = lines[-1].strip()

        # 1. Try to match the last line exactly. This is the ideal case.
        if last_line in self.current_action_map:
            return self.current_action_map[last_line]

        # 2. Fallback: Search the *entire* response (from bottom up)
        #    in case the LLM included it in its "thinking" but failed to format.
        for line in reversed(lines):
            cleaned_line = line.strip()
            if cleaned_line in self.current_action_map:
                print(f"[INFO] Parsed action '{cleaned_line}' from LLM thinking (last line was wrong).")
                return self.current_action_map[cleaned_line]
        
        # 3. Fallback: Check for partial matches in the response
        #    (e.g., if LLM says "I will discard 10 of Spades" instead of just the string)
        for action_string, action_id in self.current_action_map.items():
            if action_string in response:
                 print(f"[INFO] Parsed partial action '{action_string}' from response.")
                 return action_id

        # If no valid action found, return None
        print(f"[WARNING] Could not parse valid action string from response: {response[-200:]}...")
        print(f"       Valid options were: {list(self.current_action_map.keys())}")
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