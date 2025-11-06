"""
Action validator for LLM agent decisions.
"""
import numpy as np
from typing import Optional, List


class ActionValidator:
    """
    Validates actions from LLM against the game's action mask.
    """
    
    def __init__(self, fallback_strategy: str = "random"):
        """
        Initialize action validator.
        
        Args:
            fallback_strategy: Strategy when LLM returns invalid action
                              ("random", "first", "last")
        """
        self.fallback_strategy = fallback_strategy
        self.invalid_action_count = 0
        self.total_action_count = 0
        
    def validate_action(self, action: Optional[int], action_mask: np.ndarray) -> int:
        """
        Validate and correct action if needed.
        
        Args:
            action: Action proposed by LLM (can be None)
            action_mask: Boolean array of valid actions
            
        Returns:
            Valid action index
        """
        self.total_action_count += 1
        
        # Get list of valid actions
        valid_actions = np.where(action_mask)[0]
        
        if len(valid_actions) == 0:
            raise ValueError("No valid actions available in action mask!")
        
        # Check if LLM action is valid
        if action is not None and 0 <= action < len(action_mask) and action_mask[action]:
            return int(action)
        
        # LLM returned invalid action, use fallback
        self.invalid_action_count += 1
        print(f"[WARNING] Invalid action from LLM: {action}. Using fallback strategy: {self.fallback_strategy}")
        
        return self._get_fallback_action(valid_actions)
    
    def _get_fallback_action(self, valid_actions: np.ndarray) -> int:
        """
        Get fallback action when LLM fails.
        
        Args:
            valid_actions: Array of valid action indices
            
        Returns:
            Fallback action index
        """
        if self.fallback_strategy == "random":
            return np.random.choice(valid_actions)
        elif self.fallback_strategy == "first":
            return valid_actions[0]
        elif self.fallback_strategy == "last":
            return valid_actions[-1]
        else:
            # Default to random
            return np.random.choice(valid_actions)
    
    def get_valid_actions(self, action_mask: np.ndarray) -> List[int]:
        """
        Get list of valid action indices from mask.
        
        Args:
            action_mask: Boolean array of valid actions
            
        Returns:
            List of valid action indices
        """
        return np.where(action_mask)[0].tolist()
    
    def get_statistics(self) -> dict:
        """
        Get validation statistics.
        
        Returns:
            Dictionary with validation stats
        """
        if self.total_action_count == 0:
            return {
                "total_actions": 0,
                "invalid_actions": 0,
                "validity_rate": 0.0
            }
        
        validity_rate = 1.0 - (self.invalid_action_count / self.total_action_count)
        
        return {
            "total_actions": self.total_action_count,
            "invalid_actions": self.invalid_action_count,
            "validity_rate": validity_rate
        }
    
    def reset_statistics(self):
        """Reset validation statistics."""
        self.invalid_action_count = 0
        self.total_action_count = 0
    
    def print_statistics(self):
        """Print validation statistics."""
        stats = self.get_statistics()
        print(f"\n=== Action Validation Statistics ===")
        print(f"Total Actions: {stats['total_actions']}")
        print(f"Invalid Actions: {stats['invalid_actions']}")
        print(f"Validity Rate: {stats['validity_rate']:.2%}")
        print("=" * 35)