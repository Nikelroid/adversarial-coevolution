import numpy as np
from .agent import Agent

class RandomAgent(Agent):
    def __init__(self, env):
        self.env = env
        self.player = None  # optional: set later by tournament loop

    def do_action(self):
        """
        Selects a random valid action based on the current observation and action mask.
        """
        # Get observation in the same way as PPOAgent
        obs, _, _, _, _ = self.env.get_current_state()
        obs_array = obs['observation']
        action_mask = obs['action_mask']

        # Choose from valid actions
        valid_actions = np.where(action_mask)[0]
        if len(valid_actions) == 0:
            raise ValueError("No valid actions available!")

        action = np.random.choice(valid_actions)
        return int(action)