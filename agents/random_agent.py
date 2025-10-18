from .agent import Agent
import numpy as np

class RandomAgent(Agent):

    def __init__(self, env):
        self.env = env

    def do_action(self):
        if hasattr(self.env,'last'):
            observation, _, _, _, _ = self.env.last()
            mask = observation["action_mask"]
            return self.env.action_space(self.player).sample(mask)
        else:
            obs, _, _, _, _ = self.env.get_current_state()
            action_mask = obs['action_mask']

            # Choose from valid actions
            valid_actions = np.where(action_mask)[0]
            if len(valid_actions) == 0:
                raise ValueError("No valid actions available!")

            action = np.random.choice(valid_actions)
            return int(action)
