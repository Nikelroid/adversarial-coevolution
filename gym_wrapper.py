import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pettingzoo.classic import gin_rummy_v4
from agents import Agent
import random
from hand_scoring import score_gin_rummy_hand


class GinRummySB3Wrapper(gym.Env):
    """
    Wrapper to make PettingZoo Gin Rummy compatible with Stable-Baselines3.
    Converts the multi-agent environment to single-agent by having the opponent play randomly.
    Training agent position is randomized each episode for fair learning.
    """
    
    def __init__(self, opponent_policy, randomize_position=True):
        super().__init__()
        
        self.env = gin_rummy_v4.env(render_mode=None,knock_reward = 0.5, gin_reward = 1, opponents_hand_visible = True)
        self.opponent_policy: Agent = opponent_policy(self.env)
        self.randomize_position = randomize_position
        
        # Get a sample observation to determine spaces
        self.env.reset()
        agent = self.env.agents[0]
        sample_obs, _, _, _, _ = self.env.last()
        
        # Define observation and action spaces
        obs_shape = sample_obs['observation'].shape
        action_mask_shape = sample_obs['action_mask'].shape
        
        self.observation_space = spaces.Dict({
            'observation': spaces.Box(
                low=0, high=1, shape=obs_shape, dtype=np.int8
            ),
            'action_mask': spaces.Box(
                low=0, high=1, shape=action_mask_shape, dtype=np.int8
            )
        })

        # Action space is discrete (number of possible actions)
        action_space_size = self.env.action_space(agent).n
        self.action_space = spaces.Discrete(action_space_size)
        
        self.agents = ['player_0', 'player_1']
        
        # These will be set in reset()
        self.training_agent = None
        self.opponent_agent = None
        
        self.env.reset()
    
    def reset(self, seed=None, options=None):
        """Reset the environment."""
        if seed is not None:
            self.env.reset(seed=seed)
        else:
            self.env.reset()

        self.TURNS_LIMIT = 2
        self.turn_num = 0
        self.last_score = -1
        print("="*100)

        # Randomly assign training agent position each episode
        if self.randomize_position and random.random() < 2:
            self.training_agent = 'player_1'
            self.opponent_agent = 'player_0'
            self.opponent_policy.set_player('player_0')
            print('2ND TURN')
        else:
            self.training_agent = 'player_0'
            self.opponent_agent = 'player_1'
            self.opponent_policy.set_player('player_1')
            print('1ST TURN')
        
        # Play until it's the training agent's turn
        # while True:
        #     agent = self.env.agent_selection
        #     if agent == self.training_agent:
        #         obs, _, _, _, _ = self.env.last()
        #         return obs, {}
        #     else:
        #         # Opponent plays
        #         self._opponent_step()

        while True:
            agent = self.env.agent_selection
            
            if agent == self.training_agent:
                # It's our turn, return observation
                obs, _, _, _, _ = self.env.last()
                return obs, {}
            else:
                # Opponent's turn, make them play
                self._opponent_step()
                
                # Check if game ended during opponent's move
                _, _, term, trunc, _ = self.env.last()
                if term or trunc:
                    # Game ended before training agent could move, reset again
                    self.env.reset(seed=seed)
                    continue
    
    
    def _opponent_step(self):
        """Have the opponent take an action."""
        obs, reward, termination, truncation, info = self.env.last()
        
        if termination or truncation:
            self.env.step(None)
        else:
            action = self.opponent_policy.do_action()
            self.env.step(action)
    
    def step(self, action):
        """Take a step in the environment."""
        # Training agent takes action
        obs, reward, termination, truncation, info = self.env.last()

        # Check if action is valid
        if not termination and not truncation:
            mask = obs['action_mask']
            if not mask[action]:
                
                # Invalid action - give negative reward and sample valid action
                print("[Warning] : Invalid Action Choosed")
                reward = -1.0
                valid_actions = np.where(mask)[0]
                action = np.random.choice(valid_actions)

        print(f'Action for this hand: {action} | For Move: {self.turn_num}')
        
        self.env.step(action)

        player_hand = obs['observation'][0]
        if  sum(player_hand) == 10:
            if  self.last_score == -1:
                self.last_score = score_gin_rummy_hand(player_hand)
            else:
                r = score_gin_rummy_hand(player_hand)
                reward += r - self.last_score
                self.last_score = r

        if self.turn_num > self.TURNS_LIMIT:
            truncation = True
        self.turn_num += 1
        

        # Check if game ended
        if termination or truncation:
            next_obs, _, _, _, _ = self.env.last()
            return next_obs, reward, True, False, info
        
        # Opponent's turn(s) until it's training agent's turn again
        while True:
            agent = self.env.agent_selection
            
            if agent == self.training_agent:
                obs, reward, termination, truncation, info = self.env.last()
                done = termination or truncation  
                return obs, reward, done, False, info
            else:
                self._opponent_step()
                
                # Check if game ended during opponent's turn
                _, _, termination, truncation, _ = self.env.last()
                if termination or truncation:
                    obs, reward, _, _, info = self.env.last()
                    return obs, reward, True, False, info
    
    def render(self, mode='human'):
        """Render the environment."""
        return self.env.render()
    
    def close(self):
        """Close the environment."""
        self.env.close()