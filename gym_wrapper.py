import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pettingzoo.classic import tictactoe_v3
from agents import Agent
import random


class TicTacToeSB3Wrapper(gym.Env):
    """
    Wrapper to make PettingZoo Tic Tac Toe compatible with Stable-Baselines3.
    Supports self-play by allowing opponent to be a PPO agent.
    """
    
    def __init__(self, opponent_policy, opponent_kwargs=None, randomize_position=True):
        """
        Args:
            opponent_policy: Agent class (e.g., RandomAgent, PPOAgent)
            opponent_kwargs: Dict of kwargs to pass to opponent_policy constructor
            randomize_position: Whether to randomize player positions each episode
        """
        super().__init__()
        
        # Initialize Tic Tac Toe environment
        self.env = tictactoe_v3.env(render_mode=None)
        
        # Initialize opponent with kwargs
        if opponent_kwargs is None:
            opponent_kwargs = {}
        
        # Add env to kwargs if not present
        if 'env' not in opponent_kwargs:
            opponent_kwargs['env'] = self.env
        
        self.opponent_policy: Agent = opponent_policy(**opponent_kwargs)
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

        action_space_size = self.env.action_space(agent).n
        self.action_space = spaces.Discrete(action_space_size)
        
        # Tic Tac Toe uses 'player_1' and 'player_2'
        self.agents = ['player_1', 'player_2']
        
        self.training_agent = None
        self.opponent_agent = None
        
        self.env.reset()
    
    def reset(self, seed=None, options=None):
        """Reset the environment."""
        if seed is not None:
            self.env.reset(seed=seed)
        else:
            self.env.reset()

        # Randomly assign training agent position each episode
        if self.randomize_position and random.random() < 0.5:
            self.training_agent = 'player_2'
            self.opponent_agent = 'player_1'
            self.opponent_policy.set_player('player_1')
            print('PLAYING AS O (2ND TURN)')
        else:
            self.training_agent = 'player_1'
            self.opponent_agent = 'player_2'
            self.opponent_policy.set_player('player_2')
            print('PLAYING AS X (1ST TURN)')
        
        # Play until it's the training agent's turn
        while True:
            agent = self.env.agent_selection
            if agent == self.training_agent:
                obs, _, _, _, _ = self.env.last()
                return obs, {}
            else:
                self._opponent_step()
    
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
        obs, reward, termination, truncation, info = self.env.last()

        # Check if action is valid
        if not termination and not truncation:
            mask = obs['action_mask']
            if not mask[action]:
                print(f"[Warning] Invalid action {action} chosen!")
                reward = -10.0
                valid_actions = np.where(mask)[0]
                action = np.random.choice(valid_actions)
        
        self.env.step(action)

        if termination or truncation:
            next_obs, _, _, _, _ = self.env.last()
            return next_obs, reward, True, False, info
        
        # Opponent's turn
        while True:
            agent = self.env.agent_selection
            
            if agent == self.training_agent:
                obs, reward, termination, truncation, info = self.env.last()
                done = termination or truncation  
                return obs, reward, done, False, info
            else:
                self._opponent_step()
                
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