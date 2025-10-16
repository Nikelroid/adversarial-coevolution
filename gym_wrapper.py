import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pettingzoo.classic import tictactoe_v3
from agents import Agent
import random


class TicTacToeSB3Wrapper(gym.Env):
    """
    Wrapper to make PettingZoo Tic Tac Toe compatible with Stable-Baselines3.
    Converts the multi-agent environment to single-agent by having the opponent play randomly.
    Training agent position is randomized each episode for fair learning.
    """
    
    def __init__(self, opponent_policy, randomize_position=True, curriculum_manager=None):
        super().__init__()
        
        self.env = tictactoe_v3.env(render_mode=None)
        self.opponent_policy_class = opponent_policy  # Store class, not instance
        self.opponent_policy = None  # Will be created in reset()
        self.randomize_position = randomize_position
        
        # Curriculum learning support
        self.curriculum_manager = curriculum_manager
        self.current_model = None  # Reference to training model
        self.current_opponent_type = 'random'  # Track opponent type
        
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

        # Action space is discrete (9 positions for tic tac toe)
        action_space_size = self.env.action_space(agent).n
        self.action_space = spaces.Discrete(action_space_size)
        
        self.agents = ['player_1', 'player_2']
        
        # These will be set in reset()
        self.training_agent = None
        self.opponent_agent = None
        
        self.env.reset()

    def set_current_model(self, model):
        """Set reference to current training model for self-play"""
        self.current_model = model
    
    def reset(self, seed=None, options=None):
        """Reset the environment."""
        if seed is not None:
            self.env.reset(seed=seed)
        else:
            self.env.reset()


            # SELECT OPPONENT BASED ON CURRICULUM
        if self.curriculum_manager is not None:
            opponent_type = self.curriculum_manager.get_opponent_type()
            self.current_opponent_type = opponent_type
            
            if opponent_type == 'random':
                self.opponent_policy = self.opponent_policy_class(self.env)
            
            elif opponent_type == 'pool':
                # Load opponent from policy pool
                from agents.ppo_agent import PPOAgent
                policy_path = self.curriculum_manager.sample_policy_path(recent_n=10)
                self.opponent_policy = PPOAgent(model_path=policy_path, env=self.env)
            
            elif opponent_type == 'self':
                # Use frozen copy of current model
                from agents.ppo_agent import PPOAgent
                
                if self.current_model is None:
                    print("[WARNING] Self-play requested but current_model not set! Using random.")
                    from agents.random_agent import RandomAgent
                    self.opponent_policy = RandomAgent(self.env)
                else:
                    print(f"[Curriculum] Using current model for self-play")
                    # Create PPO agent that shares the current model
                    frozen_agent = PPOAgent(model_path=None, env=self.env)
                    frozen_agent.model = self.current_model  # Share the model reference
                    self.opponent_policy = frozen_agent
            
            else:
                # Fallback
                self.opponent_policy = self.opponent_policy_class(self.env)
        else:
            # No curriculum - use default opponent
            self.opponent_policy = self.opponent_policy_class(self.env)

        # Randomly assign training agent position each episode
        if self.randomize_position and random.random() < 0.5:
            self.training_agent = 'player_2'
            self.opponent_agent = 'player_1'
            self.opponent_policy.set_player('player_1')
        else:
            self.training_agent = 'player_1'
            self.opponent_agent = 'player_2'
            self.opponent_policy.set_player('player_2')
        
        # Play until it's the training agent's turn
        while True:
            agent = self.env.agent_selection
            if agent == self.training_agent:
                obs, _, _, _, _ = self.env.last()
                return obs, {}
            else:
                # Opponent plays
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
        # Training agent takes action
        obs, reward, termination, truncation, info = self.env.last()

        # Check if action is valid
        if not termination and not truncation:
            mask = obs['action_mask']
            if not mask[action]:
                # Invalid action - give negative reward and sample valid action
                print(f"[Warning] : Invalid Action {action} chosen")
                reward = -10.0  # Stronger penalty for invalid moves in tic tac toe
                valid_actions = np.where(mask)[0]
                if len(valid_actions) > 0:
                    action = np.random.choice(valid_actions)
                else:
                    # No valid actions (shouldn't happen)
                    action = None

        #print(f'Action: {action} (position {action if action is not None else "None"})')
        
        self.env.step(action)

        if self.curriculum_manager is not None:
            self.curriculum_manager.update_steps(1)

        # Check if game ended
        if termination or truncation:
            next_obs, final_reward, _, _, info = self.env.last()

            if self.curriculum_manager is not None:
                self.curriculum_manager.episode_complete()
            # Adjust reward for training
            # Win = +1, Loss = -1, Draw = 0 (default PettingZoo values)
            return next_obs, final_reward, True, False, info
        
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

                    if self.curriculum_manager is not None:
                        self.curriculum_manager.episode_complete()
                
                    obs, reward, _, _, info = self.env.last()
                    return obs, reward, True, False, info
    
    def render(self, mode='human'):
        """Render the environment."""
        return self.env.render()
    
    def close(self):
        """Close the environment."""
        self.env.close()