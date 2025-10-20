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
    Converts the multi-agent environment to single-agent by having the opponent play dynamically.
    Training agent position is randomized each episode for fair learning.
    """
    
    def __init__(self, opponent_policy, randomize_position=True, turns_limit=200, curriculum_manager=None, render_mode=None):
        super().__init__()
        
        self.env = gin_rummy_v4.env(render_mode=render_mode, knock_reward=0.5, gin_reward=1.5, opponents_hand_visible=False)

        self.opponent_policy_class = opponent_policy  # Store class, not instance
        self.opponent_policy = None  # Will be created in reset()
        self.randomize_position = randomize_position
        
        # Curriculum learning support
        self.curriculum_manager = curriculum_manager
        self.current_opponent_type = 'random'  # Track opponent type

        self.turns_limit = turns_limit
        
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
    
    def _select_opponent(self):
        """
        Select and initialize opponent policy based on curriculum.
        Called at the start of each episode.
        """
        if self.curriculum_manager is not None:
            opponent_type = self.curriculum_manager.get_opponent_type()
            self.current_opponent_type = opponent_type

            print(f"[DEBUG] Episode starting with opponent: {opponent_type}")
            
            if opponent_type == 'random':
                self.opponent_policy = self.opponent_policy_class(self.env)
            
            elif opponent_type == 'pool':
                # Load opponent from policy pool
                from agents.ppo_agent import PPOAgent
                policy_path = self.curriculum_manager.sample_policy_path(recent_n=10)
                self.opponent_policy = PPOAgent(model_path=policy_path, env=self.env)
            
            elif opponent_type == 'self':
                # Load the saved self-play model from disk
                from agents.ppo_agent import PPOAgent
                selfplay_path = self.curriculum_manager.get_selfplay_model_path()
                try:
                    self.opponent_policy = PPOAgent(model_path=selfplay_path, env=self.env)
                    print("[Curriculum] Loaded self-play model successfully")
                except Exception as e:
                    print(f"[Curriculum] Failed to load self-play model: {e}, using random")
                    self.opponent_policy = self.opponent_policy_class(self.env)
            
            else:
                # Fallback
                self.opponent_policy = self.opponent_policy_class(self.env)
        else:
            # No curriculum - use default opponent
            print(f"[DEBUG] THERE IS NO CURRICULUM")
            self.opponent_policy = self.opponent_policy_class(self.env)
    
    def reset(self, seed=None, options=None):
        """Reset the environment."""
        if seed is not None:
            self.env.reset(seed=seed)
            random.seed(seed)
            np.random.seed(seed)
        else:
            self.env.reset()

        self.turn_num = 0
        self.last_score = None

        # SELECT OPPONENT FOR THIS EPISODE
        self._select_opponent()

        # Randomly assign training agent position each episode
        if self.randomize_position and random.random() < 0.5:
            self.training_agent = 'player_1'
            self.opponent_agent = 'player_0'
            self.opponent_policy.set_player('player_0')
        else:
            self.training_agent = 'player_0'
            self.opponent_agent = 'player_1'
            self.opponent_policy.set_player('player_1')
        
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
                print("[Warning] : Invalid Action Choosed", " Action: ", action)
                reward = -1.0
                valid_actions = np.where(mask)[0]
                action = np.random.choice(valid_actions)
        
        self.env.step(action)

        if self.turn_num > self.turns_limit:
            truncation = True
        self.turn_num += 1
        
        # Check if game ended
        if termination or truncation:
            next_obs, reward, _, _, _ = self.env.last()

            if abs(reward - 0.2) < 0.001:
                reward = 0.5
            elif abs(reward - 1) < 0.001:
                reward = 1.5

            if self.curriculum_manager is not None:
                self.curriculum_manager.episode_complete()
            
            # CRITICAL FIX: Select new opponent for next episode
            # This is called BEFORE vectorized env's internal reset
            self._select_opponent()

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

                    if abs(reward - 0.2) < 0.001:
                        reward = 0.5
                    elif abs(reward - 1) < 0.001:
                        reward = 1.5

                    if self.curriculum_manager is not None:
                        self.curriculum_manager.episode_complete()
                    
                    # CRITICAL FIX: Select new opponent for next episode
                    self._select_opponent()

                    return obs, reward, True, False, info
    
    def render(self, mode='human'):
        """Render the environment."""
        return self.env.render()
    
    def close(self):
        """Close the environment."""
        self.env.close()