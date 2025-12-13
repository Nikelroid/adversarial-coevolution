from pettingzoo.classic import gin_rummy_v4
from agents import Agent
import random
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from hand_scoring import score_gin_rummy_hand
from agents.ppo_agent import PPOAgent



class GinRummySB3Wrapper(gym.Env):
    """
    Wrapper to make PettingZoo Gin Rummy compatible with Stable-Baselines3.
    Converts the multi-agent environment to single-agent by having the opponent play dynamically.
    Training agent position is randomized each episode for fair learning.
    """
    
    def __init__(self, opponent_policy, randomize_position=True, turns_limit=200, curriculum_manager=None, render_mode=None, rank=0, reward_system='long', evaluator=None):
        super().__init__()
        
        self.rank = rank
        self.log_buffer = [] 
        
        self.env = gin_rummy_v4.env(render_mode=render_mode, knock_reward=0.5, gin_reward=1.5, opponents_hand_visible=False)

        self.opponent_policy_class = opponent_policy  # Store class, not instance
        self.opponent_policy = None  # Will be created in reset()
        self.randomize_position = randomize_position
        
        # Curriculum learning support
        self.curriculum_manager = curriculum_manager
        self.current_opponent_type = 'random'  # Track opponent type

        # Short-term reward system
        self.reward_system = reward_system
        self.evaluator = evaluator
        
        self.log_buffer.append(
            f"[make_env rank={self.rank}] Wrapper initialized. "
            f"CurriculumManager is {'NOT ' if self.curriculum_manager is not None else ''}None."
        )

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

    def get_and_clear_logs(self):
        """Return the log buffer and clear it."""
        if not self.log_buffer:
            return []
        # Return a copy and clear the original
        logs_to_return = list(self.log_buffer)
        self.log_buffer.clear()
        return logs_to_return
    
    def _select_opponent(self):
        """
        Select and initialize opponent policy based on curriculum.
        NOW USES THE MODEL CACHE.
        """
        if self.curriculum_manager is not None:
            opponent_type,phase,step = self.curriculum_manager.get_opponent_type()
            self.current_opponent_type = opponent_type
            
            if opponent_type == 'random':
                self.opponent_policy = self.opponent_policy_class(self.env)
            
            elif opponent_type == 'pool':
                # Get a loaded model object from the cache (fast)
                ppo_model = self.curriculum_manager.get_policy_from_pool(recent_n=10)
                
                if ppo_model:
                    self.opponent_policy = PPOAgent(model=ppo_model, env=self.env)
                else:
                    self.log_buffer.append(f"[Rank {self.rank}] Pool requested, but pool is empty/failed. Using random.")
                    self.opponent_policy = self.opponent_policy_class(self.env)
            
            elif opponent_type == 'self':
                # Get the loaded self-play model from cache
                ppo_model = self.curriculum_manager.get_selfplay_policy()
                
                if ppo_model:
                    self.opponent_policy = PPOAgent(model=ppo_model, env=self.env)
                else:
                    self.log_buffer.append(f"[Rank {self.rank}] Failed to init PPOAgent with self-play model: , using random")
                    self.opponent_policy = self.opponent_policy_class(self.env)
            
            else:
                self.opponent_policy = self.opponent_policy_class(self.env)
        else:
            self.log_buffer.append(f"[Rank {self.rank}] THERE IS NO CURRICULUM")
            self.opponent_policy = self.opponent_policy_class(self.env)

    
    def last(self):
        """Delegate last to underlying env"""
        return self.env.last()

    def observe(self, agent):
        """Delegate observe to underlying env"""
        return self.env.observe(agent)

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
        step_reward = 0.0
        
        # Get pre-action state for scoring
        obs, reward, termination, truncation, info = self.env.last()
        score_before = 0.0
        
        if self.reward_system == 'short':
             try:
                 if len(obs['observation'].shape) == 2:
                     hand_mask = obs['observation'][0]
                 else:
                     hand_mask = obs['observation']
                 score_before = score_gin_rummy_hand(hand_mask)
             except Exception:
                 pass

        # Check if action is valid
        if not termination and not truncation:
            mask = obs['action_mask']
            if not mask[action]:
                # Invalid action - give negative reward and choose random valid action
                self.log_buffer.append(f"[Rank {self.rank}][Warning] Invalid Action Chosen: {action}")
                reward -= 1.0 # Penalize
                valid_actions = np.where(mask)[0]
                action = np.random.choice(valid_actions)
        
        # EXECUTE ACTION
        if not termination and not truncation:
            self.env.step(action)
        else:
            self.env.step(None)

        # Check truncation limits
        if self.turn_num > self.turns_limit:
            truncation = True
        self.turn_num += 1
        
        # Loop until it is the training agent's turn again or game ends
        while True:
            try:
                agent = self.env.agent_selection
            except:
                return obs, reward, True, False, info

            if agent == self.training_agent:
                # It is our turn again!
                obs, reward, termination, truncation, info = self.env.last()
                
                # Check additional truncation
                if self.turn_num > self.turns_limit:
                    truncation = True

                # Calculate Score Improvement (Short Term Reward)
                if self.reward_system == 'short' and not (termination or truncation):
                    try:
                        if len(obs['observation'].shape) == 2:
                            hand_mask = obs['observation'][0]
                        else:
                            hand_mask = obs['observation']
                        score_after = score_gin_rummy_hand(hand_mask)
                        
                        # Reward is the improvement
                        diff = score_after - score_before
                        step_reward += diff * 10.0 # Scale
                    except:
                        pass
                
                reward += step_reward
                
                if np.isnan(reward):
                    reward = 0.0

                done = termination or truncation
                
                if done:
                    if self.curriculum_manager is not None:
                         self.curriculum_manager.episode_complete()
                    self._select_opponent()
                
                return obs, reward, done, False, info
            
            else:
                # Opponent's turn
                obs_opp, reward_opp, term_opp, trunc_opp, info_opp = self.env.last()
                
                if term_opp or trunc_opp:
                    self.env.step(None) # Clear opponent
                    
                    if self.curriculum_manager is not None:
                         self.curriculum_manager.episode_complete()
                    self._select_opponent()
                    
                    # Return terminal state
                    # We need to return valid obs/reward/done
                    return obs_opp, 0, True, False, info_opp
                else:
                    self._opponent_step()
    
    def render(self, mode='human'):
        """Render the environment."""
        return self.env.render()
    
    def close(self):
        """Close the environment."""
        self.env.close()