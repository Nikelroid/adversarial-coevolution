import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pettingzoo.classic import gin_rummy_v4
from agents import Agent
import random


#TEMP
class SelfPlayOpponent:
    """Simple wrapper for self-play using current training model"""
    
    def __init__(self, model, env):
        """
        Args:
            model: The SB3 PPO model to use for predictions
            env: The PettingZoo environment
        """
        self.model = model
        self.env = env
        self.player = None
    
    def set_player(self, player):
        """Set which player this agent controls"""
        self.player = player
    
    def do_action(self):
        """Get action from the model"""
        obs, _, _, _, _ = self.env.last()
        
        # Use the model's predict method directly
        action, _ = self.model.predict(obs, deterministic=True)
        
        # Verify action is valid
        action_mask = obs['action_mask']
        if not action_mask[action]:
            # Fallback to random valid action
            valid_actions = np.where(action_mask)[0]
            action = np.random.choice(valid_actions)
        
        return int(action)


class GinRummySB3Wrapper(gym.Env):
    """
    Wrapper to make PettingZoo Gin Rummy compatible with Stable-Baselines3.
    Converts the multi-agent environment to single-agent by having the opponent play randomly.
    Training agent position is randomized each episode for fair learning.
    """
    
    def __init__(self, opponent_policy, randomize_position=True,curriculum_manager=None):
        super().__init__()
        
        self.env = gin_rummy_v4.env(render_mode=None,knock_reward = 0.5, gin_reward = 1, opponents_hand_visible = True)
        # self.opponent_policy: Agent = opponent_policy(self.env)
        # self.randomize_position = randomize_position

        self.opponent_policy_class = opponent_policy  # Store class, not instance
        self.opponent_policy = None  # Will be created in reset()
        self.randomize_position = randomize_position
        
        # Curriculum learning support
        self.curriculum_manager = curriculum_manager
        self.current_model = None  # Reference to training model
        self.current_opponent_type = 'random'
        
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
        
        self.env.reset()\
        
    def set_current_model(self, model):
        """Set reference to current training model for self-play"""
        self.current_model = model
    
    # def reset(self, seed=None, options=None):
    #     """Reset the environment."""
    #     if seed is not None:
    #         self.env.reset(seed=seed)
    #     else:
    #         self.env.reset()

    #     self.TURNS_LIMIT = 100
    #     self.turn_num = 0
    #     self.last_score = -1

    #     # Randomly assign training agent position each episode
    #     if self.randomize_position and random.random() < 0.5:
    #         self.training_agent = 'player_1'
    #         self.opponent_agent = 'player_0'
    #         self.opponent_policy.set_player('player_0')
    #     else:
    #         self.training_agent = 'player_0'
    #         self.opponent_agent = 'player_1'
    #         self.opponent_policy.set_player('player_1')

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        if seed is not None:
            self.env.reset(seed=seed)
        else:
            self.env.reset()

        self.TURNS_LIMIT = 10
        self.turn_num = 0
        self.last_score = -1
        

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
                if self.current_model is not None:
                    self.opponent_policy = SelfPlayOpponent(self.current_model, self.env)
                else:
                    print("[Warning] Current model not set for self-play, falling back to random")
                    self.opponent_policy = self.opponent_policy_class(self.env)
                
            else:
                # Fallback
                self.opponent_policy = self.opponent_policy_class(self.env)
        else:
            # No curriculum - use default opponent
            self.opponent_policy = self.opponent_policy_class(self.env)

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

    #     player_hand = obs['observation'][0]
    #     if  sum(player_hand) == 10:
    #         if  self.last_score == -1:
    #             self.last_score = score_gin_rummy_hand(player_hand)
    #         else:
    #             r = score_gin_rummy_hand(player_hand)
    #             reward += r - self.last_score
    #             print(f'Score for last hand: {self.last_score} | Score for this hand: {r} ')
    #             print(f'Reward for this round: {reward} | It happend in {self.turn_num} turn')
    #             self.last_score = r

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
        
        self.env.step(action)

        #TEMP
        if self.curriculum_manager is not None:
            self.curriculum_manager.update_steps(1)

        # if self.turn_num > self.TURNS_LIMIT:
        #     truncation = True
        # self.turn_num += 1
        

        # Check if game ended
        # if termination or truncation:
        #     next_obs, _, _, _, _ = self.env.last()
        #     return next_obs, reward, True, False, info

        if termination or truncation:
            next_obs, _, _, _, _ = self.env.last()
            
            # Update curriculum on episode completion
            if self.curriculum_manager is not None:
                self.curriculum_manager.episode_complete()
            
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
                # _, _, termination, truncation, _ = self.env.last()
                # if termination or truncation:
                #     obs, reward, _, _, info = self.env.last()
                #     return obs, reward, True, False, info

                _, _, termination, truncation, _ = self.env.last()
                if termination or truncation:
                    obs, reward, _, _, info = self.env.last()
                    
                    # Update curriculum on episode completion
                    if self.curriculum_manager is not None:
                        self.curriculum_manager.episode_complete()
                    
                    return obs, reward, True, False, info
    
    def render(self, mode='human'):
        """Render the environment."""
        return self.env.render()
    
    def close(self):
        """Close the environment."""
        self.env.close()