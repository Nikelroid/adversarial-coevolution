"""
Training script for PPO agent on Gin Rummy environment with action masking.

Requirements:
pip install stable-baselines3[extra] pettingzoo gymnasium wandb

Usage:
python train_ppo.py --train
"""

import os
import torch as th
from stable_baselines3.common.distributions import Distribution, CategoricalDistribution
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import CombinedExtractor
import torch
import wandb
from stable_baselines3.common.type_aliases import PyTorchObs
import torch.optim as optim
from curriculum_manager import CurriculumManager

# Import your custom components
from gym_wrapper import GinRummySB3Wrapper
from agents.random_agent import RandomAgent

# ============================================
# W&B Configuration - Works on Colab & Local
# ============================================
WANDB_PROJECT = "Adversarial-CoEvolution"

class MaskedGinRummyPolicy(ActorCriticPolicy):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_entropy = None
    """
    PPO-compatible masked MLP policy for PettingZoo Gin Rummy.
    Works with dict observation: {'observation': ..., 'action_mask': ...}
    
    FIXES:
    - Properly extracts action masks from observations
    - Overrides evaluate_actions() to apply masks during training
    - Handles batched observations correctly
    """

    def _extract_obs_and_mask(self, obs):
        """
        Extract observation vector and action mask from dict observation.
        Handles both single and batched observations.
        """

        obs_tensor = obs['observation']
        mask_tensor = obs.get('action_mask', None)
        
        # Ensure tensors are on correct device
        if not isinstance(obs_tensor, th.Tensor):
            obs_tensor = th.as_tensor(obs_tensor, device=self.device).float()
        if mask_tensor is not None and not isinstance(mask_tensor, th.Tensor):
            mask_tensor = th.as_tensor(mask_tensor, device=self.device)
            
        return obs_tensor, mask_tensor

    def _apply_action_mask(self, logits, action_mask):
        """
        Apply action mask to logits by setting invalid actions to -inf.
        """
        if action_mask is None:
            return logits
        # Ensure mask is boolean tensor on same device
        mask = action_mask.to(dtype=th.bool, device=logits.device)
        
        # Use -inf for invalid actions (safer than finfo.min)
        logits = th.where(mask, logits, th.tensor(float('-inf'), device=logits.device, dtype=logits.dtype))
        
        return logits

    def forward(self, obs, deterministic: bool = False) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass with action masking applied to logits, not latent features.
        """
        # Extract observation and mask
        _, action_mask = self._extract_obs_and_mask(obs)
        
        # Get features and latent representations
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        
        # Get action logits from latent features
        logits = self.action_net(latent_pi)  # This gives you [batch, 110]

        
        # NOW apply the mask to logits
        masked_logits = self._apply_action_mask(logits, action_mask)

        # Create distribution from masked logits
        distribution = CategoricalDistribution(self.action_space.n).proba_distribution(action_logits=masked_logits)


        # ADD THESE 2 LINES:
        entropy = distribution.entropy()
        self.last_entropy = entropy.mean().item()

        
        # Get values
        values = self.value_net(latent_vf)
        
        # Sample actions
        actions = distribution.sample() if not deterministic else th.argmax(masked_logits, dim=1)
        log_prob = distribution.log_prob(actions)
        
        return actions, values, log_prob
    
    def get_distribution(self, obs: th.Tensor) -> Distribution:
        """
        Return the masked action distribution for given observations.
        """
        _, action_mask = self._extract_obs_and_mask(obs)
        features = self.extract_features(obs)
        latent_pi, _ = self.mlp_extractor(features) if self.share_features_extractor else (
            self.mlp_extractor.forward_actor(features[0]),
            None,
        )
        logits = self.action_net(latent_pi)
        masked_logits = self._apply_action_mask(logits, action_mask)
        return CategoricalDistribution(self.action_space.n).proba_distribution(action_logits=masked_logits)

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate log-prob, entropy, and value for given actions.
        Used during policy gradient update.
        """
        _, action_mask = self._extract_obs_and_mask(obs)
        features = self.extract_features(obs)
        
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        logits = self.action_net(latent_pi)
        masked_logits = self._apply_action_mask(logits, action_mask)
        distribution = CategoricalDistribution(self.action_space.n).proba_distribution(action_logits=masked_logits)

        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        values = self.value_net(latent_vf)

        return values, log_prob, entropy

class WandbCallback(BaseCallback):
    """
    Custom callback for logging to Weights & Biases.
    """
    def __init__(self, log_freq=1_000, verbose=0):
        super(WandbCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.log_freq = log_freq
        
    def _on_step(self) -> bool:

        if hasattr(self.model.policy, 'last_entropy') and self.model.policy.last_entropy is not None and self.num_timesteps % self.log_freq == 0:
            wandb.log({
                "train/policy_entropy": self.model.policy.last_entropy,
            }, step=self.num_timesteps)
        
        # Log episode data when episodes end
        if len(self.model.ep_info_buffer) > 0 and self.num_timesteps % self.log_freq == 0:
            for info in self.model.ep_info_buffer:
                wandb.log({
                    "train/episode_reward": info['r'],
                    "train/episode_length": info['l'],
                }, step=self.num_timesteps)
        
        return True
    
    def _on_rollout_end(self) -> bool:
        # Log training metrics after each rollout
        if hasattr(self.model, 'logger') and self.model.logger is not None:
            wandb.log({
                "train/fps": self.model.logger.name_to_value.get("time/fps", 0),
            }, step=self.num_timesteps)
        return True

class SubprocessLogCallback(BaseCallback):
    """
    A callback to pull logs from subprocess environments and print them.
    W&B will automatically capture these print statements.
    """
    def __init__(self, log_freq: int = 1000, verbose: int = 0):
        super(SubprocessLogCallback, self).__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_freq == 0:
            
            # This calls 'get_and_clear_logs' on all 100 envs
            # and returns a list of lists: [ [logs_env_0], [logs_env_1], ... ]
            try:
                all_logs_per_env = self.training_env.env_method("get_and_clear_logs")
                
                for env_idx, log_list in enumerate(all_logs_per_env):
                    if log_list: # Only print if there are logs
                        for log_msg in log_list:
                            # Print to main stdout, which W&B captures
                            print(f"[Env {env_idx}] {log_msg}")
                            
            except Exception as e:
                print(f"[SubprocessLogCallback] Error collecting logs: {e}")
                
        return True

class CurriculumCallback(BaseCallback):
    """Callback to manage curriculum and save checkpoints"""
    
    def __init__(self, curriculum_manager, model_save_path, verbose=0):
        super(CurriculumCallback, self).__init__(verbose)
        self.curriculum_manager = curriculum_manager
        self.model_save_path = model_save_path
        self.last_selfplay_save_step = 0 # Renamed for clarity
        
    def _on_step(self) -> bool:
        # Get the TRUE total steps from the BaseCallback
        current_steps = self.num_timesteps
        
        # 1. Update the manager's state file so subprocesses can read it
        #    (Do this less frequently to avoid filesystem hammering)
        if current_steps % 100 == 0: # Update JSON file every 100 steps
             self.curriculum_manager.update_total_steps(current_steps)
        
        # 2. Periodically save current model for self-play
        if current_steps - self.last_selfplay_save_step >= 10_000:
            selfplay_path = self.curriculum_manager.get_selfplay_model_path()
            self.model.save(selfplay_path)
            self.last_selfplay_save_step = current_steps
            print(f"[Curriculum] Updated self-play model at {current_steps:,} steps")
        
        # 3. Check if should save checkpoint to curriculum pool
        #    (Pass the true step count)
        if self.curriculum_manager.should_save_checkpoint(current_steps):
            self.curriculum_manager.save_checkpoint(
                self.model, 
                current_steps  # Pass the true step count
            )
            
            # Log to wandb
            # Get phase based on true step count
            phase = self.curriculum_manager._get_current_phase(current_steps) 
            wandb.log({
                'curriculum/phase': phase,
                'curriculum/pool_size': len(self.curriculum_manager._get_available_policies()),
                'curriculum/total_steps': current_steps,
            })
        
        return True
    
class WandbBestModelCallback(BaseCallback):
    """
    Callback to log when a new best model is found during evaluation.
    """
    def __init__(self, verbose=0):
        super(WandbBestModelCallback, self).__init__(verbose)
        
    def _on_step(self) -> bool:
        wandb.log({"eval/new_best_model": 1, "eval/timesteps": self.num_timesteps})
        return True



# In train_ppo.py

def make_env(turns_limit=200, rank=0, curriculum_save_dir=None):
    """Create and wrap the environment."""
    
    # This will write/overwrite the debug log file
    log_file_path = f'./debug_env_{rank}.log'
    
    # --- STEP 1: LOG WHAT WE RECEIVED ---
    try:
        with open(log_file_path, 'w') as f: # 'w' = overwrite
            f.write(f"--- make_env(rank={rank}) log ---\n")
            f.write(f"Timestamp: {__import__('datetime').datetime.now()}\n")
            f.write(f"--- 1. RECEIVED ---\n")
            f.write(f"Value of curriculum_save_dir: {repr(curriculum_save_dir)}\n")
            f.write(f"Is it None? {curriculum_save_dir is None}\n")
    except Exception as e:
        pass 

    # --- STEP 2: TRY TO CREATE THE OBJECT ---
    cm_instance = None
    creation_error = "No error."
    try:
        if curriculum_save_dir is not None:
            # This line attempts to create the object
            cm_instance = CurriculumManager(
                save_dir=curriculum_save_dir,
                max_pool_size=20
            )
        else:
            creation_error = "Skipped (curriculum_save_dir was None)."
            
    except Exception as e:
        # If the import 'CurriculumManager' failed, this will catch it
        creation_error = f"CRITICAL ERROR: {e}"

    # --- STEP 3: LOG THE RESULT ---
    try:
        with open(log_file_path, 'a') as f: # 'a' = append
            f.write(f"\n--- 2. CREATION ATTEMPT ---\n")
            f.write(f"Error during creation: {creation_error}\n")
            f.write(f"Variable 'cm_instance' is None? {cm_instance is None}\n")
            f.write(f"Type of 'cm_instance': {type(cm_instance)}\n")
    except Exception as e:
        pass

    # --- STEP 4: PASS THE OBJECT TO THE WRAPPER ---
    env = GinRummySB3Wrapper(
        opponent_policy=RandomAgent, 
        randomize_position=True, 
        turns_limit=turns_limit,
        curriculum_manager=cm_instance, # Pass the instance we tried to create
        rank=rank
    )
    
    env.reset(seed=42 + rank)
    env = Monitor(env)
    return env


def train_ppo(
    total_timesteps=500_000,
    save_path='./artifacts/models/ppo_gin_rummy',
    log_path='./logs/',
    checkpoint_freq=50_000,
    eval_freq=10_000,
    log_freq=10,
    n_eval_episodes=10,
    randomize_position=True,
    wandb_project=WANDB_PROJECT,
    wandb_run_name=None,
    wandb_config=None,
    turns_limit=2000,
    num_env = 100
):
    """
    Train a PPO agent on Gin Rummy with W&B logging.
    
    Args:
        total_timesteps: Total number of training steps
        save_path: Path to save the trained model
        log_path: Path for logs
        checkpoint_freq: Frequency (in timesteps) to save checkpoints
        eval_freq: Frequency to evaluate the model
        n_eval_episodes: Number of episodes for evaluation
        randomize_position: Whether to randomize training agent position each episode
        wandb_run_name: W&B run name (optional)
        wandb_config: Additional config dict for W&B (optional)
    """
    
    # Create directories
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    curriculum_save_dir = os.path.join(save_path, 'curriculum_pool')

    curriculum_manager = CurriculumManager(
        save_dir=curriculum_save_dir,
        max_pool_size=20
    )

    
    # Initialize Weights & Biases
    # config = {
    #     "algorithm": "PPO",
    #     "policy": "MaskedGinRummyPolicy",
    #     "total_timesteps": total_timesteps,
    #     "learning_rate": 3e-4,
    #     "n_steps": 2048,          
    #     "batch_size": 256,
    #     "n_epochs": 10,
    #     "gamma": 0.99,
    #     "gae_lambda": 0.95,
    #     "clip_range": 0.2,
    #     "ent_coef": 0.001,
    #     "vf_coef": 0.5,
    #     "max_grad_norm": 0.5,
    #     "randomize_position": randomize_position,
    #     "weight_decay": 0.0001
    # }
    config = {
        "algorithm": "PPO",
        "policy": "MaskedGinRummyPolicy",
        "total_timesteps": 40_000_000,       # 20M or more for complex card games
        "learning_rate": 2.5e-4,           # slightly lower since updates are larger
        "n_steps": 512,                    # shorter rollouts, since many envs aggregate data fast
        "batch_size": 1024,                # increase batch size (divides evenly into n_steps*num_envs)
        "n_epochs": 4,                     # fewer epochs to avoid overfitting giant batches
        "gamma": 0.99,                     # standard discount
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.08,                 # slightly higher to encourage exploration
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "randomize_position": True,
        "weight_decay": 0.0,
        "normalize_advantage": True,       # essential with large batches
        "normalize_observations": True,    # if supported
    }
    if wandb_config:
        config.update(wandb_config)
    
    wandb.init(
        project=WANDB_PROJECT,
        name=wandb_run_name,
        config=config,
        sync_tensorboard=False,  # We're not using tensorboard
        monitor_gym=True,
        entity="VLAvengers",
    )
    
    # Create training environment
    print("Creating training environment with curriculum learning...")
    print(f"Position randomization: {'ENABLED' if randomize_position else 'DISABLED'}")
    print("Curriculum phases:")
    print("  Phase 1 (0-100k):     100% Random")
    print("  Phase 2 (100k-500k):  50% Random, 50% Pool")
    print("  Phase 3 (500k+):      70% Pool, 20% Random, 10% Self")
    # train_env = DummyVecEnv([lambda: make_env(turns_limit) for _ in range(50)])


    print(f"--- DEBUG: Passing this path to envs: {curriculum_save_dir} ---")
    if curriculum_save_dir is None:
        print("--- WARNING: curriculum_save_dir IS NONE! THIS IS THE PROBLEM! ---")

    train_env = SubprocVecEnv([
        lambda i=i, csd=curriculum_save_dir: make_env(
            turns_limit, 
            rank=i, 
            curriculum_save_dir=csd
        )
        for i in range(num_env)
    ])

    # Create evaluation environment
    print("Creating evaluation environment...")
    eval_env = DummyVecEnv([lambda: make_env(rank=999)])
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=save_path,
        name_prefix='ppo_gin_rummy_checkpoint'
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        log_path=log_path,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
        callback_on_new_best=WandbBestModelCallback()
    )
    
    wandb_callback = WandbCallback(log_freq=log_freq)

    curriculum_callback = CurriculumCallback(
        curriculum_manager=curriculum_manager,
        model_save_path=save_path
    )

    log_collector_callback = SubprocessLogCallback(log_freq=100) # Poll logs every 1000 steps

    
    # Create PPO model
    print("Initializing PPO model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    

    policy_kwargs_net = dict(
    features_extractor_class=CombinedExtractor,
    net_arch=dict(pi=[512, 256], vf=[512,256]),
    activation_fn=torch.nn.ReLU,
    ortho_init=True,
    optimizer_class=optim.Adam,
    optimizer_kwargs=dict(weight_decay=config["weight_decay"])
    )
    
    model = PPO(
        MaskedGinRummyPolicy,
        train_env,
        verbose=0,
        learning_rate=config["learning_rate"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        n_epochs=config["n_epochs"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_range=config["clip_range"],
        ent_coef=config["ent_coef"],
        tensorboard_log=None,
        device=device,
        policy_kwargs=policy_kwargs_net
    )
    print ('____________MODEL CREATED SUCCESSFULLY______________')
    
    # Log model architecture to W&B
    wandb.watch(model.policy, log="all", log_freq=log_freq)
    
    print(f"Training on device: {model.device}")
    print(f"Total timesteps: {config['total_timesteps']}")
    print("Starting training...")
    
    # Train the model
    try:
        model.learn(
            total_timesteps=config['total_timesteps'],
            callback=[checkpoint_callback, eval_callback, wandb_callback,curriculum_callback,log_collector_callback],
            progress_bar=True
        )
        
        # Save final model
        final_path = os.path.join(save_path, 'ppo_gin_rummy_final')
        model.save(final_path)
        print(f"\nTraining complete! Model saved to {final_path}")
        
        # Log final model to W&B
        wandb.save(final_path + ".zip")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        interrupted_path = os.path.join(save_path, 'ppo_gin_rummy_interrupted')
        model.save(interrupted_path)
        print(f"Model saved to {interrupted_path}")
        wandb.save(interrupted_path + ".zip")
    
    finally:
        train_env.close()
        eval_env.close()
        wandb.finish()
    
    return model


def test_trained_model(model_path, num_episodes=10, log_to_wandb=False, turns_limit=200):
    """
    Test a trained model.
    
    Args:
        model_path: Path to the trained model
        num_episodes: Number of episodes to test
        log_to_wandb: Whether to log results to W&B
    """
    print(f"\nTesting model: {model_path}")
    
    if log_to_wandb:
        wandb.init(project=WANDB_PROJECT, name="model_test", job_type="evaluation", entity="VLAvengers")
    
    # Load model
    model = PPO.load(model_path)
    
    # Create test environment
    env = make_env(turns_limit=turns_limit)
    
    total_rewards = []
    wins = 0
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            
            if done or truncated:
                break
        
        total_rewards.append(episode_reward)
        if episode_reward > 0:
            wins += 1
        
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")
        
        if log_to_wandb:
            wandb.log({
                "test/episode_reward": episode_reward,
                "test/episode": episode + 1,
            })
    
    env.close()
    
    avg_reward = sum(total_rewards) / len(total_rewards)
    win_rate = wins / num_episodes * 100
    
    print(f"\n=== Test Results ===")
    print(f"Episodes: {num_episodes}")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Max Reward: {max(total_rewards):.2f}")
    print(f"Min Reward: {min(total_rewards):.2f}")
    
    if log_to_wandb:
        wandb.log({
            "test/avg_reward": avg_reward,
            "test/win_rate": win_rate,
            "test/max_reward": max(total_rewards),
            "test/min_reward": min(total_rewards),
        })
        wandb.finish()


def setup_wandb_colab(api_key=None):
    """
    Setup W&B for Google Colab environment.
    
    Args:
        api_key: Your W&B API key (optional, will check environment variable)
    """
    if api_key:
        wandb.login(key=api_key)
    else:
        # Try to get from environment
        api_key = os.environ.get('WANDB_API_KEY')
        if api_key:
            wandb.login(key=api_key)
        else:
            print("⚠️  W&B API key not found!")
            print("Please provide your API key:")
            print("1. Set environment variable: os.environ['WANDB_API_KEY'] = 'your_key_here'")
            print("2. Or call: setup_wandb_colab(api_key='your_key_here')")
            print("3. Or use: wandb.login()")
            raise ValueError("W&B API key required")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train PPO on Gin Rummy')
    parser.add_argument('--train', action='store_true', help='Train a new model')
    parser.add_argument('--test', type=str, help='Test a trained model (provide path)')
    parser.add_argument('--timesteps', type=int, default=500_000, help='Training timesteps')
    parser.add_argument('--save-path', type=str, default='./artifacts/models/ppo_gin_rummy', 
                       help='Path to save models')
    parser.add_argument('--no-randomize', action='store_true',
                       help='Disable position randomization during training')
    parser.add_argument('--wandb-project', type=str, default='gin-rummy-ppo',
                       help='Weights & Biases project name')
    parser.add_argument('--wandb-run-name', type=str, default=None,
                       help='Weights & Biases run name')
    parser.add_argument('--wandb-key', type=str, default=None,
                       help='Weights & Biases API key')
    
    parser.add_argument('--turns-limit', type=int, default=2000,
                       help='Limit on turns per game')
    parser.add_argument('--num-env', type=int, default=100,
                       help='Number of running envs')
    
    args = parser.parse_args()
    
    # Setup W&B login if key provided
    if args.wandb_key:
        setup_wandb_colab(api_key=args.wandb_key)
    else:
        setup_wandb_colab()  # Try environment variable
    
    if args.train:
        # Train model
        train_ppo(
            total_timesteps=args.timesteps,
            save_path=args.save_path,
            randomize_position=not args.no_randomize,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name,
            turns_limit=args.turns_limit,
            num_env = args.num_env
        )
        
        # Test the trained model
        final_model = os.path.join(args.save_path, 'ppo_gin_rummy_final')
        if os.path.exists(final_model + '.zip'):
            test_trained_model(final_model, num_episodes=10, log_to_wandb=True, turns_limit=args.turns_limit)
    
    elif args.test:
        # Test existing model
        test_trained_model(args.test, num_episodes=20, log_to_wandb=True, turns_limit=args.turns_limit)
    
    else:
        print("Please specify --train or --test <model_path>")
        print("\nExamples:")
        print("  python train_ppo.py --train")
        print("  python train_ppo.py --train --timesteps 1000000")
        print("  python train_ppo.py --train --no-randomize")
        print("  python train_ppo.py --train --wandb-run-name experiment-1")
        print("  python train_ppo.py --test ./artifacts/models/ppo_gin_rummy/ppo_gin_rummy_final")
