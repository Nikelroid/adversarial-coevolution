import os
import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import wandb
import gymnasium as gym

# Import classes
from gym_wrapper import GinRummySB3Wrapper
import gym_wrapper
print(f"DEBUG: gym_wrapper loaded from {gym_wrapper.__file__}")
from agents.expert_agent import ExpertAgent
from agents.random_agent import RandomAgent
from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.torch_layers import CombinedExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import PyTorchObs

# Configuration
WANDB_PROJECT = "Adversarial-CoEvolution"

# Import shared policy
from agents.policy import MaskedGinRummyPolicy

# --- Utility Functions ---

def make_env(rank=0):
    """Create environment with specific settings for DAGGER"""
    # We want a predictable opponent for now, or random?
    # Expert vs Random is good for learning basic rules.
    env = GinRummySB3Wrapper(opponent_policy=RandomAgent, randomize_position=True, rank=rank)
    env.reset(seed=42 + rank)
    return env

def collect_dagger_data(env, student_policy, expert_agent, num_episodes, beta=0.0):
    """
    Collect data using DAGGER strategy.
    beta: Probability of taking expert action (Teacher Forcing).
    """
    observations = []
    expert_actions = []
    masks = []
    
    wins = 0
    total_episodes = 0
    
    print(f"Collecting {num_episodes} episodes...")
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        
        # We need to set the expert's internal env reference if it needs it per episode?
        # ExpertAgent stores self.env. The env object is persistent, so it's fine.
        
        while not done:
            # 1. Get Expert Action
            # Expert uses env.last() internally to decide
            # BUT: env.last() is only valid if it's the current player's turn.
            # In the wrapper, 'step' handles opponent turns automatically.
            # So when step returns, it stays at the Training Agent's turn (or done).
            
            # To be safe, ensure expert uses current observation from 'obs' if passed,
            # but my ExpertAgent uses self.env.last().
            # This works because wrapper logic: step() -> returns only when it's agent's turn.
            # So env.last() IS the agent's observation.
            
            expert_action = expert_agent.do_action()
            
            # 2. Get Student Prediction (for stepping if beta < 1)
            # student_policy.predict returns (action, state)
            student_action, _ = student_policy.predict(obs, deterministic=True)
            
            # 3. Choose action to execute
            if np.random.rand() < beta:
                action = expert_action
            else:
                action = student_action
                
            # Store Data: (Observation, Expert Label)
            # We store the 'obs' dict.
            # Need to flatten or store as is? 
            # Storing components for simpler batching later.
            observations.append(obs['observation'])
            masks.append(obs['action_mask'])
            expert_actions.append(expert_action)
            
            # 4. Step
            obs, reward, done, truncated, info = env.step(action)
            
            if done or truncated:
                if reward > 0.5: # Simple win check from wrapper rewards
                    wins += 1
                break
                
        total_episodes += 1
        
    win_rate = wins / total_episodes
    print(f"Collection Complete. Episodes: {total_episodes}, Win Rate: {win_rate:.2f}")
    
    return observations, masks, expert_actions, win_rate

def train_student(student_policy, observations, masks, actions, epochs=5, batch_size=64, lr=1e-4):
    """
    Train Student Policy using Supervised Learning (Cross Entropy).
    """
    print("Training Student Policy...")
    optimizer = optim.Adam(student_policy.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Convert to tensors
    obs_tensor = th.as_tensor(np.array(observations), device=student_policy.device).float()
    mask_tensor = th.as_tensor(np.array(masks), device=student_policy.device)
    action_tensor = th.as_tensor(np.array(actions), device=student_policy.device).long()
    
    dataset = TensorDataset(obs_tensor, mask_tensor, action_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    student_policy.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_obs, batch_mask, batch_action in loader:
            optimizer.zero_grad()
            
            # Forward pass: Get logits
            # We need to construct the dict input expected by policy
            dict_obs = {'observation': batch_obs, 'action_mask': batch_mask}
            
            logits = student_policy.get_action_logits(dict_obs)
            
            loss = criterion(logits, batch_action)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        wandb.log({"dagger/loss": avg_loss, "dagger/epoch": epoch})

    return avg_loss

def dagger_train(total_timesteps=100_000, episodes_per_iter=50, epochs_per_iter=4):
    """
    Main DAGGER Loop.
    """
    # 1. Setup
    env = make_env()
    expert = ExpertAgent(env.unwrapped) # Use unwrapped to access custom 'last'/'observe' methods
    expert.set_player('player_0') # Placeholder
    
    # Initialize Student Policy
    # We use PPO architecture but will train via SL
    # We can perform a dummy init using SB3 PPO to get the policy, then extract it
    from stable_baselines3 import PPO
    
    # Create a dummy model to initialize the policy network
    print("Initializing Student Policy...")
    dummy_model = PPO(
        MaskedGinRummyPolicy,
        env,
        verbose=1,
        learning_rate=3e-4,
        policy_kwargs=dict(
            features_extractor_class=CombinedExtractor,
            net_arch=dict(pi=[512, 512, 256, 128], vf=[512, 512, 256, 128]),
            activation_fn=nn.Tanh,
        )
    )
    student_policy = dummy_model.policy
    
    wandb.init(project=WANDB_PROJECT, name="dagger_training_expert", entity="VLAvengers")
    
    # Buffer
    all_obs = []
    all_masks = []
    all_actions = []
    
    iteration = 0
    cumulative_timesteps = 0
    
    while cumulative_timesteps < total_timesteps:
        iteration += 1
        print(f"\n=== DAGGER Iteration {iteration} ===")
        
        # Decay beta (Teacher Forcing)
        # Start high (mostly expert), decay to 0 (mostly student)
        beta = max(0.0, 1.0 - (iteration / 20)) # Linearly decay over 20 iterations
        print(f"Beta: {beta:.2f}")
        
        # 2. Collect Data
        obs, masks, actions, win_rate = collect_dagger_data(
            env, dummy_model, expert, episodes_per_iter, beta=beta
        )
        
        # Estimate timesteps (rough)
        steps = sum([len(obs[i].flatten())/52 for i in range(len(obs))]) # Just count samples
        steps = len(obs) # 1 sample = 1 step
        cumulative_timesteps += steps
        
        # Aggregate
        all_obs.extend(obs)
        all_masks.extend(masks)
        all_actions.extend(actions)
        
        # Limit buffer size (optional, keep last N samples if memory issue)
        MAX_BUFFER = 50000
        if len(all_obs) > MAX_BUFFER:
            all_obs = all_obs[-MAX_BUFFER:]
            all_masks = all_masks[-MAX_BUFFER:]
            all_actions = all_actions[-MAX_BUFFER:]
            
        print(f"Dataset Size: {len(all_obs)}")
        wandb.log({
            "dagger/buffer_size": len(all_obs),
            "dagger/win_rate": win_rate,
            "dagger/beta": beta,
            "dagger/timesteps": cumulative_timesteps
        })
        
        # 3. Train
        loss = train_student(student_policy, all_obs, all_masks, all_actions, epochs=epochs_per_iter)
        
        # Save Model periodically
        if iteration % 5 == 0:
            save_path = f"artifacts/models/dagger/model_iter_{iteration}"
            dummy_model.save(save_path)
            print(f"Saved model to {save_path}")
            
    # Final Save
    final_path = "artifacts/models/dagger/final_model"
    dummy_model.save(final_path)
    print(f"Training Complete. Saved to {final_path}")
    wandb.finish()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=10000)
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging")
    parser.add_argument("--episodes-per-iter", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=4)
    args = parser.parse_args()
    
    if args.no_wandb:
        os.environ["WANDB_MODE"] = "disabled"
        
    dagger_train(total_timesteps=args.timesteps, episodes_per_iter=args.episodes_per_iter, epochs_per_iter=args.epochs)
