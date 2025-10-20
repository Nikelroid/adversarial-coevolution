import os
import random
import copy
from typing import Optional
from agents.random_agent import RandomAgent

class CurriculumManager:
    """Manages opponent curriculum for progressive training"""
    
    def __init__(self, save_dir='./curriculum_policies', max_pool_size=20):
        self.save_dir = save_dir
        self.max_pool_size = max_pool_size
        self.policy_pool = []
        self.total_steps = 0
        self.episodes_completed = 0
        self.last_checkpoint_step = 0
        
        os.makedirs(save_dir, exist_ok=True)
        
        print("=" * 70)
        print("Curriculum Learning Manager Initialized")
        print(f"Save directory: {save_dir}")
        print(f"Max pool size: {max_pool_size}")
        print("=" * 70)
    
    def _get_available_policies(self):
        """Get list of available policy files from filesystem"""
        import glob
        pattern = os.path.join(self.save_dir, 'policy_step_*.zip')
        policy_files = glob.glob(pattern)
        # Sort by step number
        policy_files.sort(key=lambda x: int(x.split('_')[-1].replace('.zip', '')))
        # Remove .zip extension to get model paths
        return [f.replace('.zip', '') for f in policy_files]
    
    def get_opponent_type(self) -> str:
        """
        Decide which type of opponent to use based on curriculum phase
        Returns: 'random', 'pool', or 'self'
        """
        phase = self._get_current_phase()
        available_policies = self._get_available_policies()  # Check filesystem
        
        if phase == 1:
            # Phase 1 (0-100k): 100% Random
            return 'random'
        
        elif phase == 2:
            # Phase 2 (100k-500k): 50% Random, 50% Pool
            if random.random() < 0.5 and len(available_policies) > 0:
                print(f'Phase 2: Model selected from pool (found {len(available_policies)} models)')
                return 'pool'
            return 'random'
        
        else:  # phase == 3
            # Phase 3 (500k+): 70% Pool, 20% Random, 10% Self
            roll = random.random()
            if roll < 0.70 and len(available_policies) > 0:
                print(f'Phase 3: Model selected from pool (found {len(available_policies)} models)')
                return 'pool'
            elif roll < 0.90:
                return 'random'
            else:
                print('Phase 3: Selfplay!')
                return 'self'
            
    def sample_policy_path(self, recent_n: int = 10) -> str:
        """Sample a policy path from the pool"""
        available_policies = self._get_available_policies()  # Check filesystem
        
        if len(available_policies) == 0:
            raise ValueError("Policy pool is empty!")
        
        recent_policies = available_policies[-recent_n:] if len(available_policies) > recent_n else available_policies
        selected = random.choice(recent_policies)
        print(f"Selected policy: {os.path.basename(selected)}")
        return selected
    
    def _get_current_phase(self) -> int:
        """Determine current training phase"""
        if self.total_steps < 100_000:
            return 1
        elif self.total_steps < 500_000:
            return 2
        else:
            return 3
    
    def should_save_checkpoint(self) -> bool:
        """Check if we should save a checkpoint"""
        phase = self._get_current_phase()
        save_freq = 50_000 if phase == 1 else (25_000 if phase == 2 else 50_000)
        
        return (self.total_steps - self.last_checkpoint_step) >= save_freq
    
    def save_checkpoint(self, model, step_count: int):
        """Save current model to policy pool"""
        checkpoint_path = os.path.join(self.save_dir, f'policy_step_{step_count}')
        model.save(checkpoint_path)
        
        self.policy_pool.append(checkpoint_path)
        
        # Remove oldest if exceeding max size
        if len(self.policy_pool) > self.max_pool_size:
            old_path = self.policy_pool.pop(0)
            if os.path.exists(old_path + '.zip'):
                os.remove(old_path + '.zip')
        
        self.last_checkpoint_step = self.total_steps
        phase = self._get_current_phase()
        
        print(f"\n{'='*70}")
        print(f"✓ Checkpoint saved at {step_count:,} steps (Phase {phase})")
        print(f"  Pool size: {len(self.policy_pool)}/{self.max_pool_size}")
        print(f"  Path: {checkpoint_path}")
        print(f"{'='*70}\n")
    
    def update_steps(self, steps: int = 1):
        """Update step counter"""
        self.total_steps += steps
        if self.total_steps % 1000 == 0:
            print (f' - Step: {self.total_steps}')
    
    def episode_complete(self):
        """Mark episode as complete"""
        self.episodes_completed += 1