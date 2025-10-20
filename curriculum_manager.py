import os
import random
import json
from typing import Optional

class CurriculumManager:
    """Manages opponent curriculum for progressive training"""
    
    def __init__(self, save_dir='./curriculum_policies', max_pool_size=20):
        self.save_dir = save_dir
        self.max_pool_size = max_pool_size
        self.state_file = os.path.join(save_dir, 'curriculum_state.json')
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize or load state
        if os.path.exists(self.state_file):
            self._load_state()
        else:
            self.total_steps = 0
            self.episodes_completed = 0
            self.last_checkpoint_step = 0
            self._save_state()
        
        print("=" * 70)
        print("Curriculum Learning Manager Initialized")
        print(f"Save directory: {save_dir}")
        print(f"Max pool size: {max_pool_size}")
        # DEBUG: Print available policies on init
        available = self._get_available_policies()
        print(f"Found {len(available)} existing policies in pool")
        if available:
            print(f"  Latest: {os.path.basename(available[-1])}")
        print("=" * 70)
    
    def _save_state(self):
        """Save state to file for cross-process synchronization"""
        state = {
            'total_steps': self.total_steps,
            'episodes_completed': self.episodes_completed,
            'last_checkpoint_step': self.last_checkpoint_step
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f)
    
    def _load_state(self):
        """Load state from file"""
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
                self.total_steps = state.get('total_steps', 0)
                self.episodes_completed = state.get('episodes_completed', 0)
                self.last_checkpoint_step = state.get('last_checkpoint_step', 0)
        except Exception as e:
            print(f"[Curriculum] Warning: Could not load state: {e}")
            self.total_steps = 0
            self.episodes_completed = 0
            self.last_checkpoint_step = 0
    
    def _get_available_policies(self):
        """Get list of available policy files from filesystem"""
        import glob
        pattern = os.path.join(self.save_dir, 'policy_step_*.zip')
        policy_files = glob.glob(pattern)
        
        # DEBUG logging
        if len(policy_files) == 0:
            print(f"[Curriculum] DEBUG: No policies found matching pattern: {pattern}")
            print(f"[Curriculum] DEBUG: Directory contents: {os.listdir(self.save_dir) if os.path.exists(self.save_dir) else 'DIR NOT FOUND'}")
        
        # Sort by step number
        policy_files.sort(key=lambda x: int(x.split('_')[-1].replace('.zip', '')))
        # Remove .zip extension (stable-baselines3 adds it automatically)
        return [f.replace('.zip', '') for f in policy_files]
    
    def get_opponent_type(self) -> str:
        """Decide which type of opponent to use based on curriculum phase"""
        # Always reload state from disk to get latest
        self._load_state()
        phase = self._get_current_phase()
        available_policies = self._get_available_policies()
        
        # DEBUG: Print what we found
        if phase == 3:
            print(f"[Curriculum] DEBUG Phase 3: Found {len(available_policies)} policies in pool")
        
        # Check if self-play model exists
        self_play_path = os.path.join(self.save_dir, 'current_model_for_selfplay.zip')
        has_selfplay_model = os.path.exists(self_play_path)
        
        if phase == 1:
            return 'random'
        
        elif phase == 2:
            if random.random() < 0.5 and len(available_policies) > 0:
                print(f'[Curriculum] Phase 2: Pool opponent (found {len(available_policies)} models)')
                return 'pool'
            return 'random'
        
        else:  # phase == 3
            roll = random.random()
            if roll < 0.70 and len(available_policies) > 0:
                print(f'[Curriculum] Phase 3: Pool opponent (found {len(available_policies)} models)')
                return 'pool'
            elif roll < 0.90:
                return 'random'
            elif has_selfplay_model:
                print('[Curriculum] Phase 3: SELF-PLAY!')
                return 'self'
            else:
                print('[Curriculum] Phase 3: Self-play requested but model not available, using random')
                return 'random'
    
    def sample_policy_path(self, recent_n: int = 10) -> str:
        """Sample a policy path from the pool"""
        available_policies = self._get_available_policies()
        
        if len(available_policies) == 0:
            raise ValueError("Policy pool is empty!")
        
        recent_policies = available_policies[-recent_n:] if len(available_policies) > recent_n else available_policies
        selected = random.choice(recent_policies)
        print(f"[Curriculum] Loading opponent: {os.path.basename(selected)}")
        return selected
    
    def get_selfplay_model_path(self) -> str:
        """Get path for self-play model"""
        return os.path.join(self.save_dir, 'current_model_for_selfplay')
    
    def _get_current_phase(self) -> int:
        """Determine current training phase"""
        if self.total_steps < 10_000:
            return 1
        elif self.total_steps < 50_000:
            return 2
        else:
            return 3
    
    def should_save_checkpoint(self) -> bool:
        """Check if we should save a checkpoint"""
        self._load_state()  # Reload to get latest
        phase = self._get_current_phase()
        save_freq = 5000 if phase == 1 else (2500 if phase == 2 else 5000)
        
        return (self.total_steps - self.last_checkpoint_step) >= save_freq
    
    def save_checkpoint(self, model, step_count: int):
        """Save current model to policy pool"""
        # Save to pool
        checkpoint_path = os.path.join(self.save_dir, f'policy_step_{step_count}')
        model.save(checkpoint_path)
        print(f"[Curriculum] DEBUG: Saved checkpoint to {checkpoint_path}.zip")
        
        # Also save current model for self-play
        selfplay_path = os.path.join(self.save_dir, 'current_model_for_selfplay')
        model.save(selfplay_path)
        
        # Clean up old models if exceeding max size
        available = self._get_available_policies()
        if len(available) > self.max_pool_size:
            # Remove oldest
            for old_path in available[:len(available)-self.max_pool_size]:
                if os.path.exists(old_path + '.zip'):
                    os.remove(old_path + '.zip')
                    print(f"[Curriculum] Removed old checkpoint: {os.path.basename(old_path)}")
        
        self.last_checkpoint_step = self.total_steps
        self._save_state()
        
        phase = self._get_current_phase()
        print(f"\n{'='*70}")
        print(f"✓ Checkpoint saved at {step_count:,} steps (Phase {phase})")
        print(f"  Pool size: {min(len(available)+1, self.max_pool_size)}/{self.max_pool_size}")
        print(f"  Path: {checkpoint_path}")
        print(f"{'='*70}\n")
    
    def update_steps(self, steps: int = 1):
        """Update step counter"""
        self._load_state()  # Load latest state
        self.total_steps += steps
        self._save_state()  # Save immediately
        
        if self.total_steps % 1000 == 0:
            phase = self._get_current_phase()
            print(f'[Curriculum] Steps: {self.total_steps:,} (Phase {phase})')
    
    def episode_complete(self):
        """Mark episode as complete"""
        self._load_state()
        self.episodes_completed += 1
        self._save_state()