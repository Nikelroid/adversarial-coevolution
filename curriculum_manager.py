import os
import random
import json
from typing import Optional
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm

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
        # Use a temporary file and atomic rename for safer multiprocessing writes
        temp_file = self.state_file + '.tmp'
        try:
            with open(temp_file, 'w') as f:
                json.dump(state, f)
            os.replace(temp_file, self.state_file)
        except Exception as e:
            print(f"[CurriculumManager] Error saving state: {e}")
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def _load_state(self):
        """Load state from file"""
        if not os.path.exists(self.state_file):
            # If file doesn't exist, initialize and return
            self.total_steps = 0
            self.episodes_completed = 0
            self.last_checkpoint_step = 0
            self._save_state()
            return
            
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
                self.total_steps = state.get('total_steps', 0)
                self.episodes_completed = state.get('episodes_completed', 0)
                self.last_checkpoint_step = state.get('last_checkpoint_step', 0)
        except (json.JSONDecodeError, FileNotFoundError, Exception) as e:
            print(f"[Curriculum] Warning: Could not load state, re-initializing: {e}")
            # If file is corrupt or unreadable, reset
            self.total_steps = 0
            self.episodes_completed = 0
            self.last_checkpoint_step = 0
            self._save_state() # Create a clean one
    
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
        print ("[DEBUG] : GET OPPONENT TYPE EXECUTED")
        self._load_state() # Always reload state from disk to get latest
        
        # Use the (now correct) total_steps read from the file
        phase = self._get_current_phase(self.total_steps) 
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
                print(f'[MODEL CHOOSED] Phase 2: Pool opponent (found {len(available_policies)} models)')
                return 'pool'
            return 'random'
        
        else:  # phase == 3
            roll = random.random()
            if roll < 0.70 and len(available_policies) > 0:
                print(f'[MODEL CHOOSED] Phase 3: Pool opponent (found {len(available_policies)} models)')
                return 'pool'
            elif roll < 0.90:
                return 'random'
            elif has_selfplay_model:
                print('[MODEL CHOOSED] Phase 3: SELF-PLAY!')
                return 'self'
            else:
                print('[Curriculum] Phase 3: Self-play requested but model not available, using random')
                return 'random'
    
    # --- NEW METHOD: _prune_cache ---
    def _prune_cache(self, available_policies_on_disk: list[str]):
        """
        Removes models from the RAM cache if they no longer exist on disk.
        """
        # Add .zip to the paths from _get_available_policies for a fair comparison
        paths_on_disk = set(p + ".zip" for p in available_policies_on_disk)
        
        # Also add the self-play model, as it should not be pruned
        paths_on_disk.add(self.get_selfplay_model_path() + ".zip")
        
        # Get a copy of cache keys to avoid modifying dict during iteration
        paths_in_cache = list(self.policy_cache.keys())
        
        for path_zip in paths_in_cache:
            if path_zip not in paths_on_disk:
                print(f"[Curriculum Cache] Pruning {os.path.basename(path_zip)} from RAM.")
                del self.policy_cache[path_zip]

    # --- NEW METHOD: get_policy_from_pool ---
    def get_policy_from_pool(self, recent_n: int = 10) -> Optional[BaseAlgorithm]:
        """
        Samples a policy from the pool, using the RAM cache.
        If not in cache, loads it from disk and caches it.
        """
        available_policies = self._get_available_policies()
        
        # Prune any old models from our RAM cache
        self._prune_cache(available_policies)
        
        if len(available_policies) == 0:
            return None # Pool is empty

        recent_policies = available_policies[-recent_n:] if len(available_policies) > recent_n else available_policies
        selected_path_no_zip = random.choice(recent_policies)
        selected_path_zip = selected_path_no_zip + ".zip"
        
        # 1. Check cache (FAST RAM ACCESS)
        if selected_path_zip in self.policy_cache:
            return self.policy_cache[selected_path_zip]
        
        # 2. Not in cache, load from disk (SLOW I/O)
        print(f"[Curriculum Cache] Loading {os.path.basename(selected_path_zip)} into RAM...")
        if not os.path.exists(selected_path_zip):
            print(f"[Curriculum Cache] ERROR: File not found {selected_path_zip}")
            return None
            
        try:
            model = PPO.load(selected_path_no_zip) # SB3 handles the .zip
            self.policy_cache[selected_path_zip] = model
            return model
        except Exception as e:
            print(f"[Curriculum Cache] Failed to load {selected_path_zip}: {e}")
            return None

    # --- NEW METHOD: get_selfplay_policy ---
    def get_selfplay_policy(self) -> Optional[BaseAlgorithm]:
        """
        Gets the self-play policy, using the RAM cache.
        If not in cache, loads it from disk and caches it.
        """
        selfplay_path_no_zip = self.get_selfplay_model_path()
        selfplay_path_zip = selfplay_path_no_zip + ".zip"

        if not os.path.exists(selfplay_path_zip):
            # This is expected early in training
            return None
            
        # 1. Check cache (FAST RAM ACCESS)
        if selfplay_path_zip in self.policy_cache:
            return self.policy_cache[selfplay_path_zip]
            
        # 2. Not in cache, load from disk (SLOW I/O)
        print(f"[Curriculum Cache] Loading SELF-PLAY model {os.path.basename(selfplay_path_zip)} into RAM...")
        try:
            model = PPO.load(selfplay_path_no_zip)
            self.policy_cache[selfplay_path_zip] = model
            return model
        except Exception as e:
            print(f"[Curriculum Cache] Failed to load self-play model {selfplay_path_zip}: {e}")
            return None
    
    def get_selfplay_model_path(self) -> str:
        """Get path for self-play model"""
        return os.path.join(self.save_dir, 'current_model_for_selfplay')
    
    def _get_current_phase(self, steps: int) -> int:
        """Determine current training phase given a step count"""
        if steps < 5_000_000:
            return 1
        elif steps < 15_000_000:
            return 2
        else:
            return 3
    
    def should_save_checkpoint(self, current_total_steps: int) -> bool:
        """Check if we should save a checkpoint based on true steps"""
        self._load_state()  # Reload to get latest last_checkpoint_step
        phase = self._get_current_phase(current_total_steps)
        
        # Use frequency from your original code
        save_freq = 2_000_000 if phase == 1 else (500_000 if phase == 2 else 1_000_000)
        
        # Check against the TRUE step count
        return (current_total_steps - self.last_checkpoint_step) >= save_freq
    
    def save_checkpoint(self, model, step_count: int):
        """Save current model to policy pool and update state file"""
        checkpoint_path = os.path.join(self.save_dir, f'policy_step_{step_count}')
        model.save(checkpoint_path)
        print(f"[Curriculum] DEBUG: Saved checkpoint to {checkpoint_path}.zip")
        
        selfplay_path = os.path.join(self.save_dir, 'current_model_for_selfplay')
        model.save(selfplay_path)
        
        available = self._get_available_policies()
        if len(available) > self.max_pool_size:
            for old_path in available[:len(available)-self.max_pool_size]:
                if os.path.exists(old_path + '.zip'):
                    os.remove(old_path + '.zip')
                    print(f"[Curriculum] Removed old checkpoint: {os.path.basename(old_path)}")
        
        # CRITICAL: Update state file with the true step count
        self._load_state() # Get latest episode count
        self.last_checkpoint_step = step_count
        self.total_steps = step_count # Sync total_steps
        self._save_state()
        
        phase = self._get_current_phase(step_count)
        print(f"\n{'='*70}")
        print(f"✓ Checkpoint saved at {step_count:,} steps (Phase {phase})")
        print(f"  Pool size: {min(len(available)+1, self.max_pool_size)}/{self.max_pool_size}")
        print(f"  Path: {checkpoint_path}")
        print(f"{'='*70}\n")
    
    # --- NEW METHOD ---
    def update_total_steps(self, total_steps: int):
        """
        Update the total_steps in the state file.
        This is the 'writer' method called by the main process callback.
        """
        self._load_state() # Load to not clobber episodes_completed or last_step
        if total_steps - self.total_steps > 10000:
            self.total_steps = total_steps
            self._save_state()

    
    def episode_complete(self):
        """Mark episode as complete"""
        self._load_state()
        self.episodes_completed += 1
        if self.episodes_completed % 50 == 0:
            self._save_state()