"""
Quick test to verify curriculum manager is being created correctly in subprocesses
Run this BEFORE training to ensure everything is set up correctly
"""

import os
from stable_baselines3.common.vec_env import SubprocVecEnv
from curriculum_manager import CurriculumManager
from gym_wrapper import GinRummySB3Wrapper
from agents.random_agent import RandomAgent
from stable_baselines3.common.monitor import Monitor

def make_env_test(turns_limit=200, rank=0, curriculum_save_dir=None):
    """Test version of make_env with extensive debugging"""
    print(f"\n{'='*60}")
    print(f"[make_env_test] CALLED in process with rank={rank}")
    print(f"[make_env_test] curriculum_save_dir type: {type(curriculum_save_dir)}")
    print(f"[make_env_test] curriculum_save_dir value: {curriculum_save_dir}")
    
    curriculum_manager = None
    if curriculum_save_dir is not None:
        print(f"[make_env_test] Creating CurriculumManager...")
        try:
            curriculum_manager = CurriculumManager(
                save_dir=curriculum_save_dir,
                max_pool_size=20
            )
            print(f"[make_env_test] ✓ CurriculumManager created successfully")
        except Exception as e:
            print(f"[make_env_test] ✗ FAILED to create CurriculumManager: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"[make_env_test] ✗ curriculum_save_dir is None - NOT creating manager")
    
    env = GinRummySB3Wrapper(
        opponent_policy=RandomAgent,
        randomize_position=True,
        turns_limit=turns_limit,
        curriculum_manager=curriculum_manager
    )
    
    print(f"[make_env_test] Wrapper curriculum_manager: {env.curriculum_manager is not None}")
    print(f"{'='*60}\n")
    
    env.reset(seed=42 + rank)
    env = Monitor(env)
    return env

def test_curriculum_in_subprocess():
    """Test that curriculum manager works in SubprocVecEnv"""
    
    print("\n" + "="*70)
    print("TESTING CURRICULUM MANAGER IN SUBPROCESS")
    print("="*70)
    
    # Setup
    save_path = './test_curriculum'
    curriculum_save_dir = os.path.join(save_path, 'curriculum_pool')
    
    print(f"\n1. Setting up curriculum_save_dir: {curriculum_save_dir}")
    
    # Create the directory
    os.makedirs(curriculum_save_dir, exist_ok=True)
    
    # Create a main curriculum manager to save a dummy checkpoint
    print(f"\n2. Creating main curriculum manager and saving test checkpoint...")
    main_manager = CurriculumManager(save_dir=curriculum_save_dir, max_pool_size=5)
    print(f"   Main manager created, total_steps: {main_manager.total_steps}")
    
    # Test with 4 subprocess environments
    num_test_envs = 4
    print(f"\n3. Creating {num_test_envs} SubprocVecEnv environments...")
    
    # METHOD 1: Explicit function (RECOMMENDED)
    def make_env_with_curr(rank):
        return make_env_test(
            turns_limit=200,
            rank=rank,
            curriculum_save_dir=curriculum_save_dir
        )
    
    test_env = SubprocVecEnv([
        lambda rank=i: make_env_with_curr(rank)
        for i in range(num_test_envs)
    ])
    
    print(f"\n4. SubprocVecEnv created, testing reset...")
    obs = test_env.reset()
    print(f"   ✓ Reset successful, obs shape: {obs['observation'].shape}")
    
    print(f"\n5. Taking 5 test steps...")
    for step in range(5):
        actions = [test_env.action_space.sample() for _ in range(num_test_envs)]
        obs, rewards, dones, infos = test_env.step(actions)
        print(f"   Step {step+1}: {sum(dones)} episodes done")
    
    print(f"\n6. Closing environments...")
    test_env.close()
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)
    print("\nIf you saw '[make_env_test] ✓ CurriculumManager created successfully'")
    print("printed 4 times above, then your setup is CORRECT!")
    print("\nIf you saw '✗ curriculum_save_dir is None', then there's a problem")
    print("with how the variable is being captured in the lambda.")
    print("="*70 + "\n")

if __name__ == '__main__':
    test_curriculum_in_subprocess()