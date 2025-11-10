"""
Example: Using LLM Agent in Gin Rummy

This script demonstrates how to use the LLM agent with your Gin Rummy environment.
"""

import numpy as np  # <-- Added import
from agents.llm_agent import LLMAgent
from agents.ppo_agent import PPOAgent
from agents.random_agent import RandomAgent
# Import your Gin Rummy environment API and SB3 Wrapper
from gym_wrapper import GinRummySB3Wrapper


def test_1_single_game_random(model):
    """Test 1: LLM agent vs Random agent Single game"""
    print("\n=== Test 1: LLM agent vs Random agent Single game ===\n")
    
    env = GinRummySB3Wrapper(opponent_policy=RandomAgent)
    
    # Create LLM agent
    llm_agent = LLMAgent(env.env, model=model, prompt_name="default_prompt")
    
    obs, info = env.reset(seed=42)
    termination, truncation = False, False
    reward = 0 # Initialize reward
    
    # --- FIXED GAME LOOP ---
    while not (termination or truncation):
        
        # Check for valid actions *before* calling the agent
        valid_actions = np.where(obs["action_mask"])[0]
        if len(valid_actions) == 0:
            print("[INFO] No valid actions for LLM (game is in scoring phase or ended).")
            break # Exit the loop
            
        action = llm_agent.do_action()
        print(f"LLM chose action: {action}")

        obs, reward, termination, truncation, info = env.step(action)
    
    print(f"Game over! Final reward: {reward}")
    llm_agent.print_statistics()


def test_2_single_game_rl(model):
    """Test 2: LLM agent vs RL agent Single game"""
    print("\n=== Test 2: LLM agent vs RL agent Single game ===\n") # Fixed print
    
    env = GinRummySB3Wrapper(opponent_policy=PPOAgent)
    
    # Create LLM agent
    llm_agent = LLMAgent(env.env, model=model, prompt_name="default_prompt")
    
    obs, info = env.reset(seed=42)
    termination, truncation = False, False
    reward = 0 # Initialize reward

    # --- FIXED GAME LOOP ---
    while not (termination or truncation):
        
        # Check for valid actions *before* calling the agent
        valid_actions = np.where(obs["action_mask"])[0]
        if len(valid_actions) == 0:
            print("[INFO] No valid actions for LLM (game is in scoring phase or ended).")
            break # Exit the loop
            
        action = llm_agent.do_action()
        print(f"LLM chose action: {action}")

        obs, reward, termination, truncation, info = env.step(action)
    
    print(f"Game over! Final reward: {reward}")
    llm_agent.print_statistics()

def test_3_llm_vs_random_trial(model, trial):
    """Test 3: LLM agent vs Random agent in Trials"""
    print(f"\n=== Test 3: LLM vs Random agent in {trial} Trials ===\n")
    
    # Use RandomAgent as the opponent
    env = GinRummySB3Wrapper(opponent_policy=RandomAgent)
    
    # Create LLM agent
    llm_agent = LLMAgent(env.env, model=model, prompt_name="default_prompt")
    
    results = {"llm_wins": 0, "opponent_wins": 0, "draws": 0}
    
    for game_num in range(trial):
        print(f"\n--- Starting Game {game_num + 1}/{trial} ---")
        obs, info = env.reset()
        termination, truncation = False, False
        final_reward = 0
        
        # --- FIXED GAME LOOP ---
        while not (termination or truncation):
            # Check for valid actions *before* calling the agent
            valid_actions = np.where(obs["action_mask"])[0]
            if len(valid_actions) == 0:
                print("[INFO] No valid actions for LLM (game is in scoring phase or ended).")
                break # Exit the loop
                
            action = llm_agent.do_action()
            # print(f"LLM chose action: {action}") # Uncomment for verbose logging
            obs, reward, termination, truncation, info = env.step(action)
            final_reward = reward # Store last reward

        print(f"Game over! Final reward: {final_reward}")
        
        # Tally results
        if final_reward > 0:
            results["llm_wins"] += 1
        elif final_reward < 0:
            results["opponent_wins"] += 1
        else:
            results["draws"] += 1
        
        # Reset agent stats for the next game
        llm_agent.reset_statistics()
    
    print("\n" + "=" * 30)
    print(f"Results after {trial} games vs RandomAgent:")
    print(f"LLM wins: {results['llm_wins']}")
    print(f"RandomAgent wins: {results['opponent_wins']}")
    print(f"Draws: {results['draws']}")
    print("=" * 30)

def test_4_llm_vs_rl_trial(model, trial):
    """Test 4: LLM agent vs RL agent in Trials"""
    print(f"\n=== Test 4: LLM agent vs RL agent in {trial} Trials ===\n")
    
    # Use PPOAgent as the opponent
    env = GinRummySB3Wrapper(opponent_policy=PPOAgent)
    
    # Create LLM agent
    llm_agent = LLMAgent(env.env, model=model, prompt_name="default_prompt")
    
    results = {"llm_wins": 0, "opponent_wins": 0, "draws": 0}
    
    for game_num in range(trial):
        print(f"\n--- Starting Game {game_num + 1}/{trial} ---")
        obs, info = env.reset()
        termination, truncation = False, False
        final_reward = 0
        
        # --- FIXED GAME LOOP ---
        while not (termination or truncation):
            # Check for valid actions *before* calling the agent
            valid_actions = np.where(obs["action_mask"])[0]
            if len(valid_actions) == 0:
                print("[INFO] No valid actions for LLM (game is in scoring phase or ended).")
                break # Exit the loop
                
            action = llm_agent.do_action()
            # print(f"LLM chose action: {action}") # Uncomment for verbose logging
            obs, reward, termination, truncation, info = env.step(action)
            final_reward = reward # Store last reward

        print(f"Game over! Final reward: {final_reward}")

        # Tally results
        if final_reward > 0:
            results["llm_wins"] += 1
        elif final_reward < 0:
            results["opponent_wins"] += 1
        else:
            results["draws"] += 1
            
        # Reset agent stats for the next game
        llm_agent.reset_statistics()

    print("\n" + "=" * 30)
    print(f"Results after {trial} games vs PPOAgent:")
    print(f"LLM wins: {results['llm_wins']}")
    print(f"PPOAgent wins: {results['opponent_wins']}")
    print(f"Draws: {results['draws']}")
    print("=" * 30)


def test_5_ppo_training(model):
    """Test 5: Use LLM agent as opponent during PPO training"""
    print("\n=== Test 5: PPO Training with LLM Opponent ===\n")
    
    # Stable Baselines 3 PPO import
    from stable_baselines3 import PPO
    
    # Create the base PettingZoo environment
    # Note: Use render_mode=None for training, "ansi" is for debugging
    env_api = GinRummySB3Wrapper(render_mode=None)
    
    # Create LLM opponent
    llm_opponent = LLMAgent(
        env_api, 
        model=model, 
        prompt_name="default_prompt" # Use your preferred prompt
    )
    
    # Wrap environment for SB3, passing the LLM as the opponent
    gym_env = GinRummySB3Wrapper(
        opponent_policy=llm_opponent,
        randomize_position=True  # Good for training
    )
    
    # Train PPO
    print(f"Training PPO for 10,000 steps against LLM opponent ({model})...")
    ppo_model = PPO("MlpPolicy", gym_env, verbose=1)
    ppo_model.learn(total_timesteps=10000)
    
    print("PPO training complete.")
    
    # Check LLM opponent statistics during training
    llm_opponent.print_statistics()
    
    gym_env.close()


def test_0_test_connection(model):
    """Test 0: Test API connection"""
    print("\n=== Test 0: Test API Connection ===\n")
    
    # Use the correct API class (OllamaAPI or HuggingFaceAPI)
    # This assumes you are using the custom model_server.py
    from llm.api import OllamaAPI as ModelAPI
    
    # Create API instance
    api = ModelAPI(model=model, base_url="http://localhost:11434")
    
    # Check connection
    if api.check_connection():
        print("✓ API is connected and working!")
        
        # Test generation
        response = api.generate("Say 'ready' if you can help me play Gin Rummy")
        print(f"\nLLM Response: {response}")
        
    else:
        print("✗ Cannot connect to API server")
        print("\nMake sure:")
        print("1. Your 'model_server.py' is running.")
        print("2. It's running on http://localhost:11434")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="The configuration of the test.")
    parser.add_argument("-t", "--testnumber", help="Specify the test part (0-5).", required=True)
    parser.add_argument("-m", "--model", help="Specify the LLM model name (e.g., 'llama3.1-70b').", required=True)
    # Fixed argument from "terial" to "trial"
    parser.add_argument("-tr", "--trial", help="Specify the Trial number for test 3 and 4.", type=int, default=10)
    args = parser.parse_args()

    print("=" * 60)
    print("LLM Agent Examples for Gin Rummy")
    print("=" * 60)

    test_num = int(args.testnumber)
    
    if test_num == 1:
        test_1_single_game_random(args.model)
    elif test_num == 2:
        test_2_single_game_rl(args.model)
    elif test_num == 3:
        test_3_llm_vs_random_trial(args.model, args.trial)
    elif test_num == 4:
        test_4_llm_vs_rl_trial(args.model, args.trial)
    elif test_num == 5:
        test_5_ppo_training(args.model)
    elif test_num == 0:
        test_0_test_connection(args.model)
    else:
        print(f"Error: Test number '{args.testnumber}' not recognized. Please use a number from 0 to 5.")
    
    print("\n" + "=" * 60)
    print("Test run complete.")
    print("=" * 60)