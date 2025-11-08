"""
Example: Using LLM Agent in Gin Rummy

This script demonstrates how to use the LLM agent with your Gin Rummy environment.
Make sure Ollama is running: ollama serve
"""

from agents.llm_agent import LLMAgent
from agents.random_agent import RandomAgent
#Import your Gin Rummy environment API
from gym_wrapper import GinRummySB3Wrapper


def example_1_basic_usage():
    """Example 1: Basic LLM agent usage"""
    print("\n=== Example 1: Basic LLM Agent ===\n")
    
    env = GinRummySB3Wrapper(opponent_policy=RandomAgent)
    # env.reset(seed=42)
    
    # Create LLM agent
    llm_agent = LLMAgent(env.env, model="qwen3-vl:2b", prompt_name="default_prompt")
    # llm_agent.set_player("player_0")
    
    # Create random opponent
    # random_agent = RandomAgent(env)
    # random_agent.set_player("player_1")
    obs, info = env.reset(seed=42)
    termination, truncation = False, False
    # Play one game 
    while 1:

        if termination or truncation:
            print(f"Game over! Final reward: {reward}")
            break
        
        # print("Valid Moves:", obs["action_mask"])
        # print("Observations:",obs["observation"])
        
        action = llm_agent.do_action()
        print(f"LLM chose action: {action}")

        obs, reward, termination, truncation, info = env.step(action)
    
    # Check statistics
    llm_agent.print_statistics()


def example_2_different_strategies():
    """Example 2: Try different prompt strategies"""
    print("\n=== Example 2: Different Strategies ===\n")
    
    strategies = ["default_prompt", "aggressive_prompt", "defensive_prompt"]
    
    for strategy in strategies:
        print(f"\nTesting {strategy}...")
        
        env = GinRummyEnvAPI(render_mode="ansi")
        env.reset(seed=42)
        
        llm_agent = LLMAgent(env, model="qwen3-vl:2b", prompt_name=strategy)
        llm_agent.set_player("player_0")
        
        random_agent = RandomAgent(env)
        random_agent.set_player("player_1")
        
        # Play 5 games
        wins = 0
        for game in range(5):
            env.reset()
            
            for step_data in env.play_game():
                agent_name, obs, reward, termination, truncation, info = step_data
                
                if termination or truncation:
                    if reward > 0 and agent_name == "player_0":
                        wins += 1
                    break
                
                if agent_name == "player_0":
                    action = llm_agent.do_action()
                else:
                    action = random_agent.do_action()
                
                env.step(action)
        
        print(f"{strategy}: {wins}/5 wins")
        llm_agent.print_statistics()
        llm_agent.reset_statistics()


def example_3_llm_vs_random():
    """Example 3: LLM agent vs Random agent"""
    print("\n=== Example 3: LLM vs Random ===\n")
    
    env = GinRummyEnvAPI(render_mode="ansi")
    
    # Create agents
    llm_agent = LLMAgent(env, model="qwen3-vl:2b", prompt_name="balanced_prompt")
    llm_agent.set_player("player_0")
    
    random_agent = RandomAgent(env)
    random_agent.set_player("player_1")
    
    # Play 10 games
    results = {"llm": 0, "random": 0, "draws": 0}
    
    for game_num in range(10):
        env.reset()
        final_agent = None
        final_reward = 0
        
        for step_data in env.play_game():
            agent_name, obs, reward, termination, truncation, info = step_data
            
            if termination or truncation:
                final_agent = agent_name
                final_reward = reward
                break
            
            if agent_name == "player_0":
                action = llm_agent.do_action()
            else:
                action = random_agent.do_action()
            
            env.step(action)
        
        # Determine winner
        if final_reward > 0:
            if final_agent == "player_0":
                results["llm"] += 1
            else:
                results["random"] += 1
        elif final_reward == 0:
            results["draws"] += 1
        else:
            if final_agent == "player_0":
                results["random"] += 1
            else:
                results["llm"] += 1
    
    print(f"\nResults after 10 games:")
    print(f"LLM wins: {results['llm']}")
    print(f"Random wins: {results['random']}")
    print(f"Draws: {results['draws']}")
    
    llm_agent.print_statistics()


def example_4_ppo_training():
    """Example 4: Use LLM agent as opponent during PPO training"""
    print("\n=== Example 4: PPO Training with LLM Opponent ===\n")
    
    from stable_baselines3 import PPO
    from gym_wrapper import GinRummySB3Wrapper
    
    env_api = GinRummyEnvAPI(render_mode="ansi")
    
    # Create LLM opponent
    llm_opponent = LLMAgent(
        env_api, 
        model="qwen3-vl:2b", 
        prompt_name="balanced_prompt"
    )
    
    # Wrap environment
    gym_env = GinRummySB3Wrapper(
        opponent_policy=llm_opponent,
        randomize_position=True
    )
    
    # Train PPO
    print("Training PPO for 10,000 steps against LLM opponent...")
    model = PPO("MlpPolicy", gym_env, verbose=1)
    model.learn(total_timesteps=10000)
    
    # Check opponent statistics
    llm_opponent.print_statistics()
    
    gym_env.close()


def example_5_test_connection():
    """Example 5: Test Ollama connection"""
    print("\n=== Example 5: Test Ollama Connection ===\n")
    
    from llm.api import OllamaAPI
    
    # Create API instance
    api = OllamaAPI(model="qwen3-vl:2b")
    
    # Check connection
    if api.check_connection():
        print("✓ Ollama is connected and working!")
        
        # Test generation
        response = api.generate("Say 'ready' if you can help me play Gin Rummy")
        print(f"\nLLM Response: {response}")
        
        # Test action parsing
        test_prompt = "Choose a number from these valid actions: 0, 5, 12, 23. Respond with just the number."
        response = api.generate(test_prompt, temperature=0.3)
        print(f"\nAction Test Response: {response}")
        
    else:
        print("✗ Cannot connect to Ollama")
        print("\nMake sure:")
        print("1. Ollama is installed")
        print("2. Run 'ollama serve' in terminal")
        print("3. Run 'ollama pull qwen3-vl:2b'")


if __name__ == "__main__":
    print("=" * 60)
    print("LLM Agent Examples for Gin Rummy")
    print("=" * 60)
    
    # Run the connection test first
    # example_5_test_connection()
    
    # Uncomment to see other examples
    example_1_basic_usage()
    # example_2_different_strategies()
    # example_3_llm_vs_random()
    # example_4_ppo_training()
    
    print("\n" + "=" * 60)
    print("Uncomment examples in __main__ to try them!")
    print("=" * 60)