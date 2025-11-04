"""
Example: Using LLM Agent in Gin Rummy

This script demonstrates how to use the LLM agent with your Gin Rummy environment.
Make sure Ollama is running: ollama serve
"""

from agents.llm import LLMAgent
from agents.random_agent import RandomAgent
# Import your Gin Rummy environment API
# from src.gin_rummy_api import GinRummyEnvAPI


def example_1_basic_usage():
    """Example 1: Basic LLM agent usage"""
    print("\n=== Example 1: Basic LLM Agent ===\n")
    
    # Uncomment when you have your environment
    # env = GinRummyEnvAPI(render_mode="human")
    # env.reset(seed=42)
    
    # Create LLM agent
    # llm_agent = LLMAgent(env, model="llama3.2:1b", prompt_name="default_prompt")
    
    # Play one game
    # for step_data in env.play_game():
    #     agent_name, obs, reward, done, info = step_data
    #     
    #     if agent_name == "player_0":
    #         action = llm_agent.do_action()
    #         env.step(action)
    #         print(f"LLM chose action: {action}")
    
    # Check statistics
    # llm_agent.print_statistics()
    
    print("See code above for implementation")


def example_2_different_strategies():
    """Example 2: Try different prompt strategies"""
    print("\n=== Example 2: Different Strategies ===\n")
    
    strategies = ["default_prompt", "aggressive_prompt", "defensive_prompt"]
    
    for strategy in strategies:
        print(f"\nTesting {strategy}...")
        
        # Uncomment when you have your environment
        # env = GinRummyEnvAPI(render_mode="ansi")
        # llm_agent = LLMAgent(env, model="llama3.2:1b", prompt_name=strategy)
        
        # Play multiple games
        # wins = 0
        # for game in range(5):
        #     env.reset()
        #     # ... play game logic ...
        #     # if llm_agent wins: wins += 1
        
        # print(f"{strategy}: {wins}/5 wins")
        # llm_agent.print_statistics()
    
    print("See code above for implementation")


def example_3_llm_vs_random():
    """Example 3: LLM agent vs Random agent"""
    print("\n=== Example 3: LLM vs Random ===\n")
    
    # Uncomment when you have your environment
    # env = GinRummyEnvAPI(render_mode="human")
    
    # Create agents
    # llm_agent = LLMAgent(env, model="llama3.2:1b", prompt_name="balanced_prompt")
    # random_agent = RandomAgent(env)
    
    # Play 10 games
    # results = {"llm": 0, "random": 0, "draws": 0}
    
    # for game_num in range(10):
    #     env.reset()
    #     
    #     for step_data in env.play_game():
    #         agent_name, obs, reward, done, info = step_data
    #         
    #         if agent_name == "player_0":
    #             action = llm_agent.do_action()
    #         else:
    #             action = random_agent.do_action()
    #         
    #         env.step(action)
    #     
    #     # Determine winner
    #     # ... update results ...
    
    # print(f"\nResults after 10 games:")
    # print(f"LLM wins: {results['llm']}")
    # print(f"Random wins: {results['random']}")
    # print(f"Draws: {results['draws']}")
    
    # llm_agent.print_statistics()
    
    print("See code above for implementation")


def example_4_ppo_training():
    """Example 4: Use LLM agent as opponent during PPO training"""
    print("\n=== Example 4: PPO Training with LLM Opponent ===\n")
    
    # from stable_baselines3 import PPO
    # from gym_wrapper import GinRummyGymWrapper
    
    # env_api = GinRummyEnvAPI(render_mode="ansi")
    
    # Create LLM opponent
    # llm_opponent = LLMAgent(
    #     env_api, 
    #     model="llama3.2:1b", 
    #     prompt_name="balanced_prompt"
    # )
    
    # Wrap environment
    # gym_env = GinRummyGymWrapper(
    #     env_api=env_api,
    #     opponent_policy=llm_opponent,
    #     training_agent="player_0"
    # )
    
    # Train PPO
    # model = PPO("MlpPolicy", gym_env, verbose=1)
    # model.learn(total_timesteps=10000)
    
    # Check opponent statistics
    # llm_opponent.print_statistics()
    
    print("See code above for implementation")


def example_5_test_connection():
    """Example 5: Test Ollama connection"""
    print("\n=== Example 5: Test Ollama Connection ===\n")
    
    from llm.api import OllamaAPI
    
    # Create API instance
    api = OllamaAPI(model="llama3.2:1b")
    
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
        print("3. Run 'ollama pull llama3.2:1b'")


if __name__ == "__main__":
    print("=" * 60)
    print("LLM Agent Examples for Gin Rummy")
    print("=" * 60)
    
    # Run the connection test first
    example_5_test_connection()
    
    # Uncomment to see other examples
    # example_1_basic_usage()
    # example_2_different_strategies()
    # example_3_llm_vs_random()
    # example_4_ppo_training()
    
    print("\n" + "=" * 60)
    print("Uncomment examples in __main__ to try them!")
    print("=" * 60)