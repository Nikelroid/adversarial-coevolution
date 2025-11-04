# Gin Rummy RL Training

This repository implements adversarial RL training for a 2-player Gin Rummy game. We use PettingZoo's environment for the game logic, ma-gym for multi-agent wrappers if needed, and Stable-Baselines3 to train an RL agent (e.g., PPO) against a fixed LLM enhancer opponent. The focus is on reinforcing the RL model through iterative play to achieve high win rates.
Gin Rummy involves a 52-card deck where players draw and discard to form melds (3+ cards of same rank or sequence in suit), aiming to minimize deadwood points. Knock with ≤10 deadwood or go gin (0 deadwood) to score against the opponent.


## Requirements
- Python 3.8+
- `pettingzoo[classic]`
- `ma-gym`
- `stable-baselines3`
- `gymnasium`


## Installation
```bash
pip install pettingzoo[classic] ma-gym stable-baselines3 gymnasium
```


## Usage
Set up the environment:
```python
from pettingzoo.classic import gin_rummy_v4
env = gin_rummy_v4.env()
```

Train RL agent (PPO):
```python
from stable_baselines3 import PPO
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
model.save("gin_rummy_rl")
```

See `train.py` for details.


## Contributing

Contributions welcome! Fork, improve (e.g., LLM integration), and submit PRs.


## Description

Vision-language models (VLMs) and their extensions, such as vision-language-action (VLA) models, represent a transformative intersection of computer vision (CV) and natural language processing (NLP), enabling agents to interpret visual data through linguistic reasoning. In the context of multiplayer games like Gin Rummy, these models facilitate tasks requiring real-time decision-making, such as strategy optimization, card melding, and adversarial play. This survey synthesizes recent advancements up to 2025, drawing from key literature on model architectures, optimization strategies, datasets, and applications. It expands on the project README by providing in-depth technical details, feasibility considerations for student projects, and broader implications, while emphasizing balanced views on challenges like computational overhead and ethical concerns.
Evolution and Key Concepts of VLMs and VLA Models in Games
VLMs integrate visual encoders (e.g., Vision Transformers or ViTs) with language models (e.g., transformers like LLaMA) to process multimodal inputs. Early models like CLIP (2021) focused on image-text alignment, but by 2025, advancements include unified tokenization for vision, language, and actions, enabling end-to-end reasoning. VLA models extend this by incorporating action generation, often via policy modules for robotic or game control.
Recent phases (2022-2025) show progression: foundational integration (e.g., RT-1 for visuomotor tasks), specialization (e.g., domain-specific biases in 2024), and generalization with safety (e.g., SafeVLA in 2025). Core techniques include cross-attention for fusion, token unification, and affordance detection. For games like Gin Rummy, methods like SPAG use retrieval-augmented generation (RAG) pipelines to incorporate domain knowledge, enhancing strategic understanding in competitive settings.
Controversies arise around bias in training data—e.g., web-crawled datasets may underrepresent diverse game scenarios, leading to uneven performance across player strategies. Research suggests countering this through balanced datasets and fine-tuning, though evidence leans toward proprietary models outperforming open-source in robustness, per 2025 benchmarks.



# LLM Section:
# LLM Agent for Gin Rummy

This implementation uses Ollama to power an LLM-based agent that can play Gin Rummy strategically.

## Quick Start

### 1. Install Ollama

```bash
# On Linux/Mac
curl -fsSL https://ollama.com/install.sh | sh

# Or visit https://ollama.com for other platforms (Windows, etc.)
```

### 2. Pull a Lightweight Model

```bash
# Recommended lightweight model (1B parameters, ~700MB)
ollama pull llama3.2:1b

# Alternative options:
# ollama pull phi3:mini     # 3.8B parameters, ~2.3GB
# ollama pull gemma2:2b     # 2B parameters, ~1.6GB
# ollama pull llama3.2:3b   # 3B parameters, ~2GB
```

### 3. Start Ollama Server

```bash
ollama serve
```

Keep this running in a terminal window.

### 4. Use the LLM Agent in Your Code

```python
from agents.llm_agent import LLMAgent
from src.gin_rummy_api import GinRummyEnvAPI

# Create Gin Rummy environment
env = GinRummyEnvAPI(render_mode="human")

# Create LLM agent with default Gin Rummy strategy
llm_agent = LLMAgent(env, model="llama3.2:1b", prompt_name="default_prompt")

# Play game
env.reset(seed=42)
for step_data in env.play_game():
    agent_name, obs, reward, done, info = step_data
    
    if agent_name == "player_0":  # Adjust based on your player name
        action = llm_agent.do_action()
        env.step(action)

# Check how well the LLM performed
llm_agent.print_statistics()
```

## Architecture

```
LLMAgent (agents/llm_agent.py)
    ↓
LLMPlayerHandler (llm/player_handler.py) - Coordinates everything
    ├─→ OllamaAPI (llm/api.py) - Makes LLM API calls
    └─→ ActionValidator (llm/validator.py) - Validates actions
```

### File Structure

```
llm/
├── api.py               # Ollama API wrapper
├── validator.py         # Action validation and fallback
└── player_handler.py    # Coordinates API + validator

agents/
└── llm_agent.py         # Main LLM agent class

config/
└── prompts.yaml         # Strategy prompts for Gin Rummy
```

### Components

1. **api.py (OllamaAPI)**: 
   - Sends prompts to Ollama server
   - Formats observations for LLM
   - Parses LLM responses to extract action numbers
   - Handles API connection and errors

2. **validator.py (ActionValidator)**:
   - Validates LLM actions against action_mask
   - Provides fallback actions when LLM returns invalid moves
   - Tracks success rate and statistics
   - Supports multiple fallback strategies (random/first/last)

3. **player_handler.py (LLMPlayerHandler)**:
   - Loads prompts from YAML config
   - Coordinates between API and validator
   - Combines prompt + observation + valid actions
   - Manages statistics

4. **llm_agent.py (LLMAgent)**:
   - Implements the Agent interface
   - Integrates with your game environment
   - Uses player_handler for all decisions

## Gin Rummy Prompts

Edit `config/prompts.yaml` to customize agent behavior. Available prompts:

### default_prompt
Balanced strategy that builds melds and reduces deadwood. Good for general play.

### aggressive_prompt  
Focuses on achieving Gin (0 deadwood) for maximum points. Takes more risks.

### defensive_prompt
Conservative play that knocks early. Reduces risk and secures wins with low deadwood.

### balanced_prompt
Adaptive strategy that changes tactics based on game state (early/mid/late game).

### meld_building_prompt
Focuses on efficiently forming sets (3-4 same rank) and runs (3+ consecutive suited cards).

## Gin Rummy Strategy Concepts

The prompts teach the LLM about:

- **Melds**: Sets (3-4 cards same rank) and Runs (3+ consecutive suited cards)
- **Deadwood**: Unmelded cards that count against you
- **Deadwood Values**: A=1, 2-10=face value, J/Q/K=10 each
- **Knocking**: End round when deadwood ≤ 10 points
- **Gin**: End round with 0 deadwood for 25-point bonus
- **Strategic discards**: Remove high-value cards that don't fit melds
- **Card tracking**: Pay attention to opponent's picks and discards

## Recommended Models for Gin Rummy

| Model | Size | Speed | Quality | Best Use Case |
|-------|------|-------|---------|---------------|
| **llama3.2:1b** | ~700MB | Very Fast | Good | Testing, rapid gameplay, resource-constrained |
| **phi3:mini** | ~2.3GB | Fast | Better | Balanced performance, better reasoning |
| **gemma2:2b** | ~1.6GB | Fast | Better | Good strategic decisions |
| **llama3.2:3b** | ~2GB | Medium | Best | Competitive play, stronger strategies |

**Recommendation**: Start with `llama3.2:1b` for fast testing, upgrade to `phi3:mini` or `llama3.2:3b` for better play.

## Integration with Gym Wrapper

```python
from gym_wrapper import GinRummyGymWrapper
from agents.llm_agent import LLMAgent
from agents.random_agent import RandomAgent

# Create environment API
env_api = GinRummyEnvAPI(render_mode="ansi")

# Create LLM agent
llm_agent = LLMAgent(env_api, model="llama3.2:1b", prompt_name="default_prompt")

# Wrap for Gym interface
gym_env = GinRummyGymWrapper(
    env_api=env_api,
    opponent_policy=RandomAgent(env_api),
    training_agent="player_0"
)

# Run episodes
for episode in range(10):
    obs, info = gym_env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action = llm_agent.do_action()
        obs, reward, done, truncated, info = gym_env.step(action)
        total_reward += reward
    
    print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}")

# Print LLM agent statistics
llm_agent.print_statistics()
```

## Using LLM as Opponent in PPO Training

```python
from stable_baselines3 import PPO
from agents.llm_agent import LLMAgent

# Create LLM opponent
llm_opponent = LLMAgent(
    env_api, 
    model="llama3.2:1b", 
    prompt_name="balanced_prompt"  # Use balanced strategy
)

# Create gym environment with LLM as opponent
gym_env = GinRummyGymWrapper(
    env_api=env_api,
    opponent_policy=llm_opponent,  # LLM plays as opponent
    training_agent="player_0"
)

# Train PPO agent against LLM opponent
model = PPO("MlpPolicy", gym_env, verbose=1)
model.learn(total_timesteps=500000)

# Check LLM opponent statistics
llm_opponent.print_statistics()
```

## Troubleshooting

### Ollama Not Connecting

```bash
# Check if Ollama is running
ps aux | grep ollama

# Start Ollama server
ollama serve

# Check available models
ollama list

# Pull model if missing
ollama pull llama3.2:1b
```

### Low Action Validity Rate

If the LLM often returns invalid actions:
- **Upgrade model**: Try `phi3:mini` or `llama3.2:3b` for better parsing
- **Adjust prompt**: Make output format more explicit in prompts.yaml
- **Check temperature**: Lower temperature in api.py (currently 0.3) for more deterministic output
- **Fallback handles it**: ActionValidator automatically falls back to valid random actions

### Slow Performance

- Use smallest model: `llama3.2:1b` (~700MB)
- Reduce `max_tokens` in `api.py` generate() method
- Run Ollama on GPU for 5-10x speedup
- Consider caching common game states (future enhancement)

### LLM Makes Poor Decisions

- Try different prompts (aggressive/defensive/balanced)
- Use larger models (phi3:mini, llama3.2:3b)
- Enhance prompts with more specific Gin Rummy strategies
- Add few-shot examples in prompts.yaml

## Advanced: Custom Prompts

Add custom prompts to `config/prompts.yaml`:

```yaml
prompts:
  my_custom_prompt: |
    You are a Gin Rummy expert focused on <your strategy>.
    
    Strategy:
    - <guideline 1>
    - <guideline 2>
    
    Choose the best action from the valid actions provided.
```

Then use it:
```python
agent = LLMAgent(env, prompt_name="my_custom_prompt")
```

## Performance Expectations

With `llama3.2:1b`:
- Action selection: ~0.5-2 seconds per move
- Validity rate: 60-80% (invalid actions use fallback)
- Strategic quality: Decent for testing

With `phi3:mini` or `llama3.2:3b`:
- Action selection: ~1-3 seconds per move
- Validity rate: 70-90%
- Strategic quality: Good competitive play

## Future Enhancements

- [ ] Add game state history to prompts for better context
- [ ] Implement action caching for common positions
- [ ] Support for multi-model ensemble (vote on best action)
- [ ] Fine-tune models on Gin Rummy gameplay data
- [ ] Add chain-of-thought prompting for better reasoning
- [ ] Support for vision models (VLMs) with game visualization

## License

This code is part of your Adversarial Co-Evolution project.