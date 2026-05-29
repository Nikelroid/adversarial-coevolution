"""A throughput-oriented LLM opponent for RL-in-the-loop training.

Same plumbing as :class:`LLMAgent` (master -> worker, suit-symmetry cache,
ActionValidator fallback), but swaps the heavy chain-of-thought prompt for a
terse one that asks the model to emit ONLY the chosen action string. Paired with
a low worker --max-output-tokens, this cuts per-call latency from ~16 s (full CoT
over 1024 tokens) to ~1 s, which is what makes a PPO rollout against the LLM
feasible. Quality drops a little vs. the CoT prompt, but the opponent is still
far stronger than random (the gate that matters for teaching the RL agent).
"""
from .llm_agent import LLMAgent

FAST_PROMPT = """You are an expert Gin Rummy player. Win by forming melds:
- Set: 3-4 cards of the same rank. Run: 3+ consecutive cards of one suit.
- Card points: K/Q/J/10 = 10, others = face value, Ace = 1.
Strategy: declare Gin or knock whenever offered; take the top discard only if it
extends a meld, otherwise draw from stock; when discarding, drop your highest-
point deadwood card.

CURRENT GAME STATE:
{game_state}

VALID ACTIONS (pick exactly one):
{valid_actions}

Reply with ONLY the exact action string from the list above - no reasoning, no
extra words, just the single action line."""


class FastLLMAgent(LLMAgent):
    def __init__(self, env, model: str = "qwen2.5-7b", prompt_name: str = "default_prompt"):
        super().__init__(env, model=model, prompt_name=prompt_name)
        # replace the CoT prompt with the terse, short-output one
        self.player_handler.prompts["default_prompt"] = FAST_PROMPT
