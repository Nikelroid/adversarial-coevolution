"""Reproduce the project's Gin Rummy setup with the universal pipeline.

This shows how the game-specific extras plug in: ``env_kwargs`` for the reward knobs, a
``benchmark_agent`` (the perfect gold-standard player) to grade against, a ``reward_transform`` for
the early-knock shaping, and ``seed_models`` to practise against the strongest prior agents. The
pipeline itself is identical to the Connect Four example.

    python -m coev.examples.gin_rummy
"""
import os

from pettingzoo.classic import gin_rummy_v4

from coev import CoevConfig, train

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def gold_benchmark(env):
    from agents.gold_standard_agent import GoldStandardAgent   # game-specific expert
    return GoldStandardAgent(env)


def early_knock(reward, obs, done, info):
    # small per-decision pressure to knock early with low deadwood (training only)
    return reward - 0.0075


def main():
    seeds = [os.path.join(REPO, "game", "model", m)
             for m in ("gin_ace.zip", "gin_tactician.zip", "ppo_gin_rummy_selfplay.zip")]
    cfg = CoevConfig(
        env_fn=gin_rummy_v4.env,
        env_id="gin_rummy",
        env_kwargs=dict(knock_reward=0.5, gin_reward=0.5, opponents_hand_visible=False),
        algo="trpo",
        total_steps=12_000_000,
        num_envs=16,
        turns_limit=200,
        seed_models=[s for s in seeds if os.path.exists(s)],
        benchmark_agent=gold_benchmark,
        benchmark_name="gold",
        reward_transform=early_knock,
        init_model=os.path.join(REPO, "game", "model", "gin_ace.zip"),   # warm-start
        eval_every=1_000_000,
        eval_games=300,
        final_eval_games=800,
        name="gin_rummy_trpo",
    )
    train(cfg)


if __name__ == "__main__":
    main()
