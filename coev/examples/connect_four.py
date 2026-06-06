"""Universality demo: train an agent on Connect Four with the SAME pipeline used for Gin Rummy.

The only thing that changes from one game to another is ``env_fn`` (and, if you have them,
``seed_models``, ``benchmark_agent``, and ``reward_transform``). Nothing else is game-specific.

    python -m coev.examples.connect_four
"""
from pettingzoo.classic import connect_four_v3

from coev import CoevConfig, train


def main():
    cfg = CoevConfig(
        env_fn=connect_four_v3.env,
        env_id="connect_four",
        algo="trpo",
        total_steps=2_000_000,
        num_envs=8,
        eval_every=500_000,
        eval_games=200,
        name="connect_four_trpo",
    )
    train(cfg)


if __name__ == "__main__":
    main()
