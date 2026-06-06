"""coev: a small, universal pipeline for training a masked RL agent against an opponent curriculum
in any PettingZoo AEC game (or your own AEC environment).

Quick start:

    from pettingzoo.classic import connect_four_v3
    from coev import CoevConfig, train

    cfg = CoevConfig(env_fn=connect_four_v3.env, env_id="connect_four",
                     algo="trpo", total_steps=2_000_000)
    train(cfg)

The agent, opponents, masking, and curriculum are all game-agnostic. Supply ``seed_models`` to give
it prior agents to practise against, ``benchmark_agent`` to grade it against an expert, and
``reward_transform`` to shape the reward. See coev/examples/.
"""
from coev.config import CoevConfig
from coev.train import train, evaluate, build_model
from coev.env import MaskedCoevEnv
from coev.policy import MaskedPolicy
from coev.agents import RandomAgent, PolicyAgent, masked_argmax
from coev.curriculum import Curriculum, default_stages, seed_pool, load_any

__all__ = ["CoevConfig", "train", "evaluate", "build_model", "MaskedCoevEnv", "MaskedPolicy",
           "RandomAgent", "PolicyAgent", "masked_argmax", "Curriculum", "default_stages",
           "seed_pool", "load_any"]
