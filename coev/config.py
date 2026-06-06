"""Configuration for one training run of the universal co-evolution pipeline."""
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class CoevConfig:
    # --- the environment (the only required field) ---
    env_fn: Callable                       # callable returning a fresh PettingZoo AEC env
    env_id: str = "custom"                 # name, for logging only
    env_kwargs: dict = field(default_factory=dict)

    # --- algorithm + hyper-parameters ---
    algo: str = "ppo"                      # "ppo" or "trpo"
    total_steps: int = 1_000_000
    num_envs: int = 8
    n_steps: int = 256
    batch_size: int = 1024
    n_epochs: int = 4
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.01                 # PPO only
    clip_range: float = 0.2                # PPO only
    target_kl: float = 0.01                # TRPO only
    net_arch: tuple = (256, 128)
    normalize_reward: bool = True
    seed: int = 0

    # --- opponent curriculum ---
    stages: Optional[list] = None          # None -> coev.curriculum.default_stages()
    seed_models: tuple = ()                # paths to prior agents to practise against
    turns_limit: int = 1000
    reward_transform: Optional[Callable] = None   # (reward, obs, done, info) -> reward

    # --- evaluation ---
    benchmark_agent: Optional[Callable] = None    # (env) -> agent (e.g. a perfect/expert player)
    benchmark_name: str = "benchmark"
    eval_games: int = 200
    eval_every: int = 1_000_000            # 0 disables the in-training learning curve
    final_eval_games: int = 400
    keep_best: bool = True                 # ship the best checkpoint, not the drifted final one

    # --- io ---
    init_model: str = ""                   # warm-start from this model path
    save_freq: int = 1_000_000
    out_dir: str = "coev_runs"
    name: str = "run"
