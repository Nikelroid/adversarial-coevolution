"""Pool self-play (AlphaZero-style): resume run_5 and train it against a GROWING
pool of its own past checkpoints (plus some random), so it generalizes instead
of overfitting one frozen opponent.

- Seeds the curriculum pool with run_5 and forces phase 3 (pool + self + random)
  from step 0 via CURRICULUM_PHASE2/3_STEP=0.
- A callback checkpoints the current model into the pool every --save-freq steps;
  env workers then sample recent past selves as opponents (curriculum_manager).
- Evals the final agent vs a *panel* -- random, run_5, and the pool of selves --
  reporting win rate AND mean reward (decisiveness / gin-tendency), since
  win-rate-vs-one-opponent is a poor proxy for general strength.

    python sweep/poolplay_one.py --steps 12000000
"""
from __future__ import annotations
from agents.action_utils import masked_argmax

import argparse
import functools
import json
import os
import shutil
import sys
import time

import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from agents.random_agent import RandomAgent  # noqa: E402
from agents.ppo_agent import PPOAgent  # noqa: E402
from gym_wrapper import GinRummySB3Wrapper  # noqa: E402
from curriculum_manager import CurriculumManager  # noqa: E402
import ppo_train  # noqa: E402,F401  (registers MaskedGinRummyPolicy)

try:
    import wandb  # noqa: E402
    _WANDB = True
except Exception:
    _WANDB = False

KNOCK, GIN = 0.5, 1.5                                   # baseline (best in the sweep)
RESUME = os.path.join(PROJECT_ROOT, "game", "model", "ppo_gin_rummy_winrate.zip")
OUT_DIR = os.path.join(PROJECT_ROOT, "sweep", "poolplay")
POOL_DIR = os.path.join(OUT_DIR, "pool")


def _wandb_enabled():
    if not _WANDB or os.environ.get("WANDB_DISABLED", "").lower() in ("1", "true", "yes"):
        return False
    return bool(os.environ.get("WANDB_API_KEY")) or os.path.exists(
        os.path.expanduser("~/.netrc"))


def seed_pool(run5_path):
    """Fresh pool seeded with run_5 as the first 'past self' + self-play model."""
    if os.path.isdir(POOL_DIR):
        shutil.rmtree(POOL_DIR)
    os.makedirs(POOL_DIR, exist_ok=True)
    shutil.copy(run5_path, os.path.join(POOL_DIR, "policy_step_0.zip"))
    shutil.copy(run5_path, os.path.join(POOL_DIR, "current_model_for_selfplay.zip"))


def make_env(rank, seed):
    def _thunk():
        env = GinRummySB3Wrapper(
            opponent_policy=RandomAgent, randomize_position=True, turns_limit=200,
            curriculum_manager=CurriculumManager(save_dir=POOL_DIR, max_pool_size=20),
            rank=rank, knock_reward=KNOCK, gin_reward=GIN)
        env.reset(seed=seed + rank)
        return Monitor(env)
    return _thunk


class PoolCallback(BaseCallback):
    """Checkpoint the current model into the pool so opponents get stronger."""

    def __init__(self, cm, save_freq):
        super().__init__()
        self.cm = cm
        self.save_freq = save_freq
        self._last_update = 0
        self._last_ckpt = 0

    def _on_step(self) -> bool:
        s = self.num_timesteps
        if s - self._last_update >= 10_000:
            self.cm.update_total_steps(s)
            self._last_update = s
        if s - self._last_ckpt >= self.save_freq:
            self.cm.save_checkpoint(self.model, s)
            self._last_ckpt = s
            n = len(self.cm._get_available_policies())
            print(f"[pool] checkpoint @ {s:,} steps; pool size {n}", flush=True)
            if _wandb_enabled() and wandb.run is not None:
                wandb.log({"curriculum/pool_size": n, "curriculum/total_steps": s},
                          step=s)
        return True


class _WandbRollout(BaseCallback):
    def _on_step(self):
        return True

    def _on_rollout_end(self):
        if not (_wandb_enabled() and wandb.run is not None):
            return
        snap = {}
        for k, v in self.model.logger.name_to_value.items():
            try:
                # Present loss terms as positive (PPO policy/entropy loss is
                # legitimately negative; abs() is purely a dashboard preference).
                snap[k] = abs(float(v)) if "loss" in k else float(v)
            except Exception:
                pass
        if snap:
            wandb.log(snap, step=self.num_timesteps)


def evaluate(model, opponent, frozen_run5, n_episodes):
    """opponent in {'random','run5','pool'}; returns win_rate + mean_reward."""
    cm = None
    if opponent == "pool":
        cm = CurriculumManager(save_dir=POOL_DIR, max_pool_size=20)
        opp_cls = RandomAgent
    elif opponent == "run5":
        opp_cls = functools.partial(PPOAgent, model=frozen_run5)
    else:
        opp_cls = RandomAgent
    env = GinRummySB3Wrapper(opponent_policy=opp_cls, randomize_position=True,
                             turns_limit=200, curriculum_manager=cm,
                             knock_reward=KNOCK, gin_reward=GIN)
    rewards, wins, losses = [], 0, 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_r, done = 0.0, False
        while not done:
            action = masked_argmax(model, obs)
            obs, r, done, trunc, _ = env.step(action)
            ep_r += float(r)
            if trunc:
                done = True
        rewards.append(ep_r)
        wins += ep_r > 0
        losses += ep_r < 0
    env.close()
    return dict(win_rate=wins / n_episodes, loss_rate=losses / n_episodes,
                mean_reward=float(np.mean(rewards)), n=n_episodes)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=12_000_000)
    p.add_argument("--save-freq", type=int, default=1_500_000)
    p.add_argument("--num-env", type=int, default=96)
    p.add_argument("--eval-episodes", type=int, default=500)
    p.add_argument("--resume", default=RESUME)
    args = p.parse_args()

    # Start pool/self-play immediately (not after 5M random steps).
    os.environ["CURRICULUM_PHASE2_STEP"] = "0"
    os.environ["CURRICULUM_PHASE3_STEP"] = "0"

    os.makedirs(os.path.join(OUT_DIR, "results"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "models"), exist_ok=True)
    print(f"=== pool self-play: resume={args.resume} steps={args.steps:,} "
          f"save_freq={args.save_freq:,} ===", flush=True)

    th.manual_seed(100)
    np.random.seed(100)
    seed_pool(args.resume)

    train_env = SubprocVecEnv([make_env(i, 100) for i in range(args.num_env)],
                              start_method="fork")
    model = PPO.load(args.resume, env=train_env, device="cpu")

    main_cm = CurriculumManager(save_dir=POOL_DIR, max_pool_size=20)
    callbacks = [PoolCallback(main_cm, args.save_freq)]
    if _wandb_enabled():
        try:
            wandb.init(project=os.environ.get("WANDB_PROJECT", "Adversarial-CoEvolution"),
                       entity=os.environ.get("WANDB_ENTITY", "VLAvengers"),
                       name="poolplay_run5", group="phase2-poolselfplay",
                       tags=["poolselfplay"],
                       config=dict(steps=args.steps, save_freq=args.save_freq,
                                   knock=KNOCK, gin=GIN), reinit=True)
            callbacks.append(_WandbRollout())
            print("[wandb] logging enabled", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[wandb] init failed, continuing: {exc}", flush=True)

    t0 = time.time()
    model.learn(total_timesteps=args.steps, progress_bar=False, callback=callbacks)
    train_seconds = time.time() - t0
    train_env.close()

    model_path = os.path.join(OUT_DIR, "models", "poolplay_final")
    model.save(model_path)

    frozen_run5 = PPO.load(args.resume, device="cpu")
    panel = {opp: evaluate(model, opp, frozen_run5, args.eval_episodes)
             for opp in ("random", "run5", "pool")}
    result = dict(steps=args.steps, save_freq=args.save_freq, knock=KNOCK, gin=GIN,
                  train_seconds=train_seconds, model_path=model_path + ".zip",
                  pool_size=len(main_cm._get_available_policies()), panel=panel)
    with open(os.path.join(OUT_DIR, "results", "poolplay.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2), flush=True)

    if _wandb_enabled() and wandb.run is not None:
        wandb.log({f"eval/{opp}_win": panel[opp]["win_rate"] for opp in panel}
                  | {f"eval/{opp}_reward": panel[opp]["mean_reward"] for opp in panel})
        wandb.finish()


if __name__ == "__main__":
    main()
