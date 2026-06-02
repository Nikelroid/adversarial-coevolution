"""One self-play fine-tuning run + eval, for a (knock_reward, gin_reward) combo.

Resumes the strongest Phase-1 agent (run_5) and continues training it against a
FROZEN copy of run_5 -- one generation of self-play. PPO exploits any weakness
in the frozen opponent, so the result should beat its progenitor (>50% vs run_5)
while staying dominant vs random. Sweeping the reward shaping tells us which
opponent plays strongest.

The frozen opponent is loaded once in the parent and inherited by the forked
SubprocVecEnv workers (copy-on-write), so there's no per-worker reload.

    python sweep/selfplay_one.py --combo 0
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import time

import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from agents.random_agent import RandomAgent  # noqa: E402
from agents.ppo_agent import PPOAgent  # noqa: E402
from gym_wrapper import GinRummySB3Wrapper  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
import ppo_train  # noqa: E402,F401  (registers MaskedGinRummyPolicy for unpickling)

try:
    import wandb  # noqa: E402
    _WANDB = True
except Exception:
    _WANDB = False


def _wandb_enabled() -> bool:
    if not _WANDB or os.environ.get("WANDB_DISABLED", "").lower() in ("1", "true", "yes"):
        return False
    return bool(os.environ.get("WANDB_API_KEY")) or os.path.exists(
        os.path.expanduser("~/.netrc"))


class _WandbRollout(BaseCallback):
    """Push SB3 train/* metrics to W&B once per rollout."""

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if not _wandb_enabled():
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


class _PeriodicGinEval(BaseCallback):
    """Every ``eval_freq`` env-steps, run a quick gin-rate eval vs random, log the
    point to W&B (-> gin-rate-vs-steps curve), and keep the best-by-gin-rate
    checkpoint. Best-effort: any failure is swallowed so it never kills training,
    and it guards the longer run against late divergence (we keep the best, not
    just the final)."""

    def __init__(self, eval_freq, n_games, best_path, verbose=0):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.n_games = n_games
        self.best_path = best_path
        self.best_gin = -1.0
        self.best_step = 0
        self._last = 0
        self.history = []

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last < self.eval_freq:
            return True
        self._last = self.num_timesteps
        try:
            res = evaluate(self.model, RandomAgent, self.n_games)
            res["step"] = int(self.num_timesteps)
            self.history.append(res)
            print(f"[eval@{self.num_timesteps}] gin_rate={res['gin_rate']:.3f} "
                  f"win={res['win_rate']:.3f}", flush=True)
            if _wandb_enabled() and wandb.run is not None:
                wandb.log({"curve/gin_rate_vs_random": res["gin_rate"],
                           "curve/win_rate_vs_random": res["win_rate"]},
                          step=self.num_timesteps)
            if res["gin_rate"] > self.best_gin:
                self.best_gin = res["gin_rate"]
                self.best_step = int(self.num_timesteps)
                self.model.save(self.best_path)
        except Exception as exc:  # noqa: BLE001 - eval must never kill training
            print(f"[eval] periodic eval failed (continuing): {exc}", flush=True)
        return True


# Reward-shaping configs to find the strongest opponent.
REWARD_CONFIGS = [
    dict(knock_reward=0.5, gin_reward=1.5),    # 0: Phase-1 baseline (low gin rate)
    dict(knock_reward=0.5, gin_reward=2.5),    # 1: reward gin harder
    dict(knock_reward=0.25, gin_reward=2.5),   # 2: discourage knock + reward gin
    dict(knock_reward=0.1, gin_reward=3.0),    # 3: strong push toward gin
    dict(knock_reward=0.0, gin_reward=2.0),    # 4: knocking earns nothing (gin-only)
    dict(knock_reward=0.25, gin_reward=4.0),   # 5: extreme gin incentive
]

# Eval always scores with the SAME rewards so the gin rate is comparable across
# configs (training rewards differ; a gin is a gin regardless).
EVAL_KNOCK, EVAL_GIN = 0.5, 1.5

DEFAULT_RESUME = os.path.join(PROJECT_ROOT, "game", "model", "ppo_gin_rummy_winrate.zip")


def _make_env(rank, seed, opponent_factory, knock, gin, turns_limit=200):
    def _thunk():
        env = GinRummySB3Wrapper(
            opponent_policy=opponent_factory, randomize_position=True,
            turns_limit=turns_limit, curriculum_manager=None, rank=rank,
            knock_reward=knock, gin_reward=gin)
        env.reset(seed=seed + rank)
        return Monitor(env)
    return _thunk


def evaluate(model, opponent_factory, n_episodes, turns_limit=200):
    """Score the policy with FIXED rewards (EVAL_KNOCK/EVAL_GIN) so the gin rate is
    comparable across configs trained with different shaping. A win whose terminal
    payoff matches EVAL_GIN is a gin; matching EVAL_KNOCK is a knock-win."""
    env = GinRummySB3Wrapper(
        opponent_policy=opponent_factory, randomize_position=True,
        turns_limit=turns_limit, curriculum_manager=None,
        knock_reward=EVAL_KNOCK, gin_reward=EVAL_GIN)
    wins = losses = gin_wins = knock_wins = 0
    gin_thresh = 0.5 * (EVAL_KNOCK + EVAL_GIN)   # midpoint -> classify win type
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_r, done = 0.0, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, trunc, _ = env.step(action)
            ep_r += float(r)
            if trunc:
                done = True
        if ep_r > 0:
            wins += 1
            if ep_r >= gin_thresh:
                gin_wins += 1
            else:
                knock_wins += 1
        elif ep_r < 0:
            losses += 1
    env.close()
    return dict(win_rate=wins / n_episodes, loss_rate=losses / n_episodes,
                gin_rate=gin_wins / n_episodes,           # gins per game played
                knock_rate=knock_wins / n_episodes,
                gin_share=gin_wins / max(wins, 1),        # of wins, how many are gins
                n=n_episodes)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--combo", type=int,
                   default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    p.add_argument("--steps", type=int,
                   default=int(os.environ.get("STEPS", 8_000_000)))
    p.add_argument("--num-env", type=int, default=96)
    p.add_argument("--eval-episodes", type=int, default=1000)
    p.add_argument("--eval-freq", type=int,
                   default=int(os.environ.get("EVAL_FREQ", 1_000_000)),
                   help="env-steps between periodic gin-rate evals (0 disables)")
    p.add_argument("--eval-games", type=int, default=300)
    p.add_argument("--seed", type=int,
                   default=int(os.environ.get("SEED", 100)))
    p.add_argument("--resume", default=DEFAULT_RESUME)
    args = p.parse_args()

    cfg = REWARD_CONFIGS[args.combo]
    knock, gin = cfg["knock_reward"], cfg["gin_reward"]
    tag = f"run_{args.combo}_s{args.seed}"
    print(f"=== selfplay combo={args.combo} seed={args.seed} knock={knock} "
          f"gin={gin} steps={args.steps} resume={args.resume} ===", flush=True)

    out_dir = os.path.join(PROJECT_ROOT, "sweep", "selfplay")
    os.makedirs(os.path.join(out_dir, "results"), exist_ok=True)
    run_dir = os.path.join(out_dir, "models", tag)
    os.makedirs(run_dir, exist_ok=True)

    th.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Frozen opponent (run_5), loaded once -> inherited by forked workers.
    frozen = PPO.load(args.resume, device="cpu")
    opp_factory = functools.partial(PPOAgent, model=frozen)

    train_env = SubprocVecEnv(
        [_make_env(i, args.seed, opp_factory, knock, gin)
         for i in range(args.num_env)],
        start_method="fork")

    # Resume the agent's weights and continue training vs the frozen opponent.
    model = PPO.load(args.resume, env=train_env, device="cpu")

    callbacks = []
    if _wandb_enabled():
        try:
            wandb.init(
                project=os.environ.get("WANDB_PROJECT", "Adversarial-CoEvolution"),
                entity=os.environ.get("WANDB_ENTITY", "VLAvengers"),
                name=f"selfplay_c{args.combo}_s{args.seed}_k{knock}_g{gin}",
                group="phase3-ginshape",
                tags=["selfplay", "ginshape", f"combo{args.combo}",
                      f"seed{args.seed}"],
                config=dict(combo=args.combo, seed=args.seed,
                            knock_reward=knock, gin_reward=gin, steps=args.steps),
                reinit=True)
            callbacks.append(_WandbRollout())
            print("[wandb] logging enabled", flush=True)
        except Exception as exc:  # noqa: BLE001 - never let W&B kill training
            print(f"[wandb] init failed, continuing without: {exc}", flush=True)

    gin_eval = None
    if args.eval_freq > 0:
        gin_eval = _PeriodicGinEval(
            eval_freq=args.eval_freq, n_games=args.eval_games,
            best_path=os.path.join(run_dir, "best"))
        callbacks.append(gin_eval)

    t0 = time.time()
    model.learn(total_timesteps=args.steps, progress_bar=False,
                callback=callbacks or None)
    train_seconds = time.time() - t0
    train_env.close()

    model_path = os.path.join(run_dir, "final")
    model.save(model_path)

    vs_run5 = evaluate(model, functools.partial(PPOAgent, model=frozen),
                       args.eval_episodes)
    vs_random = evaluate(model, RandomAgent, args.eval_episodes)

    result = dict(combo=args.combo, seed=args.seed, knock_reward=knock,
                  gin_reward=gin, steps=args.steps, train_seconds=train_seconds,
                  model_path=model_path + ".zip",
                  vs_run5=vs_run5, vs_random=vs_random)
    if gin_eval is not None:
        result["curve"] = gin_eval.history          # gin-rate vs steps
        result["best_gin_rate"] = gin_eval.best_gin
        result["best_step"] = gin_eval.best_step
        result["best_model_path"] = os.path.join(run_dir, "best.zip")
    with open(os.path.join(out_dir, "results", f"{tag}.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2), flush=True)

    if _wandb_enabled() and wandb.run is not None:
        wandb.log({"eval/vs_run5_win": vs_run5["win_rate"],
                   "eval/vs_random_win": vs_random["win_rate"],
                   # gin metrics are the headline of this sweep -> surface them
                   "eval/gin_rate_vs_random": vs_random["gin_rate"],
                   "eval/gin_share_vs_random": vs_random["gin_share"],
                   "eval/gin_rate_vs_run5": vs_run5["gin_rate"],
                   "eval/gin_share_vs_run5": vs_run5["gin_share"],
                   "eval/train_seconds": train_seconds})
        # also store as run summary so they're sortable in the runs table
        wandb.run.summary.update({
            "gin_rate_vs_random": vs_random["gin_rate"],
            "gin_rate_vs_run5": vs_run5["gin_rate"],
            "win_rate_vs_random": vs_random["win_rate"],
            "win_rate_vs_run5": vs_run5["win_rate"]})
        wandb.finish()


if __name__ == "__main__":
    main()
