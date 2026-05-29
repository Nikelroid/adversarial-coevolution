"""Phase-2 RL-vs-LLM training run.

PPO (the action-masked Gin Rummy policy) trains against the LLM opponent served
by the distributed master/worker stack. Each forked env runs a FastLLMAgent whose
OllamaAPI points at $GINLLM_MASTER_URL; the master batches/caches and load-balances
across the GPU worker pool. We warm-start from the self-play champion and
fine-tune against the LLM, so this measures whether an LLM teacher moves the agent
(co-evolution) rather than training a blank net from scratch.

Env knobs: NUM_ENV, TOTAL_STEPS, N_STEPS, INIT_MODEL, SAVE_PATH, WANDB (0/1).
"""
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy  # noqa: F401
# Fully import cv2 in the PARENT before any fork. SB3 pulls in opencv lazily;
# if a forked SubprocVecEnv child imports it mid-fork, cv2/typing/__init__.py
# collides with the partially-initialized stdlib `typing` and crashes the worker.
try:
    import cv2  # noqa: F401
except Exception:
    pass

import torch
import torch.optim as optim
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from gym_wrapper import GinRummySB3Wrapper
from agents.fast_llm_agent import FastLLMAgent
from ppo_train import MaskedGinRummyPolicy, CombinedExtractor


def make_env(rank, turns_limit=200):
    def _f():
        env = GinRummySB3Wrapper(
            opponent_policy=FastLLMAgent, randomize_position=True,
            turns_limit=turns_limit, curriculum_manager=None, rank=rank)
        env.reset(seed=1000 + rank)
        return Monitor(env)
    return _f


class ThroughputLogger:
    """Print steps/s every rollout so we can see the LLM-in-loop throughput."""
    def __init__(self):
        self.t0 = time.time()
        self.last = 0


def main():
    num_env = int(os.environ.get("NUM_ENV", "64"))
    total = int(os.environ.get("TOTAL_STEPS", "40000"))
    n_steps = int(os.environ.get("N_STEPS", "128"))
    init = os.environ.get("INIT_MODEL", "game/model/ppo_gin_rummy_selfplay.zip")
    save = os.environ.get("SAVE_PATH", "game/model/ppo_gin_rummy_llm.zip")
    master = os.environ.get("GINLLM_MASTER_URL", "(unset)")
    print(f"[llmplay] num_env={num_env} total={total} n_steps={n_steps} "
          f"master={master} init={init}", flush=True)

    env = SubprocVecEnv([make_env(i) for i in range(num_env)], start_method="fork")
    pk = dict(
        features_extractor_class=CombinedExtractor,
        net_arch=dict(pi=[512, 512, 256, 128], vf=[512, 512, 256, 128]),
        activation_fn=torch.nn.Tanh, ortho_init=True,
        optimizer_class=optim.Adam, optimizer_kwargs=dict(weight_decay=0.0))
    model = PPO(
        MaskedGinRummyPolicy, env, verbose=1, learning_rate=1e-4, n_steps=n_steps,
        batch_size=1024, n_epochs=4, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        ent_coef=0.01, device="cpu", policy_kwargs=pk)

    if init and os.path.exists(init):
        try:
            model.set_parameters(init, device="cpu")
            print(f"[llmplay] warm-started weights from {init}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[llmplay] warm-start failed ({exc}); training from scratch", flush=True)
    else:
        print("[llmplay] no init model; training from scratch", flush=True)

    t0 = time.time()
    model.learn(total_timesteps=total, progress_bar=False)
    dt = time.time() - t0
    os.makedirs(os.path.dirname(save), exist_ok=True)
    model.save(save)
    rollout = num_env * n_steps
    print(f"[llmplay] DONE {total} steps in {dt:.0f}s = {total/max(dt,1):.0f} steps/s "
          f"| ~{rollout} steps/rollout, ~{total/max(rollout,1):.1f} rollouts | saved {save}",
          flush=True)
    print("LLMPLAY_DONE", flush=True)


if __name__ == "__main__":
    main()
