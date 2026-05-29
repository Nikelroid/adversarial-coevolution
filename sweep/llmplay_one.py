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
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

try:
    import wandb
    _WANDB = True
except Exception:
    _WANDB = False


def _wandb_enabled():
    if not _WANDB or os.environ.get("WANDB_DISABLED", "").lower() in ("1", "true", "yes"):
        return False
    return bool(os.environ.get("WANDB_API_KEY")) or os.path.exists(
        os.path.expanduser("~/.netrc"))


class _WandbRollout(BaseCallback):
    """Push SB3 train/* + rollout/* metrics to W&B once per rollout."""
    def _on_step(self):
        return True

    def _on_rollout_end(self):
        if not _wandb_enabled():
            return
        snap = {}
        for k, v in self.model.logger.name_to_value.items():
            try:
                snap[k] = abs(float(v)) if "loss" in k else float(v)
            except Exception:
                pass
        if snap:
            wandb.log(snap, step=self.num_timesteps)

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

    ckpt_dir = os.environ.get("CKPT_DIR", "sweep/llmplay/ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)
    save_every = max(int(os.environ.get("CKPT_EVERY", "250000")) // num_env, 1)
    callbacks = [CheckpointCallback(save_freq=save_every, save_path=ckpt_dir,
                                    name_prefix="llm_ppo")]
    print(f"[llmplay] checkpoint every ~{save_every*num_env} steps -> {ckpt_dir}", flush=True)

    if _wandb_enabled():
        try:
            wandb.init(
                project=os.environ.get("WANDB_PROJECT", "Adversarial-CoEvolution"),
                entity=os.environ.get("WANDB_ENTITY", "VLAvengers"),
                name=os.environ.get("WANDB_NAME", f"rl_vs_llm_{total//1000}k"),
                group="phase2-rl-vs-llm",
                tags=["rl-vs-llm", "qwen2.5-7b"],
                config=dict(num_env=num_env, total_steps=total, n_steps=n_steps,
                            init=init, opponent="qwen2.5-7b", master=master),
                reinit=True)
            callbacks.append(_WandbRollout())
            print("[llmplay] W&B logging enabled", flush=True)
        except Exception as exc:  # noqa: BLE001 - never let W&B kill training
            print(f"[llmplay] W&B init failed, continuing without: {exc}", flush=True)
    else:
        print("[llmplay] W&B not enabled (no key) — training without logging", flush=True)

    t0 = time.time()
    model.learn(total_timesteps=total, progress_bar=False, callback=callbacks)
    dt = time.time() - t0
    os.makedirs(os.path.dirname(save), exist_ok=True)
    model.save(save)
    rollout = num_env * n_steps
    print(f"[llmplay] DONE {total} steps in {dt:.0f}s = {total/max(dt,1):.0f} steps/s "
          f"| ~{rollout} steps/rollout, ~{total/max(rollout,1):.1f} rollouts | saved {save}",
          flush=True)
    if _wandb_enabled() and wandb.run is not None:
        wandb.log({"train/steps_per_s": total / max(dt, 1), "train/wall_seconds": dt})
        wandb.finish()
    print("LLMPLAY_DONE", flush=True)


if __name__ == "__main__":
    main()
