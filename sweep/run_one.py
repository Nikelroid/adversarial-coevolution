"""Train one sweep config + deterministic eval vs RandomAgent.

Reads the HP combo from --combo (or $SLURM_ARRAY_TASK_ID), trains a masked PPO
agent against random opponents only (curriculum manager disabled), then runs a
deterministic eval over `eval_episodes` games and writes a JSON summary to
sweep/results/run_<combo>.json.
"""
from __future__ import annotations
from agents.action_utils import masked_argmax

import argparse
import json
import os
import sys
import time

import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import CombinedExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv

# Optional W&B logging: only active when WANDB_API_KEY (or `wandb login`)
# is set. Avoids blocking SLURM jobs on auth.
try:
    import wandb  # type: ignore

    _WANDB_AVAILABLE = True
except Exception:
    _WANDB_AVAILABLE = False


def _wandb_enabled() -> bool:
    if not _WANDB_AVAILABLE:
        return False
    if os.environ.get("WANDB_DISABLED", "").lower() in ("1", "true", "yes"):
        return False
    return bool(os.environ.get("WANDB_API_KEY")) or os.path.exists(
        os.path.expanduser("~/.netrc")
    )


class WandbRolloutLogger(BaseCallback):
    """Pushes SB3 train/* logger values to W&B once per rollout."""

    def _on_step(self) -> bool:  # noqa: D401
        return True

    def _on_rollout_end(self) -> None:  # noqa: D401
        if not _wandb_enabled():
            return
        snap = {}
        for k, v in self.model.logger.name_to_value.items():
            try:
                snap[k] = float(v)
            except Exception:
                pass
        if snap:
            wandb.log(snap, step=self.num_timesteps)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from agents.random_agent import RandomAgent  # noqa: E402
from gym_wrapper import GinRummySB3Wrapper  # noqa: E402
from ppo_train import MaskedGinRummyPolicy  # noqa: E402

from sweep.configs import get_config  # noqa: E402


def _make_env(rank: int, seed: int, turns_limit: int):
    """Build one env factory. NO curriculum_manager -> opponent is always
    RandomAgent, which is exactly what we want for the vs-random sweep."""

    def _thunk():
        env = GinRummySB3Wrapper(
            opponent_policy=RandomAgent,
            randomize_position=True,
            turns_limit=turns_limit,
            curriculum_manager=None,
            rank=rank,
        )
        env.reset(seed=seed + rank)
        return Monitor(env)

    return _thunk


def evaluate(model: PPO, n_episodes: int, turns_limit: int) -> dict:
    """Deterministic policy vs RandomAgent over n_episodes games."""
    env = GinRummySB3Wrapper(
        opponent_policy=RandomAgent,
        randomize_position=True,
        turns_limit=turns_limit,
        curriculum_manager=None,
    )
    rewards, lengths = [], []
    wins = losses = 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_r = 0.0
        ep_l = 0
        done = False
        while not done:
            action = masked_argmax(model, obs)
            obs, r, done, trunc, _ = env.step(action)
            ep_r += float(r)
            ep_l += 1
            if trunc:
                done = True
        rewards.append(ep_r)
        lengths.append(ep_l)
        if ep_r > 0:
            wins += 1
        elif ep_r < 0:
            losses += 1
    env.close()
    return dict(
        eval_episodes=n_episodes,
        win_rate=wins / n_episodes,
        loss_rate=losses / n_episodes,
        draw_rate=(n_episodes - wins - losses) / n_episodes,
        mean_reward=float(np.mean(rewards)),
        std_reward=float(np.std(rewards)),
        mean_length=float(np.mean(lengths)),
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--combo",
        type=int,
        default=None,
        help="HP index from sweep.configs; defaults to $SLURM_ARRAY_TASK_ID.",
    )
    p.add_argument("--save-root", default=os.path.join(PROJECT_ROOT, "sweep"))
    args = p.parse_args()

    combo = (
        args.combo
        if args.combo is not None
        else int(os.environ["SLURM_ARRAY_TASK_ID"])
    )
    cfg = get_config(combo)
    print(f"=== run_one combo={combo} ===")
    print(json.dumps(cfg, indent=2))

    os.makedirs(os.path.join(args.save_root, "results"), exist_ok=True)
    run_dir = os.path.join(args.save_root, "models", f"run_{combo}")
    os.makedirs(run_dir, exist_ok=True)

    th.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    # start_method='fork' avoids a cv2/numpy circular-import crash that hits
    # the default 'forkserver' path (children re-import everything from scratch).
    train_env = SubprocVecEnv(
        [_make_env(i, cfg["seed"], cfg["turns_limit"]) for i in range(cfg["num_env"])],
        start_method="fork",
    )

    device = "cuda" if th.cuda.is_available() else "cpu"
    print(f"device={device}")
    if device == "cuda":
        print(f"gpu={th.cuda.get_device_name(0)}")

    policy_kwargs = dict(
        features_extractor_class=CombinedExtractor,
        net_arch=dict(pi=[512, 512, 256, 128], vf=[512, 512, 256, 128]),
        activation_fn=th.nn.Tanh,
        ortho_init=True,
        optimizer_kwargs=dict(weight_decay=cfg["weight_decay"]),
    )

    model = PPO(
        MaskedGinRummyPolicy,
        train_env,
        learning_rate=cfg["learning_rate"],
        n_steps=cfg["n_steps"],
        batch_size=cfg["batch_size"],
        n_epochs=cfg["n_epochs"],
        gamma=cfg["gamma"],
        gae_lambda=cfg["gae_lambda"],
        clip_range=cfg["clip_range"],
        ent_coef=cfg["ent_coef"],
        vf_coef=cfg["vf_coef"],
        max_grad_norm=cfg["max_grad_norm"],
        device=device,
        policy_kwargs=policy_kwargs,
        verbose=0,
        seed=cfg["seed"],
    )

    callbacks = []
    if _wandb_enabled():
        run_name = (
            f"sweep_combo{combo}"
            f"_lr{cfg['learning_rate']:.0e}_ent{cfg['ent_coef']}"
            f"_ns{cfg['n_steps']}_ep{cfg['n_epochs']}"
        )
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "Adversarial-CoEvolution"),
            entity=os.environ.get("WANDB_ENTITY", "VLAvengers"),
            name=run_name,
            group="phase1-hp-sweep",
            tags=["phase1", "vs-random", f"combo{combo}"],
            config=cfg,
            reinit=True,
        )
        callbacks.append(WandbRolloutLogger())

    t0 = time.time()
    model.learn(
        total_timesteps=cfg["total_timesteps"],
        progress_bar=False,
        callback=callbacks or None,
    )
    train_seconds = time.time() - t0
    train_env.close()

    model_path = os.path.join(run_dir, "final")
    model.save(model_path)
    print(f"trained in {train_seconds:.0f}s; eval over {cfg['eval_episodes']} episodes")

    eval_stats = evaluate(model, cfg["eval_episodes"], cfg["turns_limit"])

    result = dict(
        combo=combo,
        train_seconds=train_seconds,
        model_path=model_path + ".zip",
        config=cfg,
        **eval_stats,
    )
    out_path = os.path.join(args.save_root, "results", f"run_{combo}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"WROTE {out_path}")
    print(json.dumps(eval_stats, indent=2))

    if _wandb_enabled():
        wandb.log({f"eval/{k}": v for k, v in eval_stats.items()})
        wandb.finish()


if __name__ == "__main__":
    main()
