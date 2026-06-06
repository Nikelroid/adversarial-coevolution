"""A simple, game-agnostic opponent curriculum.

It schedules opponents over training: ``random`` early, then a growing ``pool`` (curated seed
models plus the agent's own past checkpoints), then ``self`` (the current model). The wrapper asks
it for an opponent type each episode and for a loaded model to play. State (total steps) is shared
across forked worker processes through one small JSON file, the same proven mechanism used in the
Gin Rummy sweeps.

stages: a list like
    [{"until": 0.25, "mix": {"random": 1.0}},
     {"until": 0.60, "mix": {"random": 0.4, "pool": 0.6}},
     {"until": 1.01, "mix": {"random": 0.2, "pool": 0.5, "self": 0.3}}]
where ``until`` is a fraction of total training and ``mix`` sums to 1.
"""
import glob
import json
import os
import random
import shutil

import numpy as np
from stable_baselines3 import PPO

_LOAD_CACHE: dict = {}


def load_any(path):
    """Load an SB3 zip as a frozen inference policy, trying PPO then TRPO. Cached per path."""
    if path in _LOAD_CACHE:
        return _LOAD_CACHE[path]
    last = None
    for kind in ("ppo", "trpo"):
        try:
            if kind == "ppo":
                m = PPO.load(path, device="cpu")
            else:
                from sb3_contrib import TRPO
                m = TRPO.load(path, device="cpu")
            _LOAD_CACHE[path] = m
            return m
        except Exception as e:  # noqa: BLE001
            last = e
    print(f"[curriculum] could not load {path}: {last}", flush=True)
    return None


def default_stages():
    return [
        {"until": 0.25, "mix": {"random": 1.0}},
        {"until": 0.60, "mix": {"random": 0.40, "pool": 0.60}},
        {"until": 1.01, "mix": {"random": 0.20, "pool": 0.50, "self": 0.30}},
    ]


class Curriculum:
    def __init__(self, pool_dir, stages, total_steps, recent_n=10):
        self.pool_dir = pool_dir
        self.stages = stages
        self.total = max(int(total_steps), 1)
        self.recent_n = recent_n
        self.state_file = os.path.join(pool_dir, "state.json")
        os.makedirs(pool_dir, exist_ok=True)
        self.total_steps = 0
        self._last_ckpt = -10 ** 18
        self._load()

    # ---- shared step counter ----
    def _load(self):
        try:
            self.total_steps = json.load(open(self.state_file)).get("total_steps", 0)
        except Exception:
            self.total_steps = 0

    def _save(self):
        tmp = self.state_file + f".tmp{os.getpid()}"
        try:
            json.dump({"total_steps": self.total_steps}, open(tmp, "w"))
            os.replace(tmp, self.state_file)
        except Exception as e:  # noqa: BLE001
            print(f"[curriculum] save error: {e}", flush=True)

    def update_total_steps(self, steps):
        self._load()
        if steps - self.total_steps > 5000:
            self.total_steps = steps
            self._save()

    def episode_complete(self):
        pass

    # ---- pool discovery ----
    def _seeds(self):
        return sorted(glob.glob(os.path.join(self.pool_dir, "seed_*.zip")))

    def _selves(self):
        fs = [f for f in glob.glob(os.path.join(self.pool_dir, "policy_step_*.zip"))
              if f.split("_")[-1].replace(".zip", "").isdigit()]
        fs.sort(key=lambda x: int(x.split("_")[-1].replace(".zip", "")))
        return fs

    def _get_available_policies(self):
        return [f.replace(".zip", "") for f in self._seeds() + self._selves()]

    def selfplay_path(self):
        return os.path.join(self.pool_dir, "current_self")

    # ---- opponent choice ----
    def _stage(self, steps):
        frac = steps / self.total
        for i, st in enumerate(self.stages):
            if frac < st["until"]:
                return i, st
        return len(self.stages) - 1, self.stages[-1]

    def get_opponent_type(self):
        self._load()
        idx, st = self._stage(self.total_steps)
        kinds = list(st["mix"]); w = np.array([st["mix"][k] for k in kinds], float); w /= w.sum()
        choice = kinds[int(min(np.searchsorted(np.cumsum(w), random.random(), side="right"),
                               len(kinds) - 1))]
        have_pool = bool(self._seeds() or self._selves())
        if choice == "pool" and not have_pool:
            choice = "random"
        if choice == "self" and not os.path.exists(self.selfplay_path() + ".zip"):
            choice = "pool" if have_pool else "random"
        return choice, idx + 1, self.total_steps

    def get_policy_from_pool(self, recent_n=None):
        recent_n = recent_n or self.recent_n
        selves = self._selves()
        cands = self._seeds() + (selves[-recent_n:] if len(selves) > recent_n else selves)
        if not cands:
            return None
        return load_any(random.choice(cands).replace(".zip", ""))

    def get_selfplay_policy(self):
        p = self.selfplay_path()
        return load_any(p) if os.path.exists(p + ".zip") else None

    # ---- checkpointing (main process only) ----
    def should_save_checkpoint(self, step, save_freq):
        return step - self._last_ckpt >= save_freq

    def save_checkpoint(self, model, step, max_self=15):
        model.save(os.path.join(self.pool_dir, f"policy_step_{step}"))
        model.save(self.selfplay_path())
        self._last_ckpt = step
        for old in self._selves()[:-max_self]:
            try:
                os.remove(old)
            except OSError:
                pass


def seed_pool(pool_dir, seed_models, fresh=True):
    """(Re)create the pool dir and copy curated seed opponents into it. ``seed_models`` is a list
    of paths to SB3 ``.zip`` models (any prior agents you want the new one to practise against)."""
    if fresh and os.path.isdir(pool_dir):
        shutil.rmtree(pool_dir)
    os.makedirs(pool_dir, exist_ok=True)
    kept = []
    for i, src in enumerate(seed_models or []):
        src = src if src.endswith(".zip") else src + ".zip"
        if os.path.exists(src):
            shutil.copy(src, os.path.join(pool_dir, f"seed_{i:02d}.zip"))
            kept.append(os.path.basename(src))
        else:
            print(f"[curriculum] seed not found, skipped: {src}", flush=True)
    print(f"[curriculum] {pool_dir}: seeded {kept}", flush=True)
    return kept
