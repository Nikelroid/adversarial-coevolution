"""Lightweightness benchmark for the shipped champion policy (gin_ace).

Measures, on ONE CPU thread:
  - parameter count + on-disk size
  - per-move latency of the exact eval path (agents.action_utils.masked_argmax)
Prints a small JSON at the end.
"""
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch

torch.set_num_threads(1)

from stable_baselines3 import PPO
from agents.action_utils import masked_argmax

ZIP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "game", "model", "gin_ace.zip")
model = PPO.load(ZIP, device="cpu")

n_params = sum(p.numel() for p in model.policy.parameters())
disk_mb = os.path.getsize(ZIP) / 1e6
space = model.observation_space
print("obs space:", space)

rng = np.random.default_rng(0)


def make_obs():
    s = space.sample()
    if isinstance(s, dict):
        if "action_mask" in s:
            m = np.zeros_like(np.asarray(s["action_mask"]))
            legal = rng.choice(m.shape[-1], size=12, replace=False)
            m.reshape(-1)[legal] = 1
            s["action_mask"] = m
        return s
    return s


obs_pool = [make_obs() for _ in range(64)]

# warmup
for i in range(200):
    masked_argmax(model, obs_pool[i % 64])

N = 2000
times = []
for i in range(N):
    o = obs_pool[i % 64]
    t0 = time.perf_counter()
    masked_argmax(model, o)
    times.append(time.perf_counter() - t0)

ms = sorted(t * 1e3 for t in times)
res = dict(
    model="gin_ace (Curriculum Ace, headline agent)",
    torch_threads=torch.get_num_threads(),
    params=int(n_params),
    disk_mb=round(disk_mb, 2),
    moves_timed=N,
    ms_per_move_median=round(ms[N // 2], 3),
    ms_per_move_mean=round(sum(ms) / N, 3),
    ms_per_move_p95=round(ms[int(N * 0.95)], 3),
    moves_per_sec_1core=round(1000.0 / (sum(ms) / N)),
)
print(json.dumps(res, indent=2))
