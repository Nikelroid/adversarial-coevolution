"""Integration smoke for the Phase-2 RL<->LLM path: run Gin Rummy with the LLM
(served via the master) as the OPPONENT, exactly as training would.

Requires a running master reachable at $GINLLM_MASTER_URL. Plays a few episodes
with a random 'training agent'; every opponent turn queries the LLM through
OllamaAPI -> master -> worker (and through the suit-symmetry canonical cache).
Reports cache hit-rate + timing so we can see the cache working.

    GINLLM_MASTER_URL=http://127.0.0.1:11434 python -m llm.llm_opponent_smoke
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import requests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from gym_wrapper import GinRummySB3Wrapper  # noqa: E402
from agents.llm_agent import LLMAgent  # noqa: E402


def main():
    url = os.environ.get("GINLLM_MASTER_URL", "http://127.0.0.1:11434")
    print("master health:", requests.get(url + "/health", timeout=15).json(), flush=True)

    n_ep = int(os.environ.get("N_EP", "3"))
    env = GinRummySB3Wrapper(opponent_policy=LLMAgent, randomize_position=True,
                             turns_limit=200, curriculum_manager=None)
    t0 = time.time()
    for ep in range(n_ep):
        obs, _ = env.reset()
        done, steps, r = False, 0, 0.0
        while not done and steps < 300:
            legal = np.where(obs["action_mask"])[0]
            obs, r, done, trunc, _ = env.step(int(np.random.choice(legal)))
            steps += 1
            if trunc:
                done = True
        print(f"ep {ep}: {steps} agent-steps, final reward {r:.3f}", flush=True)
    dt = time.time() - t0

    stats = requests.get(url + "/stats", timeout=15).json()
    cache = stats.get("cache", {})
    print(f"cache: hits={cache.get('hits')} misses={cache.get('misses')} "
          f"hit_rate={cache.get('hit_rate')}", flush=True)
    print(f"workers: {stats.get('pool', {}).get('n_healthy')} healthy", flush=True)
    print(f"completed {n_ep} episodes in {dt:.1f}s "
          f"({cache.get('total', 0)} LLM queries, "
          f"{cache.get('misses', 0)} reached a worker)", flush=True)
    print("LLM_OPPONENT_SMOKE_OK", flush=True)


if __name__ == "__main__":
    main()
