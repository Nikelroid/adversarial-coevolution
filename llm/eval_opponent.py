"""Measure how STRONG the LLM opponent is (not just that the pipeline runs).

Plays N full Gin Rummy games with the LLM (served via the master) as the
OPPONENT against a chosen "hero" agent, and reports the LLM's win/loss/draw
record. LLM-vs-random win-rate is the decisive competence signal: a model that
can't parse or reason falls back to random moves and lands near 50%; a capable
model wins clearly more. (OLMoE-1B sat at chance — that's why it was useless.)

    GINLLM_MASTER_URL=http://127.0.0.1:11434 N_EP=20 HERO=random \
        python -m llm.eval_opponent
    HERO=ppo PPO_PATH=game/model/ppo_gin_rummy_selfplay.zip python -m llm.eval_opponent

The wrapper returns the terminal reward from the HERO's perspective:
    reward > 0  -> hero won  (LLM lost)
    reward < 0  -> LLM won
    reward ~ 0  -> draw / dead hand
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
from agents.llm_agent import LLMAgent       # noqa: E402


def _make_hero(kind, env):
    """Return a function obs->action for the hero (the agent the LLM plays)."""
    if kind == "ppo":
        from agents.ppo_agent import PPOAgent
        path = os.environ.get("PPO_PATH", "game/model/ppo_gin_rummy_selfplay.zip")
        agent = PPOAgent(model_path=path, env=env.env)
        print(f"[hero] PPO from {path}", flush=True)

        def act(obs):
            agent.set_player(env.training_agent)
            return agent.do_action()
        return act

    print("[hero] random", flush=True)

    def act(obs):
        legal = np.where(obs["action_mask"])[0]
        return int(np.random.choice(legal))
    return act


def main():
    url = os.environ.get("GINLLM_MASTER_URL", "http://127.0.0.1:11434")
    print("master health:", requests.get(url + "/health", timeout=15).json(), flush=True)

    n_ep = int(os.environ.get("N_EP", "20"))
    hero_kind = os.environ.get("HERO", "random")
    env = GinRummySB3Wrapper(opponent_policy=LLMAgent, randomize_position=True,
                             turns_limit=200, curriculum_manager=None)
    hero = _make_hero(hero_kind, env)

    llm_w = llm_l = draw = 0
    total_turns = 0
    t0 = time.time()
    for ep in range(n_ep):
        obs, _ = env.reset()
        done, steps, r = False, 0, 0.0
        while not done and steps < 300:
            obs, r, done, trunc, _ = env.step(hero(obs))
            steps += 1
            if trunc:
                done = True
        total_turns += steps
        if r > 1e-6:
            llm_l += 1; res = "LLM_loss"
        elif r < -1e-6:
            llm_w += 1; res = "LLM_win"
        else:
            draw += 1; res = "draw"
        print(f"ep {ep:2d}: {steps:3d} hero-steps, hero_reward={r:+.3f} -> {res}", flush=True)

    dt = time.time() - t0
    decided = llm_w + llm_l
    wr = (llm_w / decided) if decided else float("nan")
    print("-" * 60, flush=True)
    print(f"LLM as opponent vs {hero_kind}: "
          f"W={llm_w} L={llm_l} D={draw}  (N={n_ep})", flush=True)
    print(f"LLM win-rate (decided games): {wr:.1%}", flush=True)
    print(f"avg game length: {total_turns / max(n_ep,1):.1f} hero-steps | "
          f"{dt:.1f}s total ({dt/max(n_ep,1):.1f}s/game)", flush=True)

    try:
        stats = requests.get(url + "/stats", timeout=15).json()
        c = stats.get("cache", {})
        print(f"cache: hits={c.get('hits')} misses={c.get('misses')} "
              f"hit_rate={c.get('hit_rate')} | workers="
              f"{stats.get('pool', {}).get('n_healthy')}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print("stats unavailable:", exc, flush=True)
    print("EVAL_OPPONENT_DONE", flush=True)


if __name__ == "__main__":
    main()
