"""Round-robin head-to-head among the trained agents (+ random). CPU-only,
no LLM — runs in parallel with GPU training. Writes sweep/h2h.json."""
import os, sys, json, functools, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: F401
from stable_baselines3 import PPO
from gym_wrapper import GinRummySB3Wrapper
from agents.random_agent import RandomAgent
from agents.ppo_agent import PPOAgent
import ppo_train  # noqa: F401  registers MaskedGinRummyPolicy

MODELS = {
    "selfplay": "game/model/ppo_gin_rummy_selfplay.zip",
    "pool":     "game/model/ppo_gin_rummy_pool.zip",
    "winrate":  "game/model/ppo_gin_rummy_winrate.zip",
    "reward":   "game/model/ppo_gin_rummy_reward.zip",
}
for k, p in [("llm_full", "game/model/ppo_gin_rummy_llm_full.zip"),
             ("llm_40k", "game/model/ppo_gin_rummy_llm.zip")]:
    if os.path.exists(p):
        MODELS[k] = p
        break

N = int(os.environ.get("H2H_N", "300"))
loaded = {k: PPO.load(p, device="cpu") for k, p in MODELS.items()}
print("loaded:", list(loaded), flush=True)


def play(hero, opp_factory, n):
    env = GinRummySB3Wrapper(opponent_policy=opp_factory, randomize_position=True,
                             turns_limit=200, curriculum_manager=None)
    w = 0
    for _ in range(n):
        obs, _ = env.reset(); done = False; r = 0.0
        while not done:
            a, _ = hero.predict(obs, deterministic=True)
            obs, r, done, trunc, _ = env.step(a)
            if trunc:
                done = True
        if r > 1e-6:
            w += 1
    env.close()
    return w / n


opps = {**{k: functools.partial(PPOAgent, model=m) for k, m in loaded.items()},
        "random": RandomAgent}
matrix, t0 = {}, time.time()
for hk, hm in loaded.items():
    matrix[hk] = {}
    for ok, of in opps.items():
        if ok == hk:
            matrix[hk][ok] = None
            continue
        wr = play(hm, of, N)
        matrix[hk][ok] = round(wr, 3)
        print(f"{hk:9s} vs {ok:9s}: {wr:.3f}", flush=True)

out = dict(n=N, models=list(loaded), matrix=matrix, seconds=round(time.time() - t0))
json.dump(out, open("sweep/h2h.json", "w"), indent=2)
print("H2H_DONE -> sweep/h2h.json", flush=True)
