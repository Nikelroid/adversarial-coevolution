"""Grade the PIMC/ISMCTS search agent against the fixed expert (gold), random, and the champion --
using the SAME seat-swapped scoring as curriculum_train.eval_full, so the win-rate vs the expert is
directly comparable to the architecture sweep and the gold benchmark.

    OPP=gold N=300 ISMCTS_ROLLOUTS=20 python sweep/ismcts_eval.py
"""
import json
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
try:
    import cv2  # noqa: F401
except Exception:
    pass
from pettingzoo.classic import gin_rummy_v4
from stable_baselines3 import PPO

from agents.ismcts_agent import ISMCTSAgent
from agents.gold_standard_agent import GoldStandardAgent
from agents.random_agent import RandomAgent
from agents.ppo_agent import PPOAgent
import ppo_train  # noqa: F401
from sweep.llm_dagger_one import ENVK, GIN


def play(opp_kind, n, seed0, champ=None):
    env = gin_rummy_v4.env(**ENVK)
    hero = ISMCTSAgent(env)
    opp = (GoldStandardAgent(env) if opp_kind == "gold"
           else RandomAgent(env) if opp_kind == "random" else PPOAgent(env, model=champ))
    win = gin = loss = 0
    lengths = []
    for g in range(n):
        h_seat = "player_0" if g % 2 == 0 else "player_1"
        agents = ({"player_0": hero, "player_1": opp} if h_seat == "player_0"
                  else {"player_0": opp, "player_1": hero})
        for k, a in agents.items():
            a.set_player(k)
        env.reset(seed=seed0 + g)
        tot = {"player_0": 0.0, "player_1": 0.0}
        steps = 0
        for ag in env.agent_iter():
            obs, rew, term, trunc, info = env.last()
            tot[ag] += rew
            if term or trunc:
                env.step(None)
                continue
            env.step(agents[ag].do_action())
            steps += 1
        lengths.append(steps)
        r = tot[h_seat]
        if r > 0:
            win += 1
            gin += int(r >= GIN - 0.1)
        elif r < 0:
            loss += 1
    env.close()
    return dict(win_rate=win / n, gin_rate=gin / n, loss_rate=loss / n,
                mean_len=float(np.mean(lengths)), n=n)


def main():
    opp = os.environ.get("OPP", "gold")
    n = int(os.environ.get("N", 300))
    name = os.environ.get("NAME", f"ismcts_vs_{opp}")
    champ = None
    if opp == "champion":
        cp = os.environ.get("CHAMP_MODEL", "game/model/gin_curriculum_champion.zip")
        champ = PPO.load(cp[:-4] if cp.endswith(".zip") else cp, device="cpu")
    print(f"=== ISMCTS eval vs {opp}: {n} games, rollouts="
          f"{os.environ.get('ISMCTS_ROLLOUTS', 20)} ===", flush=True)
    t0 = time.time()
    res = play(opp, n, 10_000, champ)
    secs = time.time() - t0
    out = os.path.join(PROJECT_ROOT, "sweep", "curriculum", f"{name}.json")
    result = dict(name=name, agent="ismcts_pimc", opponent=opp, rollouts=int(os.environ.get("ISMCTS_ROLLOUTS", 20)),
                  eval_seconds=secs, **{f"vs_{opp}": res})
    tmp = out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(result, f, indent=2)
    os.replace(tmp, out)
    print(f"[done] {name}: win_rate={res['win_rate']:.3f} gin_rate={res['gin_rate']:.3f} "
          f"loss_rate={res['loss_rate']:.3f} mean_len={res['mean_len']:.1f} in {secs:.0f}s -> {out}",
          flush=True)


if __name__ == "__main__":
    main()
