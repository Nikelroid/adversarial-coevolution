"""Benchmark the gold-standard agent against random and every trained PPO agent,
in the real PettingZoo gin_rummy_v4 env. Seats are swapped each half so neither
player gets a dealer advantage. Writes sweep/gold_bench.json.

    python sweep/bench_gold.py            # default 400 games / matchup
    GOLD_N=1000 python sweep/bench_gold.py
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pettingzoo.classic import gin_rummy_v4
from stable_baselines3 import PPO

from agents.gold_standard_agent import GoldStandardAgent
from agents.random_agent import RandomAgent
from agents.ppo_agent import PPOAgent
import ppo_train  # noqa: F401  registers MaskedGinRummyPolicy for unpickling

N = int(os.environ.get("GOLD_N", "400"))
KNOCK, GIN = 0.5, 1.5

MODEL_PATHS = {
    "champion": "game/model/ppo_gin_rummy_selfplay.zip",
    "pool":     "game/model/ppo_gin_rummy_pool.zip",
    "winrate":  "game/model/ppo_gin_rummy_winrate.zip",
    "reward":   "game/model/ppo_gin_rummy_reward.zip",
    "llm_full": "game/model/ppo_gin_rummy_llm_full.zip",
}
loaded = {k: PPO.load(p, device="cpu") for k, p in MODEL_PATHS.items()
          if os.path.exists(p)}
print("loaded PPO models:", list(loaded), flush=True)


def make_agent(kind, env):
    if kind == "gold":
        return GoldStandardAgent(env)
    if kind == "gold_hold":
        return GoldStandardAgent(env, hold_for_gin=True)
    if kind == "random":
        return RandomAgent(env)
    return PPOAgent(env, model=loaded[kind])


def play_match(a_kind, b_kind, n):
    """Play n games of a_kind vs b_kind, swapping seats each half. Returns stats
    from a_kind's perspective."""
    env = gin_rummy_v4.env(knock_reward=KNOCK, gin_reward=GIN,
                           opponents_hand_visible=False)
    def build(p0_kind, p1_kind):
        ags = {"player_0": make_agent(p0_kind, env),
               "player_1": make_agent(p1_kind, env)}
        for seat, ag in ags.items():
            ag.set_player(seat)
        return ags

    agents = build(a_kind, b_kind)
    res = dict(a_win=0, a_gin=0, a_loss=0, draw=0, b_gin=0, illegal=0, n=n)
    for g in range(n):
        # swap which seat a_kind occupies halfway through
        if g == n // 2:
            agents = build(b_kind, a_kind)
        a_seat = "player_0" if g < n // 2 else "player_1"
        b_seat = "player_1" if g < n // 2 else "player_0"
        env.reset(seed=g)
        tot = {"player_0": 0.0, "player_1": 0.0}
        for ag in env.agent_iter():
            obs, rew, term, trunc, info = env.last()
            tot[ag] += rew
            if term or trunc:
                env.step(None); continue
            act = agents[ag].do_action()
            if obs["action_mask"][act] != 1:
                res["illegal"] += 1
            env.step(act)
        ra, rb = tot[a_seat], tot[b_seat]
        if ra > 0:
            res["a_win"] += 1
            if ra >= GIN - 0.1:
                res["a_gin"] += 1
        elif ra < 0:
            res["a_loss"] += 1
        else:
            res["draw"] += 1
        if rb >= GIN - 0.1:
            res["b_gin"] += 1
    env.close()
    for k in ("a_win", "a_gin", "a_loss", "draw", "b_gin"):
        res[k + "_rate"] = round(res[k] / n, 4)
    return res


def main():
    t0 = time.time()
    out = {"n": N, "matchups": {}}
    opponents = ["random"] + list(loaded) + ["gold"]  # gold vs gold = sanity ~0.5
    for opp in opponents:
        r = play_match("gold", opp, N)
        out["matchups"][opp] = r
        print(f"gold vs {opp:9s}: win={r['a_win_rate']:.3f} "
              f"gin={r['a_gin_rate']:.3f} loss={r['a_loss_rate']:.3f} "
              f"(opp gin={r['b_gin_rate']:.3f}) illegal={r['illegal']}", flush=True)
    # gin-vs-knock study: holding for gin instead of knocking ASAP.
    out["gold_hold_vs_random"] = play_match("gold_hold", "random", N)
    h = out["gold_hold_vs_random"]
    print(f"gold_hold vs random: win={h['a_win_rate']:.3f} gin={h['a_gin_rate']:.3f}",
          flush=True)
    if "champion" in loaded:
        out["gold_hold_vs_champion"] = play_match("gold_hold", "champion", N)
        hc = out["gold_hold_vs_champion"]
        print(f"gold_hold vs champion: win={hc['a_win_rate']:.3f} "
              f"gin={hc['a_gin_rate']:.3f}", flush=True)
    out["seconds"] = round(time.time() - t0)
    json.dump(out, open("sweep/gold_bench.json", "w"), indent=2)
    print(f"\nwrote sweep/gold_bench.json in {out['seconds']}s")


if __name__ == "__main__":
    main()
