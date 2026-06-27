"""Head-to-head: a trained RL model vs the ISMCTS search agent, to see how the learned agents
actually stand against principled search. Same seat-swapped scoring as the rest of the study; the
model picks the highest-probability LEGAL action (masked-argmax, never a random fallback), and the
opponent is the determinized ISMCTS (no peeking at the model's hidden cards) by default.

    MODEL_PATH=game/model/gin_ace.zip MODEL_LABEL=ace ISMCTS_ROLLOUTS=60 N=300 python sweep/model_vs_ismcts.py
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
from agents.ppo_agent import PPOAgent
import ppo_train  # noqa: F401  (registers MaskedGinRummyPolicy)
from sweep.llm_dagger_one import ENVK, GIN
from sweep.wandb_util import wandb_init, wandb_finish


def load_model(path):
    p = path[:-4] if path.endswith(".zip") else path
    try:
        return PPO.load(p, device="cpu")
    except Exception:
        from sb3_contrib import TRPO
        return TRPO.load(p, device="cpu")


def play(model, n, rollouts, determinize, seed0=10_000):
    env = gin_rummy_v4.env(**ENVK)
    hero = PPOAgent(env=env, model=model)
    opp = ISMCTSAgent(env, rollouts=rollouts, determinize=determinize)
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
    model_path = os.environ.get("MODEL_PATH", "game/model/gin_ace.zip")
    label = os.environ.get("MODEL_LABEL", os.path.splitext(os.path.basename(model_path))[0])
    rollouts = int(os.environ.get("ISMCTS_ROLLOUTS", 60))
    determinize = bool(int(os.environ.get("ISMCTS_DETERMINIZE", 1)))
    n = int(os.environ.get("N", 300))
    mode = "det" if determinize else "oracle"
    name = os.environ.get("NAME", f"h2h_{label}_vs_ismcts_{mode}_r{rollouts}")

    print(f"=== {label} vs ISMCTS ({mode}, rollouts={rollouts}): {n} games ===", flush=True)
    model = load_model(os.path.join(PROJECT_ROOT, model_path) if not os.path.isabs(model_path) else model_path)
    wandb_init(name=name, group="phase8-head2head",
               config=dict(model=label, model_path=model_path, opponent="ismcts",
                           rollouts=rollouts, determinize=determinize, n=n),
               tags=["head2head", "ismcts", label])
    t0 = time.time()
    res = play(model, n, rollouts, determinize)
    secs = time.time() - t0
    out = os.path.join(PROJECT_ROOT, "sweep", "curriculum", f"{name}.json")
    result = dict(name=name, model=label, model_path=model_path, opponent="ismcts",
                  rollouts=rollouts, determinize=determinize, eval_seconds=secs,
                  vs_ismcts=res)
    tmp = out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(result, f, indent=2)
    os.replace(tmp, out)
    wandb_finish(summary=dict(win_rate=res["win_rate"], gin_rate=res["gin_rate"],
                              loss_rate=res["loss_rate"], rollouts=rollouts))
    print(f"[done] {label} vs ISMCTS({mode},r{rollouts}): model win_rate={res['win_rate']:.3f} "
          f"(gin {res['gin_rate']:.3f}, loss {res['loss_rate']:.3f}) in {secs:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
