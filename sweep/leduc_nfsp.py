"""Phase-8 baseline (Wave 4): Neural Fictitious Self-Play (Heinrich & Silver 2016) on Leduc
Hold'em, graded against the CFR-OPTIMAL expert -- the canonical imperfect-information learning
baseline, slotted into the same expert-yardstick methodology as the rest of the study.

NFSP learns an approximate Nash equilibrium by self-play (a DQN best-response head + a supervised
average-policy head). We train it, then grade its AVERAGE policy (the equilibrium approximation)
against the fixed CFR expert, seat-averaged -- exactly the "grade vs a fixed strong expert" yardstick
used for Gin Rummy and the tabular-Q Leduc agent. A near-zero return vs the CFR optimum means NFSP
has approached the game-theoretic solution.

Dependency: OpenSpiel + PyTorch NFSP (+ dm-tree). No TensorFlow.

    EPISODES=400000 SEED=0 python sweep/leduc_nfsp.py
"""
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pyspiel
from open_spiel.python import rl_environment
from open_spiel.python.pytorch import nfsp

from sweep.leduc_train import cfr_expert  # reuse the exact CFR expert
from sweep.wandb_util import wandb_init, wandb_log, wandb_finish


def grade_nfsp(env, agent, agent_seat, expert, n, rng):
    """Mean return of NFSP's AVERAGE policy in `agent_seat` vs the CFR `expert`, over n games."""
    tot = 0.0
    for _ in range(n):
        ts = env.reset()
        while not ts.last():
            cur = ts.observations["current_player"]
            if cur == agent_seat:
                with agent.temp_mode_as(nfsp.MODE.AVERAGE_POLICY):
                    a = agent.step(ts, is_evaluation=True).action
            else:
                ap = expert.action_probabilities(env.get_state)
                acts = list(ap.keys())
                ps = np.array([ap[k] for k in acts], dtype=float)
                ps = ps / ps.sum()
                a = int(rng.choice(acts, p=ps))
            ts = env.step([a])
        tot += ts.rewards[agent_seat]
    return tot / n


def main():
    episodes = int(os.environ.get("EPISODES", 400_000))
    cfr_iters = int(os.environ.get("CFR_ITERS", 400))
    grade_games = int(os.environ.get("GRADE_GAMES", 2000))
    seed = int(os.environ.get("SEED", 0))

    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    game = pyspiel.load_game("leduc_poker")
    env = rl_environment.Environment("leduc_poker")
    env.seed(seed)
    info_state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    print(f"=== Leduc NFSP (seed={seed}, {episodes:,} eps): solving CFR expert ({cfr_iters} iters) ===",
          flush=True)
    expert, expl = cfr_expert(game, cfr_iters)
    print(f"CFR expert exploitability = {expl:.4f}", flush=True)
    wandb_init(name=f"leduc_nfsp_vs_cfr_s{seed}", group="phase8-leduc",
               config=dict(game="leduc_poker", agent="nfsp_selfplay", expert="cfr",
                           episodes=episodes, seed=seed, cfr_iters=cfr_iters),
               tags=["leduc", "nfsp", "baseline"])

    kwargs = dict(
        hidden_layers_sizes=[128, 128],
        reservoir_buffer_capacity=int(2e5),
        anticipatory_param=0.1,
        batch_size=128,
        rl_learning_rate=0.01,
        sl_learning_rate=0.01,
        min_buffer_size_to_learn=1000,
        learn_every=64,
        optimizer_str="sgd",
    )
    agents = [nfsp.NFSP(idx, info_state_size, num_actions, **kwargs) for idx in range(2)]

    print(f"=== NFSP self-play, {episodes:,} episodes ===", flush=True)
    curve = []
    every = max(1, episodes // 15)
    for ep in range(episodes):
        ts = env.reset()
        while not ts.last():
            pid = ts.observations["current_player"]
            ts = env.step([agents[pid].step(ts).action])
        for ag in agents:
            ag.step(ts)
        if (ep + 1) % every == 0 or ep + 1 == episodes:
            r0 = grade_nfsp(env, agents[0], 0, expert, grade_games, rng)
            r1 = grade_nfsp(env, agents[1], 1, expert, grade_games, rng)
            r = 0.5 * (r0 + r1)
            curve.append(dict(episode=ep + 1, return_vs_expert=round(r, 4)))
            wandb_log(dict(episode=ep + 1, return_vs_expert=r, return_seat0=r0, return_seat1=r1))
            print(f"ep {ep + 1:>8,}: NFSP avg-policy return vs CFR expert = {r:+.3f} "
                  f"(seat0 {r0:+.3f}, seat1 {r1:+.3f})", flush=True)

    out = os.path.join(PROJECT_ROOT, "sweep", "curriculum", f"leduc_nfsp_vs_cfr_s{seed}.json")
    result = dict(name=f"leduc_nfsp_vs_cfr_s{seed}", game="leduc_poker", agent="nfsp_selfplay",
                  expert="cfr", seed=seed, cfr_iters=cfr_iters, cfr_exploitability=round(expl, 4),
                  episodes=episodes, final_return_vs_expert=curve[-1]["return_vs_expert"], curve=curve)
    tmp = out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(result, f, indent=2)
    os.replace(tmp, out)
    wandb_finish(summary=dict(final_return_vs_expert=curve[-1]["return_vs_expert"],
                              cfr_exploitability=round(expl, 4)))
    print(f"[done] Leduc NFSP final return vs CFR expert = {curve[-1]['return_vs_expert']:+.3f} "
          f"(random baseline was about -0.78, tabular-Q reached near parity) -> {out}", flush=True)


if __name__ == "__main__":
    main()
