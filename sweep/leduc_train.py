"""Phase-8 generality (Wave 3): train an RL agent on Leduc Hold'em and grade it against the
CFR-OPTIMAL expert over training -- the same methodology as the Gin Rummy study, on a second
imperfect-information game where the optimum is computable. Dependency-light (OpenSpiel tabular
Q-learner + CFR; no TensorFlow).

The yardstick = mean return of the (greedy) RL agent vs the fixed CFR expert, seat-averaged. As the
agent learns by self-play, its return-vs-expert should rise from badly-losing toward parity --
exactly the "grade an agent against a fixed strong expert" yardstick, transferred to a new game.

    EPISODES=300000 python sweep/leduc_train.py
"""
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pyspiel
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import cfr, tabular_qlearner, exploitability


def cfr_expert(game, iters):
    solver = cfr.CFRSolver(game)
    for _ in range(iters):
        solver.evaluate_and_update_policy()
    return solver.average_policy(), exploitability.exploitability(game, solver.average_policy())


def grade(env, agent, agent_seat, expert, n, rng):
    """Mean return of `agent` (greedy) in seat `agent_seat` vs the CFR `expert`."""
    tot = 0.0
    for _ in range(n):
        ts = env.reset()
        while not ts.last():
            cur = ts.observations["current_player"]
            if cur == agent_seat:
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
    episodes = int(os.environ.get("EPISODES", 300_000))
    cfr_iters = int(os.environ.get("CFR_ITERS", 400))
    grade_games = int(os.environ.get("GRADE_GAMES", 1000))
    game = pyspiel.load_game("leduc_poker")
    env = rl_environment.Environment("leduc_poker")
    nA = env.action_spec()["num_actions"]
    rng = np.random.default_rng(0)

    print(f"=== Leduc: solving CFR expert ({cfr_iters} iters) ===", flush=True)
    expert, expl = cfr_expert(game, cfr_iters)
    print(f"CFR expert exploitability = {expl:.4f}", flush=True)

    agents = [tabular_qlearner.QLearner(player_id=i, num_actions=nA) for i in range(2)]
    print(f"=== self-play tabular Q-learning, {episodes:,} episodes ===", flush=True)
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
            r0 = grade(env, agents[0], 0, expert, grade_games, rng)
            r1 = grade(env, agents[1], 1, expert, grade_games, rng)
            r = 0.5 * (r0 + r1)
            curve.append(dict(episode=ep + 1, return_vs_expert=round(r, 4)))
            print(f"ep {ep + 1:>8,}: return vs CFR expert = {r:+.3f} "
                  f"(seat0 {r0:+.3f}, seat1 {r1:+.3f})", flush=True)

    out = os.path.join(PROJECT_ROOT, "sweep", "curriculum", "leduc_rl_vs_cfr.json")
    result = dict(name="leduc_rl_vs_cfr", game="leduc_poker", agent="tabular_qlearning_selfplay",
                  expert="cfr", cfr_iters=cfr_iters, cfr_exploitability=round(expl, 4),
                  episodes=episodes, final_return_vs_expert=curve[-1]["return_vs_expert"],
                  curve=curve)
    tmp = out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(result, f, indent=2)
    os.replace(tmp, out)
    print(f"[done] Leduc RL final return vs CFR expert = {curve[-1]['return_vs_expert']:+.3f} "
          f"(random baseline was about -0.78) -> {out}", flush=True)


if __name__ == "__main__":
    main()
