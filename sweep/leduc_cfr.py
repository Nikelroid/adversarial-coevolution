"""Phase-8 second game (generality): a CFR-optimal expert for Leduc Hold'em via OpenSpiel.

Leduc is small enough to SOLVE, so its game-theoretic optimum is computable by CFR -- a clean
validation that the expert-yardstick methodology (grade agents against a fixed strong/optimal
expert) transfers to a second imperfect-information game. We solve leduc_poker by CFR, report
exploitability over iterations (-> 0 means near-optimal), grade a spectrum of agents (random and
partially-trained CFR) against the converged expert to show the yardstick is meaningful, and save
the expert policy.

    python sweep/leduc_cfr.py
"""
import os
import pickle
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pyspiel
from open_spiel.python.algorithms import cfr, exploitability
from open_spiel.python import policy as policy_lib


def head_to_head(game, p_hero, p_opp, n=2000, seed=0):
    """Average return of p_hero vs p_opp over n random playouts (hero in seat 0 and 1 split)."""
    rng = np.random.default_rng(seed)
    tot = 0.0
    for g in range(n):
        hero_seat = g % 2
        state = game.new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node():
                outcomes, probs = zip(*state.chance_outcomes())
                state.apply_action(int(rng.choice(outcomes, p=probs)))
                continue
            cur = state.current_player()
            pol = p_hero if cur == hero_seat else p_opp
            ap = pol.action_probabilities(state)
            acts, ps = list(ap.keys()), np.array(list(ap.values()))
            ps = ps / ps.sum()
            state.apply_action(int(rng.choice(acts, p=ps)))
        tot += state.returns()[hero_seat]
    return tot / n


def main():
    iters = int(os.environ.get("CFR_ITERS", 400))
    game = pyspiel.load_game("leduc_poker")
    solver = cfr.CFRSolver(game)

    snapshots = {}
    print(f"=== CFR on leduc_poker, {iters} iters ===", flush=True)
    for i in range(iters):
        solver.evaluate_and_update_policy()
        n = i + 1
        if n in (10, 50) or n % 100 == 0 or n == 1:
            expl = exploitability.exploitability(game, solver.average_policy())
            print(f"iter {n:4d}  exploitability={expl:.5f}", flush=True)
            if n in (10, 50):
                snapshots[n] = solver.average_policy().to_tabular()
    expert = solver.average_policy()
    final_expl = exploitability.exploitability(game, expert)
    print(f"FINAL exploitability={final_expl:.5f}  (0 = game-theoretic optimum)", flush=True)

    # The yardstick is meaningful: weaker agents lose to the converged expert, near-optimal ~ties.
    uniform = policy_lib.UniformRandomPolicy(game)
    grades = {"random": head_to_head(game, uniform, expert),
              "cfr_10iter": head_to_head(game, snapshots.get(10, uniform), expert),
              "cfr_50iter": head_to_head(game, snapshots.get(50, uniform), expert)}
    print("\n=== mean return vs the CFR expert (more negative = weaker; ~0 = near-optimal) ===")
    for k, v in grades.items():
        print(f"  {k:12s} {v:+.3f}")

    out = os.path.join(PROJECT_ROOT, "game", "model", "leduc_cfr_expert.pkl")
    with open(out, "wb") as f:
        pickle.dump({"tabular": expert.to_tabular().action_probability_array,
                     "iters": iters, "exploitability": final_expl, "grades": grades}, f)
    print(f"\nsaved expert + grades -> {out}")


if __name__ == "__main__":
    main()
