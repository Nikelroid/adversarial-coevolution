"""Print a leaderboard of sweep results sorted by win_rate vs RandomAgent."""
from __future__ import annotations

import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    results = []
    for path in sorted(glob.glob(os.path.join(HERE, "results", "run_*.json"))):
        with open(path) as f:
            results.append(json.load(f))

    if not results:
        print("no results yet")
        return

    results.sort(key=lambda r: r["win_rate"], reverse=True)
    print(
        f"{'combo':>5} {'win%':>6} {'loss%':>6} {'draw%':>6} {'mean_r':>8} "
        f"{'ep_len':>7} {'train_s':>8} {'lr':>8} {'ent':>5} {'n_steps':>8} {'epochs':>7}"
    )
    print("-" * 96)
    for r in results:
        c = r["config"]
        print(
            f"{r['combo']:5d} "
            f"{100*r['win_rate']:6.2f} {100*r['loss_rate']:6.2f} {100*r['draw_rate']:6.2f} "
            f"{r['mean_reward']:8.3f} {r['mean_length']:7.1f} {r['train_seconds']:8.0f} "
            f"{c['learning_rate']:8.0e} {c['ent_coef']:5.2f} {c['n_steps']:8d} {c['n_epochs']:7d}"
        )


if __name__ == "__main__":
    main()
