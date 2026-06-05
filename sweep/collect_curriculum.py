"""Aggregate the Phase-6 sweep cells (sweep/curriculum/*.json) into one summary the report
and figures consume. Prints a readable table, picks the best cell per axis, and writes
sweep/curriculum/_summary.json (+ a tidy CSV). Safe to run while cells are still landing --
it simply reports whatever has finished.

    python sweep/collect_curriculum.py
"""
import csv
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(HERE, "curriculum")


def load_cells():
    cells = []
    for f in sorted(glob.glob(os.path.join(DIR, "*.json"))):
        if os.path.basename(f).startswith("_"):
            continue
        try:
            cells.append(json.load(open(f)))
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {f}: {e}")
    return cells


def row(c):
    return dict(
        name=c.get("name", "?"), algo=c.get("algo", "?"),
        curriculum=c.get("curriculum", "?"),
        knock=c.get("reward", {}).get("knock"), gin=c.get("reward", {}).get("gin"),
        loss_scale=c.get("reward", {}).get("loss_scale"),
        seed=c.get("seed"),
        vs_random=round(c.get("vs_random", {}).get("win_rate", float("nan")), 3),
        vs_champion=round(c.get("vs_champion", {}).get("win_rate", float("nan")), 3),
        vs_gold=round(c.get("vs_gold", {}).get("win_rate", float("nan")), 3),
        gin_vs_gold=round(c.get("vs_gold", {}).get("gin_rate", float("nan")), 3),
        steps=c.get("steps"), secs=round(c.get("train_seconds", 0)),
    )


def main():
    cells = load_cells()
    if not cells:
        print(f"no cells yet in {DIR}")
        return
    rows = [row(c) for c in cells]
    rows.sort(key=lambda r: (r["vs_gold"] if r["vs_gold"] == r["vs_gold"] else -1), reverse=True)

    hdr = ["name", "algo", "curriculum", "knock", "gin", "loss_scale", "seed",
           "vs_random", "vs_champion", "vs_gold", "gin_vs_gold"]
    w = {h: max(len(h), *(len(str(r[h])) for r in rows)) for h in hdr}
    print("  ".join(h.ljust(w[h]) for h in hdr))
    print("  ".join("-" * w[h] for h in hdr))
    for r in rows:
        print("  ".join(str(r[h]).ljust(w[h]) for h in hdr))

    finished = [r for r in rows if r["vs_gold"] == r["vs_gold"]]
    best = finished[0] if finished else None
    if best:
        print(f"\nBEST vs gold: {best['name']} -> {best['vs_gold']:.3f} "
              f"(champion {best['vs_champion']:.3f}, random {best['vs_random']:.3f})")
        ppo = [r for r in finished if r["algo"] == "ppo"]
        trpo = [r for r in finished if r["algo"] == "trpo"]
        if ppo and trpo:
            print(f"best PPO  vs gold: {max(r['vs_gold'] for r in ppo):.3f}")
            print(f"best TRPO vs gold: {max(r['vs_gold'] for r in trpo):.3f}")

    os.makedirs(DIR, exist_ok=True)
    json.dump(dict(n_cells=len(cells), best=best, rows=rows),
              open(os.path.join(DIR, "_summary.json"), "w"), indent=2)
    with open(os.path.join(DIR, "_summary.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)
    print(f"\nwrote {os.path.join(DIR, '_summary.json')} and _summary.csv ({len(cells)} cells)")


if __name__ == "__main__":
    main()
