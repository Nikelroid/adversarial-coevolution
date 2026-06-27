"""Organized eval summary for the Phase-8 architecture sweep.

Scans sweep/curriculum/arch_*.json and reports, per run and per architecture, the headline
evaluation against the FIXED EXPERT (gold) yardstick, plus the champion / random references and
the gin-rate + mean game length. Writes a clean CSV, a Markdown table, and a JSON, and prints a
readable leaderboard. Safe on a partial sweep -- it reports whatever has finished.

    python sweep/arch_collect.py
"""
import csv
import glob
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(HERE, "curriculum")
ANCHOR = "arch_mlp_default"          # the baseline architecture cell (winner net, from scratch)
NAN = float("nan")


def _na(c):
    na = c.get("net_arch")
    if na is None:
        return "[256,128]"
    if isinstance(na, dict):
        return str(na.get("pi"))
    return str(na)


def load():
    rows = []
    for f in sorted(glob.glob(os.path.join(DIR, "arch_*.json"))):
        try:
            c = json.load(open(f))
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {f}: {e}")
            continue
        g = c.get("vs_gold", {}) or {}
        ch = c.get("vs_champion", {}) or {}
        rd = c.get("vs_random", {}) or {}
        name = c.get("name", "?")
        rows.append(dict(
            name=name, cell=name.rsplit("_s", 1)[0], arch=c.get("arch", "mlp"),
            activation=c.get("activation", "tanh"), net_arch=_na(c), seed=c.get("seed"),
            vs_gold=round(g.get("win_rate", NAN), 4), gin_vs_gold=round(g.get("gin_rate", NAN), 4),
            len_vs_gold=round(g.get("mean_len", NAN), 2),
            vs_champion=round(ch.get("win_rate", NAN), 4), vs_random=round(rd.get("win_rate", NAN), 4),
            best_step=c.get("best_step"), secs=round(c.get("train_seconds", 0)),
        ))
    return rows


def _table(rows, cols):
    w = {h: max(len(h), *(len(str(r[h])) for r in rows)) for h in cols}
    print("  ".join(h.ljust(w[h]) for h in cols))
    print("  ".join("-" * w[h] for h in cols))
    for r in rows:
        print("  ".join(str(r[h]).ljust(w[h]) for h in cols))


def main():
    rows = load()
    if not rows:
        print(f"no arch results yet in {DIR} (sweep still running)")
        return
    rows.sort(key=lambda r: (r["vs_gold"] if r["vs_gold"] == r["vs_gold"] else -1), reverse=True)

    print(f"PER-RUN ({len(rows)} finished, sorted by win-rate vs the fixed expert):")
    _table(rows, ["name", "arch", "activation", "net_arch", "seed", "vs_gold", "gin_vs_gold",
                  "len_vs_gold", "vs_champion", "vs_random", "best_step", "secs"])

    # per-architecture aggregate across seeds
    byc = {}
    for r in rows:
        if r["vs_gold"] == r["vs_gold"]:
            byc.setdefault(r["cell"], []).append(r["vs_gold"])
    anchor = st.mean(byc[ANCHOR]) if ANCHOR in byc else None
    agg = []
    for cell, vs in byc.items():
        m = st.mean(vs)
        agg.append(dict(cell=cell, n=len(vs), mean_vs_gold=round(m, 4),
                        std=round(st.pstdev(vs) if len(vs) > 1 else 0.0, 4),
                        delta_vs_anchor=(round(m - anchor, 4) if anchor is not None else None)))
    agg.sort(key=lambda a: a["mean_vs_gold"], reverse=True)
    print("\nPER-ARCHITECTURE (mean win-rate vs expert over seeds; delta vs the MLP anchor):")
    _table(agg, ["cell", "n", "mean_vs_gold", "std", "delta_vs_anchor"])
    if agg and anchor is not None:
        b = agg[0]
        print(f"\nBEST: {b['cell']} -> {b['mean_vs_gold']:.3f} vs expert "
              f"(MLP anchor {anchor:.3f}, delta {b['delta_vs_anchor']:+.3f})")

    # write artifacts (land in sweep/curriculum/, which the publish step already commits)
    with open(os.path.join(DIR, "_arch_summary.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    with open(os.path.join(DIR, "_arch_summary.md"), "w") as f:
        f.write("# Phase-8 architecture sweep — evaluation vs the fixed expert\n\n")
        f.write(f"_{len(rows)} runs finished across {len(byc)} architectures. "
                "Win-rate is vs the fixed deterministic expert (benchmark-only)._\n\n")
        f.write("## Per architecture (mean over seeds)\n\n")
        f.write("| architecture | n | win% vs expert | std | Δ vs MLP anchor |\n|---|---|---|---|---|\n")
        for a in agg:
            d = "" if a["delta_vs_anchor"] is None else f"{a['delta_vs_anchor']:+.3f}"
            f.write(f"| {a['cell']} | {a['n']} | {a['mean_vs_gold']:.3f} | {a['std']:.3f} | {d} |\n")
        f.write("\n## Per run\n\n")
        f.write("| run | arch | act | net | seed | win% vs expert | gin% | mean len | vs champ | vs random | secs |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['name']} | {r['arch']} | {r['activation']} | {r['net_arch']} | {r['seed']} | "
                    f"{r['vs_gold']:.3f} | {r['gin_vs_gold']:.3f} | {r['len_vs_gold']} | "
                    f"{r['vs_champion']:.3f} | {r['vs_random']:.3f} | {r['secs']} |\n")
    json.dump(dict(per_run=rows, per_arch=agg, anchor=ANCHOR, anchor_mean=anchor),
              open(os.path.join(DIR, "_arch_summary.json"), "w"), indent=2)
    print(f"\nwrote {DIR}/_arch_summary.{{csv,md,json}}")


if __name__ == "__main__":
    main()
