"""One-stop Phase-8 results collector. Scans sweep/curriculum/*.json and organizes EVERYTHING that
grades against a fixed expert into a single, readable summary, so after an unattended 24h run you can
see at a glance what landed and where each agent stands vs the yardstick:

  - Gin Rummy architecture sweep (win-rate vs the GOLD expert), per cell + per-architecture IQM/CI
  - Stage-C cross-recipe robustness (win-rate vs gold, grouped by recipe x architecture)
  - Recurrence (LSTM vs PPO-MLP control) vs gold
  - ISMCTS search baseline vs gold, by rollout budget
  - Leduc Hold'em generality: tabular-Q and NFSP return vs the CFR-optimal expert, per seed + mean

Writes sweep/curriculum/_phase8_summary.md and _phase8_summary.json. Pure stdlib + numpy.

    python sweep/collect_phase8.py
"""
import glob
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "sweep", "curriculum")


def load_all():
    out = {}
    for f in glob.glob(os.path.join(RES, "*.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        out[os.path.splitext(os.path.basename(f))[0]] = d
    return out


def _wr_vs_gold(d):
    """Best win-rate vs the gold expert from a curriculum/arch/stagec result, if present."""
    if "best_vs_gold" in d and isinstance(d["best_vs_gold"], (int, float)):
        return float(d["best_vs_gold"])
    vg = d.get("vs_gold")
    if isinstance(vg, dict) and "win_rate" in vg:
        return float(vg["win_rate"])
    return None


def iqm(xs):
    xs = np.sort(np.asarray(xs, dtype=float))
    if len(xs) == 0:
        return None, None, None
    lo, hi = int(0.25 * len(xs)), int(np.ceil(0.75 * len(xs)))
    core = xs[lo:hi] if hi > lo else xs
    m = float(np.mean(core))
    # stratified bootstrap CI over the raw seeds
    if len(xs) >= 3:
        rng = np.random.default_rng(0)
        boots = [np.mean(np.sort(rng.choice(xs, len(xs)))[lo:hi] if hi > lo else rng.choice(xs, len(xs)))
                 for _ in range(2000)]
        return m, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    return m, float(xs.min()), float(xs.max())


def cell_base(name):
    """Strip a trailing _sN seed tag to group seeds of the same cell."""
    import re
    return re.sub(r"_s\d+$", "", name)


def main():
    A = load_all()
    md = ["# Phase-8 results summary", ""]
    summary = {}

    # ---- Gin Rummy architecture sweep (vs GOLD) ----
    arch = {k: v for k, v in A.items() if k.startswith("arch_")}
    rows = []
    for k, d in arch.items():
        wr = _wr_vs_gold(d)
        if wr is not None:
            rows.append((wr, k, d.get("arch", "?"), d.get("activation", "?")))
    rows.sort(reverse=True)
    md += ["## Gin Rummy: architecture sweep vs GOLD expert (win-rate, best ckpt)",
           f"_{len(rows)} cells with results_", "", "| win% vs gold | cell | arch | act |",
           "|---:|---|---|---|"]
    for wr, k, a, act in rows:
        md.append(f"| {wr:.3f} | {k} | {a} | {act} |")
    # per-architecture-family IQM
    fam = {}
    for wr, k, a, act in rows:
        fam.setdefault(cell_base(k), []).append(wr)
    md += ["", "### Per-cell IQM (across seeds) with 95% bootstrap CI", "",
           "| cell | n | IQM | CI low | CI high |", "|---|---:|---:|---:|---:|"]
    fam_summary = {}
    for base, xs in sorted(fam.items(), key=lambda kv: -np.mean(kv[1])):
        m, lo, hi = iqm(xs)
        fam_summary[base] = dict(n=len(xs), iqm=m, ci=[lo, hi], seeds=xs)
        md.append(f"| {base} | {len(xs)} | {m:.3f} | {lo:.3f} | {hi:.3f} |")
    summary["arch_vs_gold"] = dict(cells={k: wr for wr, k, *_ in rows}, families=fam_summary)

    # ---- Stage-C cross-recipe robustness (vs GOLD) ----
    sc = {k: v for k, v in A.items() if k.startswith("sc_")}
    if sc:
        md += ["", "## Stage-C: cross-recipe robustness vs GOLD (win-rate, best ckpt)", "",
               "| win% vs gold | cell (recipe_arch_seed) |", "|---:|---|"]
        scr = sorted(((_wr_vs_gold(d), k) for k, d in sc.items() if _wr_vs_gold(d) is not None),
                     reverse=True)
        for wr, k in scr:
            md.append(f"| {wr:.3f} | {k} |")
        summary["stagec_vs_gold"] = {k: wr for wr, k in scr}

    # ---- Recurrence ----
    rec = {k: v for k, v in A.items() if k.startswith("rec_")}
    if rec:
        md += ["", "## Recurrence (LSTM vs PPO-MLP control) vs GOLD", "",
               "| win% vs gold | cell |", "|---:|---|"]
        for k, d in sorted(rec.items()):
            wr = _wr_vs_gold(d) or (d.get("vs_gold", {}) or {}).get("win_rate")
            md.append(f"| {wr if wr is None else round(wr,3)} | {k} |")

    # ---- ISMCTS search baseline vs gold, by rollout budget; split fair (determinized) vs oracle ----
    ism = {k: v for k, v in A.items() if k.startswith("ismcts_")}
    if ism:
        irows = []
        for k, d in ism.items():
            vg = d.get("vs_gold") or {}
            if vg:
                det = bool(d.get("determinize", False))   # old files (no field) were oracle
                irows.append((det, d.get("rollouts", 0), vg.get("win_rate"), vg.get("gin_rate"),
                              vg.get("mean_len"), k))
        for det, title in [(True, "determinized = FAIR imperfect-info baseline"),
                           (False, "oracle = perfect-info UPPER BOUND (sees opponent cards)")]:
            sub = sorted([r for r in irows if r[0] == det], key=lambda r: r[1])
            if not sub:
                continue
            md += ["", f"## ISMCTS vs GOLD by rollout budget &mdash; {title}", "",
                   "| rollouts | win% vs gold | gin% | mean len | cell |", "|---:|---:|---:|---:|---|"]
            for _, r, wr, gr, ml, k in sub:
                md.append(f"| {r} | {wr} | {gr} | {ml} | {k} |")
        summary["ismcts_vs_gold"] = [dict(rollouts=r, win_rate=wr, determinize=det, cell=k)
                                     for det, r, wr, _, _, k in sorted(irows, key=lambda x: (x[0], x[1]))]

    # ---- Head-to-head: trained models vs ISMCTS ----
    h2h = {k: v for k, v in A.items() if k.startswith("h2h_")}
    if h2h:
        md += ["", "## Head-to-head: trained models vs ISMCTS (model win-rate)", "",
               "| model | win% vs ISMCTS | rollouts | mode | cell |", "|---|---:|---:|---|---|"]
        hrows = []
        for k, d in h2h.items():
            vi = d.get("vs_ismcts") or {}
            if vi:
                hrows.append((vi.get("win_rate"), d.get("model"), d.get("rollouts"),
                              "det" if d.get("determinize") else "oracle", k))
        for wr, mdl, r, mode, k in sorted(hrows, reverse=True):
            md.append(f"| {mdl} | {wr} | {r} | {mode} | {k} |")
        summary["model_vs_ismcts"] = [dict(model=mdl, win_rate=wr, rollouts=r, mode=mode)
                                      for wr, mdl, r, mode, k in sorted(hrows, reverse=True)]

    # ---- Leduc generality vs CFR-optimal expert ----
    leduc = {k: v for k, v in A.items() if k.startswith("leduc_")}
    if leduc:
        md += ["", "## Leduc Hold'em generality: return vs CFR-optimal expert", "",
               "_(0 = parity with the game-theoretic optimum; random baseline ~ -0.78)_", "",
               "| agent | seed | return vs CFR |", "|---|---:|---:|"]
        groups = {"tabular_q": [], "nfsp": []}
        for k, d in sorted(leduc.items()):
            kind = "nfsp" if "nfsp" in k else "tabular_q"
            r = d.get("final_return_vs_expert")
            if r is not None:
                groups[kind].append(r)
            md.append(f"| {kind} | {d.get('seed','?')} | {r} |")
        md += ["", "| agent | n seeds | mean return vs CFR |", "|---|---:|---:|"]
        leduc_sum = {}
        for kind, xs in groups.items():
            if xs:
                leduc_sum[kind] = dict(n=len(xs), mean=float(np.mean(xs)))
                md.append(f"| {kind} | {len(xs)} | {np.mean(xs):.3f} |")
        summary["leduc_vs_cfr"] = leduc_sum

    out_md = os.path.join(RES, "_phase8_summary.md")
    out_json = os.path.join(RES, "_phase8_summary.json")
    open(out_md, "w").write("\n".join(md) + "\n")
    json.dump(summary, open(out_json, "w"), indent=2)
    print(f"[collect_phase8] wrote {out_md} and {out_json}")
    print(f"  arch cells={len(rows)}  stagec={len(sc)}  recurrence={len(rec)}  "
          f"ismcts={len(ism)}  leduc={len(leduc)}")


if __name__ == "__main__":
    main()
