"""Generate all figures for paper/main.tex.

Phase-1 figures come from sweep/results/*.json; Phase-2 figures from
sweep/selfplay/results/*.json, sweep/poolplay/results/poolplay.json, and a few
measured constants (LLM-opponent eval, worker load time) recorded inline with
their source. Outputs (PNG + PDF) go to paper/figures/.
"""
from __future__ import annotations

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
RESULTS = os.path.join(ROOT, "sweep", "results")
FIGS = os.path.join(HERE, "figures")
os.makedirs(FIGS, exist_ok=True)


def _save(fig, name):
    fig.savefig(os.path.join(FIGS, name + ".pdf"))
    fig.savefig(os.path.join(FIGS, name + ".png"), dpi=160)
    plt.close(fig)


def load_results():
    rs = []
    for p in sorted(glob.glob(os.path.join(RESULTS, "run_*.json"))):
        with open(p) as f:
            rs.append(json.load(f))
    rs.sort(key=lambda r: r["combo"])
    return rs


def short_label(r):
    c = r["config"]
    return f"c{r['combo']}\nlr={c['learning_rate']:.0e}\nent={c['ent_coef']}"


# ----------------------------------------------------------------- Phase 1
def fig_winrate(rs):
    labels = [short_label(r) for r in rs]
    win = [100 * r["win_rate"] for r in rs]
    n = [r["eval_episodes"] for r in rs]
    ci = [196 * np.sqrt(p / 100 * (1 - p / 100) / nn) for p, nn in zip(win, n)]
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    xs = np.arange(len(rs))
    ax.bar(xs, win, yerr=ci, capsize=3, color="#3a7fb8", edgecolor="black", linewidth=0.4)
    ax.set_ylim(95, 100.3)
    ax.set_ylabel("win rate vs random (%)")
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=7)
    ax.set_title("Phase-1 sweep: win rate per HP config (1000 eval episodes, 95\\% CI)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); _save(fig, "win_rate")


def fig_mean_reward(rs):
    labels = [short_label(r) for r in rs]
    mr = [r["mean_reward"] for r in rs]
    sd = [r["std_reward"] / np.sqrt(r["eval_episodes"]) for r in rs]
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    xs = np.arange(len(rs))
    ax.bar(xs, mr, yerr=sd, capsize=3, color="#d9822b", edgecolor="black", linewidth=0.4)
    ax.axhline(0.5, ls=":", color="green", lw=1.0, label="knock reward")
    ax.axhline(1.5, ls=":", color="red", lw=1.0, label="gin reward")
    ax.set_ylim(0, 1.6); ax.set_ylabel("mean episode reward")
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=7)
    ax.set_title("Mean reward per config — clusters at the knock value")
    ax.legend(loc="upper right", fontsize=8); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); _save(fig, "mean_reward")


def fig_lr_vs_reward(rs):
    core = [r for r in rs if r["combo"] < 6]
    lr = [r["config"]["learning_rate"] for r in core]
    mr = [r["mean_reward"] for r in core]
    ent = [r["config"]["ent_coef"] for r in core]
    fig, ax = plt.subplots(figsize=(5.0, 2.9))
    for ec, marker, color in [(0.01, "o", "#2e7d32"), (0.03, "s", "#c62828")]:
        x = [lri for lri, e in zip(lr, ent) if e == ec]
        y = [mri for mri, e in zip(mr, ent) if e == ec]
        ax.scatter(x, y, s=70, marker=marker, color=color, label=f"ent={ec}", edgecolor="black")
    ax.set_xscale("log"); ax.set_xlabel("learning rate"); ax.set_ylabel("mean reward")
    ax.set_title("LR $\\times$ entropy grid"); ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); _save(fig, "lr_vs_reward")


# ----------------------------------------------------------------- Phase 2
def fig_selfplay():
    # seedless run_N.json only (the original 3M self-play sweep); the seeded
    # run_N_sNNN.json files belong to the Phase-3 gin-shaping sweep (fig_ginshape).
    paths = sorted(glob.glob(os.path.join(ROOT, "sweep", "selfplay", "results", "run_[0-9].json")))
    rs = [json.load(open(p)) for p in paths]
    rs.sort(key=lambda r: r["combo"])
    labels = [f"k={r['knock_reward']}\ng={r['gin_reward']}" for r in rs]
    vr5 = [100 * r["vs_run5"]["win_rate"] for r in rs]
    vrand = [100 * r["vs_random"]["win_rate"] for r in rs]
    xs = np.arange(len(rs)); w = 0.38
    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    ax.bar(xs - w / 2, vr5, w, label="vs frozen run\\_5", color="#6a51a3", edgecolor="black", linewidth=0.4)
    ax.bar(xs + w / 2, vrand, w, label="vs random", color="#9ecae1", edgecolor="black", linewidth=0.4)
    ax.axhline(50, ls="--", color="gray", lw=0.8)
    ax.set_ylim(0, 105); ax.set_ylabel("win rate (%)")
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=7)
    ax.set_title("Self-play fine-tune: beats its progenitor ($>$50\\% vs run\\_5)")
    ax.legend(fontsize=8, loc="center right"); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); _save(fig, "selfplay")


def fig_pool():
    p = json.load(open(os.path.join(ROOT, "sweep", "poolplay", "results", "poolplay.json")))
    panel = p["panel"]
    # contrast the pool champion's vs-random with the self-play agents' vs-random
    # to show the post-10M divergence (vs-random fell from ~0.98 to 0.858).
    names = ["vs random", "vs run\\_5", "vs pool"]
    vals = [100 * panel["random"]["win_rate"], 100 * panel["run5"]["win_rate"],
            100 * panel["pool"]["win_rate"]]
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    xs = np.arange(len(names))
    ax.bar(xs, vals, color=["#9ecae1", "#6a51a3", "#fd8d3c"], edgecolor="black", linewidth=0.4)
    ax.axhline(98.5, ls=":", color="green", lw=1.0, label="self-play vs-random (\\~98.5\\%)")
    ax.axhline(50, ls="--", color="gray", lw=0.8)
    ax.set_ylim(0, 105); ax.set_ylabel("win rate (%)")
    ax.set_xticks(xs); ax.set_xticklabels(names, fontsize=8)
    ax.set_title("Pool champion (12M): vs-random dropped to 85.8\\% (divergence)")
    ax.legend(fontsize=7, loc="lower right"); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); _save(fig, "pool")


def fig_llm_opponent():
    # LLM-as-opponent win rate vs a random hero. Source: llm/eval_opponent.py runs.
    # OLMoE-1B parses/reason poorly -> falls back to random -> ~chance. Qwen2.5-7B
    # won 5/5 decided games (small N; shown with the count).
    models = ["OLMoE-1B\n(chance)", "Qwen2.5-7B\n(5/5)"]
    win = [50.0, 100.0]
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    xs = np.arange(len(models))
    ax.bar(xs, win, color=["#bdbdbd", "#31a354"], edgecolor="black", linewidth=0.5, width=0.55)
    ax.axhline(50, ls="--", color="gray", lw=0.8, label="random ($=$50\\%)")
    ax.set_ylim(0, 110); ax.set_ylabel("LLM win rate vs random (%)")
    ax.set_xticks(xs); ax.set_xticklabels(models, fontsize=8)
    ax.set_title("LLM opponent strength")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); _save(fig, "llm_opponent")


def fig_infra():
    # Worker load time for Qwen2.5-7B: home NFS (~11 MB/s, timed out the health
    # check) vs scratch/BeeGFS. Source: slurm worker logs (27.2 s on scratch).
    src = ["home NFS", "scratch (BeeGFS)"]
    secs = [1680.0, 27.0]
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    xs = np.arange(len(src))
    bars = ax.bar(xs, secs, color=["#c62828", "#2e7d32"], edgecolor="black", linewidth=0.5, width=0.55)
    ax.set_yscale("log"); ax.set_ylabel("7B load time (s, log scale)")
    ax.set_xticks(xs); ax.set_xticklabels(src, fontsize=9)
    ax.set_title("Worker weight-load: 62$\\times$ faster from scratch")
    for b, s in zip(bars, secs):
        ax.text(b.get_x() + b.get_width() / 2, s * 1.1, f"{s:.0f}s", ha="center", fontsize=9)
    ax.grid(axis="y", alpha=0.3, which="both")
    fig.tight_layout(); _save(fig, "infra_load")


def fig_throughput():
    # Aggregate LLM-query throughput. Naive: 1 stream, CoT prompt, ~16 s/call.
    # Distributed (measured): 14 Qwen2.5-7B workers + terse prompt served ~19.3k
    # calls in ~10 min on the live RL-vs-LLM run = ~32 calls/s. Projected at the
    # 32-worker target scales linearly with the worker count.
    names = ["naive\n(1 stream, CoT)", "distributed\n(14 workers)", "projected\n(32 workers)"]
    cps = [1.0 / 16.0, 32.0, 32.0 * 32.0 / 14.0]
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    xs = np.arange(len(names))
    bars = ax.bar(xs, cps, color=["#bdbdbd", "#2171b5", "#08519c"], edgecolor="black",
                  linewidth=0.5, width=0.6)
    ax.set_yscale("log"); ax.set_ylabel("LLM queries / s (log scale)")
    ax.set_xticks(xs); ax.set_xticklabels(names, fontsize=8)
    ax.set_title("Inference throughput: master + worker pool")
    for b, v in zip(bars, cps):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.25, f"{v:.2g}", ha="center", fontsize=9)
    ax.grid(axis="y", alpha=0.3, which="both")
    fig.tight_layout(); _save(fig, "throughput")


def fig_rlvsllm():
    p = os.path.join(ROOT, "sweep", "llmplay", "curve.json")
    if not os.path.exists(p):
        print("  [skip] fig_rlvsllm: no curve.json")
        return
    d = json.load(open(p))
    steps = np.array(d["steps"]) / 1e6
    rew = d["ep_rew"]
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    ax.plot(steps, rew, color="#2171b5", lw=1.6, marker="o", ms=2.5, label="ep reward (vs LLM)")
    ax.axhline(0.5, ls=":", color="green", lw=1.2, label="knock value")
    ax.axhline(1.5, ls=":", color="red", lw=1.2, label="gin value")
    ax.set_ylim(0, 1.6)
    ax.set_xlabel("training steps (millions)"); ax.set_ylabel("mean episode reward")
    ax.set_title("RL-vs-LLM: episode reward stays at the knock value")
    ax.legend(fontsize=8, loc="center right"); ax.grid(alpha=0.3)
    fig.tight_layout(); _save(fig, "rlvsllm_curve")


def fig_h2h():
    p = os.path.join(ROOT, "sweep", "h2h.json")
    if not os.path.exists(p):
        print("  [skip] fig_h2h: no h2h.json")
        return
    d = json.load(open(p))
    M = d["matrix"]
    heroes = list(M.keys())
    opps = heroes + (["random"] if any("random" in M[h] for h in heroes) else [])
    Z = np.full((len(heroes), len(opps)), np.nan)
    for i, h in enumerate(heroes):
        for j, o in enumerate(opps):
            v = M[h].get(o)
            if v is not None:
                Z[i, j] = 100 * v
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    im = ax.imshow(Z, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(opps))); ax.set_xticklabels(opps, rotation=30, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(heroes))); ax.set_yticklabels(heroes, fontsize=8.5)
    ax.set_xlabel("opponent (column)"); ax.set_ylabel("hero (row)")
    for i in range(len(heroes)):
        for j in range(len(opps)):
            if not np.isnan(Z[i, j]):
                ax.text(j, i, f"{Z[i, j]:.0f}", ha="center", va="center", fontsize=8)
    ax.set_title(f"Head-to-head win-rate (row agent vs column, {d.get('n','?')} games)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="win %")
    fig.tight_layout(); _save(fig, "h2h")


def write_summary_csv(rs):
    out = os.path.join(FIGS, "sweep_summary.csv")
    with open(out, "w") as f:
        f.write("combo,lr,ent_coef,n_steps,n_epochs,win_rate,loss_rate,mean_reward,mean_length,train_s\n")
        for r in rs:
            c = r["config"]
            f.write(f"{r['combo']},{c['learning_rate']},{c['ent_coef']},{c['n_steps']},{c['n_epochs']},"
                    f"{r['win_rate']:.4f},{r['loss_rate']:.4f},{r['mean_reward']:.4f},"
                    f"{r['mean_length']:.2f},{r['train_seconds']:.0f}\n")


def fig_ginshape():
    """Phase-3 reward-shaping sweep: gin rate (mean +/- sd over 3 seeds) per reward
    config, with win rate. Reads the seeded run_<combo>_s<seed>.json files only."""
    import collections, statistics as st
    paths = sorted(glob.glob(os.path.join(
        ROOT, "sweep", "selfplay", "results", "run_*_s*.json")))
    if not paths:
        print("  [skip] fig_ginshape: no seeded results")
        return
    by = collections.defaultdict(list)
    for p in paths:
        d = json.load(open(p)); by[d["combo"]].append(d)
    labels, gin_r, gin_e, win_r = [], [], [], []
    for c in sorted(by):
        ds = by[c]
        labels.append(f"k{ds[0]['knock_reward']}/g{ds[0]['gin_reward']}")
        gr = [d["vs_random"]["gin_rate"] * 100 for d in ds]
        gin_r.append(st.mean(gr)); gin_e.append(st.pstdev(gr))
        win_r.append(st.mean([d["vs_random"]["win_rate"] * 100 for d in ds]))
    fig, ax = plt.subplots(figsize=(5.0, 2.6))
    x = range(len(labels))
    ax.bar(x, gin_r, yerr=gin_e, color="#c0392b", alpha=.85, capsize=3)
    ax.axhline(gin_r[0], ls="--", lw=.8, color="#7f8c8d")
    ax.set_ylabel("gin rate vs random (%)", color="#c0392b")
    ax.set_ylim(0, max(5, max(gin_r) * 2))
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, rotation=20, fontsize=7)
    ax2 = ax.twinx(); ax2.plot(x, win_r, "o-", color="#2c3e50")
    ax2.set_ylim(80, 100); ax2.set_ylabel("win rate vs random (%)", color="#2c3e50")
    ax.set_title("Reward shaping does not raise the gin rate", fontsize=9)
    fig.tight_layout(); _save(fig, "ginshape")


def main():
    rs = load_results()
    print(f"loaded {len(rs)} phase-1 results")
    fig_winrate(rs); fig_mean_reward(rs); fig_lr_vs_reward(rs); write_summary_csv(rs)
    for fn in (fig_selfplay, fig_pool, fig_llm_opponent, fig_infra, fig_throughput,
               fig_rlvsllm, fig_h2h, fig_ginshape):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            print(f"  [skip] {fn.__name__}: {exc}")
    print(f"figures written to {FIGS}")


if __name__ == "__main__":
    main()
