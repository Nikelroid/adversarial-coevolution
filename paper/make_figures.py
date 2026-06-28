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

# One restrained palette for the whole paper: deep green (primary / RL), a lighter green
# (second series), gold (the gold standard / "vs gold"), and a neutral grey (weak / random).
C_GREEN = "#0b5b39"
C_GREEN2 = "#4a9c78"
C_GOLD = "#c9962b"
C_GREY = "#9aa6a0"
C_INK = "#16242d"
OPP_COLORS = {"random": C_GREY, "champion": C_GREEN2, "gold": C_GOLD}
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12.5, "axes.labelsize": 11,
                     "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9.5,
                     "axes.edgecolor": "#444", "axes.linewidth": 0.9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlepad": 7,
                     "legend.frameon": False, "figure.dpi": 200,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
                     "lines.linewidth": 2.0, "lines.markersize": 6,
                     "font.family": "serif", "mathtext.fontset": "dejavuserif"})


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


def fig_gold():
    """Gold-standard benchmark: win rate vs every agent, and the gin/knock
    trade-off (hold-for-gin vs random vs vs the champion)."""
    p = os.path.join(ROOT, "sweep", "gold_bench.json")
    if not os.path.exists(p):
        print("  [skip] fig_gold: no gold_bench.json")
        return
    d = json.load(open(p))
    order = ["random", "reward", "pool", "winrate", "llm_full", "champion"]
    labels = [k for k in order if k in d["matchups"]]
    wins = [d["matchups"][k]["a_win_rate"] * 100 for k in labels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.8))
    colors = ["#7f8c8d"] + ["#2171b5"] * (len(labels) - 1)
    ax1.barh(labels[::-1], wins[::-1], color=colors[::-1])
    ax1.axvline(50, ls="--", lw=.8, color="#c0392b")
    ax1.set_xlim(0, 100); ax1.set_xlabel("gold-standard win rate (%)")
    ax1.set_title("Gold beats every learned agent", fontsize=9)
    for i, w in enumerate(wins[::-1]):
        ax1.text(w + 1, i, f"{w:.0f}", va="center", fontsize=7)

    # gin/knock trade-off
    grp = ["vs random", "vs champion"]
    knock_win = [d["matchups"]["random"]["a_win_rate"] * 100,
                 d["matchups"]["champion"]["a_win_rate"] * 100]
    hold_win = [d["gold_hold_vs_random"]["a_win_rate"] * 100,
                d.get("gold_hold_vs_champion", {}).get("a_win_rate", 0) * 100]
    hold_gin = [d["gold_hold_vs_random"]["a_gin_rate"] * 100,
                d.get("gold_hold_vs_champion", {}).get("a_gin_rate", 0) * 100]
    x = np.arange(len(grp)); w = 0.27
    ax2.bar(x - w, knock_win, w, label="knock ASAP: win", color="#2171b5")
    ax2.bar(x, hold_win, w, label="hold for gin: win", color="#27ae60")
    ax2.bar(x + w, hold_gin, w, label="hold for gin: gin%", color="#d9a521")
    ax2.set_xticks(x); ax2.set_xticklabels(grp); ax2.set_ylim(0, 100)
    ax2.set_ylabel("rate (%)"); ax2.set_title("Hold-for-gin: great vs weak,\nfatal vs strong", fontsize=9)
    ax2.legend(fontsize=6, loc="upper right")
    fig.tight_layout(); _save(fig, "gold_bench")


def fig_algo():
    """Algorithm study: PPO vs TRPO win rate vs random/champion/gold (mean +/- sd
    over seeds). Reads sweep/algo/algo_<algo>_s<seed>.json (skips smokes)."""
    import collections, statistics as st
    files = [f for f in glob.glob(os.path.join(ROOT, "sweep", "algo", "algo_*.json"))
             if "smoke" not in f]
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for f in files:
        d = json.load(open(f))
        for o in ("random", "champion", "gold"):
            by[d["algo"]][o].append(d[f"vs_{o}"]["win_rate"] * 100)
    algos = [a for a in ("ppo", "trpo") if a in by]
    if not algos:
        print("  [skip] fig_algo: no results")
        return
    opps = ["random", "champion", "gold"]
    fig, ax = plt.subplots(figsize=(4.6, 3.4))     # matched to fig_reward_gin (paper Fig. 4)
    x = np.arange(len(opps)); w = 0.36
    col = {"ppo": C_GREY, "trpo": C_GREEN}
    for i, a in enumerate(algos):
        m = [st.mean(by[a][o]) for o in opps]
        e = [st.pstdev(by[a][o]) for o in opps]
        ax.bar(x + (i - 0.5) * w, m, w, yerr=e, capsize=3, label=a.upper(), color=col[a])
    ax.set_xticks(x); ax.set_xticklabels([f"vs {o}" for o in opps])
    ax.set_ylabel("win rate (%)"); ax.set_ylim(0, 100)
    ax.set_title("TRPO beats PPO (2 seeds)"); ax.legend()
    fig.tight_layout(); _save(fig, "algo_compare")


def fig_reward_gin():
    """Single-panel companion to fig_algo (matched size/fonts for paper Fig. 4): gin-rate stays
    under ~1% for every reward design, even one paying 3x for a gin."""
    by, st = _curriculum_cells()
    order = [("03_rew_R0_highgin", "high-gin\n(pays 3x)"), ("01_base_trpo", "balanced"),
             ("04_rew_R2_knockfwd", "knock-fwd"), ("05_rew_R4_earlyknk", "early-knock")]
    order = [(c, lab) for c, lab in order if c in by]
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    if not order:
        ax.text(0.5, 0.5, "pending", ha="center"); ax.axis("off")
        fig.tight_layout(); _save(fig, "reward_gin"); return
    x = np.arange(len(order))
    gin = [st.mean(by[c]["gin"]) for c, _ in order]
    ax.bar(x, gin, 0.62, color=C_GREEN)
    ax.axhline(GOLD_GIN, ls="--", lw=1.2, color=C_GOLD)
    ax.text(len(order) - 0.5, GOLD_GIN + 0.04, "gold's gin rate", fontsize=8, ha="right", color=C_GOLD)
    ax.set_xticks(x); ax.set_xticklabels([lab for _, lab in order])
    ax.set_ylabel("gin rate vs gold (%)"); ax.set_ylim(0, 1.6)
    ax.set_title("Rewarding gin does not make it gin")
    fig.tight_layout(); _save(fig, "reward_gin")
    print("  fig_reward_gin: ok")


def fig_phase5():
    """Representation study: sparse vs temporal-embed vs LLM-embed win rate. Reads
    sweep/phase5/phase5_{sparse,temporal,llm}.json; missing condition shown 'pending'."""
    cond = [("sparse", "sparse 208-d", "#7f8c8d"),
            ("temporal", "temporal embed 20-d", "#27ae60"),
            ("llm", "LLM embed 20-d", "#c0392b")]
    opps = ["random", "champion", "gold"]
    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    x = np.arange(len(opps)); n = len(cond); w = 0.8 / n
    for i, (tag, label, c) in enumerate(cond):
        f = os.path.join(ROOT, "sweep", "phase5", f"phase5_{tag}.json")
        off = (i - (n - 1) / 2) * w
        if os.path.exists(f):
            d = json.load(open(f))
            vals = [d[f"vs_{o}"]["win_rate"] * 100 for o in opps]
            ax.bar(x + off, vals, w, label=label, color=c)
        else:
            ax.bar(x + off, [0, 0, 0], w, label=label + " (pending)", color=c,
                   alpha=0.25, hatch="//")
    ax.set_xticks(x); ax.set_xticklabels([f"vs {o}" for o in opps])
    ax.set_ylabel("win rate (%)"); ax.set_ylim(0, 100)
    ax.set_title("Representation: sparse vs learned embeddings", fontsize=10)
    ax.legend(fontsize=7)
    fig.tight_layout(); _save(fig, "phase5_compare")


CHAMP_VS_GOLD = 29.8     # champion's own win-rate vs gold (gold_bench: gold beats champ 70.2%)
GOLD_GIN = 0.7           # the optimal player's own gin-rate vs champion (gold_bench)

_CELL_LABEL = {
    "01_base_trpo": "TRPO baseline", "02_base_ppo": "PPO", "03_rew_R0_highgin": "reward: high-gin",
    "04_rew_R2_knockfwd": "reward: knock-forward", "05_rew_R4_earlyknk": "reward: early-knock",
    "06_cur_noRandTail": "curric: no-random-tail", "07_cur_pfsp": "curric: PFSP",
    "08_trpo_gamma_high": "gamma 0.997", "09_trpo_kl_small": "small trust region",
    "10_trpo_kl_large": "large trust region", "11_trpo_gae_batch": "GAE+batch",
    "12_ppo_ent_hi": "PPO high-explore",
}


def _curriculum_cells():
    import collections, re, statistics as st
    files = [f for f in glob.glob(os.path.join(ROOT, "sweep", "curriculum", "*.json"))
             if not os.path.basename(f).startswith("_")]
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for f in files:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        cell = re.sub(r"_s\d+$", "", d.get("name", ""))
        by[cell]["gold"].append(d.get("vs_gold", {}).get("win_rate", 0) * 100)
        by[cell]["bestgold"].append(d.get("best_vs_gold", d.get("vs_gold", {}).get("win_rate", 0)) * 100)
        by[cell]["champ"].append(d.get("vs_champion", {}).get("win_rate", 0) * 100)
        by[cell]["gin"].append(d.get("vs_gold", {}).get("gin_rate", 0) * 100)
        by[cell]["len"].append(d.get("vs_gold", {}).get("mean_len", 0))
    return by, st


def fig_curriculum():
    """Phase-6 main result: per-cell win-rate vs champion and vs gold (mean +/- sd over
    seeds), sorted by vs-gold, with the champion's own vs-gold score as the reference line.
    Reads sweep/curriculum/*.json; placeholder until cells land."""
    by, st = _curriculum_cells()
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    if not by:
        ax.text(0.5, 0.5, "Phase-6 curriculum / algorithm / reward sweep\n"
                "30 runs queued -- results pending", ha="center", va="center", fontsize=11)
        ax.axis("off"); fig.tight_layout(); _save(fig, "curriculum")
        print("  fig_curriculum: pending"); return
    cells = sorted(by, key=lambda c: st.mean(by[c]["gold"]))
    labels = [_CELL_LABEL.get(c, c) for c in cells]
    y = np.arange(len(cells)); h = 0.4
    champ = [st.mean(by[c]["champ"]) for c in cells]
    champ_e = [st.pstdev(by[c]["champ"]) if len(by[c]["champ"]) > 1 else 0 for c in cells]
    gold = [st.mean(by[c]["gold"]) for c in cells]
    gold_e = [st.pstdev(by[c]["gold"]) if len(by[c]["gold"]) > 1 else 0 for c in cells]
    ax.barh(y + h / 2, champ, h, xerr=champ_e, capsize=2, label="vs champion", color="#2171b5")
    ax.barh(y - h / 2, gold, h, xerr=gold_e, capsize=2, label="vs gold (perfect player)", color="#c0392b")
    ax.axvline(CHAMP_VS_GOLD, ls="--", lw=1.3, color="#444")
    ax.text(CHAMP_VS_GOLD + 0.6, len(cells) - 0.4, "champion's own\nscore vs gold",
            fontsize=7.5, color="#444", va="top")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("win rate (%)"); ax.set_xlim(0, 60)
    ax.set_title("Phase-6: every recipe lands near champion-strength vs gold", fontsize=10.5)
    ax.legend(fontsize=8, loc="lower right"); fig.tight_layout(); _save(fig, "curriculum")
    print(f"  fig_curriculum: {len(cells)} cells")


def fig_curriculum_reward():
    """The reward finding: gin-rate stays under ~1% for EVERY reward design, even the one
    that pays 3x for a gin -- the optimal player almost never gins, and the policy learns
    that regardless. Bars = gin-rate per reward cell; dashed line = the gold agent's own gin-rate."""
    by, st = _curriculum_cells()
    order = [("03_rew_R0_highgin", "high-gin\n(gin pays 3x)"), ("01_base_trpo", "balanced\n(baseline)"),
             ("04_rew_R2_knockfwd", "knock-forward"), ("05_rew_R4_earlyknk", "early-knock\n(speed penalty)")]
    order = [(c, lab) for c, lab in order if c in by]
    if not order:
        fig, ax = plt.subplots(figsize=(5.6, 3.2))
        ax.text(0.5, 0.5, "reward study -- results pending", ha="center", va="center")
        ax.axis("off"); fig.tight_layout(); _save(fig, "curriculum_reward"); return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.2, 3.3))
    labs = [lab for _, lab in order]
    x = np.arange(len(order))
    gin = [st.mean(by[c]["gin"]) for c, _ in order]
    a1.bar(x, gin, 0.6, color=["#c0392b", "#7f8c8d", "#2e86c1", "#27ae60"][:len(order)])
    a1.axhline(GOLD_GIN, ls="--", lw=1.2, color="#444")
    a1.text(len(order) - 1, GOLD_GIN + 0.03, "gold's own gin-rate", fontsize=7.5, ha="right", color="#444")
    a1.set_xticks(x); a1.set_xticklabels(labs, fontsize=7.5)
    a1.set_ylabel("gin-rate vs gold (%)"); a1.set_ylim(0, 1.6)
    a1.set_title("Rewarding gin does NOT make it gin", fontsize=9.5)
    ln = [st.mean(by[c]["len"]) for c, _ in order]
    a2.bar(x, ln, 0.6, color=["#c0392b", "#7f8c8d", "#2e86c1", "#27ae60"][:len(order)])
    a2.set_xticks(x); a2.set_xticklabels(labs, fontsize=7.5)
    a2.set_ylabel("mean game length (turns)"); a2.set_ylim(min(ln) - 1.5, max(ln) + 1.0)
    a2.set_title("Early-knock reward shortens games", fontsize=9.5)
    fig.tight_layout(); _save(fig, "curriculum_reward")
    print(f"  fig_curriculum_reward: {len(order)} reward cells")


def fig_architecture():
    """Clean orthogonal system flowchart (PNG): a central spine
    gold -- learner -- curriculum -- LLM, with the environment on the left and the game on the
    right. Every arrow is strictly horizontal or vertical."""
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(7.6, 5.0)); ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.axis("off")

    def box(cx, cy, w, h, text, fc, tc="white", fs=10):
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                    boxstyle="round,pad=0.4,rounding_size=2.2", fc=fc, ec="none"))
        ax.text(cx, cy, text, ha="center", va="center", color=tc, fontsize=fs,
                fontweight="bold", linespacing=1.35)

    def harrow(x1, x2, y, c, two=False, label=None, ly=None):
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="<|-|>" if two else "-|>", color=c, lw=1.7,
                                    shrinkA=0, shrinkB=0))
        if label:
            ax.text((x1 + x2) / 2, ly, label, ha="center", va="bottom", fontsize=8.5, color=c)

    def varrow(y1, y2, x, c, two=False, label=None, lx=None):
        ax.annotate("", xy=(x, y2), xytext=(x, y1),
                    arrowprops=dict(arrowstyle="<|-|>" if two else "-|>", color=c, lw=1.7,
                                    shrinkA=0, shrinkB=0))
        if label:
            ax.text(lx, (y1 + y2) / 2, label, ha="left", va="center", fontsize=8.5, color=c)

    SLATE = "#46606e"
    ax.text(50, 97, "Framework overview", ha="center", fontsize=12, fontweight="bold", color=C_INK)
    box(50, 86, 33, 12, "Gold standard\n(perfect — benchmark only)", C_GOLD, tc="#3a2e00")
    box(15, 60, 24, 14, "Gin Rummy\nenvironment", C_GREEN)
    box(50, 60, 27, 14, "RL learner\n(masked PPO / TRPO)", C_GREEN)
    box(85, 60, 24, 14, "Web game\n(human vs agent)", SLATE)
    box(50, 36, 34, 12, "Opponent curriculum\n(random → selves → strong)", C_GREEN)
    box(50, 13, 31, 12, "Distributed LLM\n(master + GPU workers)", SLATE)
    # central vertical spine + horizontal env/game arms -- all straight
    varrow(80, 67, 50, C_GOLD, label="eval only", lx=52)            # learner -> gold (up)
    harrow(27, 36.5, 60, C_GREEN, two=True)                         # env <-> learner (play loop)
    harrow(63.5, 73, 60, SLATE)                                     # learner -> game
    varrow(53, 42, 50, C_GREEN, two=True, label="opponents", lx=52)  # learner <-> curriculum
    varrow(19, 30, 50, SLATE, label="LLM opponent", lx=52)         # LLM -> curriculum (up)
    fig.tight_layout(); _save(fig, "architecture")
    print("  fig_architecture: ok")


def _best_p7_gold():
    """Strongest Phase-7 agent's precise win-rate vs gold, from the high-N re-eval if present."""
    import json as _j
    p = os.path.join(ROOT, "sweep", "curriculum", "_best_models_precise.json")
    try:
        rows = _j.load(open(p))
        return max(r["gold"] for r in rows) * 100, max(r["gold"] for r in rows)
    except Exception:
        return 35.7, 0.357      # fallback to the 1000-game headline until the precise eval lands


def fig_gold_bench():
    """The gold-standard expert beats every learned agent, yet gins under 2% of the time -- it
    wins by knocking low, not by chasing gin. Reads sweep/gold_bench.json."""
    p = os.path.join(ROOT, "sweep", "gold_bench.json")
    if not os.path.exists(p):
        print("  [skip] fig_gold_bench: no gold_bench.json"); return
    d = json.load(open(p))["matchups"]
    order = [("random", "Random"), ("winrate", "Win-rate PPO"), ("reward", "Reward PPO"),
             ("pool", "Pool PPO"), ("llm_full", "LLM-tutored"), ("champion", "Self-play champion")]
    order = [(k, lab) for k, lab in order if k in d]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.4), gridspec_kw={"width_ratios": [1.55, 1]})
    y = np.arange(len(order))
    wins = [d[k]["a_win_rate"] * 100 for k, _ in order]
    a1.barh(y, wins, color=C_GOLD)
    for i, w in enumerate(wins):
        a1.text(w - 2, i, f"{w:.0f}%", va="center", ha="right", color="#3a2e00", fontsize=8.5, fontweight="bold")
    a1.set_yticks(y); a1.set_yticklabels([lab for _, lab in order])
    a1.set_xlabel("gold's win rate vs this agent (%)"); a1.set_xlim(0, 100)
    a1.set_title("The gold expert beats every learned agent")
    # gin-rate panel: gold's own gin-rate is tiny everywhere
    gins = [d[k]["a_gin_rate"] * 100 for k, _ in order]
    a2.barh(y, gins, color=C_GREEN)
    a2.set_yticks(y); a2.set_yticklabels([]); a2.set_xlim(0, 5)
    a2.axvline(2, ls="--", lw=1, color="#444"); a2.text(2.1, len(order) - 0.6, "2%", fontsize=8, color="#444")
    a2.set_xlabel("gold's gin rate (%)")
    a2.set_title("...yet gins under 2%: it knocks low")
    fig.tight_layout(); _save(fig, "gold_bench")
    print("  fig_gold_bench: ok")


def fig_regimes():
    """The whole landscape: best win-rate vs the gold standard for every TRAINING regime we
    tried, from imitation (DAgger) at the bottom to the Phase-7 curriculum agent at the top."""
    champ = 0.298
    try:
        gb = json.load(open(os.path.join(ROOT, "sweep", "gold_bench.json")))["matchups"]
        champ = 1 - gb["champion"]["a_win_rate"]
    except Exception:
        pass
    scratch = 0.085
    try:                                  # the no-imitation from-scratch RL baseline (2M steps)
        scratch = json.load(open(os.path.join(ROOT, "sweep", "dagger", "dagger_baseline.json")))["final_eval"]["gold"]["win_rate"]
    except Exception:
        pass
    p7 = _best_p7_gold()[1]
    # Every bar is a win-rate vs gold we actually MEASURED. (DAgger-from-expert and short-term
    # reward shaping collapsed in a separate tournament -- discussed in the text, not plotted on
    # this axis since they were not benchmarked head-to-head vs gold.)
    bars = [
        ("PPO from scratch, short (2M)", scratch * 100),
        ("Frozen state-embedding (Phase 5)", 13.8),
        ("PPO from scratch, sparse obs (Phase 5)", 15.0),
        ("TRPO from scratch (2M)", 22.5),
        ("Self-play champion (12M)", champ * 100),
        ("Curriculum sweep best (Phase 6)", 33.0),
        ("Keep-best + warm-start (Phase 7)", p7 * 100),
    ]
    bars.sort(key=lambda b: b[1])
    import matplotlib.colors as mcolors
    ramp = mcolors.LinearSegmentedColormap.from_list("g", [C_GREY, C_GREEN])
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    y = np.arange(len(bars))
    cols = [ramp(i / (len(bars) - 1)) for i in range(len(bars))]   # weak (grey) -> strong (green)
    ax.barh(y, [b[1] for b in bars], color=cols)
    for i, b in enumerate(bars):
        ax.text(b[1] + 0.6, i, f"{b[1]:.1f}%", va="center", fontsize=8.5, fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels([b[0] for b in bars])
    ax.set_xlabel("best win rate vs the gold standard (%)"); ax.set_xlim(0, 45)
    ax.set_title("Every regime we tried, ranked by strength vs perfect play")
    fig.tight_layout(); _save(fig, "regimes")
    print("  fig_regimes: ok")


def fig_journey():
    """The climb: best RL win-rate vs gold at each project milestone, with the perfect player
    as the ceiling. Tells the headline story in one picture."""
    p7pct, _ = _best_p7_gold()
    champ = 29.8
    try:
        gb = json.load(open(os.path.join(ROOT, "sweep", "gold_bench.json")))["matchups"]
        champ = (1 - gb["champion"]["a_win_rate"]) * 100
    except Exception:
        pass
    steps = [("Self-play\nchampion", champ), ("Algorithm\n(TRPO)", 22.5 if champ > 25 else 22.5),
             ("Curriculum\nsweep (P6)", 33.0), ("Keep-best +\nwarm-start (P7)", p7pct)]
    # order as a narrative climb (champion is the starting reference, then the systematic push)
    steps = [("Self-play\nchampion", champ), ("Curriculum\nsweep (P6)", 33.0),
             ("Keep-best +\nwarm-start (P7)", p7pct)]
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    x = np.arange(len(steps))
    vals = [s[1] for s in steps]
    ax.plot(x, vals, "-o", color=C_GREEN, lw=2.5, ms=9, zorder=3)
    for i, v in enumerate(vals):
        ax.text(i, v + 1.3, f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold", color=C_GREEN)
    ax.axhline(100, ls="--", lw=1.4, color=C_GOLD)
    ax.text(len(steps) - 1, 96, "gold standard = the ceiling (perfect play)", ha="right",
            fontsize=8.5, color=C_GOLD)
    ax.set_xticks(x); ax.set_xticklabels([s[0] for s in steps], fontsize=9)
    ax.set_ylabel("win rate vs the gold standard (%)"); ax.set_ylim(0, 105)
    ax.set_title("Closing the gap to perfect play, step by step", fontsize=11)
    ax.grid(axis="y", ls=":", alpha=0.4)
    fig.tight_layout(); _save(fig, "journey")
    print(f"  fig_journey: champion {champ:.0f}% -> P7 {p7pct:.0f}%")


def fig_learning_curves():
    """How the curriculum drives learning: win-rate vs the champion over training steps for a few
    representative runs, with the curriculum stage boundaries (random -> +pool -> +self -> +strong)
    marked. Reads the per-run learning curve saved in sweep/curriculum/*.json."""
    runs = [("01_base_trpo_s0", "TRPO baseline", C_GREEN),
            ("05_rew_R4_earlyknk_s0", "early-knock reward", C_GREEN2),
            ("12_ppo_ent_hi_s0", "PPO", C_GREY)]
    fig, ax = plt.subplots(figsize=(6.8, 3.7))
    drew = False
    for name, lab, c in runs:
        p = os.path.join(ROOT, "sweep", "curriculum", name + ".json")
        if not os.path.exists(p):
            continue
        curve = json.load(open(p)).get("curve", [])
        if not curve:
            continue
        xs = [s["step"] / 1e6 for s in curve]
        ys = [s["champion"]["win"] * 100 for s in curve]
        ax.plot(xs, ys, "-o", color=c, lw=2.2, ms=5, label=lab); drew = True
    if not drew:
        ax.text(0.5, 0.5, "learning curves pending", ha="center", va="center")
        ax.axis("off"); fig.tight_layout(); _save(fig, "learning_curves"); return
    ax.set_ylim(0, 60); ax.set_xlim(0, 12.3)
    for frac, lab in [(0.25, "+ pool"), (0.45, "+ self"), (0.70, "+ strong")]:
        x = frac * 12
        ax.axvline(x, ls=":", lw=1, color="#9aa6a0")
        ax.text(x + 0.1, 57, lab, fontsize=8, color="#5c6b73", va="top")
    ax.text(0.2, 57, "vs random", fontsize=8, color="#5c6b73", va="top")
    ax.set_xlabel("training steps (millions)"); ax.set_ylabel("win rate vs champion (%)")
    ax.set_title("Skill rises as the curriculum adds tougher opponents")
    ax.legend(loc="lower right"); ax.grid(ls=":", alpha=0.4)
    fig.tight_layout(); _save(fig, "learning_curves")
    print("  fig_learning_curves: ok")


# ----- Phase-8: architecture comparison with IQM + 95% stratified bootstrap CIs (rliable-style)
def _iqm(x):
    """Interquartile mean (Agarwal et al. 2021): mean of the middle 50%. With <4 samples there is
    nothing to trim, so it degrades to the plain mean."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n >= 4:
        return float(x[int(np.floor(n * 0.25)):int(np.ceil(n * 0.75))].mean())
    return float(x.mean())


def _bootstrap_ci(x, B=2000, alpha=0.05):
    """95% bootstrap CI of the IQM by resampling the seeds with replacement."""
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return float(x.mean()), float(x.mean())
    rng = np.random.default_rng(0)
    stats = np.array([_iqm(rng.choice(x, size=len(x), replace=True)) for _ in range(B)])
    return float(np.percentile(stats, 100 * alpha / 2)), float(np.percentile(stats, 100 * (1 - alpha / 2)))


def fig_arch_rliable():
    """Phase-8: win-rate vs the fixed expert per architecture, as the IQM with 95% bootstrap CIs
    over seeds. Reads sweep/curriculum/arch_*.json; skips until >=2 architectures have finished."""
    paths = sorted(glob.glob(os.path.join(ROOT, "sweep", "curriculum", "arch_*.json")))
    by = {}
    for p in paths:
        try:
            c = json.load(open(p))
        except Exception:  # noqa: BLE001
            continue
        wr = (c.get("vs_gold") or {}).get("win_rate")
        if wr is None:
            continue
        by.setdefault(c.get("name", "?").rsplit("_s", 1)[0], []).append(100.0 * wr)
    if len(by) < 2:
        print("  [skip] fig_arch_rliable: <2 architectures finished")
        return
    rows = []
    for cell, vs in by.items():
        lo, hi = _bootstrap_ci(vs)
        rows.append((cell.replace("arch_", ""), _iqm(vs), lo, hi, len(vs)))
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(7.2, max(2.4, 0.42 * len(rows) + 1.0)))
    ys = np.arange(len(rows))
    pts = [r[1] for r in rows]
    err = [[r[1] - r[2] for r in rows], [r[3] - r[1] for r in rows]]
    ax.errorbar(pts, ys, xerr=err, fmt="o", color=C_GREEN, ecolor=C_GREY, capsize=3, ms=5, lw=1.2)
    anchor = next((r[1] for r in rows if r[0] == "mlp_default"), None)
    if anchor is not None:
        ax.axvline(anchor, ls="--", color=C_GOLD, lw=1, label="MLP anchor")
        ax.legend(loc="lower right")
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r[0]} (n={r[4]})" for r in rows])
    ax.set_xlabel("win rate vs the fixed expert (%) — IQM, 95% bootstrap CI")
    ax.set_title("Does network architecture move the agent past the expert ceiling?")
    ax.grid(ls=":", axis="x", alpha=0.4)
    fig.tight_layout(); _save(fig, "arch_rliable")
    print(f"  fig_arch_rliable: ok ({len(rows)} architectures)")


def fig_ismcts():
    """Phase-8: ISMCTS win-rate vs the fixed expert by per-move rollout budget, FAIR (determinized,
    re-deals hidden cards) vs ORACLE (peeks at the true hand). Reads sweep/curriculum/ismcts_*vs_gold*.
    The fair curve sitting below the trained agents while the oracle soars is the information-bound
    evidence: the gap is the value of the hidden information."""
    fair, orac = {}, {}
    for p in glob.glob(os.path.join(ROOT, "sweep", "curriculum", "ismcts_*vs_gold*.json")):
        try:
            c = json.load(open(p))
        except Exception:  # noqa: BLE001
            continue
        wr = (c.get("vs_gold") or {}).get("win_rate")
        if wr is None:
            continue
        (fair if c.get("determinize") else orac)[int(c.get("rollouts", 0))] = 100.0 * wr
    if not fair and not orac:
        print("  [skip] fig_ismcts: no ismcts vs gold results")
        return
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    if orac:
        xs = sorted(orac)
        ax.plot(xs, [orac[x] for x in xs], "-o", color=C_GOLD, lw=1.8, ms=5,
                label="oracle search (peeks at hand)")
    if fair:
        xs = sorted(fair)
        ax.plot(xs, [fair[x] for x in xs], "-o", color=C_GREEN, lw=1.8, ms=5,
                label="fair search (cards hidden)")
    ax.axhline(34.2, ls="--", lw=1.2, color=C_INK)
    ax.text(ax.get_xlim()[1], 35.4, "best trained agent (34%)", fontsize=8, ha="right", color=C_INK)
    ax.set_xlabel("search rollouts per move")
    ax.set_ylabel("win rate vs the fixed expert (%)")
    ax.set_title("A fair search is weak; only an oracle beats the expert")
    ax.grid(ls=":", alpha=0.4)
    ax.legend(loc="center right", fontsize=8)
    fig.tight_layout(); _save(fig, "ismcts_fair_oracle")
    print(f"  fig_ismcts: ok (fair={len(fair)}, oracle={len(orac)})")


def fig_archbase():
    """Full-width Phase-8 headline: LEFT = win-rate vs the expert by architecture (IQM, 95% bootstrap
    CI over seeds); RIGHT = ISMCTS vs the expert by rollout budget, fair (cards hidden) vs oracle
    (peeks). One figure spanning both columns so the text stays legible. Reads arch_*.json and
    ismcts_*vs_gold*.json; skips if neither is ready."""
    # --- architecture data ---
    by = {}
    for p in sorted(glob.glob(os.path.join(ROOT, "sweep", "curriculum", "arch_*.json"))):
        try:
            c = json.load(open(p))
        except Exception:  # noqa: BLE001
            continue
        wr = (c.get("vs_gold") or {}).get("win_rate")
        if wr is not None:
            by.setdefault(c.get("name", "?").rsplit("_s", 1)[0], []).append(100.0 * wr)
    # --- ISMCTS data ---
    fair, orac = {}, {}
    for p in glob.glob(os.path.join(ROOT, "sweep", "curriculum", "ismcts_*vs_gold*.json")):
        try:
            c = json.load(open(p))
        except Exception:  # noqa: BLE001
            continue
        wr = (c.get("vs_gold") or {}).get("win_rate")
        if wr is None:
            continue
        (fair if c.get("determinize") else orac)[int(c.get("rollouts", 0))] = 100.0 * wr
    if len(by) < 2 and not (fair or orac):
        print("  [skip] fig_archbase: data not ready")
        return

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.6, 4.0), gridspec_kw={"width_ratios": [1.18, 1]})

    # LEFT: architecture IQM with CIs
    NICE = {"mlp_default": "MLP (anchor)", "mlp_wide": "MLP wide", "mlp_narrow": "MLP narrow",
            "mlp_deep": "MLP deep", "mlp_xwide": "MLP x-wide", "mlp_asym": "MLP asym",
            "act_relu": "ReLU", "act_gelu": "GELU", "conv1d": "Conv1D", "deepsets": "DeepSets",
            "attn": "Attention", "attn_deep": "Attention (deep)", "deepsets_big": "DeepSets (big)",
            "wd_lo": "Weight decay (lo)", "wd_hi": "Weight decay (hi)"}
    rows = []
    for cell, vs in by.items():
        lo, hi = _bootstrap_ci(vs)
        rows.append((NICE.get(cell.replace("arch_", ""), cell.replace("arch_", "")),
                     _iqm(vs), lo, hi, len(vs), cell.replace("arch_", "")))
    rows.sort(key=lambda r: r[1])
    ys = np.arange(len(rows))
    err = [[r[1] - r[2] for r in rows], [r[3] - r[1] for r in rows]]
    structured = {"conv1d", "deepsets", "attn", "attn_deep", "deepsets_big"}
    cols = [C_GOLD if r[5] in structured else C_GREEN for r in rows]
    for y, r, c in zip(ys, rows, cols):
        axL.errorbar(r[1], y, xerr=[[r[1] - r[2]], [r[3] - r[1]]], fmt="o", color=c,
                     ecolor=C_GREY, capsize=3, ms=6, lw=1.4, zorder=3)
    anchor = next((r[1] for r in rows if r[5] == "mlp_default"), None)
    if anchor is not None:
        axL.axvline(anchor, ls="--", color=C_INK, lw=1, alpha=0.6)
    axL.set_yticks(ys)
    axL.set_yticklabels([f"{r[0]}" for r in rows])
    axL.set_xlabel("win rate vs the expert (%)")
    axL.set_title("(a) Architecture is not the lever")
    axL.grid(ls=":", axis="x", alpha=0.45)
    axL.margins(y=0.03)

    # RIGHT: ISMCTS fair vs oracle
    if orac:
        xs = sorted(orac)
        axR.plot(xs, [orac[x] for x in xs], "-o", color=C_GOLD, label="oracle (peeks)")
    if fair:
        xs = sorted(fair)
        axR.plot(xs, [fair[x] for x in xs], "-o", color=C_GREEN, label="fair (hidden)")
    axR.axhline(34.2, ls="--", lw=1.2, color=C_INK, alpha=0.7)
    axR.text(0.98, 0.40, "best trained agent", transform=axR.transAxes, fontsize=8.5,
             ha="right", color=C_INK)
    axR.set_xlabel("search rollouts per move")
    axR.set_ylabel("win rate vs the expert (%)")
    axR.set_title("(b) A fair search is weak")
    axR.grid(ls=":", alpha=0.45)
    axR.legend(loc="center right")
    fig.tight_layout(w_pad=2.0)
    _save(fig, "archbase")
    print(f"  fig_archbase: ok (arch={len(rows)}, fair={len(fair)}, oracle={len(orac)})")


def main():
    rs = load_results()
    print(f"loaded {len(rs)} phase-1 results")
    fig_winrate(rs); fig_mean_reward(rs); fig_lr_vs_reward(rs); write_summary_csv(rs)
    for fn in (fig_selfplay, fig_pool, fig_llm_opponent, fig_infra, fig_throughput,
               fig_rlvsllm, fig_h2h, fig_ginshape, fig_gold, fig_algo, fig_phase5,
               fig_curriculum, fig_curriculum_reward, fig_reward_gin,
               fig_gold_bench, fig_regimes, fig_journey, fig_architecture, fig_learning_curves,
               fig_arch_rliable, fig_ismcts, fig_archbase):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            print(f"  [skip] {fn.__name__}: {exc}")
    print(f"figures written to {FIGS}")


if __name__ == "__main__":
    main()
