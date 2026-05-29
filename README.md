<h1 align="center">Adversarial Co-Evolution of RL and LLM Agents in Gin Rummy</h1>

<p align="center">
  <i>A hybrid system where a lightweight action-masked <b>PPO</b> agent and a strong-but-slow
  <b>LLM</b> opponent train against each other — without paying full LLM latency on every RL step.</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/RL-action--masked%20PPO-0b5b39"/>
  <img src="https://img.shields.io/badge/env-PettingZoo%20%2F%20RLCard-0b5b39"/>
  <img src="https://img.shields.io/badge/LLM-Qwen2.5--7B-0b5b39"/>
  <img src="https://img.shields.io/badge/serving-master%2Fworker%2Fcache-0b5b39"/>
  <img src="https://img.shields.io/badge/cluster-SLURM%20%C2%B7%20up%20to%2032%20V100-0b5b39"/>
</p>

<p align="center">
  📊 <b><a href="https://htmlpreview.github.io/?https://github.com/Nikelroid/adversarial-coevolution/blob/main/docs/report.html">Read the full HTML report</a></b>
  &nbsp;·&nbsp; 📄 <b><a href="paper/main.pdf">4-page PDF</a></b>
  &nbsp;·&nbsp; 🎮 <a href="game/">Play the web game</a>
</p>

---

## TL;DR

| | |
|---|---|
| 🏆 **Best PPO vs random** | **99.6%** (8-config sweep all in 98.3–99.6%) |
| 🤖 **Qwen2.5-7B opponent** | **beats random 5/5** — a genuinely useful teacher (OLMoE-1B plays at chance) |
| ⚡ **Worker weight load** | **62× faster** from scratch/BeeGFS than home NFS (27 s vs ~28 min) |
| 🚀 **LLM-in-loop throughput** | **~32 queries/s** over 14 workers; per-call latency **16 s → 0.4 s** |
| 🧩 **First RL-vs-LLM run** | 40k steps in 17.5 min; agent **retains 98.2% vs random** |

> Gin Rummy needs both short-horizon arithmetic (deadwood counting) and long-horizon planning
> (meld formation). Our roadmap: **(1)** train a strong RL backbone vs weak opponents →
> **(2)** train it vs an LLM opponent that is richer but far slower → **(3)** let them co-evolve.
> This repo covers **Phases 1–2**.

<p align="center">
  <img src="paper/figures/game_ui.png" width="640" alt="Human vs RL web game"/><br/>
  <sub><b>The human-vs-RL web client</b> (debug view, opponent hand revealed) — 3-D card animations, four trained opponents.</sub>
</p>

---

## Phase 1 — the RL backbone saturates against random

Action-masked PPO (illegal-action logits → −∞) on PettingZoo `gin_rummy_v4`. An eight-configuration
sweep (3×2 grid over learning rate × entropy, plus two ablations), 2M steps each, 96 parallel envs,
evaluated over 1000 deterministic games. **Every config lands in the 98.3–99.6% band** — statistically
indistinguishable (binomial 95% CI ±0.6 pp). The shared failure mode is the lever: mean reward sits at
the **knock** value (0.5), not the **gin** value (1.5) — a risk/reward call only a *thinking* opponent
can teach. That motivates Phase 2.

<p align="center"><img src="paper/figures/win_rate.png" width="540" alt="Phase-1 win rates"/></p>

| Config | win% | loss% | note |
|---|--:|--:|---|
| cfg5 · lr 5e-5, ent .03 | **99.6** | 0.4 | best of sweep |
| cfg7 · lr 1e-4, 10 epochs | 99.5 | 0.5 | ablation |
| cfg0 · lr 3e-4, ent .01 | 99.4 | 0.6 | baseline |
| cfg3 · lr 1e-4, ent .03 | 99.4 | 0.6 | |
| cfg6 · lr 1e-4, n_steps 1024 | 99.3 | 0.7 | ablation |
| cfg4 · lr 5e-5, ent .01 | 99.2 | 0.8 | |
| cfg1 · lr 3e-4, ent .03 | 98.8 | 1.2 | |
| cfg2 · lr 1e-4, ent .01 | 98.3 | 1.7 | worst of sweep |

---

## Phase 2 — LLM-in-the-loop infrastructure

A single PPO rollout fires up to ~50k opponent queries; at 0.5–3 s per 7B call a naïve loop takes hours.
We decouple inference from RL with a **three-tier stack**:

```
   env subprocess ─▶  Master (CPU, FastAPI)  ─▶ suit-symmetry cache (hit? return)
   (per-step query)   Ollama-compatible API          │ miss
                              │ round-robin           ▼
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
         vLLM/HF worker  worker  …  worker   (1 GPU each, registers in shared-FS registry)
         Qwen2.5-7B      (master discovers + health-checks + load-balances)
```

- **Workers** load the model and self-register in a shared-filesystem registry (`llm/discovery.py`).
- **Master** (`llm/master.py`) discovers live workers, load-balances, and owns a **suit-symmetry prompt
  cache** — Gin Rummy is invariant under suit relabeling, so suit-equivalent states share one entry.
- **Client** reaches the master through the existing Ollama-compatible API — no RL code changes.

<table>
<tr>
<td align="center"><img src="paper/figures/selfplay.png" width="250"/><br/><sub><b>Self-play</b> beats its progenitor</sub></td>
<td align="center"><img src="paper/figures/pool.png" width="250"/><br/><sub><b>Pool</b> diverged after ~10M</sub></td>
</tr>
<tr>
<td align="center"><img src="paper/figures/llm_opponent.png" width="250"/><br/><sub><b>Qwen2.5-7B</b> is competent; OLMoE isn't</sub></td>
<td align="center"><img src="paper/figures/throughput.png" width="250"/><br/><sub><b>Throughput</b>: ~32 q/s, scales with pool</sub></td>
</tr>
</table>

| Agent / opponent | vs | win% | loss% | note |
|---|---|--:|--:|---|
| Self-play (3M) | frozen run_5 | **61.1** | 38.9 | beats progenitor |
| Self-play (3M) | random | 98.7 | 1.3 | stays dominant |
| Pool champion (12M) | random | 85.8 | 14.2 | ⚠️ diverged |
| OLMoE-1B opponent | random | ≈50 | — | plays at chance |
| Qwen2.5-7B opponent | random | **100** | 0 | 5/5 (small N) |

> 💡 **Infra finding:** loading a 7B worker from home NFS runs at ~11 MB/s (~28 min — blows the
> health-check timeout). Staging weights on **scratch/BeeGFS cuts this to 27 s (62×)** — mandatory at scale.

---

## First RL-vs-LLM run

PPO **warm-started from the self-play champion**, 64 env subprocesses, playing through the master against
a pool of up to 32 Qwen2.5-7B workers (granted 14 under the shared GPU quota — elastic). Ran **40k steps
(~5 rollouts) in 17.5 min** at 39 env-steps/s; the terse prompt + 64-token cap dropped per-call latency
from ~16 s to ~0.4 s.

| Fine-tuned agent (40k steps) | vs | win% | loss% |
|---|---|--:|--:|
| LLM-finetuned PPO | random | **98.2** | 1.8 |
| LLM-finetuned PPO | self-play champion | 45.4 | 54.6 |

> **Honest read:** the agent *retains* 98.2% vs random but sits at 45.4% vs the champion it started from —
> 40k steps (against 3M for self-play) is far too short to improve. This run validates the **pipeline** and
> confirms competence is retained; showing the LLM teacher actually *helps* is the **Phase-3** experiment
> (a much longer fine-tune to raise the gin rate, the headroom Phase 1 exposed).

---

## Repository layout

| Path | What |
|---|---|
| `ppo_train.py`, `gym_wrapper.py` | masked-PPO policy + single-agent wrapper over `gin_rummy_v4` |
| `sweep/` | Phase-1 sweep, self-play, pool, and `llmplay_one.py` (RL-vs-LLM) training scripts |
| `llm/` | master / worker / cache / discovery + `eval_opponent.py`, `fast_llm_agent` prompt |
| `agents/` | `RandomAgent`, `PPOAgent`, `LLMAgent`, `FastLLMAgent` |
| `slurm/` | SLURM jobs: `master`, `worker` (array), `llm_train`, `llm_eval` |
| `game/` | zero-dependency human-vs-RL web client (server + HTML/JS) |
| `paper/` | the report (`main.tex` → `main.pdf`), figures, `make_figures.py`, `make_report_html.py` |
| `docs/` | `report.html` (full report), `llm_architecture.md` |

---

## Quickstart

```bash
# 1) Play against the trained agent (web game)
python game/server.py --host 127.0.0.1 --port 8000      # then open http://127.0.0.1:8000

# 2) Phase-1 sweep (SLURM, 64-core CPU node)
sbatch sweep/sweep.slurm

# 3) LLM opponent stack (SLURM): master + GPU worker pool
sbatch slurm/master.slurm
sbatch --array=0-31 --export=ALL,WORKER_PRESET=qwen2.5-7b,WORKER_MAXTOK=64 slurm/worker.slurm

# 4) RL-vs-LLM fine-tune (finds the master via runtime/master.json)
sbatch slurm/llm_train.slurm

# regenerate report figures + HTML
python paper/make_figures.py && python paper/make_report_html.py
```

> ⚠️ Load LLM worker weights from **scratch/BeeGFS** (`HF_HOME=/scratch.../hf_cache`), not home NFS.

---

## Authors

Nima Kelidari · Mahdi Salmani · Mohammadsaeed Haghi — University of Southern California

<sub>Built on PPO + action masking, PettingZoo/RLCard, Stable-Baselines3, and Qwen2.5. Figures and report
are regenerated from measured JSON results — see <a href="docs/report.html">the full report</a>.</sub>
