# Phase-8 architecture sweep — evaluation vs the fixed expert

_1 runs finished across 1 architectures. Win-rate is vs the fixed deterministic expert (benchmark-only)._

## Per architecture (mean over seeds)

| architecture | n | win% vs expert | std | Δ vs MLP anchor |
|---|---|---|---|---|
| arch_mlp_default | 1 | 0.264 | 0.000 | +0.000 |

## Per run

| run | arch | act | net | seed | win% vs expert | gin% | mean len | vs champ | vs random | secs |
|---|---|---|---|---|---|---|---|---|---|---|
| arch_mlp_default_s0 | mlp | tanh | [256,128] | 0 | 0.264 | 0.006 | 33.81 | 0.437 | 0.985 | 8189 |
