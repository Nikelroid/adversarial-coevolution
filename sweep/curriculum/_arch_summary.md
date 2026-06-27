# Phase-8 architecture sweep — evaluation vs the fixed expert

_5 runs finished across 4 architectures. Win-rate is vs the fixed deterministic expert (benchmark-only)._

## Per architecture (mean over seeds)

| architecture | n | win% vs expert | std | Δ vs MLP anchor |
|---|---|---|---|---|
| arch_mlp_asym | 1 | 0.295 | 0.000 | +0.030 |
| arch_act_relu | 1 | 0.290 | 0.000 | +0.025 |
| arch_mlp_narrow | 1 | 0.281 | 0.000 | +0.017 |
| arch_mlp_default | 2 | 0.265 | 0.001 | +0.000 |

## Per run

| run | arch | act | net | seed | win% vs expert | gin% | mean len | vs champ | vs random | secs |
|---|---|---|---|---|---|---|---|---|---|---|
| arch_mlp_asym_s0 | mlp | tanh | [128, 64] | 0 | 0.295 | 0.010 | 34.18 | 0.436 | 0.990 | 10805 |
| arch_act_relu_s0 | mlp | relu | [256,128] | 0 | 0.290 | 0.003 | 34.71 | 0.445 | 0.996 | 10285 |
| arch_mlp_narrow_s0 | mlp | tanh | [128, 64] | 0 | 0.281 | 0.002 | 34.18 | 0.426 | 0.988 | 8623 |
| arch_mlp_default_s1 | mlp | tanh | [256,128] | 1 | 0.265 | 0.004 | 35.07 | 0.392 | 0.986 | 8668 |
| arch_mlp_default_s0 | mlp | tanh | [256,128] | 0 | 0.264 | 0.006 | 33.81 | 0.437 | 0.985 | 8189 |
