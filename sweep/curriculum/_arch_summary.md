# Phase-8 architecture sweep — evaluation vs the fixed expert

_14 runs finished across 9 architectures. Win-rate is vs the fixed deterministic expert (benchmark-only)._

## Per architecture (mean over seeds)

| architecture | n | win% vs expert | std | Δ vs MLP anchor |
|---|---|---|---|---|
| arch_mlp_narrow | 1 | 0.288 | 0.000 | +0.024 |
| arch_mlp_deep | 1 | 0.280 | 0.000 | +0.015 |
| arch_act_relu | 1 | 0.279 | 0.000 | +0.015 |
| arch_mlp_wide | 2 | 0.276 | 0.024 | +0.011 |
| arch_mlp_asym | 4 | 0.275 | 0.017 | +0.011 |
| arch_act_gelu | 1 | 0.268 | 0.000 | +0.004 |
| arch_mlp_default | 2 | 0.265 | 0.001 | +0.000 |
| arch_wd_lo | 1 | 0.264 | 0.000 | -0.001 |
| arch_wd_hi | 1 | 0.258 | 0.000 | -0.006 |

## Per run

| run | arch | act | net | seed | win% vs expert | gin% | mean len | vs champ | vs random | secs |
|---|---|---|---|---|---|---|---|---|---|---|
| arch_mlp_wide_s1 | mlp | tanh | [512, 256] | 1 | 0.299 | 0.008 | 34.28 | 0.432 | 0.995 | 18615 |
| arch_mlp_asym_s0 | mlp | tanh | [128, 64] | 0 | 0.295 | 0.010 | 34.18 | 0.436 | 0.990 | 10805 |
| arch_mlp_narrow_s0 | mlp | tanh | [128, 64] | 0 | 0.288 | 0.002 | 33.73 | 0.466 | 0.988 | 12234 |
| arch_mlp_asym_s3 | mlp | tanh | [128, 64] | 3 | 0.285 | 0.003 | 34.4 | 0.428 | 0.987 | 10766 |
| arch_mlp_deep_s0 | mlp | tanh | [256, 256, 128] | 0 | 0.280 | 0.004 | 34.26 | 0.458 | 0.991 | 16511 |
| arch_act_relu_s0 | mlp | relu | [256,128] | 0 | 0.279 | 0.007 | 34.86 | 0.417 | 0.989 | 13633 |
| arch_mlp_asym_s2 | mlp | tanh | [128, 64] | 2 | 0.272 | 0.005 | 34.61 | 0.396 | 0.990 | 10615 |
| arch_act_gelu_s0 | mlp | gelu | [256,128] | 0 | 0.268 | 0.002 | 34.91 | 0.451 | 0.988 | 11644 |
| arch_mlp_default_s1 | mlp | tanh | [256,128] | 1 | 0.265 | 0.004 | 35.07 | 0.392 | 0.986 | 8668 |
| arch_mlp_default_s0 | mlp | tanh | [256,128] | 0 | 0.264 | 0.006 | 33.81 | 0.437 | 0.985 | 8189 |
| arch_wd_lo_s0 | mlp | tanh | [256,128] | 0 | 0.264 | 0.006 | 35.05 | 0.399 | 0.992 | 8355 |
| arch_wd_hi_s0 | mlp | tanh | [256,128] | 0 | 0.258 | 0.004 | 35.17 | 0.407 | 0.993 | 9670 |
| arch_mlp_wide_s0 | mlp | tanh | [512, 256] | 0 | 0.252 | 0.007 | 35.65 | 0.395 | 0.996 | 14482 |
| arch_mlp_asym_s1 | mlp | tanh | [128, 64] | 1 | 0.249 | 0.002 | 35.65 | 0.407 | 0.991 | 10696 |
