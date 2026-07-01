# Phase-8 results summary

## Gin Rummy: architecture sweep vs GOLD expert (win-rate, best ckpt)
_24 cells with results_

| win% vs gold | cell | arch | act |
|---:|---|---|---|
| 0.331 | arch_conv1d_s0 | conv1d | tanh |
| 0.307 | arch_deepsets_s0 | deepsets | tanh |
| 0.306 | arch_deepsets_s1 | deepsets | tanh |
| 0.302 | arch_deepsets_s2 | deepsets | tanh |
| 0.296 | arch_mlp_wide_s1 | mlp | tanh |
| 0.296 | arch_mlp_default_s2 | mlp | tanh |
| 0.295 | arch_mlp_asym_s0 | mlp | tanh |
| 0.291 | arch_conv1d_s1 | conv1d | tanh |
| 0.289 | arch_mlp_xwide_s0 | mlp | tanh |
| 0.289 | arch_deepsets_s3 | deepsets | tanh |
| 0.285 | arch_mlp_asym_s3 | mlp | tanh |
| 0.281 | arch_mlp_narrow_s0 | mlp | tanh |
| 0.280 | arch_mlp_deep_s0 | mlp | tanh |
| 0.277 | arch_act_relu_s0 | mlp | relu |
| 0.272 | arch_mlp_asym_s2 | mlp | tanh |
| 0.271 | arch_mlp_xwide_s1 | mlp | tanh |
| 0.269 | arch_mlp_default_s3 | mlp | tanh |
| 0.268 | arch_act_gelu_s0 | mlp | gelu |
| 0.265 | arch_mlp_default_s1 | mlp | tanh |
| 0.264 | arch_wd_lo_s0 | mlp | tanh |
| 0.264 | arch_mlp_default_s0 | mlp | tanh |
| 0.252 | arch_mlp_wide_s0 | mlp | tanh |
| 0.241 | arch_wd_hi_s0 | mlp | tanh |
| 0.221 | arch_mlp_asym_s1 | mlp | tanh |

### Per-cell IQM (across seeds) with 95% bootstrap CI

| cell | n | IQM | CI low | CI high |
|---|---:|---:|---:|---:|
| arch_conv1d | 2 | 0.311 | 0.291 | 0.331 |
| arch_deepsets | 4 | 0.304 | 0.289 | 0.307 |
| arch_mlp_narrow | 1 | 0.281 | 0.281 | 0.281 |
| arch_mlp_xwide | 2 | 0.280 | 0.271 | 0.289 |
| arch_mlp_deep | 1 | 0.280 | 0.280 | 0.280 |
| arch_act_relu | 1 | 0.277 | 0.277 | 0.277 |
| arch_mlp_wide | 2 | 0.274 | 0.252 | 0.296 |
| arch_mlp_default | 4 | 0.267 | 0.264 | 0.296 |
| arch_mlp_asym | 4 | 0.278 | 0.221 | 0.295 |
| arch_act_gelu | 1 | 0.268 | 0.268 | 0.268 |
| arch_wd_lo | 1 | 0.264 | 0.264 | 0.264 |
| arch_wd_hi | 1 | 0.241 | 0.241 | 0.241 |

## Stage-C: cross-recipe robustness vs GOLD (win-rate, best ckpt)

| win% vs gold | cell (recipe_arch_seed) |
|---:|---|
| 0.302 | sc_ppo_asym_s0 |
| 0.287 | sc_pfsp_wide_s0 |
| 0.283 | sc_pfsp_wide_s1 |
| 0.280 | sc_ppo_mlpdef_s1 |
| 0.278 | sc_ppo_wide_s0 |
| 0.278 | sc_ppo_asym_s1 |
| 0.276 | sc_ppo_wide_s1 |
| 0.271 | sc_ppo_mlpdef_s0 |
| 0.270 | sc_pfsp_mlpdef_s1 |
| 0.253 | sc_pfsp_asym_s1 |
| 0.250 | sc_pfsp_mlpdef_s0 |
| 0.243 | sc_pfsp_asym_s0 |

## Recurrence (LSTM vs PPO-MLP control) vs GOLD

| win% vs gold | cell |
|---:|---|
| 0.243 | rec_lstm_s0 |
| 0.202 | rec_lstm_s1 |
| 0.152 | rec_mlp_ctrl_s0 |
| 0.192 | rec_mlp_ctrl_s1 |

## ISMCTS vs GOLD by rollout budget &mdash; determinized = FAIR imperfect-info baseline

| rollouts | win% vs gold | gin% | mean len | cell |
|---:|---:|---:|---:|---|
| 10 | 0.10333333333333333 | 0.02 | 33.013333333333335 | ismcts_det_vs_gold_r10 |
| 30 | 0.17333333333333334 | 0.04666666666666667 | 31.553333333333335 | ismcts_det_vs_gold_r30 |
| 60 | 0.21 | 0.03666666666666667 | 31.773333333333333 | ismcts_det_vs_gold_r60 |
| 120 | 0.256 | 0.048 | 30.52 | ismcts_det_vs_gold_r120 |

## ISMCTS vs GOLD by rollout budget &mdash; oracle = perfect-info UPPER BOUND (sees opponent cards)

| rollouts | win% vs gold | gin% | mean len | cell |
|---:|---:|---:|---:|---|
| 10 | 0.4066666666666667 | 0.09333333333333334 | 47.82333333333333 | ismcts_vs_gold_r10 |
| 20 | 0.55 | 0.125 | 47.65 | ismcts_vs_gold |
| 30 | 0.59 | 0.11666666666666667 | 48.49333333333333 | ismcts_vs_gold_r30 |
| 60 | 0.7533333333333333 | 0.14333333333333334 | 43.53 | ismcts_vs_gold_r60 |
| 120 | 0.8533333333333334 | 0.21 | 40.28333333333333 | ismcts_vs_gold_r120 |

## Head-to-head: trained models vs ISMCTS (model win-rate)

| model | win% vs ISMCTS | rollouts | mode | cell |
|---|---:|---:|---|---|
| tactician | 0.6933333333333334 | 60 | det | h2h_tactician_vs_ismcts_det_r60 |
| ace | 0.6933333333333334 | 60 | det | h2h_ace_vs_ismcts_det_r60 |
| goldhunter | 0.68 | 60 | det | h2h_goldhunter_vs_ismcts_det_r60 |
| selfplay | 0.6633333333333333 | 60 | det | h2h_selfplay_vs_ismcts_det_r60 |

## Leduc Hold'em generality: return vs CFR-optimal expert

_(0 = parity with the game-theoretic optimum; random baseline ~ -0.78)_

| agent | seed | return vs CFR |
|---|---:|---:|
| nfsp | 0 | -0.7622 |
| nfsp | 1 | -0.6803 |
| nfsp | 2 | -0.619 |
| nfsp | 3 | -0.7803 |
| tabular_q | 0 | -0.169 |
| tabular_q | 1 | -0.1517 |
| tabular_q | 2 | 0.0053 |
| tabular_q | 3 | -0.1058 |
| tabular_q | 4 | -0.052 |
| tabular_q | 5 | -0.072 |
| tabular_q | 6 | -0.105 |
| tabular_q | 7 | -0.0292 |

| agent | n seeds | mean return vs CFR |
|---|---:|---:|
| tabular_q | 8 | -0.085 |
| nfsp | 4 | -0.710 |
