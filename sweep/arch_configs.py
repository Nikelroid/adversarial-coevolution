"""Phase-8 = "does ARCHITECTURE break the ceiling?". Generate the architecture-sweep configs.

Holds the winning RECIPE fixed (TRPO, knock=0.5/gin=0.5, step_penalty=0.0075, deadwood_coef=0.05,
a designed curriculum, recent sampling, the proven HP bundle) and varies ONLY the policy network:
MLP depth/width, activation function, and three structured encoders over the (4,52) card planes
(Conv1D, permutation-invariant DeepSets, self-attention). Every cell trains FROM SCRATCH, because a
different-shaped network cannot warm-start from the MLP champion checkpoint (algo_cls.load restores
the saved module shapes). That is the correct framing: a clean RELATIVE architecture ranking at a
fixed from-scratch budget, graded against the fixed expert yardstick.

    python sweep/arch_configs.py     # writes sweep/arch_cfgs/*.json + INDEX.txt
"""
import json
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arch_cfgs")

# Winner recipe in SCRATCH form (init_model="" -> build_model runs, so the configured net is the
# one actually trained). Mirrors p7_05_scratch_deadwood but with the winner's deadwood_coef=0.05.
BASE = dict(algo="trpo", init_model="", curriculum="designed", sampling="recent",
            strong_from=0.70, knock=0.5, gin=0.5, win_scale=1.0, loss_scale=1.0,
            step_penalty=0.0075, deadwood_coef=0.05, target_kl=0.01, ent_coef=0.01,
            clip_range=0.2, lr=3e-4, gamma=0.99, gae_lambda=0.95, n_steps=256,
            steps=10_000_000, eval_every=1_000_000, ckpt_eval_games=400, final_eval_games=1000,
            save_freq=1_500_000,
            arch="mlp", net_arch=None, activation="tanh", extractor_kwargs={}, weight_decay=0.0)

# (name, overrides, seeds, purpose) -- vary ONLY the network.
CELLS = [
    ("arch_mlp_default", {}, [0, 1],
     "Anchor: winner recipe from scratch, MLP [256,128]. Baseline for every architecture."),
    ("arch_mlp_wide", dict(net_arch=dict(pi=[512, 256], vf=[512, 256])), [0, 1],
     "Wider MLP. Andrychowicz'21: policy width has a sweet spot -- does more help here?"),
    ("arch_mlp_narrow", dict(net_arch=dict(pi=[128, 64], vf=[128, 64])), [0],
     "Narrower MLP: a capacity floor."),
    ("arch_mlp_deep", dict(net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128])), [0],
     "Deeper MLP (3 hidden layers). Expect a plateau -- 2 layers usually suffice."),
    ("arch_mlp_asym", dict(net_arch=dict(pi=[128, 64], vf=[512, 256])), [0],
     "Narrow policy + wide value net (Andrychowicz'21)."),
    ("arch_act_relu", dict(activation="relu"), [0], "ReLU vs the default Tanh."),
    ("arch_act_gelu", dict(activation="gelu"), [0], "GELU vs the default Tanh."),
    ("arch_conv1d", dict(arch="conv1d", extractor_kwargs=dict(features_dim=256)), [0, 1],
     "1-D conv over the (4,52) planes: adjacency ~ runs if the deck is rank-ordered."),
    ("arch_deepsets", dict(arch="deepsets", extractor_kwargs=dict(features_dim=128)), [0, 1],
     "Permutation-invariant set encoder (Deep Sets): principled for an unordered hand."),
    ("arch_attn", dict(arch="attn", extractor_kwargs=dict(features_dim=128, layers=2)), [0, 1],
     "Self-attention over the 52 card tokens (light transformer)."),
    # --- regularization (Wave 2): safe weight-decay lever on the winner MLP net ---
    ("arch_wd_lo", dict(weight_decay=1e-4), [0], "MLP + weight-decay 1e-4 (regularization)."),
    ("arch_wd_hi", dict(weight_decay=1e-3), [0], "MLP + weight-decay 1e-3 (regularization)."),
]


def build():
    os.makedirs(OUT, exist_ok=True)
    for f in os.listdir(OUT):
        if f.endswith(".json") or f == "INDEX.txt":
            os.remove(os.path.join(OUT, f))
    names = []
    for cell_name, ov, seeds, purpose in CELLS:
        for sd in seeds:
            c = dict(BASE); c.update(ov)
            name = f"{cell_name}_s{sd}"
            cfg = dict(name=name, seed=sd, purpose=purpose, **c)
            json.dump(cfg, open(os.path.join(OUT, f"{name}.json"), "w"), indent=2)
            names.append(name)
    with open(os.path.join(OUT, "INDEX.txt"), "w") as f:
        f.write("\n".join(names) + "\n")
    print(f"wrote {len(names)} Phase-8 architecture configs ({len(CELLS)} cells) to {OUT}")
    for n in names:
        print(" ", n)
    return names


if __name__ == "__main__":
    build()
