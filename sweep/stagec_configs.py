"""Phase-8 Stage-C: is the architecture finding RECIPE-ROBUST, or an artifact of the one winning
recipe? We take the from-scratch architecture BASE (same fixed budget as the Stage-A arch sweep) and
cross a small set of architectures with ALTERNATIVE training recipes (a different algorithm and a
different opponent-sampling rule). If the architecture ranking (anchor vs asym vs wide) holds across
recipes, the Stage-A conclusion is not a single-config fluke -- the standard "is this just one lucky
config?" reviewer worry.

Every cell still trains FROM SCRATCH (so the configured network is the one trained), graded against
the same fixed expert yardstick. Consumed by slurm/curriculum.slurm with CFG_DIR=stagec_cfgs.

    python sweep/stagec_configs.py     # writes sweep/stagec_cfgs/*.json + INDEX.txt
"""
import json
import os

from sweep.arch_configs import BASE  # the from-scratch winner-recipe base (TRPO, recent sampling)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stagec_cfgs")

# Architectures to carry across recipes (the MLP variants already ranked in Stage-A; encoders can be
# added on resume once Stage-A names the best one).
ARCHS = {
    "mlpdef": {},                                                  # anchor MLP [256,128]
    "asym":   dict(net_arch=dict(pi=[128, 64], vf=[512, 256])),    # Stage-A leader
    "wide":   dict(net_arch=dict(pi=[512, 256], vf=[512, 256])),   # wide MLP
}

# Alternative recipes (each a delta on BASE). "trpo_recent" is the Stage-A recipe itself = the bridge
# that lets us check the ranking is consistent; "ppo" swaps the algorithm; "pfsp" swaps the sampler.
RECIPES = {
    "ppo":  dict(algo="ppo"),
    "pfsp": dict(sampling="pfsp"),
}

SEEDS = [0, 1]


def build():
    os.makedirs(OUT, exist_ok=True)
    for f in os.listdir(OUT):
        if f.endswith(".json") or f == "INDEX.txt":
            os.remove(os.path.join(OUT, f))
    names = []
    for rname, rov in RECIPES.items():
        for aname, aov in ARCHS.items():
            for sd in SEEDS:
                c = dict(BASE)
                c.update(rov)
                c.update(aov)
                name = f"sc_{rname}_{aname}_s{sd}"
                cfg = dict(name=name, seed=sd,
                           purpose=f"Stage-C: arch={aname} under recipe={rname} (recipe-robustness).",
                           **c)
                json.dump(cfg, open(os.path.join(OUT, f"{name}.json"), "w"), indent=2)
                names.append(name)
    with open(os.path.join(OUT, "INDEX.txt"), "w") as f:
        f.write("\n".join(names) + "\n")
    print(f"wrote {len(names)} Stage-C configs "
          f"({len(RECIPES)} recipes x {len(ARCHS)} archs x {len(SEEDS)} seeds) to {OUT}")
    for n in names:
        print(" ", n)
    return names


if __name__ == "__main__":
    build()
