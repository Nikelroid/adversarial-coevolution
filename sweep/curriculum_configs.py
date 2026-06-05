"""Generate the Phase-6 sweep matrix as one JSON config per (cell, seed).

A one-factor-at-a-time ablation around a baseline B, refined by a 5-lens design review, so
each cell isolates a single variable and the result is paper-grade rather than a blind grid.
Three questions, three axes (+ drop-in algorithm levers and within-algo sensitivity strips):
  * ALGORITHM  -- trpo (our prior winner, 0.52 vs champion @2M) vs ppo, + gamma / GAE / batch
  * REWARD     -- de-emphasise gin (optimal gins <2%): R1 balanced baseline, R0 high-gin
                  negative control, R2 knock-forward decisiveness, R4 early-knock speed penalty
  * CURRICULUM -- designed league-lite vs no-random-tail (forgetting) vs PFSP sampling rule

Seeds: 2 for large effects (algo, R0/R1/R2), 3 for the small decision-relevant cells
(curriculum variants + within-algo HP). Core cells (answer the 3 questions) are emitted FIRST
so a concurrency-capped array runs them before the sensitivity refinements.

    python sweep/curriculum_configs.py     # writes sweep/curriculum_cfgs/*.json + INDEX.txt
"""
import json
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "curriculum_cfgs")

# Reward designs. knock/gin are the env terminal-payoff bonuses for a knock/gin WIN (the loser
# always gets -deadwood/100), so changing them shifts the win:loss ratio -- a shape change that
# survives return/advantage normalization. step_penalty is a dense per-decision speed pressure.
REWARDS = {
    "R0_highgin": dict(knock=0.5,  gin=1.5,  win_scale=1.0, loss_scale=1.0, step_penalty=0.0),
    "R1_balanced":dict(knock=0.5,  gin=0.5,  win_scale=1.0, loss_scale=1.0, step_penalty=0.0),
    "R2_knockfwd":dict(knock=0.75, gin=0.75, win_scale=1.0, loss_scale=1.0, step_penalty=0.0),
    "R4_earlyknk":dict(knock=0.5,  gin=0.5,  win_scale=1.0, loss_scale=1.0, step_penalty=0.0075),
}

# Baseline B: TRPO, balanced reward, designed curriculum + recency sampling, kl 0.01, g 0.99.
BASE = dict(algo="trpo", reward="R1_balanced", curriculum="designed", sampling="recent",
            target_kl=0.01, ent_coef=0.01, clip_range=0.2, lr=3e-4, gamma=0.99,
            gae_lambda=0.95, n_steps=256, steps=12_000_000)

# (name, overrides, seeds, core?, purpose). Each overrides ONE thing from B.
CELLS = [
    # -- core: answer the 3 user questions --
    ("01_base_trpo",      {}, [0, 1], True, "BASELINE B; algo answer (vs 02)"),
    ("02_base_ppo",       dict(algo="ppo"), [0, 1], True, "ALGO ablation: PPO vs TRPO"),
    ("03_rew_R0_highgin", dict(reward="R0_highgin"), [0, 1], True,
     "REWARD control: does privileging gin help? (expect gin-rate stays <2%)"),
    ("04_rew_R2_knockfwd",dict(reward="R2_knockfwd"), [0, 1], True,
     "REWARD: decisiveness (higher win bonus, gin not privileged)"),
    ("05_rew_R4_earlyknk", dict(reward="R4_earlyknk"), [0, 1, 2], True,
     "REWARD mechanism: reward EARLY low-deadwood knocking (speed penalty); report game length"),
    ("06_cur_noRandTail", dict(curriculum="no_random_tail"), [0, 1, 2], True,
     "CURRICULUM: drop random after stage 1 (catastrophic-forgetting test)"),
    ("07_cur_pfsp",       dict(sampling="pfsp"), [0, 1, 2], True,
     "CURRICULUM: PFSP loss-weighted pool sampling vs recency-uniform (the sampling RULE)"),
    # -- drop-in algorithm levers + within-algo sensitivity strips --
    ("08_trpo_gamma_high",dict(gamma=0.997), [0, 1, 2], False,
     "ALGO drop-in: longer credit reach for terminal-only reward (0.997^40=0.89 vs 0.67)"),
    ("09_trpo_kl_small",  dict(target_kl=0.005), [0, 1, 2], False,
     "TRPO exploration strip: tighter trust region"),
    ("10_trpo_kl_large",  dict(target_kl=0.02), [0, 1, 2], False,
     "TRPO exploration strip: looser trust region"),
    ("11_trpo_gae_batch", dict(gae_lambda=0.98, n_steps=512), [0, 1], False,
     "ALGO drop-in: lower-variance advantage + bigger on-policy batch (self-play stability)"),
    ("12_ppo_ent_hi",     dict(algo="ppo", ent_coef=0.03), [0, 1], False,
     "PPO exploration strip (separate from TRPO; not a joint 'exploit_factor' axis)"),
]


def build():
    os.makedirs(OUT, exist_ok=True)
    for f in os.listdir(OUT):
        if f.endswith(".json") or f == "INDEX.txt":
            os.remove(os.path.join(OUT, f))
    names = []
    # core cells first, then refinements -- so a %-capped array runs the decisive ones first
    for cell_name, ov, seeds, _core, purpose in sorted(CELLS, key=lambda c: (not c[3], c[0])):
        for sd in seeds:
            c = dict(BASE); c.update(ov)
            rew = REWARDS[c.pop("reward")]
            name = f"{cell_name}_s{sd}"
            cfg = dict(name=name, algo=c["algo"], curriculum=c["curriculum"],
                       sampling=c["sampling"], knock=rew["knock"], gin=rew["gin"],
                       win_scale=rew["win_scale"], loss_scale=rew["loss_scale"],
                       step_penalty=rew["step_penalty"], target_kl=c["target_kl"],
                       ent_coef=c["ent_coef"], clip_range=c["clip_range"], lr=c["lr"],
                       gamma=c["gamma"], gae_lambda=c["gae_lambda"], n_steps=c["n_steps"],
                       seed=sd, steps=c["steps"], cell=cell_name, purpose=purpose)
            json.dump(cfg, open(os.path.join(OUT, f"{name}.json"), "w"), indent=2)
            names.append(name)
    with open(os.path.join(OUT, "INDEX.txt"), "w") as f:
        f.write("\n".join(names) + "\n")
    print(f"wrote {len(names)} run configs ({len(CELLS)} cells) to {OUT}")
    for n in names:
        print(" ", n)
    return names


if __name__ == "__main__":
    build()
