"""Phase-7 = "break the champion ceiling". Generate the next series of runs as JSON configs.

Phase 6 plateaued at ~30-33% vs gold, and the saved 'final' model often drifted below its own
mid-training peak. Phase 7 fixes and pushes on that:
  * KEEP-THE-BEST: every 1M steps the harness evaluates and saves the best checkpoint as
    best.zip; the shipped model is the peak, not the drifted end. (Built into curriculum_train.)
  * WARM-START from Phase-6's strongest agent (the early-knock champion) instead of from scratch.
  * STRONGER OPPONENTS: the strong seed tier now includes the two best Phase-6 agents (~33% vs
    gold), so the agent practises against genuinely strong play.
  * NEW LEVERS: a dense DEADWOOD-reduction reward that coaches the optimal 'knock low' style
    directly, the early-knock speed penalty that already worked, PFSP opponent sampling, and a
    'warm' curriculum that skips the wasteful random-only phase.

    python sweep/phase7_configs.py     # writes sweep/phase7_cfgs/*.json + INDEX.txt
"""
import json
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phase7_cfgs")
WARM = "game/model/gin_curriculum_champion.zip"   # the best Phase-6 agent (early-knock, ~33% vs gold)

# Baseline B7: warm-start the champion, keep its winning R4 reward (early-knock), a warm
# curriculum that faces strong opponents from the start, keep-the-best, 1M-step evals.
BASE = dict(algo="trpo", init_model=WARM, curriculum="warm", sampling="recent",
            strong_from=0.10, knock=0.5, gin=0.5, win_scale=1.0, loss_scale=1.0,
            step_penalty=0.0075, deadwood_coef=0.0, target_kl=0.01, ent_coef=0.01,
            clip_range=0.2, lr=3e-4, gamma=0.99, gae_lambda=0.95, n_steps=256,
            steps=16_000_000, eval_every=1_000_000, ckpt_eval_games=400, final_eval_games=1000,
            save_freq=1_500_000)

# (name, overrides, seeds, purpose)
CELLS = [
    ("p7_01_warm_long", {}, [0, 1],
     "Warm-start the champion + R4 reward + keep-best, train 16M more. Does squeezing the proven recipe help?"),
    ("p7_02_warm_deadwood", dict(deadwood_coef=0.02), [0, 1],
     "+ dense deadwood-reduction reward: coach the optimal 'knock low' style directly."),
    ("p7_03_warm_deadwood_hi", dict(deadwood_coef=0.05), [0],
     "Stronger dense deadwood reward."),
    ("p7_04_warm_pfsp", dict(sampling="pfsp"), [0, 1],
     "+ PFSP opponent sampling: practise most against whoever is currently beating you."),
    ("p7_05_scratch_deadwood", dict(init_model="", curriculum="designed", strong_from=0.70,
                                    deadwood_coef=0.02, steps=14_000_000), [0, 1],
     "From scratch with the dense deadwood reward: isolates whether the new reward helps on its own."),
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
    print(f"wrote {len(names)} Phase-7 run configs ({len(CELLS)} cells) to {OUT}")
    for n in names:
        print(" ", n)
    return names


if __name__ == "__main__":
    build()
