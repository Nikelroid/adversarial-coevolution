"""Post-hoc upload of finished sweep runs to Weights & Biases.

Use this when you ran the sweep with WANDB_DISABLED=true (as we did the
first time) and want the results visible in the W&B UI.

Each `sweep/results/run_*.json` becomes a separate W&B run inside the
project specified below.  We log the full config and a single final metric
row (we don't have intra-training metrics from these JSONs).

Usage::

    wandb login                 # one time
    PYTHONNOUSERSITE=1 /scratch1/kelidari/envs/coev/bin/python sweep/to_wandb.py

"""
from __future__ import annotations

import glob
import json
import os

PROJECT = "Adversarial-CoEvolution"
ENTITY = "VLAvengers"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def main() -> None:
    import wandb

    paths = sorted(glob.glob(os.path.join(RESULTS_DIR, "run_*.json")))
    if not paths:
        print("no results to upload")
        return

    for p in paths:
        with open(p) as f:
            r = json.load(f)
        cfg = r["config"]
        run_name = (
            f"sweep_combo{r['combo']}"
            f"_lr{cfg['learning_rate']:.0e}"
            f"_ent{cfg['ent_coef']}"
            f"_ns{cfg['n_steps']}"
            f"_ep{cfg['n_epochs']}"
        )
        run = wandb.init(
            project=PROJECT,
            entity=ENTITY,
            name=run_name,
            group="phase1-hp-sweep",
            tags=["phase1", "vs-random", f"combo{r['combo']}"],
            config=cfg,
            reinit=True,
        )
        wandb.log(
            {
                "eval/win_rate": r["win_rate"],
                "eval/loss_rate": r["loss_rate"],
                "eval/draw_rate": r["draw_rate"],
                "eval/mean_reward": r["mean_reward"],
                "eval/std_reward": r["std_reward"],
                "eval/mean_length": r["mean_length"],
                "train/wall_seconds": r["train_seconds"],
            }
        )
        wandb.finish()
        print(f"uploaded {os.path.basename(p)} as run {run_name}")


if __name__ == "__main__":
    main()
