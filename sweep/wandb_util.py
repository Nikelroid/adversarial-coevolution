"""Tiny guarded Weights & Biases helper for the standalone (non-SB3) experiment scripts
(Leduc tabular-Q, Leduc NFSP, ISMCTS). Mirrors the convention in curriculum_train.py: every call
is wrapped so a missing key or a wandb hiccup can NEVER interfere with or kill the experiment.

Honors WANDB_PROJECT / WANDB_ENTITY / WANDB_GROUP / WANDB_DISABLED / WANDB_API_KEY from the env
(the SLURM wrappers export these, extracting the key from ~/.bashrc WANDB_TOKEN when needed).
"""
import os

try:
    import wandb
    _WANDB = True
except Exception:
    _WANDB = False

_run = None


def wandb_on():
    if not _WANDB or os.environ.get("WANDB_DISABLED", "").lower() in ("1", "true", "yes"):
        return False
    return bool(os.environ.get("WANDB_API_KEY")) or os.path.exists(os.path.expanduser("~/.netrc"))


def wandb_init(name, group, config=None, tags=None):
    """Best-effort init. Returns True if a run started. Group defaults via env WANDB_GROUP."""
    global _run
    if not wandb_on():
        return False
    try:
        _run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "Adversarial-CoEvolution"),
            entity=os.environ.get("WANDB_ENTITY", "VLAvengers"),
            name=name,
            group=os.environ.get("WANDB_GROUP", group),
            tags=tags or [],
            config=config or {},
            reinit=True,
        )
        print("[wandb] logging enabled", flush=True)
        return True
    except Exception as e:
        print(f"[wandb] init failed, continuing without it: {e}", flush=True)
        _run = None
        return False


def wandb_log(d, step=None):
    if _run is None:
        return
    try:
        wandb.log(d, step=step)
    except Exception:
        pass


def wandb_finish(summary=None):
    global _run
    if _run is None:
        return
    try:
        if summary:
            for k, v in summary.items():
                wandb.run.summary[k] = v
        wandb.finish()
    except Exception:
        pass
    _run = None
