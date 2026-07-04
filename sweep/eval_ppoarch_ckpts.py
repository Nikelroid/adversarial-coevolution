"""Evaluate the PPO arch best-checkpoints (attention vs MLP anchor) vs the gold expert.
The 10M runs hung at ~2M (curriculum deadlock under PPO+OMP), so we grade the best
checkpoints each run saved (all ~2M, a matched budget). Writes _ppoarch_eval.json."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
from stable_baselines3 import PPO
from sweep.curriculum_train import eval_full, load_any, SEED_STRONG, SCRATCH

N = int(os.environ.get("N", "1000"))
champ = PPO.load(SEED_STRONG["champion"], device="cpu")
cells = ["ppoarch_attn_s0", "ppoarch_attn_s1", "ppoarch_attn_s2", "ppoarch_attn_s3",
         "ppoarch_mlp_s0", "ppoarch_mlp_s1", "ppoarch_mlp_s2", "ppoarch_mlp_s3"]
res = []
for c in cells:
    p = os.path.join(SCRATCH, "sweep_curriculum", c, "best")
    if not os.path.exists(p + ".zip"):
        print(f"{c}: NO best.zip", flush=True); continue
    try:
        m = load_any(p, prefer="ppo")
        r = eval_full(m, "gold", champ, N)
        arch = "attn" if "attn" in c else "mlp"
        res.append({"cell": c, "arch": arch, "gold": r["win_rate"], "n": N})
        print(f"{c:18s} arch={arch:4s} vs_gold={r['win_rate']*100:.1f}%  (n={N})", flush=True)
    except Exception as e:
        print(f"{c}: EVAL FAILED: {e}", flush=True)

def iqm(xs):
    xs = sorted(xs); k = len(xs) // 4
    core = xs[k:len(xs) - k] if len(xs) >= 4 else xs
    return sum(core) / len(core) if core else float("nan")

for a in ("attn", "mlp"):
    gs = [x["gold"] for x in res if x["arch"] == a]
    if gs:
        print(f"== {a}: n={len(gs)} IQM={iqm(gs)*100:.1f}%  mean={sum(gs)/len(gs)*100:.1f}%  "
              f"range={min(gs)*100:.1f}-{max(gs)*100:.1f}", flush=True)
json.dump(res, open("sweep/curriculum/_ppoarch_eval.json", "w"), indent=2)
print("wrote sweep/curriculum/_ppoarch_eval.json", flush=True)
