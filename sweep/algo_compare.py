"""Is plain PPO suboptimal for our masked Gin Rummy task? Compare PPO vs TRPO under
IDENTICAL conditions: both reuse MaskedGinRummyPolicy (obs-based action masking), the
same net/extractor, the same self-play-vs-champion task and the same eval. Isolates the
RL ALGORITHM as the only variable.

Scope note (see the report): GRPO and DPO are LLM-alignment methods (group-relative
reward-model / preference fine-tuning) that do not map onto per-step env-interaction game
RL, so they are analyzed, not run. MaskablePPO (canonical masking) and RecurrentPPO (LSTM
memory for the imperfect-information state) are relevant but need a masked policy + masked
eval; flagged as the next experiment.

Env: ALGO (ppo|trpo), TAG, RL_STEPS, NUM_ENV, N_STEPS, EVAL_GAMES, SEED.
"""
import os, sys, json, time, functools
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
try:
    import cv2  # noqa: F401  (parent import before fork)
except Exception:
    pass
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.torch_layers import CombinedExtractor

from gym_wrapper import GinRummySB3Wrapper
from agents.ppo_agent import PPOAgent
import ppo_train  # noqa: F401  registers MaskedGinRummyPolicy
from sweep.llm_dagger_one import evaluate, KNOCK, GIN   # reuse the eval harness


class FiniteMaskedPolicy(ppo_train.MaskedGinRummyPolicy):
    """Mask illegal actions with a large FINITE negative instead of -inf. PPO tolerates
    -inf, but TRPO's KL / conjugate-gradient math produces NaNs on infinities. -1e8 gives
    ~0 probability while keeping the logits finite, so the same masked policy works for
    BOTH algorithms (fair comparison)."""
    def _apply_action_mask(self, logits, action_mask):
        if action_mask is None:
            return logits
        mask = action_mask.to(dtype=th.bool, device=logits.device)
        return th.where(mask, logits, th.full_like(logits, -1e8))


def build_model(algo, env, n_steps):
    pkw = dict(features_extractor_class=CombinedExtractor,
               net_arch=dict(pi=[256, 128], vf=[256, 128]),
               activation_fn=th.nn.Tanh, ortho_init=True)
    if algo == "ppo":
        return PPO(FiniteMaskedPolicy, env, n_steps=n_steps, device="cpu",
                   verbose=0, policy_kwargs=pkw)
    if algo == "trpo":
        from sb3_contrib import TRPO
        return TRPO(FiniteMaskedPolicy, env, n_steps=n_steps, device="cpu",
                    verbose=0, policy_kwargs=pkw)
    raise ValueError(f"unknown ALGO={algo}")


def main():
    algo = os.environ.get("ALGO", "ppo")
    tag = os.environ.get("TAG", f"{algo}_s{os.environ.get('SEED', 0)}")
    rl_steps = int(os.environ.get("RL_STEPS", 2_000_000))
    num_env = int(os.environ.get("NUM_ENV", 64))
    n_steps = int(os.environ.get("N_STEPS", 256))
    eval_games = int(os.environ.get("EVAL_GAMES", 400))
    champ_path = os.environ.get("CHAMP_MODEL", "game/model/ppo_gin_rummy_selfplay.zip")
    seed = int(os.environ.get("SEED", 0))
    th.manual_seed(seed); np.random.seed(seed)

    print(f"=== algo-compare algo={algo} tag={tag} rl_steps={rl_steps} ===", flush=True)
    champ = PPO.load(champ_path, device="cpu")

    def _mk(rank):
        def _t():
            e = GinRummySB3Wrapper(
                opponent_policy=functools.partial(PPOAgent, model=champ),
                randomize_position=True, turns_limit=200, curriculum_manager=None,
                rank=rank, knock_reward=KNOCK, gin_reward=GIN)
            e.reset(seed=seed + rank)
            return Monitor(e)
        return _t
    env = SubprocVecEnv([_mk(i) for i in range(num_env)], start_method="fork")

    model = build_model(algo, env, n_steps)
    t0 = time.time()
    model.learn(total_timesteps=rl_steps)
    secs = time.time() - t0
    env.close()

    res = {k: evaluate(model, k, champ, eval_games) for k in ("random", "champion", "gold")}
    out_dir = os.path.join(PROJECT_ROOT, "sweep", "algo")
    os.makedirs(out_dir, exist_ok=True)
    result = dict(algo=algo, tag=tag, seed=seed, rl_steps=rl_steps, train_seconds=secs,
                  vs_random=res["random"], vs_champion=res["champion"], vs_gold=res["gold"])
    json.dump(result, open(os.path.join(out_dir, f"algo_{tag}.json"), "w"), indent=2)
    print(f"[algo {tag}] vs_random={res['random']['win_rate']:.3f} "
          f"vs_champion={res['champion']['win_rate']:.3f} "
          f"vs_gold={res['gold']['win_rate']:.3f} (in {secs:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
