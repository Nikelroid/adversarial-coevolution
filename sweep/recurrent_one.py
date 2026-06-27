"""Phase-8 recurrence side-study (ISOLATED from the main sweep).

Trains a MASKED PPO-LSTM (RecurrentPPO + MaskedRecurrentPolicy) on Gin Rummy against the frozen
champion, OR a matched PPO-MLP control (ARCH=mlp), then grades it vs gold / champion / random with a
recurrent-aware game loop (LSTM state threaded across each game, reset per game). Result JSON is
written to sweep/curriculum/<name>.json so arch_collect picks it up (arch='lstm' | 'mlp_ctrl').

Because RecurrentPPO is PPO-only, the PPO-MLP control isolates the recurrence effect from the
PPO-vs-TRPO change. Self-play here is vs a FIXED frozen champion (no pool), so recurrent opponents
are never needed -- keeps this fully isolated from the running curriculum sweep.

Env: ARCH (lstm|mlp), SEED, STEPS, NUM_ENV, N_STEPS, EVAL_GAMES, CHAMP_MODEL, NAME, WANDB_* .
"""
import functools
import json
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
try:
    import cv2  # noqa: F401  (parent import before fork, like the other entrypoints)
except Exception:
    pass
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import CombinedExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv

from gym_wrapper import GinRummySB3Wrapper
from agents.ppo_agent import PPOAgent
from agents.random_agent import RandomAgent
from agents.gold_standard_agent import GoldStandardAgent
import ppo_train  # noqa: F401  registers MaskedGinRummyPolicy
from sweep.algo_compare import FiniteMaskedPolicy
from sweep.llm_dagger_one import KNOCK, GIN

try:
    import wandb
    _WANDB = True
except Exception:
    _WANDB = False


def _wandb_on():
    if not _WANDB or os.environ.get("WANDB_DISABLED", "").lower() in ("1", "true", "yes"):
        return False
    return bool(os.environ.get("WANDB_API_KEY"))


def make_train_env(champ, num_env, seed):
    def _mk(rank):
        def _t():
            e = GinRummySB3Wrapper(
                opponent_policy=functools.partial(PPOAgent, model=champ),
                randomize_position=True, turns_limit=200, curriculum_manager=None,
                rank=rank, knock_reward=KNOCK, gin_reward=GIN)
            e.reset(seed=seed + rank)
            return Monitor(e)
        return _t
    return SubprocVecEnv([_mk(i) for i in range(num_env)], start_method="fork")


def build(arch, env, seed, n_steps):
    if arch == "lstm":
        from sb3_contrib import RecurrentPPO
        from sweep.recurrent_policy import MaskedRecurrentPolicy
        pkw = dict(net_arch=dict(pi=[128], vf=[128]), activation_fn=th.nn.Tanh,
                   lstm_hidden_size=128, n_lstm_layers=1, enable_critic_lstm=True)
        return RecurrentPPO(MaskedRecurrentPolicy, env, n_steps=n_steps, batch_size=256, n_epochs=4,
                            learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, ent_coef=0.01,
                            clip_range=0.2, device="cpu", verbose=0, seed=seed, policy_kwargs=pkw)
    pkw = dict(features_extractor_class=CombinedExtractor,  # dict obs needs CombinedExtractor
               net_arch=dict(pi=[256, 128], vf=[256, 128]), activation_fn=th.nn.Tanh, ortho_init=True)
    return PPO(FiniteMaskedPolicy, env, n_steps=n_steps, batch_size=256, n_epochs=4,
               learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, ent_coef=0.01, clip_range=0.2,
               device="cpu", verbose=0, seed=seed, policy_kwargs=pkw)


def eval_vs(model, arch, opp_policy, n, seed0):
    """Recurrent-aware eval via the wrapper (opponent is internal). LSTM state threaded per game.
    Terminal-only payoff under (knock=0.5, gin=1.5): tot>0 win, tot>=1.4 gin."""
    env = GinRummySB3Wrapper(opponent_policy=opp_policy, randomize_position=True, turns_limit=200,
                             curriculum_manager=None, knock_reward=0.5, gin_reward=1.5)
    win = gin = loss = 0
    lengths = []
    for g in range(n):
        r = env.reset(seed=seed0 + g)
        obs = r[0] if isinstance(r, tuple) else r
        lstm, ep, done, tot, steps = None, True, False, 0.0, 0
        while not done:
            if arch == "lstm":
                a, lstm = model.predict(obs, state=lstm, episode_start=np.array([ep]),
                                        deterministic=True)
            else:
                a, _ = model.predict(obs, deterministic=True)
            obs, rw, term, trunc, _ = env.step(int(np.asarray(a).reshape(-1)[0]))
            tot += float(rw); ep = False; steps += 1; done = bool(term or trunc)
        lengths.append(steps)
        if tot > 0:
            win += 1
            gin += int(tot >= 1.4)
        elif tot < 0:
            loss += 1
    env.close()
    return dict(win_rate=win / n, gin_rate=gin / n, loss_rate=loss / n,
                mean_len=float(np.mean(lengths)), n=n)


def main():
    arch = os.environ.get("ARCH", "lstm")
    seed = int(os.environ.get("SEED", 0))
    steps = int(os.environ.get("STEPS", 4_000_000))
    num_env = int(os.environ.get("NUM_ENV", 8))
    n_steps = int(os.environ.get("N_STEPS", 128))
    eval_games = int(os.environ.get("EVAL_GAMES", 600))
    # 'rec_' prefix keeps these OUT of the arch_collect leaderboard: this study uses a different
    # (wrapper) eval harness than the arch sweep's eval_full, so the two are not directly comparable.
    # The LSTM vs its matched PPO-MLP control ARE comparable (same eval here).
    name = os.environ.get("NAME", f"rec_{'lstm' if arch == 'lstm' else 'mlp_ctrl'}_s{seed}")
    champ_path = os.environ.get("CHAMP_MODEL", "game/model/gin_curriculum_champion.zip")
    th.manual_seed(seed); np.random.seed(seed)
    print(f"=== recurrent-one arch={arch} name={name} steps={steps:,} seed={seed} "
          f"champ={champ_path} ===", flush=True)

    champ = PPO.load(champ_path[:-4] if champ_path.endswith(".zip") else champ_path, device="cpu")

    if _wandb_on():
        try:
            wandb.init(project=os.environ.get("WANDB_PROJECT", "Adversarial-CoEvolution"),
                       entity=os.environ.get("WANDB_ENTITY", "VLAvengers"), name=name,
                       group=os.environ.get("WANDB_GROUP", "phase8-recurrent"),
                       tags=[f"arch:{arch}", f"seed{seed}", "recurrence-sidestudy"],
                       config=dict(arch=arch, seed=seed, steps=steps), reinit=True)
            print("[wandb] logging enabled", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[wandb] init failed, continuing: {e}", flush=True)

    env = make_train_env(champ, num_env, seed)
    model = build(arch, env, seed, n_steps)
    t0 = time.time()
    model.learn(total_timesteps=steps, progress_bar=False)
    secs = time.time() - t0
    env.close()

    out_dir = os.path.join(PROJECT_ROOT, "sweep", "curriculum")
    os.makedirs(out_dir, exist_ok=True)
    scratch = os.environ.get("SCRATCH", "/scratch1/kelidari/advcoev_store")
    mp = os.path.join(scratch, "sweep_recurrent", name)
    os.makedirs(os.path.dirname(mp), exist_ok=True)
    model.save(mp)

    res = {k: eval_vs(model, arch, opp, eval_games, 10_000)
           for k, opp in (("gold", GoldStandardAgent), ("random", RandomAgent),
                          ("champion", functools.partial(PPOAgent, model=champ)))}
    result = dict(name=name, algo=("recurrent_ppo" if arch == "lstm" else "ppo"),
                  arch=("lstm" if arch == "lstm" else "mlp_ctrl"),
                  activation="tanh", net_arch=([128] if arch == "lstm" else [256, 128]),
                  seed=seed, steps=steps, train_seconds=secs, model_path=mp + ".zip",
                  best_step=steps, vs_gold=res["gold"], vs_champion=res["champion"],
                  vs_random=res["random"])
    out = os.path.join(out_dir, f"{name}.json")
    tmp = out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(result, f, indent=2)
    os.replace(tmp, out)
    print(f"[done] {name} vs_gold={res['gold']['win_rate']:.3f} "
          f"vs_champion={res['champion']['win_rate']:.3f} vs_random={res['random']['win_rate']:.3f} "
          f"in {secs:.0f}s -> {out}", flush=True)
    if _wandb_on() and wandb.run is not None:
        wandb.log({"final/win_vs_gold": res["gold"]["win_rate"],
                   "final/win_vs_champion": res["champion"]["win_rate"],
                   "final/win_vs_random": res["random"]["win_rate"],
                   "final/gin_vs_gold": res["gold"]["gin_rate"]})
        wandb.finish()


if __name__ == "__main__":
    main()
