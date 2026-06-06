"""The universal training pipeline.

Point it at any PettingZoo AEC game (or your own AEC env) via ``CoevConfig.env_fn`` and call
``train(cfg)``. It trains a masked PPO or TRPO agent through an opponent curriculum, keeps the best
checkpoint, evaluates against random play (and an optional benchmark agent you supply), and writes a
``result.json``. Nothing in here is specific to any one game.
"""
import json
import os
import random
import time

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import CombinedExtractor, FlattenExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from coev.agents import PolicyAgent, RandomAgent
from coev.curriculum import Curriculum, default_stages, load_any, seed_pool
from coev.env import MaskedCoevEnv
from coev.policy import MaskedPolicy


# --------------------------------------------------------------------- evaluation
def evaluate(model, env_fn, env_kwargs, opponent_fn, n, seed0=10_000):
    """Win-rate of ``model`` vs an opponent, seat-rotated for fairness, on a fresh raw env.
    ``opponent_fn(env)`` returns an agent with ``set_player`` / ``do_action``."""
    env = env_fn(**(env_kwargs or {}))
    hero = PolicyAgent(env, model)
    opp = opponent_fn(env)
    win = loss = draw = 0
    for g in range(n):
        env.reset(seed=seed0 + g)
        agents = list(env.agents)
        hero_seat = agents[g % len(agents)]
        tot = {a: 0.0 for a in agents}
        for ag in env.agent_iter():
            _obs, rew, term, trunc, _info = env.last()
            tot[ag] = tot.get(ag, 0.0) + rew
            if term or trunc:
                env.step(None); continue
            actor = hero if ag == hero_seat else opp
            actor.set_player(ag)
            env.step(actor.do_action())
        r = tot.get(hero_seat, 0.0)
        win += r > 0; loss += r < 0; draw += r == 0
    env.close()
    return dict(win_rate=win / n, loss_rate=loss / n, draw_rate=draw / n, n=n)


# --------------------------------------------------------------------- model builder
def build_model(env, cfg):
    # Dict observations (the usual case, they carry the action_mask) need CombinedExtractor;
    # plain Box observations use FlattenExtractor. SB3's base policy defaults to Flatten, so we
    # pick it explicitly to stay universal.
    is_dict = isinstance(env.observation_space, gym.spaces.Dict)
    pkw = dict(net_arch=list(cfg.net_arch), activation_fn=th.nn.Tanh, ortho_init=True,
               features_extractor_class=CombinedExtractor if is_dict else FlattenExtractor)
    common = dict(env=env, n_steps=cfg.n_steps, batch_size=cfg.batch_size, device="cpu",
                  verbose=0, policy_kwargs=pkw, seed=cfg.seed, gamma=cfg.gamma,
                  gae_lambda=cfg.gae_lambda, learning_rate=cfg.lr, normalize_advantage=True)
    if cfg.algo == "ppo":
        return PPO(MaskedPolicy, n_epochs=cfg.n_epochs, clip_range=cfg.clip_range,
                   ent_coef=cfg.ent_coef, **common)
    if cfg.algo == "trpo":
        from sb3_contrib import TRPO
        return TRPO(MaskedPolicy, target_kl=cfg.target_kl, **common)
    raise ValueError(f"unknown algo {cfg.algo!r} (use 'ppo' or 'trpo')")


# --------------------------------------------------------------------- callback
class _Callback(BaseCallback):
    """Grows the opponent pool, records a learning curve, and keeps the best checkpoint."""

    def __init__(self, cfg, curriculum, eval_opps, best_path, curve):
        super().__init__()
        self.cfg = cfg
        self.cm = curriculum
        self.eval_opps = eval_opps          # {name: opponent_fn}
        self.key = list(eval_opps)[-1]      # keep-best on the hardest opponent (last one)
        self.best_path = best_path
        self.curve = curve
        self.best = -1.0
        self.best_step = 0
        self._last_update = 0
        self._last_eval = 0

    def _on_step(self) -> bool:
        s = self.num_timesteps
        if s - self._last_update >= 10_000:
            self.cm.update_total_steps(s); self._last_update = s
        if self.cm.should_save_checkpoint(s, self.cfg.save_freq):
            self.cm.save_checkpoint(self.model, s)
        if self.cfg.eval_every and s - self._last_eval >= self.cfg.eval_every:
            self._last_eval = s
            snap = {"step": s}
            for name, opp in self.eval_opps.items():
                snap[name] = evaluate(self.model, self.cfg.env_fn, self.cfg.env_kwargs, opp,
                                      self.cfg.eval_games)["win_rate"]
            self.curve.append(snap)
            improved = self.cfg.keep_best and snap[self.key] > self.best
            if improved:
                self.best = snap[self.key]; self.best_step = s
                self.model.save(self.best_path)
            print(f"[coev] step {s:,} " + " ".join(f"{k}={snap[k]:.3f}" for k in self.eval_opps)
                  + ("  <- new best" if improved else ""), flush=True)
        return True


# --------------------------------------------------------------------- entrypoint
def train(cfg):
    th.manual_seed(cfg.seed); np.random.seed(cfg.seed); random.seed(cfg.seed)
    run_dir = os.path.join(cfg.out_dir, cfg.name)
    pool_dir = os.path.join(run_dir, "pool")
    os.makedirs(run_dir, exist_ok=True)
    stages = cfg.stages or default_stages()
    seed_pool(pool_dir, list(cfg.seed_models))
    print(f"[coev] training '{cfg.name}' on '{cfg.env_id}' with {cfg.algo.upper()} "
          f"for {cfg.total_steps:,} steps", flush=True)

    def mk(rank):
        def _thunk():
            cur = Curriculum(pool_dir, stages, cfg.total_steps)
            e = MaskedCoevEnv(cfg.env_fn, curriculum=cur, env_kwargs=cfg.env_kwargs,
                              turns_limit=cfg.turns_limit, reward_transform=cfg.reward_transform,
                              rank=rank)
            e.reset(seed=cfg.seed + rank)
            return Monitor(e)
        return _thunk

    venv = SubprocVecEnv([mk(i) for i in range(cfg.num_envs)], start_method="fork")
    env = (VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=cfg.gamma)
           if cfg.normalize_reward else venv)

    if cfg.init_model:
        cls = PPO if cfg.algo == "ppo" else __import__("sb3_contrib", fromlist=["TRPO"]).TRPO
        model = cls.load(cfg.init_model.replace(".zip", ""), env=env, device="cpu")
        print(f"[coev] warm-started from {cfg.init_model}", flush=True)
    else:
        model = build_model(env, cfg)

    main_cur = Curriculum(pool_dir, stages, cfg.total_steps)
    eval_opps = {"random": lambda e: RandomAgent(e)}
    if cfg.benchmark_agent is not None:
        eval_opps[cfg.benchmark_name] = cfg.benchmark_agent
    curve = []
    best_path = os.path.join(run_dir, "best")
    cb = _Callback(cfg, main_cur, eval_opps, best_path, curve)

    t0 = time.time()
    model.learn(total_timesteps=cfg.total_steps, callback=cb, progress_bar=False)
    secs = time.time() - t0
    env.close()

    final_path = os.path.join(run_dir, "final")
    model.save(final_path)
    best_src = best_path if (cfg.keep_best and os.path.exists(best_path + ".zip")) else final_path
    best_model = load_any(best_src)
    results = {name: evaluate(best_model, cfg.env_fn, cfg.env_kwargs, opp, cfg.final_eval_games)
               for name, opp in eval_opps.items()}
    out = dict(name=cfg.name, env_id=cfg.env_id, algo=cfg.algo, steps=cfg.total_steps,
               seed=cfg.seed, train_seconds=secs, best_step=cb.best_step,
               best_model=best_src + ".zip", final_model=final_path + ".zip",
               curve=curve, eval=results)
    json.dump(out, open(os.path.join(run_dir, "result.json"), "w"), indent=2)
    print(f"[coev] done '{cfg.name}': "
          + " ".join(f"{k}={results[k]['win_rate']:.3f}" for k in results)
          + f"  ({secs:.0f}s) -> {run_dir}/result.json", flush=True)
    return out
