"""Phase-6 sweep: best ALGORITHM x best REWARD x best CURRICULUM for Gin Rummy.

One run = one config. An RL algorithm (PPO or TRPO) is trained through a multi-STAGE
opponent curriculum:  random  ->  a growing POOL of the run's own past selves and a
curated MID tier of pre-divergence checkpoints  ->  self-play  ->  a non-evicting STRONG
tier (champion + LLM-derived models) sampled only late.  Under a chosen REWARD design (the
env's knock/gin terminal-payoff ratio, an asymmetric win/loss scale, and an optional dense
per-decision speed penalty).  This answers the user's three questions head-on:
  1. best algorithm  -- PPO vs TRPO (our prior study: TRPO > PPO under identical masking)
  2. best reward     -- the OPTIMAL player gins <2% of the time (gold_bench), so a reward
                        that privileges gin is miscalibrated; we test de-emphasising it and
                        rewarding decisive, early, low-deadwood knocking instead.
  3. best curriculum -- a league-lite schedule + the pool-SAMPLING rule (recency vs PFSP).

REDLINE: the gold-standard agent is BENCHMARK-ONLY.  It never trains the RL and is never a
training opponent; it informs reward DESIGN only as a conceptual insight, and is the eval
yardstick.  Opponent diversity comes from random + the model pool (past selves + seeds),
never from gold.

Design verified by a 5-lens review; fixes folded in: generic PPO/TRPO opponent loader,
per-cell isolated pool dirs, VecNormalize reward normalization (de-confounds the reward
study), mid/strong seed tiers (don't crush a fresh agent), PFSP sampling, a <10s startup
self-test, and mean-game-length as the R4 mechanism check.

Config: a JSON file path in CFG (see sweep/curriculum_configs.py), or env knobs.  Models +
the per-run opponent pool live on scratch (isolated per run).  Results (full learning curve
+ best-of-curve + final eval vs random/champion/gold) -> sweep/curriculum/<name>.json.
"""
from __future__ import annotations

import glob
import json
import os
import random
import shutil
import sys
import tempfile
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

try:
    import cv2  # noqa: F401  (import in parent before any fork; matches other entrypoints)
except Exception:
    pass
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import CombinedExtractor

from pettingzoo.classic import gin_rummy_v4
from gym_wrapper import GinRummySB3Wrapper
from agents.ppo_agent import PPOAgent
from agents.random_agent import RandomAgent
import ppo_train  # noqa: F401  registers MaskedGinRummyPolicy (needed to unpickle checkpoints)
from sweep.algo_compare import FiniteMaskedPolicy          # masked policy that survives TRPO
from sweep.arch_extractors import (Conv1DCardExtractor, DeepSetsCardExtractor,  # Phase-8 arch sweep
                                   SetAttentionExtractor)
from sweep.llm_dagger_one import evaluate, KNOCK, GIN, ENVK  # seat-swapped eval vs random/champ/gold

try:
    import wandb
    _WANDB = True
except Exception:
    _WANDB = False


def _wandb_on():
    """W&B is on when a key/.netrc is present and it is not explicitly disabled. Never required
    -- every wandb call is guarded so logging can never interfere with or kill training."""
    if not _WANDB or os.environ.get("WANDB_DISABLED", "").lower() in ("1", "true", "yes"):
        return False
    return bool(os.environ.get("WANDB_API_KEY")) or os.path.exists(
        os.path.expanduser("~/.netrc"))


SCRATCH = os.environ.get("ADVCOEV_STORE", "/scratch1/kelidari/advcoev_store")
POOLPLAY = os.path.join(SCRATCH, "sweep_poolplay_pool")
HERO = os.path.join(PROJECT_ROOT, "game", "model")

# Curated, fast (checkpoint-based) opponents that seed every run's pool.  MID = a difficulty
# ramp of PRE-divergence poolplay milestones + the winrate anchor (the 10.5M/12M post-
# divergence checkpoints are excluded -- they poison training).  STRONG = the champion plus
# the LLM-derived models (dagbase = LLM-DAgger baseline, llm = trained-vs-LLM), which give the
# curriculum its 'play vs LLM' flavour cheaply (no live LLM in the loop -- too slow, phase-2
# finding).  Strong opponents are sampled ONLY late, so a fresh agent is not crushed early.
SEED_MID = {
    "winrate": os.path.join(HERO, "ppo_gin_rummy_winrate.zip"),
    "pool1_5M": os.path.join(POOLPLAY, "policy_step_1500000.zip"),
    "pool4_5M": os.path.join(POOLPLAY, "policy_step_4500000.zip"),
    "pool7_5M": os.path.join(POOLPLAY, "policy_step_7500000.zip"),
    "pool9M":   os.path.join(POOLPLAY, "policy_step_9000000.zip"),
}
SEED_STRONG = {
    "champion": os.path.join(HERO, "ppo_gin_rummy_selfplay.zip"),
    "llm":      os.path.join(HERO, "ppo_gin_rummy_llm.zip"),
    "dagbase":  os.path.join(HERO, "ppo_gin_rummy_dagbase.zip"),
    # the two strongest agents from the Phase-6 sweep (~33% vs gold) -- so later runs practise
    # against genuinely strong opponents, not just the older ~30% champion.
    "p6champ":  os.path.join(HERO, "gin_curriculum_champion.zip"),
    "p6gold":   os.path.join(HERO, "gin_gold_hunter.zip"),
}

# ----------------------------------------------------------------- generic loader
# A run trained with TRPO checkpoints TRPO models; the curated seeds are PPO. Within one pool
# we therefore mix classes, so we must load with the right class. PPOAgent only ever calls
# model.predict(), which is class-agnostic, so a TRPO model works fine as a frozen opponent
# once loaded with TRPO.load. We always know the algo of a file (seeds = ppo; self-checkpoints
# = the run's algo), so we try the correct class first and fall back for safety.
_LOAD_CACHE: dict[str, object] = {}


def load_any(path, prefer="ppo"):
    if path in _LOAD_CACHE:
        return _LOAD_CACHE[path]
    order = ["ppo", "trpo"] if prefer == "ppo" else ["trpo", "ppo"]
    last_err = None
    for algo in order:
        try:
            if algo == "ppo":
                m = PPO.load(path, device="cpu")
            else:
                from sb3_contrib import TRPO
                m = TRPO.load(path, device="cpu")
            _LOAD_CACHE[path] = m
            return m
        except Exception as e:  # noqa: BLE001
            last_err = e
    print(f"[load_any] FAILED to load {path}: {last_err}", flush=True)
    return None


# ----------------------------------------------------------------- reward shaping
def _hand_deadwood(obs):
    """Best achievable deadwood of the agent's OWN hand (plane 0 of the observation), using
    RLCard's rules-based meld scorer. This is the game's own scoring of a PUBLIC hand, not the
    gold agent's policy -- the same deadwood the terminal payoff already uses, just read per
    step. Returns None for non-standard hand sizes so the caller can skip cheaply (we only
    score the 10-card hand, which is one meld call; the 11-card draw state is skipped)."""
    try:
        from agents.gold_standard_agent import _best_deadwood
        hand = np.where(np.asarray(obs["observation"])[0].reshape(-1) == 1)[0].tolist()
        if len(hand) == 10:
            return _best_deadwood(hand)
    except Exception:
        pass
    return None


class ShapedWrapper(GinRummySB3Wrapper):
    """knock_reward / gin_reward (the env's terminal-payoff bonuses) are the PRIMARY reward
    levers and pass straight through.  On top we optionally (a) rescale the TERMINAL outcome
    asymmetrically -- win_scale on a win, loss_scale on a loss; (b) apply a small dense
    per-decision penalty (step_penalty) to knock EARLY; and (c) a dense DEADWOOD-reduction
    reward (deadwood_coef): each turn the agent lowers its own best-achievable deadwood, it is
    rewarded -- coaching the optimal 'knock low' style directly (potential-based, so it cannot
    change which policy is optimal, only speed learning).  Eval always uses the raw env, so
    none of this leaks into scoring."""

    def __init__(self, *args, win_scale=1.0, loss_scale=1.0, step_penalty=0.0,
                 deadwood_coef=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.win_scale = float(win_scale)
        self.loss_scale = float(loss_scale)
        self.step_penalty = float(step_penalty)
        self.deadwood_coef = float(deadwood_coef)
        self._prev_dw = None

    def reset(self, *a, **k):
        self._prev_dw = None
        return super().reset(*a, **k)

    def step(self, action):
        obs, reward, done, trunc, info = super().step(action)
        if done and reward:
            if reward > 0:
                reward *= self.win_scale
            elif reward < 0:
                reward *= self.loss_scale
        if self.step_penalty:
            reward -= self.step_penalty           # dense per-decision speed pressure
        if self.deadwood_coef and not done:
            dw = _hand_deadwood(obs)
            if dw is not None:
                if self._prev_dw is not None:
                    reward += self.deadwood_coef * (self._prev_dw - dw)  # reward shedding deadwood
                self._prev_dw = dw
        return obs, reward, done, trunc, info


# ----------------------------------------------------------------- stage curriculum
class StageCurriculum:
    """Configurable multi-stage opponent curriculum implementing the duck-typed interface
    GinRummySB3Wrapper._select_opponent calls (types 'random'/'pool'/'self', no wrapper
    change).  Diversity within 'pool' = a MID seed tier + the run's growing self-checkpoints;
    a STRONG seed tier is folded into 'pool' sampling only after strong_from (so a fresh agent
    is not crushed early).  Sampling rule is 'recent' (uniform over mid+recent-self) or 'pfsp'
    (weighted by the agent's recent loss-rate vs each opponent, from a single-writer weights
    file the main callback refreshes; falls back to uniform when absent).  total_steps is
    shared across forked workers via an atomic JSON write, like the proven CurriculumManager."""

    def __init__(self, pool_dir, stages, total_target, algo="ppo",
                 recent_n=10, sampling="recent", strong_from=0.70):
        self.pool_dir = pool_dir
        self.stages = stages
        self.total_target = max(int(total_target), 1)
        self.algo = algo
        self.recent_n = recent_n
        self.sampling = sampling
        self.strong_from = strong_from
        self.state_file = os.path.join(pool_dir, "curriculum_state.json")
        self.weights_file = os.path.join(pool_dir, "pfsp_weights.json")
        os.makedirs(pool_dir, exist_ok=True)
        self.total_steps = 0
        self._last_ckpt = -10 ** 18
        self._load_state()

    # ---- file-state sync (cross-process) ----
    def _load_state(self):
        try:
            with open(self.state_file) as f:
                self.total_steps = json.load(f).get("total_steps", 0)
        except Exception:
            self.total_steps = 0

    def _save_state(self):
        tmp = self.state_file + f".tmp{os.getpid()}"
        try:
            with open(tmp, "w") as f:
                json.dump({"total_steps": self.total_steps}, f)
            os.replace(tmp, self.state_file)
        except Exception as e:  # noqa: BLE001
            print(f"[StageCurriculum] save_state error: {e}", flush=True)

    def update_total_steps(self, total_steps):
        self._load_state()
        if total_steps - self.total_steps > 5000:
            self.total_steps = total_steps
            self._save_state()

    def episode_complete(self):
        pass

    # ---- pool discovery ----
    def _mid_seeds(self):
        return sorted(glob.glob(os.path.join(self.pool_dir, "seed_mid_*.zip")))

    def _strong_seeds(self):
        return sorted(glob.glob(os.path.join(self.pool_dir, "seed_strong_*.zip")))

    def _self_files(self):
        fs = [f for f in glob.glob(os.path.join(self.pool_dir, "policy_step_*.zip"))
              if f.split("_")[-1].replace(".zip", "").isdigit()]   # skip any malformed name
        fs.sort(key=lambda x: int(x.split("_")[-1].replace(".zip", "")))
        return fs

    def _get_available_policies(self):
        return [f.replace(".zip", "") for f in
                self._mid_seeds() + self._strong_seeds() + self._self_files()]

    # ---- stage logic ----
    def _frac(self):
        return self.total_steps / self.total_target

    def _stage(self, steps):
        frac = steps / self.total_target
        for i, st in enumerate(self.stages):
            if frac < st["until"]:
                return i, st
        return len(self.stages) - 1, self.stages[-1]

    def get_opponent_type(self):
        self._load_state()
        idx, st = self._stage(self.total_steps)
        mix = st["mix"]
        kinds = list(mix.keys())
        weights = np.array([mix[k] for k in kinds], dtype=float)
        weights /= weights.sum()
        r = random.random()
        choice = kinds[int(min(np.searchsorted(np.cumsum(weights), r, side="right"),
                               len(kinds) - 1))]
        if choice == "pool" and not (self._mid_seeds() or self._strong_seeds()
                                     or self._self_files()):
            choice = "random"
        if choice == "self" and not os.path.exists(self.get_selfplay_model_path() + ".zip"):
            choice = "pool" if (self._mid_seeds() or self._self_files()) else "random"
        return choice, idx + 1, self.total_steps

    # ---- opponent providers ----
    def _pfsp_weights(self, paths):
        try:
            with open(self.weights_file) as f:
                w = json.load(f)
        except Exception:
            return None
        floor = 0.05            # keys are basenames WITHOUT .zip (must match _refresh_pfsp)
        return [floor + float(w.get(os.path.basename(p).replace(".zip", ""), 0.5))
                for p in paths]

    def get_policy_from_pool(self, recent_n=None):
        recent_n = recent_n or self.recent_n
        selves = self._self_files()
        recent = selves[-recent_n:] if len(selves) > recent_n else selves
        cands = self._mid_seeds() + recent
        if self._frac() >= self.strong_from:               # strong tier only late
            cands += self._strong_seeds()
        if not cands:
            return None
        weights = self._pfsp_weights(cands) if self.sampling == "pfsp" else None
        if weights and sum(weights) > 0:
            pick = random.choices(cands, weights=weights, k=1)[0]
        else:
            pick = random.choice(cands)
        path = pick.replace(".zip", "")
        prefer = self.algo if os.path.basename(path).startswith("policy_step_") else "ppo"
        return load_any(path, prefer=prefer)

    def get_selfplay_model_path(self):
        return os.path.join(self.pool_dir, "current_model_for_selfplay")

    def get_selfplay_policy(self):
        p = self.get_selfplay_model_path()
        if not os.path.exists(p + ".zip"):
            return None
        return load_any(p, prefer=self.algo)

    # ---- checkpointing (main callback only) ----
    def should_save_checkpoint(self, step, save_freq):
        return step - self._last_ckpt >= save_freq

    def save_checkpoint(self, model, step, max_self=15):
        model.save(os.path.join(self.pool_dir, f"policy_step_{step}"))
        model.save(self.get_selfplay_model_path())
        self._last_ckpt = step
        for old in self._self_files()[:-max_self]:           # trim oldest self checkpoints
            try:
                os.remove(old)
            except OSError:
                pass


def seed_pool(pool_dir, fresh=True):
    """(Re)create the per-run pool dir and copy the curated MID + STRONG seeds into it. Each
    run gets its OWN isolated pool (namespaced by cell name), so concurrent array tasks never
    collide -- no shared rmtree, no overlapping policy_step ids, no state clobber."""
    if fresh and os.path.isdir(pool_dir):
        shutil.rmtree(pool_dir)
    os.makedirs(pool_dir, exist_ok=True)
    kept = {"mid": [], "strong": []}
    for tier, lib in (("mid", SEED_MID), ("strong", SEED_STRONG)):
        for name, src in lib.items():
            if os.path.exists(src):
                shutil.copy(src, os.path.join(pool_dir, f"seed_{tier}_{name}.zip"))
                kept[tier].append(name)
            else:
                print(f"[seed_pool] missing {tier} seed '{name}' ({src}) -- skipped", flush=True)
    print(f"[seed_pool] {pool_dir}: mid={kept['mid']} strong={kept['strong']}", flush=True)
    return kept


# ----------------------------------------------------------------- model builder
# Phase-8 architecture sweep: the policy network is config-driven. The defaults below reproduce
# the Phase-6/7 winner policy (CombinedExtractor, [256,128] pi/vf, Tanh, ortho-init) BYTE-FOR-BYTE,
# so every existing cell is unaffected; arch/net_arch/activation/extractor_kwargs in a config JSON
# select a different network. Masking is preserved for ALL of them: FiniteMaskedPolicy reads the
# action mask from the raw dict obs and applies -1e8 to logits, independent of the features
# extractor (which only ever produces the policy/value features).
ACTIVATIONS = {"tanh": th.nn.Tanh, "relu": th.nn.ReLU, "gelu": th.nn.GELU}
EXTRACTORS = {"mlp": CombinedExtractor, "conv1d": Conv1DCardExtractor,
              "deepsets": DeepSetsCardExtractor, "attn": SetAttentionExtractor}


def _make_policy_kwargs(arch="mlp", net_arch=None, activation="tanh",
                        extractor_kwargs=None, weight_decay=0.0):
    if net_arch is None:
        na = dict(pi=[256, 128], vf=[256, 128])
    elif isinstance(net_arch, dict):
        na = dict(pi=list(net_arch["pi"]), vf=list(net_arch["vf"]))
    else:                                                  # bare list/tuple -> shared pi/vf
        na = dict(pi=list(net_arch), vf=list(net_arch))
    pkw = dict(features_extractor_class=EXTRACTORS[str(arch)],
               net_arch=na, activation_fn=ACTIVATIONS[str(activation).lower()],
               ortho_init=True)
    if str(arch) != "mlp":
        pkw["features_extractor_kwargs"] = dict(extractor_kwargs or {})
    if weight_decay and float(weight_decay) > 0:           # safe regularization lever
        pkw["optimizer_kwargs"] = dict(weight_decay=float(weight_decay))
    return pkw


def build_model(algo, env, hp):
    """PPO or TRPO on the masked policy. The network comes from _make_policy_kwargs (config-driven;
    defaults = the proven winner net so comparisons stay unconfounded). Reuses FiniteMaskedPolicy
    (works for BOTH algorithms)."""
    pkw = _make_policy_kwargs(hp.get("arch", "mlp"), hp.get("net_arch"),
                              hp.get("activation", "tanh"), hp.get("extractor_kwargs"),
                              hp.get("weight_decay", 0.0))
    common = dict(env=env, n_steps=hp["n_steps"], device="cpu", verbose=0,
                  policy_kwargs=pkw, seed=hp["seed"], gamma=hp["gamma"],
                  gae_lambda=hp["gae_lambda"], learning_rate=hp["lr"],
                  batch_size=hp["batch_size"], normalize_advantage=True)
    if algo == "ppo":
        return PPO(FiniteMaskedPolicy, n_epochs=hp["n_epochs"],
                   clip_range=hp["clip_range"], ent_coef=hp["ent_coef"], **common)
    if algo == "trpo":
        from sb3_contrib import TRPO
        return TRPO(FiniteMaskedPolicy, target_kl=hp["target_kl"], **common)
    raise ValueError(f"unknown algo {algo}")


# ----------------------------------------------------------------- eval helpers
def eval_full(model, opp_kind, champ_model, n, seed0=10_000):
    """Seat-swapped eval vs random/champion/gold that ALSO returns mean game length (the R4
    mechanism check). Mirrors sweep.llm_dagger_one.evaluate exactly + a step counter; scored
    under the FIXED (knock=0.5,gin=1.5) env so win/gin-rate stay comparable across reward cells."""
    from agents.gold_standard_agent import GoldStandardAgent
    env = gin_rummy_v4.env(**ENVK)
    hero = PPOAgent(env, model=model)
    if opp_kind == "gold":
        opp = GoldStandardAgent(env)
    elif opp_kind == "random":
        opp = RandomAgent(env)
    else:
        opp = PPOAgent(env, model=champ_model)
    win = gin = loss = 0
    lengths = []
    for g in range(n):
        h_seat = "player_0" if g % 2 == 0 else "player_1"
        agents = ({"player_0": hero, "player_1": opp} if h_seat == "player_0"
                  else {"player_0": opp, "player_1": hero})
        for k, a in agents.items():
            a.set_player(k)
        env.reset(seed=seed0 + g)
        tot = {"player_0": 0.0, "player_1": 0.0}
        steps = 0
        for ag in env.agent_iter():
            obs, rew, term, trunc, info = env.last()
            tot[ag] += rew
            if term or trunc:
                env.step(None); continue
            env.step(agents[ag].do_action()); steps += 1
        lengths.append(steps)
        r = tot[h_seat]
        if r > 0:
            win += 1
            if r >= GIN - 0.1:
                gin += 1
        elif r < 0:
            loss += 1
    env.close()
    return dict(win_rate=win / n, gin_rate=gin / n, loss_rate=loss / n,
                mean_len=float(np.mean(lengths)), n=n)


def eval_vs_model(model, opp_model, n, seed0=77_000):
    """Loss-rate of `model` vs a single frozen `opp_model` (for PFSP weighting). Fast (no gold)."""
    env = gin_rummy_v4.env(**ENVK)
    hero = PPOAgent(env, model=model)
    opp = PPOAgent(env, model=opp_model)
    loss = 0
    for g in range(n):
        h_seat = "player_0" if g % 2 == 0 else "player_1"
        agents = ({"player_0": hero, "player_1": opp} if h_seat == "player_0"
                  else {"player_0": opp, "player_1": hero})
        for k, a in agents.items():
            a.set_player(k)
        env.reset(seed=seed0 + g)
        tot = {"player_0": 0.0, "player_1": 0.0}
        for ag in env.agent_iter():
            o, rew, term, trunc, info = env.last()
            tot[ag] += rew
            if term or trunc:
                env.step(None); continue
            env.step(agents[ag].do_action())
        if tot[h_seat] < 0:
            loss += 1
    env.close()
    return loss / n


# ----------------------------------------------------------------- train callback
class SweepCallback(BaseCallback):
    """Drives the curriculum file-state, grows the opponent pool, records a learning curve by
    evaluating vs random/champion/gold at a step cadence, and (PFSP cells only) refreshes the
    single-writer opponent-difficulty weights the workers sample from."""

    def __init__(self, cm, champ_model, save_freq, eval_every, eval_games, curve,
                 pfsp=False, pfsp_games=15, best_path=None):
        super().__init__()
        self.cm = cm
        self.champ = champ_model
        self.save_freq = save_freq
        self.eval_every = eval_every
        self.eval_games = eval_games
        self.curve = curve
        self.pfsp = pfsp
        self.pfsp_games = pfsp_games
        self.best_path = best_path          # keep-the-best: save the peak checkpoint here
        self.best_metric = -1.0
        self.best_step = 0
        self._last_update = 0
        self._last_eval = 0

    def _refresh_pfsp(self):
        try:
            paths = [p.replace(".zip", "") for p in
                     self.cm._mid_seeds() + self.cm._self_files()[-self.cm.recent_n:]
                     + self.cm._strong_seeds()]
            w = {}
            for p in paths:
                opp = load_any(p, prefer=self.cm.algo if os.path.basename(p)
                               .startswith("policy_step_") else "ppo")
                if opp is not None:
                    w[os.path.basename(p)] = eval_vs_model(self.model, opp, self.pfsp_games)
            tmp = self.cm.weights_file + f".tmp{os.getpid()}"
            with open(tmp, "w") as f:
                json.dump(w, f)
            os.replace(tmp, self.cm.weights_file)
            print(f"[pfsp] refreshed {len(w)} opponent weights @ {self.num_timesteps:,}",
                  flush=True)
        except Exception as e:  # noqa: BLE001  PFSP is best-effort; uniform fallback on failure
            print(f"[pfsp] refresh skipped: {e}", flush=True)

    def _on_step(self) -> bool:
        s = self.num_timesteps
        if s - self._last_update >= 10_000:
            self.cm.update_total_steps(s)
            self._last_update = s
        if self.cm.should_save_checkpoint(s, self.save_freq):
            self.cm.save_checkpoint(self.model, s)
            print(f"[pool] checkpoint @ {s:,}; pool={len(self.cm._get_available_policies())}",
                  flush=True)
            if self.pfsp:
                self._refresh_pfsp()
        if self.eval_every and s - self._last_eval >= self.eval_every:
            self._last_eval = s
            snap = {"step": s, "stage": self.cm._stage(s)[0] + 1}
            for k in ("random", "champion", "gold"):
                r = evaluate(self.model, k, self.champ, self.eval_games)
                snap[k] = {"win": r["win_rate"], "gin": r["gin_rate"], "loss": r["loss_rate"]}
            self.curve.append(snap)
            # keep-the-best: rank by win-vs-gold, break ties by win-vs-champion (lower variance).
            metric = snap["gold"]["win"] + 0.001 * snap["champion"]["win"]
            improved = metric > self.best_metric
            if improved and self.best_path:
                self.best_metric = metric
                self.best_step = s
                self.model.save(self.best_path)
            print(f"[curve] @ {s:,} stage {snap['stage']} "
                  f"rand={snap['random']['win']:.3f} champ={snap['champion']['win']:.3f} "
                  f"gold={snap['gold']['win']:.3f} gin(vsgold)={snap['gold']['gin']:.3f}"
                  f"{'  <- NEW BEST (saved)' if improved else ''}", flush=True)
            if _wandb_on() and wandb.run is not None:
                wandb.log({
                    "curriculum/stage": snap["stage"],
                    "curriculum/pool_size": len(self.cm._get_available_policies()),
                    "eval/win_vs_random": snap["random"]["win"],
                    "eval/win_vs_champion": snap["champion"]["win"],
                    "eval/win_vs_gold": snap["gold"]["win"],
                    "eval/gin_rate_vs_gold": snap["gold"]["gin"],
                    "eval/best_vs_gold_so_far": max(self.best_metric, snap["gold"]["win"]),
                }, step=s)
        return True


# ----------------------------------------------------------------- curricula
def make_stages(name):
    if name == "designed":          # league-lite: ramp difficulty, keep some random throughout
        return [
            {"until": 0.25, "mix": {"random": 1.0}},
            {"until": 0.45, "mix": {"random": 0.70, "pool": 0.30}},
            {"until": 0.70, "mix": {"random": 0.30, "pool": 0.50, "self": 0.20}},
            {"until": 1.01, "mix": {"random": 0.15, "pool": 0.60, "self": 0.25}},
        ]
    if name == "no_random_tail":    # drop random after stage 1 -- catastrophic-forgetting test
        return [
            {"until": 0.25, "mix": {"random": 1.0}},
            {"until": 0.50, "mix": {"random": 0.30, "pool": 0.70}},
            {"until": 1.01, "mix": {"pool": 0.70, "self": 0.30}},
        ]
    if name == "warm":              # for WARM-STARTED strong agents: skip the random-only phase
        return [                    # and face real (incl. strong) opponents from the start
            {"until": 0.10, "mix": {"random": 0.50, "pool": 0.50}},
            {"until": 0.45, "mix": {"random": 0.20, "pool": 0.55, "self": 0.25}},
            {"until": 1.01, "mix": {"random": 0.10, "pool": 0.55, "self": 0.35}},
        ]
    raise ValueError(f"unknown curriculum {name}")


# ----------------------------------------------------------------- config loading
def _load_config():
    cfg = {}
    cfg_path = os.environ.get("CFG", "").strip()
    if cfg_path:
        with open(cfg_path) as f:
            cfg = json.load(f)

    def g(key, env, default, cast=str):
        if key in cfg:
            return cfg[key]
        v = os.environ.get(env)
        return cast(v) if v is not None else default

    d = dict(
        name=g("name", "NAME", "curr_run"),
        algo=g("algo", "ALGO", "trpo"),
        knock=float(g("knock", "KNOCK_R", 0.5)),
        gin=float(g("gin", "GIN_R", 0.5)),
        win_scale=float(g("win_scale", "WIN_SCALE", 1.0)),
        loss_scale=float(g("loss_scale", "LOSS_SCALE", 1.0)),
        step_penalty=float(g("step_penalty", "STEP_PENALTY", 0.0)),
        deadwood_coef=float(g("deadwood_coef", "DEADWOOD_COEF", 0.0)),
        init_model=g("init_model", "INIT_MODEL", ""),
        strong_from=float(g("strong_from", "STRONG_FROM", 0.70)),
        curriculum=g("curriculum", "CURRICULUM", "designed"),
        sampling=g("sampling", "SAMPLING", "recent"),
        steps=int(g("steps", "STEPS", 12_000_000)),
        num_env=int(g("num_env", "NUM_ENV", 16)),
        n_steps=int(g("n_steps", "N_STEPS", 256)),
        batch_size=int(g("batch_size", "BATCH", 1024)),
        n_epochs=int(g("n_epochs", "N_EPOCHS", 4)),
        lr=float(g("lr", "LR", 3e-4)),
        ent_coef=float(g("ent_coef", "ENT_COEF", 0.01)),
        clip_range=float(g("clip_range", "CLIP", 0.2)),
        target_kl=float(g("target_kl", "TARGET_KL", 0.01)),
        gamma=float(g("gamma", "GAMMA", 0.99)),
        gae_lambda=float(g("gae_lambda", "GAE_LAMBDA", 0.95)),
        seed=int(g("seed", "SEED", 0)),
        save_freq=int(g("save_freq", "SAVE_FREQ", 1_000_000)),
        eval_every=int(g("eval_every", "EVAL_EVERY", 3_000_000)),
        ckpt_eval_games=int(g("ckpt_eval_games", "CKPT_EVAL_GAMES", 150)),
        final_eval_games=int(g("final_eval_games", "FINAL_EVAL_GAMES", 800)),
        # Phase-8 architecture knobs (default = the winner net). net_arch/extractor_kwargs are
        # structured, so they are JSON-only via cfg.get (not env-castable).
        arch=g("arch", "ARCH", "mlp"),
        net_arch=cfg.get("net_arch", None),
        activation=g("activation", "ACTIVATION", "tanh"),
        extractor_kwargs=cfg.get("extractor_kwargs", {}),
        weight_decay=float(g("weight_decay", "WEIGHT_DECAY", 0.0)),
    )
    # OPERATIONAL knobs: an explicit env value always wins (so a smoke run can shrink a real
    # config's steps/envs/eval without editing its JSON). Scientific knobs stay JSON-driven.
    for key, env in (("steps", "STEPS"), ("num_env", "NUM_ENV"), ("save_freq", "SAVE_FREQ"),
                     ("eval_every", "EVAL_EVERY"), ("ckpt_eval_games", "CKPT_EVAL_GAMES"),
                     ("final_eval_games", "FINAL_EVAL_GAMES")):
        if os.environ.get(env) is not None:
            d[key] = int(os.environ[env])
    return d


# ----------------------------------------------------------------- startup self-test
def _selftest(c):
    """<10s fast-fail: build the trainee policy WITH THE CONFIGURED ARCHITECTURE, save it, reload
    it via the generic loader (catches the TRPO-reload + missing-policy-import traps and any broken
    custom extractor), and load one curated seed. Abort the whole run immediately if any of these
    fail -- before burning the env setup or training."""
    algo = c["algo"]
    tmp = tempfile.mkdtemp(prefix="curr_selftest_")
    try:
        dummy = GinRummySB3Wrapper(opponent_policy=RandomAgent, randomize_position=True,
                                   turns_limit=200, curriculum_manager=None)
        hp = dict(n_steps=16, batch_size=16, n_epochs=1, lr=3e-4, ent_coef=0.01,
                  clip_range=0.2, target_kl=0.01, gamma=0.99, gae_lambda=0.95, seed=0,
                  arch=c.get("arch", "mlp"), net_arch=c.get("net_arch"),
                  activation=c.get("activation", "tanh"),
                  extractor_kwargs=c.get("extractor_kwargs"),
                  weight_decay=c.get("weight_decay", 0.0))
        m = build_model(algo, dummy, hp)
        p = os.path.join(tmp, "probe")
        m.save(p)
        _LOAD_CACHE.clear()
        reloaded = load_any(p, prefer=algo)
        assert reloaded is not None, "generic reload of the trainee checkpoint returned None"
        a_seed = next((s for s in SEED_STRONG.values() if os.path.exists(s)), None)
        if a_seed:
            assert load_any(a_seed.replace(".zip", ""), prefer="ppo") is not None, \
                "could not load a curated seed model"
        _LOAD_CACHE.clear()
        dummy.close()
        print(f"[selftest] OK (algo={algo}: build->save->reload->seed-load all pass)", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ----------------------------------------------------------------- main
def main():
    c = _load_config()
    th.manual_seed(c["seed"]); np.random.seed(c["seed"]); random.seed(c["seed"])
    _selftest(c)                                            # abort in <10s on a load/arch trap

    pool_dir = os.path.join(SCRATCH, "sweep_curriculum", c["name"], "pool")
    out_dir = os.path.join(PROJECT_ROOT, "sweep", "curriculum")
    os.makedirs(out_dir, exist_ok=True)
    print(f"=== curriculum-train name={c['name']} algo={c['algo']} "
          f"reward(knock={c['knock']},gin={c['gin']},win={c['win_scale']},"
          f"loss={c['loss_scale']},step_pen={c['step_penalty']}) curriculum={c['curriculum']}"
          f"/{c['sampling']} gamma={c['gamma']} steps={c['steps']:,} seed={c['seed']} ===",
          flush=True)

    if _wandb_on():
        try:
            wandb.init(project=os.environ.get("WANDB_PROJECT", "Adversarial-CoEvolution"),
                       entity=os.environ.get("WANDB_ENTITY", "VLAvengers"),
                       name=c["name"], group=os.environ.get("WANDB_GROUP", "phase6-curriculum"),
                       tags=[c["algo"], f"arch:{c['arch']}", f"cur:{c['curriculum']}",
                             f"samp:{c['sampling']}", f"knock{c['knock']}", f"gin{c['gin']}",
                             "steppen" if c["step_penalty"] else "nopen",
                             "deadwood" if c.get("deadwood_coef") else "nodw",
                             "warmstart" if c.get("init_model") else "scratch", f"seed{c['seed']}"],
                       config=c, reinit=True)
            print("[wandb] logging enabled", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[wandb] init failed, continuing without it: {e}", flush=True)

    seed_pool(pool_dir, fresh=True)
    champ = PPO.load(SEED_STRONG["champion"], device="cpu")
    stages = make_stages(c["curriculum"])

    def _mk(rank):
        def _t():
            cm = StageCurriculum(pool_dir, stages, c["steps"], algo=c["algo"],
                                 sampling=c["sampling"], strong_from=c["strong_from"])
            e = ShapedWrapper(opponent_policy=RandomAgent, randomize_position=True,
                              turns_limit=200, curriculum_manager=cm, rank=rank,
                              knock_reward=c["knock"], gin_reward=c["gin"],
                              win_scale=c["win_scale"], loss_scale=c["loss_scale"],
                              step_penalty=c["step_penalty"],
                              deadwood_coef=c.get("deadwood_coef", 0.0))
            e.reset(seed=c["seed"] + rank)
            return Monitor(e)
        return _t

    venv = SubprocVecEnv([_mk(i) for i in range(c["num_env"])], start_method="fork")
    # Return normalization de-confounds the reward study (R0 gin=1.5 vs R1=0.5 otherwise act as
    # different effective learning rates). obs are binary planes -> norm_obs=False; eval uses a
    # fresh RAW env so scoring is unaffected and no stats need saving.
    env = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=c["gamma"])

    hp = dict(n_steps=c["n_steps"], batch_size=c["batch_size"], n_epochs=c["n_epochs"],
              lr=c["lr"], ent_coef=c["ent_coef"], clip_range=c["clip_range"],
              target_kl=c["target_kl"], gamma=c["gamma"], gae_lambda=c["gae_lambda"],
              seed=c["seed"], arch=c["arch"], net_arch=c["net_arch"],
              activation=c["activation"], extractor_kwargs=c["extractor_kwargs"],
              weight_decay=c["weight_decay"])
    init = c.get("init_model", "").strip()
    if init:                                               # warm-start from a prior best model
        ipath = init[:-4] if init.endswith(".zip") else init
        algo_cls = PPO if c["algo"] == "ppo" else __import__("sb3_contrib", fromlist=["TRPO"]).TRPO
        model = algo_cls.load(ipath, env=env, device="cpu")
        print(f"[warm-start] loaded {ipath} as the starting policy", flush=True)
    else:
        model = build_model(c["algo"], env, hp)

    main_cm = StageCurriculum(pool_dir, stages, c["steps"], algo=c["algo"],
                              sampling=c["sampling"], strong_from=c["strong_from"])
    curve = []
    best_path = os.path.join(SCRATCH, "sweep_curriculum", c["name"], "best")
    cb = SweepCallback(main_cm, champ, c["save_freq"], c["eval_every"], c["ckpt_eval_games"],
                       curve, pfsp=(c["sampling"] == "pfsp"), best_path=best_path)

    t0 = time.time()
    model.learn(total_timesteps=c["steps"], callback=cb, progress_bar=False)
    train_seconds = time.time() - t0
    env.close()

    model_path = os.path.join(SCRATCH, "sweep_curriculum", c["name"], "final")
    model.save(model_path)

    final = {k: eval_full(model, k, champ, c["final_eval_games"])
             for k in ("random", "champion", "gold")}
    # keep-the-best: the shipped model is the best CHECKPOINT (re-confirmed at full N), not the
    # possibly-drifted final. Fall back to the final model if no best was saved.
    best_src = best_path if os.path.exists(best_path + ".zip") else model_path
    best_model = load_any(best_src, prefer=c["algo"])
    best_eval = {k: eval_full(best_model, k, champ, c["final_eval_games"])
                 for k in ("random", "champion", "gold")}
    headline = best_eval if best_eval["gold"]["win_rate"] >= final["gold"]["win_rate"] else final
    result = dict(name=c["name"], algo=c["algo"], curriculum=c["curriculum"],
                  sampling=c["sampling"], init_model=c.get("init_model", ""),
                  arch=c["arch"], net_arch=c["net_arch"], activation=c["activation"],
                  reward=dict(knock=c["knock"], gin=c["gin"], win_scale=c["win_scale"],
                              loss_scale=c["loss_scale"], step_penalty=c["step_penalty"],
                              deadwood_coef=c.get("deadwood_coef", 0.0)),
                  hp=dict(lr=c["lr"], ent_coef=c["ent_coef"], clip_range=c["clip_range"],
                          target_kl=c["target_kl"], gamma=c["gamma"],
                          gae_lambda=c["gae_lambda"], n_steps=c["n_steps"]),
                  steps=c["steps"], seed=c["seed"], train_seconds=train_seconds,
                  model_path=model_path + ".zip", best_model_path=best_src + ".zip",
                  best_step=cb.best_step,
                  pool_size=len(main_cm._get_available_policies()), curve=curve,
                  # headline metrics now come from the kept-best model (confirmed at full N)
                  best_vs_gold=best_eval["gold"]["win_rate"],
                  best_vs_champion=best_eval["champion"]["win_rate"],
                  final_vs_gold=final["gold"], final_vs_champion=final["champion"],
                  vs_random=headline["random"], vs_champion=headline["champion"],
                  vs_gold=headline["gold"])
    out = os.path.join(out_dir, f"{c['name']}.json")
    tmp = out + ".tmp"                                      # atomic: never leave a partial JSON
    with open(tmp, "w") as f:
        json.dump(result, f, indent=2)
    os.replace(tmp, out)
    print(f"[done] {c['name']} HEADLINE(best) vs_random={headline['random']['win_rate']:.3f} "
          f"vs_champion={headline['champion']['win_rate']:.3f} "
          f"vs_gold={headline['gold']['win_rate']:.3f} | "
          f"final vs_gold={final['gold']['win_rate']:.3f} best_step={cb.best_step:,} "
          f"in {train_seconds:.0f}s -> {out}", flush=True)

    if _wandb_on() and wandb.run is not None:
        wandb.log({
            "final/win_vs_random": final["random"]["win_rate"],
            "final/win_vs_champion": final["champion"]["win_rate"],
            "final/win_vs_gold": final["gold"]["win_rate"],
            "final/gin_rate_vs_gold": final["gold"]["gin_rate"],
            "final/mean_len_vs_gold": final["gold"]["mean_len"],
            "best/win_vs_gold": best_eval["gold"]["win_rate"],
            "best/win_vs_champion": best_eval["champion"]["win_rate"],
            "best/step": cb.best_step,
            "final/train_seconds": train_seconds,
        }, step=c["steps"])
        wandb.finish()


if __name__ == "__main__":
    main()
