"""Phase-4 LLM-guided RL: LLM-DAgger.

The LLM is the GUIDE (never the optimal solver). We (1) collect (obs, llm_action)
labels by letting the LLM play against a strong opponent, (2) behavioural-clone the
masked PPO policy on those labels, then (3) PPO self-play so the agent improves
BEYOND the imperfect LLM. The gold-standard agent is used ONLY to score the result.

Research question: can a sub-optimal LLM bootstrap a strong RL policy? Headline
metric: win rate vs the gold-standard yardstick (the champion sits at ~30%).

Env knobs: LLM_MODEL, BC_STATES, BC_EPOCHS, BC_LR, RL_STEPS, NUM_ENV, N_STEPS,
COLLECT_THREADS, INIT_MODEL (BC from scratch if empty), CHAMP_MODEL, SAVE_PATH,
EVAL_GAMES, STUB_LLM (1 = random-legal fake labeler for smoke tests).
"""
import os, sys, json, time, threading

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
try:
    import cv2  # noqa: F401  (import in parent before any fork; see llmplay_one)
except Exception:
    pass
import torch as th
from concurrent.futures import ThreadPoolExecutor

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from pettingzoo.classic import gin_rummy_v4
from gym_wrapper import GinRummySB3Wrapper
from agents.ppo_agent import PPOAgent
from agents.random_agent import RandomAgent
from agents.gold_standard_agent import GoldStandardAgent
import ppo_train  # noqa: F401  registers MaskedGinRummyPolicy

try:
    import wandb
    _WANDB = True
except Exception:
    _WANDB = False

KNOCK, GIN = 0.5, 1.5
ENVK = dict(knock_reward=KNOCK, gin_reward=GIN, opponents_hand_visible=False)


def _wandb_on():
    if not _WANDB or os.environ.get("WANDB_DISABLED", "").lower() in ("1", "true", "yes"):
        return False
    return bool(os.environ.get("WANDB_API_KEY")) or os.path.exists(
        os.path.expanduser("~/.netrc"))


def cfg_int(name, default):
    return int(os.environ.get(name, default))


# --------------------------------------------------------------------- collect
class _StubLabeler:
    """Fake 'LLM' that returns a random legal action (smoke tests, no GPU)."""
    def __init__(self, env):
        self.env = env

    def set_player(self, p):
        self.player = p

    def do_action(self):
        obs, *_ = self.env.last()
        legal = np.where(obs["action_mask"] == 1)[0]
        return int(np.random.choice(legal))


def _make_labeler(env, stub, llm_model):
    if stub:
        return _StubLabeler(env)
    from agents.fast_llm_agent import FastLLMAgent
    return FastLLMAgent(env, model=llm_model)


def collect_labels(target, threads, champ_model, stub, llm_model, seed0=0):
    """Run games where the LLM (hero) plays a strong opponent; record every LLM
    decision as a label. Threaded so concurrent calls saturate the master."""
    records = []          # list of (observation[4,52] int8, mask[110] int8, action int)
    lock = threading.Lock()
    stop = threading.Event()

    def worker(wid):
        env = gin_rummy_v4.env(**ENVK)
        hero = _make_labeler(env, stub, llm_model)
        opp = (RandomAgent(env) if champ_model is None
               else PPOAgent(env, model=champ_model))
        hero.set_player("player_0"); opp.set_player("player_1")
        g = 0
        while not stop.is_set():
            hero_seat = "player_0" if (g % 2 == 0) else "player_1"
            env.reset(seed=seed0 + wid * 100000 + g)
            g += 1
            local = []
            for ag in env.agent_iter():
                obs, rew, term, trunc, info = env.last()
                if term or trunc:
                    env.step(None); continue
                if ag == hero_seat:
                    a = hero.do_action()
                    local.append((obs["observation"].astype(np.int8).copy(),
                                  obs["action_mask"].astype(np.int8).copy(), int(a)))
                else:
                    a = opp.do_action()
                env.step(a)
            with lock:
                records.extend(local)
                if len(records) >= target:
                    stop.set()
        env.close()

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=threads) as ex:
        list(ex.map(worker, range(threads)))
    dt = time.time() - t0
    n = min(len(records), target)
    print(f"[collect] {n} labels in {dt:.0f}s ({n/max(dt,1):.1f}/s)", flush=True)
    return records[:n]


# -------------------------------------------------------------------- BC train
def behavioural_clone(model, records, epochs, lr, batch=512):
    obs = np.stack([r[0] for r in records]).astype(np.float32)
    msk = np.stack([r[1] for r in records]).astype(np.float32)
    act = np.array([r[2] for r in records], dtype=np.int64)
    dev = model.policy.device
    act_t = th.as_tensor(act, device=dev)
    opt = th.optim.Adam(model.policy.parameters(), lr=lr)
    n = len(records)
    model.policy.train()
    last = {}
    for ep in range(epochs):
        idx = np.random.permutation(n)
        tot, correct, seen = 0.0, 0, 0
        for s in range(0, n, batch):
            b = idx[s:s + batch]
            obs_b = {"observation": obs[b], "action_mask": msk[b]}
            obs_t, _ = model.policy.obs_to_tensor(obs_b)
            dist = model.policy.get_distribution(obs_t)
            loss = -dist.log_prob(act_t[b]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(b)
            pred = dist.distribution.probs.argmax(dim=1)
            correct += int((pred == act_t[b]).sum()); seen += len(b)
        last = dict(bc_loss=tot / n, bc_acc=correct / max(seen, 1))
        print(f"[bc] epoch {ep+1}/{epochs} loss={last['bc_loss']:.3f} "
              f"acc={last['bc_acc']:.3f}", flush=True)
        if _wandb_on() and wandb.run is not None:
            wandb.log({f"bc/{k}": v for k, v in last.items()}, step=ep)
    return last


# ----------------------------------------------------------------------- eval
def evaluate(model, opp_kind, champ_model, n, seed0=10_000):
    env = gin_rummy_v4.env(**ENVK)
    hero = PPOAgent(env, model=model)
    if opp_kind == "gold":
        opp = GoldStandardAgent(env)
    elif opp_kind == "random":
        opp = RandomAgent(env)
    else:
        opp = PPOAgent(env, model=champ_model)
    hero.set_player("player_0"); opp.set_player("player_1")
    win = gin = loss = 0
    for g in range(n):
        h_seat = "player_0" if g % 2 == 0 else "player_1"
        # rebind seats by swapping which agent answers for which id
        agents = ({"player_0": hero, "player_1": opp} if h_seat == "player_0"
                  else {"player_0": opp, "player_1": hero})
        for k, a in agents.items():
            a.set_player(k)
        env.reset(seed=seed0 + g)
        tot = {"player_0": 0.0, "player_1": 0.0}
        for ag in env.agent_iter():
            obs, rew, term, trunc, info = env.last()
            tot[ag] += rew
            if term or trunc:
                env.step(None); continue
            env.step(agents[ag].do_action())
        r = tot[h_seat]
        if r > 0:
            win += 1
            if r >= GIN - 0.1:
                gin += 1
        elif r < 0:
            loss += 1
    env.close()
    return dict(win_rate=win / n, gin_rate=gin / n, loss_rate=loss / n, n=n)


# ----------------------------------------------------------------------- main
def main():
    stub = os.environ.get("STUB_LLM", "0") == "1"
    llm_model = os.environ.get("LLM_MODEL", "qwen3-30b")
    bc_states = cfg_int("BC_STATES", 20000)
    bc_epochs = cfg_int("BC_EPOCHS", 8)
    bc_lr = float(os.environ.get("BC_LR", 3e-4))
    rl_steps = cfg_int("RL_STEPS", 2_000_000)
    num_env = cfg_int("NUM_ENV", 64)
    n_steps = cfg_int("N_STEPS", 256)
    threads = cfg_int("COLLECT_THREADS", 48)
    eval_games = cfg_int("EVAL_GAMES", 400)
    init_model = os.environ.get("INIT_MODEL", "").strip()
    champ_path = os.environ.get("CHAMP_MODEL", "game/model/ppo_gin_rummy_selfplay.zip")
    save_path = os.environ.get("SAVE_PATH", "game/model/ppo_gin_rummy_dagger.zip")
    seed = cfg_int("SEED", 0)

    th.manual_seed(seed); np.random.seed(seed)
    champ_model = PPO.load(champ_path, device="cpu")
    print(f"=== LLM-DAgger stub={stub} llm={llm_model} bc_states={bc_states} "
          f"bc_epochs={bc_epochs} rl_steps={rl_steps} init={init_model or 'scratch'} ===",
          flush=True)

    if _wandb_on():
        try:
            wandb.init(project=os.environ.get("WANDB_PROJECT", "Adversarial-CoEvolution"),
                       entity=os.environ.get("WANDB_ENTITY", "VLAvengers"),
                       name=os.environ.get("WANDB_NAME", f"dagger_{llm_model}_s{seed}"),
                       group="phase4-llm-dagger",
                       tags=["dagger", llm_model, "stub" if stub else "real"],
                       config=dict(llm_model=llm_model, bc_states=bc_states,
                                   bc_epochs=bc_epochs, rl_steps=rl_steps,
                                   init=init_model or "scratch"), reinit=True)
            print("[wandb] on", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[wandb] off: {e}", flush=True)

    # 1) collect LLM labels (skip entirely for the no-LLM baseline, BC_STATES=0)
    records = []
    if bc_states > 0:
        records = collect_labels(bc_states, threads, champ_model, stub, llm_model,
                                 seed0=seed)
        if not records:
            sys.exit("no labels collected")
    else:
        print("[collect] BC_STATES=0 -> no-LLM baseline (pure RL)", flush=True)

    # 2) build the env we will train on, then the policy on top of it
    import functools
    need_rl = rl_steps > 0
    frozen = PPO.load(champ_path, device="cpu")
    def _mk(rank):
        def _t():
            e = GinRummySB3Wrapper(
                opponent_policy=functools.partial(PPOAgent, model=frozen),
                randomize_position=True, turns_limit=200, curriculum_manager=None,
                rank=rank, knock_reward=KNOCK, gin_reward=GIN)
            e.reset(seed=seed + rank)
            return Monitor(e)
        return _t
    if need_rl:
        base_env = SubprocVecEnv([_mk(i) for i in range(num_env)], start_method="fork")
    else:
        base_env = GinRummySB3Wrapper(opponent_policy=RandomAgent, randomize_position=True,
                                      turns_limit=200, curriculum_manager=None)
    if init_model:
        model = PPO.load(init_model, env=base_env, device="cpu")
    else:
        from stable_baselines3.common.torch_layers import CombinedExtractor
        pkw = dict(features_extractor_class=CombinedExtractor,
                   net_arch=dict(pi=[512, 512, 256, 128], vf=[512, 512, 256, 128]),
                   activation_fn=th.nn.Tanh, ortho_init=True)
        model = PPO(ppo_train.MaskedGinRummyPolicy, base_env, n_steps=n_steps,
                    device="cpu", verbose=0, policy_kwargs=pkw)

    # behavioural-clone the policy on the LLM labels (skipped for the baseline)
    bc_stats, bc_eval = {}, {}
    if records:
        bc_stats = behavioural_clone(model, records, bc_epochs, bc_lr)
        bc_eval = {k: evaluate(model, k, champ_model, eval_games) for k in
                   ("random", "champion", "gold")}
        print("[bc-eval]", json.dumps(bc_eval), flush=True)

    # 3) PPO self-play (vs the frozen champion) to improve beyond the LLM
    class _WB(BaseCallback):
        def _on_step(self): return True
        def _on_rollout_end(self):
            if _wandb_on() and wandb.run is not None:
                snap = {}
                for k, v in self.model.logger.name_to_value.items():
                    try: snap[k] = abs(float(v)) if "loss" in k else float(v)
                    except Exception: pass
                if snap: wandb.log(snap, step=self.num_timesteps)

    t0 = time.time()
    if need_rl:
        model.learn(total_timesteps=rl_steps, callback=_WB())
    rl_secs = time.time() - t0
    if need_rl:
        base_env.close()
    model.save(save_path)

    final_eval = {k: evaluate(model, k, champ_model, eval_games) for k in
                  ("random", "champion", "gold")}
    print("[final-eval]", json.dumps(final_eval), flush=True)

    result = dict(stub=stub, llm_model=llm_model, bc_states=len(records),
                  bc_epochs=bc_epochs, rl_steps=rl_steps, init=init_model or "scratch",
                  bc=bc_stats, bc_eval=bc_eval, final_eval=final_eval,
                  rl_seconds=rl_secs, save_path=save_path)
    os.makedirs(os.path.join(PROJECT_ROOT, "sweep", "dagger"), exist_ok=True)
    # distinct filename per run (baseline vs LLM run share model+seed otherwise)
    tag = os.environ.get("WANDB_NAME") or f"dagger_{llm_model}_s{seed}"
    out = os.path.join(PROJECT_ROOT, "sweep", "dagger", f"{tag}.json")
    json.dump(result, open(out, "w"), indent=2)
    print("wrote", out, flush=True)
    if _wandb_on() and wandb.run is not None:
        wandb.log({f"eval/final_{k}_win": v["win_rate"] for k, v in final_eval.items()})
        wandb.finish()


if __name__ == "__main__":
    main()
