"""Phase-4(alt2): self-supervised TEMPORAL-context state embedding (word2vec-style),
the LLM-free baseline.

Play games with the heroes (+ random), record each game's ordered sequence of one
seat's decision states (a "sentence" of states). Then: states close together in the
SAME game = similar (positive, graded by the time gap); states from DIFFERENT games
= dissimilar (negative, score 0). Writes the SAME (obs1, obs2, score) format as the
LLM dataset, so embed_train.py and the Phase-5 harness work unchanged -> directly
comparable to the LLM-similarity embedding and to the raw sparse vector. CPU only.

Env: N_GAMES, N_PAIRS, WINDOW, POS_FRAC, OUT, SEED.
"""
import os, sys, time
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from pettingzoo.classic import gin_rummy_v4
from stable_baselines3 import PPO
from agents.random_agent import RandomAgent
from agents.gold_standard_agent import GoldStandardAgent
from agents.ppo_agent import PPOAgent
import ppo_train  # noqa: F401  registers MaskedGinRummyPolicy

ENVK = dict(knock_reward=0.5, gin_reward=1.5, opponents_hand_visible=False)
MODEL_PATHS = {
    "champion": "game/model/ppo_gin_rummy_selfplay.zip",
    "pool":     "game/model/ppo_gin_rummy_pool.zip",
    "winrate":  "game/model/ppo_gin_rummy_winrate.zip",
    "reward":   "game/model/ppo_gin_rummy_reward.zip",
    "llm_full": "game/model/ppo_gin_rummy_llm_full.zip",
}


def _planes(obs):
    p = np.asarray(obs["observation"])
    return p.T if (p.ndim == 2 and p.shape[0] not in (4, 5)) else p


def _make(kind, env, models):
    if kind == "random":
        return RandomAgent(env)
    if kind == "gold":
        return GoldStandardAgent(env)
    return PPOAgent(env, model=models[kind])


def collect_trajectories(n_games, kinds, models, seed0=0):
    """Each game -> ordered list of player_0's decision-state observation vectors."""
    env = gin_rummy_v4.env(**ENVK)
    trajs = []
    for g in range(n_games):
        a0 = _make(kinds[g % len(kinds)], env, models)
        a1 = _make(kinds[(g + 1) % len(kinds)], env, models)
        a0.set_player("player_0"); a1.set_player("player_1")
        env.reset(seed=seed0 + g)
        seq = []
        for ag in env.agent_iter():
            obs, r, term, trunc, info = env.last()
            if term or trunc:
                env.step(None); continue
            mask = np.asarray(obs["action_mask"]).reshape(-1)
            if ag == "player_0" and int(mask.sum()) >= 2:
                seq.append(_planes(obs)[:4].reshape(-1).astype(np.int8).copy())
            env.step((a0 if ag == "player_0" else a1).do_action())
        if len(seq) >= 2:
            trajs.append(seq)
    env.close()
    return trajs


def make_temporal_pairs(trajs, n_pairs, window, pos_frac, seed):
    rng = np.random.default_rng(seed)
    pairs = []
    n_pos = int(n_pairs * pos_frac)
    nt = len(trajs)
    # positives: same trajectory, gap 1..window, score graded by closeness
    tries = 0
    while sum(1 for _ in pairs) < n_pos and tries < n_pos * 5:
        tries += 1
        t = trajs[rng.integers(nt)]
        if len(t) < 2:
            continue
        i = int(rng.integers(len(t) - 1))
        gmax = min(window, len(t) - 1 - i)
        if gmax < 1:
            continue
        gap = 1 + int(rng.integers(gmax))
        score = int(round(100 * (1 - (gap - 1) / max(window, 1))))
        pairs.append((t[i], t[i + gap], max(5, score)))   # floor 5 > negatives
    # negatives: two states from DIFFERENT trajectories
    for _ in range(n_pairs - len(pairs)):
        a = int(rng.integers(nt)); b = int(rng.integers(nt))
        while b == a and nt > 1:
            b = int(rng.integers(nt))
        ta, tb = trajs[a], trajs[b]
        pairs.append((ta[int(rng.integers(len(ta)))], tb[int(rng.integers(len(tb)))], 0))
    rng.shuffle(pairs)
    return pairs


def main():
    n_games = int(os.environ.get("N_GAMES", 6000))
    n_pairs = int(os.environ.get("N_PAIRS", 200000))
    window = int(os.environ.get("WINDOW", 4))
    pos_frac = float(os.environ.get("POS_FRAC", 0.5))
    out = os.environ.get("OUT", "/scratch1/kelidari/advcoev_store/embed/temporal_dataset.npz")
    seed = int(os.environ.get("SEED", 0))
    os.makedirs(os.path.dirname(out), exist_ok=True)

    models = {k: PPO.load(p, device="cpu") for k, p in MODEL_PATHS.items()
              if os.path.exists(p)}
    kinds = ["random", "gold"] + list(models)   # diverse trajectory sources
    print(f"=== temporal-embed n_games={n_games} n_pairs={n_pairs} window={window} "
          f"pos_frac={pos_frac} kinds={kinds} ===", flush=True)
    t0 = time.time()
    trajs = collect_trajectories(n_games, kinds, models, seed0=seed)
    lens = [len(t) for t in trajs]
    print(f"[traj] {len(trajs)} trajectories (mean len {np.mean(lens):.1f}) "
          f"in {time.time()-t0:.0f}s", flush=True)
    pairs = make_temporal_pairs(trajs, n_pairs, window, pos_frac, seed)

    O1 = np.stack([p[0] for p in pairs]).astype(np.int8)
    O2 = np.stack([p[1] for p in pairs]).astype(np.int8)
    S = np.array([p[2] for p in pairs], dtype=np.int16)
    np.savez_compressed(out, obs1=O1, obs2=O2, score=S)
    import collections
    ranks = np.argsort(np.argsort(S)); bins = (ranks * 6 // len(S)).clip(0, 5)
    print(f"[done] {len(S)} pairs in {time.time()-t0:.0f}s -> {out}", flush=True)
    print(f"[score] zeros(neg)={int((S==0).sum())} pos={int((S>0).sum())}; "
          f"rank-binned 0-5 {dict(sorted(collections.Counter(bins.tolist()).items()))}",
          flush=True)


if __name__ == "__main__":
    main()
