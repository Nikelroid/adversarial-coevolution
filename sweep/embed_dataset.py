"""Phase-4(alt): build an LLM-judged state-similarity dataset for learning a dense
embedding of the sparse Gin Rummy observation.

Pipeline: (1) build a pool of LEGIT reachable states by playing mixed-agent games
and snapshotting observations; (2) sample stratified pairs (random + near-duplicate
so both similarity extremes are covered); (3) describe each state in plain language
(hand, best melds, deadwood, top discard, phase, legal options) via the exact
melding utils; (4) ask a strong LLM how strategically similar the two states are
(0-5); (5) write (obs1[208], obs2[208], score) records.

The LLM judges SIMILARITY (a reasoning task), never plays. Serve a model with
Ollama (set GINLLM_MASTER_URL); gpt-oss:20b on an L40S is fast + strong.

Env: N_STATES, N_PAIRS, NEAR_FRAC, LLM_MODEL, COLLECT_THREADS, OUT, STUB_LLM.
"""
import os, sys, json, time, threading, re
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor

from pettingzoo.classic import gin_rummy_v4
from rlcard.games.gin_rummy.utils import melding
from rlcard.games.gin_rummy.utils import utils as gu
from rlcard.games.gin_rummy.utils.utils import card_from_card_id

A_DRAW, A_PICKUP, A_GIN, A_DISCARD, A_KNOCK = 2, 3, 5, 6, 58
ENVK = dict(knock_reward=0.5, gin_reward=1.5, opponents_hand_visible=False)
RULES = ("Gin Rummy: form melds (sets = 3-4 same rank; runs = 3+ consecutive same "
         "suit). Card points: A=1, 2-9 face, T/J/Q/K=10. Unmelded cards are deadwood. "
         "Knock when deadwood <= 10; gin when deadwood = 0 (worth more). Each turn: "
         "draw (stock or the up-card) then discard.")


def _planes(obs):
    p = np.asarray(obs["observation"])
    return p.T if (p.ndim == 2 and p.shape[0] not in (4, 5)) else p


def _best_deadwood(card_ids):
    if len(card_ids) != 10:
        return None
    cards = [card_from_card_id(int(i)) for i in card_ids]
    cl = melding.get_best_meld_clusters(hand=cards)
    if not cl:
        return gu.get_deadwood_count(hand=cards, meld_cluster=[]), []
    dw = min(gu.get_deadwood_count(hand=cards, meld_cluster=c) for c in cl)
    melds = [[str(c) for c in m] for m in cl[0]]
    return dw, melds


def _to_planes(x):
    if isinstance(x, dict):
        return _planes(x)
    a = np.asarray(x)
    return a.reshape(4, 52) if a.ndim == 1 else a


def describe_state(obs):
    """Plain-language, strategically grounded description, derived ONLY from the
    observation planes (no action mask needed): hand size -> phase, deadwood ->
    knock/gin. Works for real and synthesized states alike."""
    p = _to_planes(obs)
    hand_ids = [int(c) for c in np.where(p[0] == 1)[0]]
    top_ids = [int(c) for c in np.where(p[1] == 1)[0]]
    hand = sorted(str(card_from_card_id(c)) for c in hand_ids)
    top = str(card_from_card_id(top_ids[0])) if top_ids else "none"
    # best melds + deadwood: the 10-card hand, or the best 10 of an 11-card hand.
    dw, melds = None, []
    if len(hand_ids) == 10:
        r = _best_deadwood(hand_ids)
        if r: dw, melds = r
    elif len(hand_ids) == 11:
        best = None
        for d in hand_ids:
            r = _best_deadwood([c for c in hand_ids if c != d])
            if r and (best is None or r[0] < best[0]):
                best = r
        if best: dw, melds = best
    phase = "must discard (holds 11)" if len(hand_ids) == 11 else "to draw (holds 10)"
    knockable = dw is not None and dw <= 10
    lines = [f"Hand ({len(hand)} cards): {', '.join(hand)}",
             f"Best melds: {melds if melds else 'none'}",
             f"Deadwood (best): {dw if dw is not None else 'n/a'}",
             f"Top of discard pile: {top}",
             f"Phase: {phase}; can knock: {knockable}; gin possible: {dw == 0}"]
    return "\n".join(lines)


# --------------------------------------------------------------- state pool
def build_pool(n_states, seed0=0):
    """Play mixed-agent games; snapshot decision-state observations (the 208-vec +
    its description). Mixed skill -> diverse, legit, reachable states."""
    from agents.random_agent import RandomAgent
    from agents.gold_standard_agent import GoldStandardAgent
    pool = []
    env = gin_rummy_v4.env(**ENVK)
    agents_pool = [RandomAgent(env), GoldStandardAgent(env)]
    for a, seat in zip(agents_pool, ("player_0", "player_1")):
        a.set_player(seat)
    g = 0
    while len(pool) < n_states:
        env.reset(seed=seed0 + g); g += 1
        # randomly assign which agent sits where for variety
        for ag in env.agent_iter():
            obs, rew, term, trunc, info = env.last()
            if term or trunc:
                env.step(None); continue
            mask = np.asarray(obs["action_mask"]).reshape(-1)
            # snapshot only real decision states (>=2 legal actions)
            if int(mask.sum()) >= 2 and len(pool) < n_states:
                vec = _planes(obs)[:4].reshape(-1).astype(np.int8).copy()
                pool.append(dict(vec=vec, desc=describe_state(obs)))
            a = agents_pool[0 if ag == "player_0" else 1].do_action()
            env.step(a)
    env.close()
    return pool[:n_states]


def _swap_one_card(vec):
    """Make a near-duplicate state: swap one hand card for an absent card."""
    v = vec.copy().reshape(4, 52)
    hand = np.where(v[0] == 1)[0]
    absent = np.where(v[0] == 0)[0]
    if len(hand) and len(absent):
        v[0, np.random.choice(hand)] = 0
        v[0, np.random.choice(absent)] = 1
    return v.reshape(-1)


def make_pairs(pool, n_pairs, near_frac):
    """Stratified: near-duplicate pairs (high-sim anchor) + random pairs."""
    pairs = []
    n = len(pool)
    n_near = int(n_pairs * near_frac)
    for _ in range(n_near):
        i = np.random.randint(n)
        v2 = _swap_one_card(pool[i]["vec"])
        pairs.append((pool[i]["vec"], pool[i]["desc"], v2, describe_state(v2)))
    for _ in range(n_pairs - n_near):
        i, j = np.random.randint(n), np.random.randint(n)
        pairs.append((pool[i]["vec"], pool[i]["desc"], pool[j]["vec"], pool[j]["desc"]))
    return pairs


# --------------------------------------------------------------- LLM scoring
def _score_prompt(desc1, desc2):
    # Ask on a FINE 0-100 scale; we rank-bin to 0-5 later (debiases the LLM's
    # tendency to cluster coarse scores).
    return (f"You are a Gin Rummy expert. {RULES}\n\n"
            f"Two game situations follow. Judge how STRATEGICALLY SIMILAR they are: "
            f"would the optimal next move, the available options, and the strategy to "
            f"follow be the same kind of decision?\n\n"
            f"SITUATION 1:\n{desc1}\n\nSITUATION 2:\n{desc2}\n\n"
            f"Think briefly about the optimal move and strategy in each, then reply on "
            f"the LAST line with ONLY a single integer 0-100 (0 = not relevant at all, "
            f"100 = strategically near-identical). Use the full range.")


def _extract_score(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    nums = re.findall(r"\d{1,3}", text)
    return max(0, min(100, int(nums[-1]))) if nums else None  # last number = the answer


def llm_score(desc1, desc2, model, url, stub=False):
    """Raw 0-100 similarity from the LLM (rank-binned to 0-5 at train time). Lets the
    model reason at LOW effort (good judgment, still fast on an L40S)."""
    if stub:
        return int(np.random.randint(0, 101))
    prompt = _score_prompt(desc1, desc2)
    base = {"model": model, "prompt": prompt, "stream": False,
            "options": {"num_predict": int(os.environ.get("NUM_PREDICT", 512)),
                        "temperature": 0.2}}
    timeout = int(os.environ.get("LLM_TIMEOUT", 90))

    def _call(extra):
        r = requests.post(f"{url}/api/generate", json={**base, **extra}, timeout=timeout)
        r.raise_for_status()
        return r.json()

    think = os.environ.get("THINK", "low")   # brief reasoning, not runaway
    try:
        j = _call({"think": think})           # gpt-oss reasoning effort (low/medium/high)
    except requests.HTTPError:
        try:
            j = _call({"think": True})         # some versions only accept a bool
        except requests.HTTPError:
            j = _call({})                      # last resort: model's default
    # prefer the final answer; fall back to the reasoning channel if needed
    return _extract_score(j.get("response", "") or "") or \
        _extract_score(j.get("thinking", "") or "")


def main():
    n_states = int(os.environ.get("N_STATES", 4000))
    n_pairs = int(os.environ.get("N_PAIRS", 20000))
    near_frac = float(os.environ.get("NEAR_FRAC", 0.35))
    model = os.environ.get("LLM_MODEL", "gpt-oss:20b")
    threads = int(os.environ.get("COLLECT_THREADS", 24))
    url = os.environ.get("GINLLM_MASTER_URL", "http://127.0.0.1:11434")
    stub = os.environ.get("STUB_LLM", "0") == "1"
    out = os.environ.get("OUT", os.path.join(PROJECT_ROOT, "sweep", "embed",
                                              "sim_dataset.npz"))
    seed = int(os.environ.get("SEED", 0)); np.random.seed(seed)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    print(f"=== embed-dataset states={n_states} pairs={n_pairs} near={near_frac} "
          f"model={model} stub={stub} ===", flush=True)
    t0 = time.time()
    pool = build_pool(n_states, seed0=seed)
    print(f"[pool] {len(pool)} legit states in {time.time()-t0:.0f}s", flush=True)
    pairs = make_pairs(pool, n_pairs, near_frac)
    print(f"[pairs] {len(pairs)} sampled", flush=True)

    O1, O2, S = [], [], []
    lock = threading.Lock(); done = [0]
    def work(pr):
        v1, d1, v2, d2 = pr
        try:
            sc = llm_score(d1, d2, model, url, stub)
        except Exception:
            sc = None
        if sc is not None:
            with lock:
                O1.append(v1); O2.append(v2); S.append(sc); done[0] += 1
                if done[0] % 500 == 0:
                    print(f"  scored {done[0]}/{len(pairs)} "
                          f"({done[0]/(time.time()-t0):.1f}/s)", flush=True)
    with ThreadPoolExecutor(max_workers=threads) as ex:
        list(ex.map(work, pairs))

    O1 = np.stack(O1).astype(np.int8); O2 = np.stack(O2).astype(np.int8)
    S = np.array(S, dtype=np.int16)            # RAW 0-100 (rank-binned to 0-5 at train)
    np.savez_compressed(out, obs1=O1, obs2=O2, score=S)
    print(f"[done] {len(S)} labeled pairs in {time.time()-t0:.0f}s -> {out}", flush=True)
    print(f"[raw 0-100] min={S.min()} max={S.max()} mean={S.mean():.1f} "
          f"std={S.std():.1f} uniques={len(np.unique(S))}", flush=True)
    # preview the debiased 0-5 rank-bins (should be ~equal counts)
    ranks = np.argsort(np.argsort(S)); bins = (ranks * 6 // max(len(S), 1)).clip(0, 5)
    import collections
    print(f"[rank-binned 0-5 counts] {dict(sorted(collections.Counter(bins.tolist()).items()))}",
          flush=True)


if __name__ == "__main__":
    main()
