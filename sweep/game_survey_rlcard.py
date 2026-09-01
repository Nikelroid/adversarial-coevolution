"""Second measurement path: RLCard environments, for games OpenSpiel does not ship.

The OpenSpiel survey (sweep/game_survey.py) is the primary instrument, because pyspiel exposes
information-state strings and world resampling, which is what makes exact and resampled hidden-bit
estimates possible. RLCard exposes neither. What it does expose is enough for everything else:
action-space size, observation size, episode length, and the empirical branching factor.

So for RLCard games we measure what the API supports and compute hidden information the one way
that needs no engine support: the closed form for how many ways the opponents' hands could have
been dealt from the cards the acting player cannot see. That is the same "closed form" column the
OpenSpiel survey already reports, so the two paths are comparable on that one number and clearly
marked as different instruments everywhere else.

Usage:
  /scratch1/kelidari/envs/coev/bin/python sweep/game_survey_rlcard.py [--games 60]

Writes sweep/game_survey_rlcard.json.
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import rlcard

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def deal_bits(unseen, hands):
    """log2 of the number of ways to deal `hands` (a list of sizes) out of `unseen` cards.

    Multinomial rather than a single binomial, because with three opponents it matters which
    opponent holds which cards. Returns None if the deal does not fit.
    """
    total = sum(hands)
    if total > unseen or total < 0:
        return None
    bits, pool = 0.0, unseen
    for k in hands:
        bits += math.log2(math.comb(pool, k))
        pool -= k
    return bits


# name -> (label, family, closed-form hidden bits, one-line note on what is hidden)
# Deal arithmetic is spelled out so the numbers can be checked by hand.
GAMES = {
    # 108-card deck, 7 each. From my seat: 108 - my 7 - the face-up card = 100 unseen,
    # of which the opponent holds 7.
    "uno":            ("UNO 2p", "card", deal_bits(100, [7]),
                       "opponent's hand, drawn from the unseen deck"),
    # 136 tiles, 13 each to four players. From my seat 136 - 13 = 123 unseen, three
    # opponents hold 13 apiece.
    "mahjong":        ("Mahjong 4p", "multi", deal_bits(123, [13, 13, 13]),
                       "three opponents' hands out of the unseen wall"),
    # 54-card deck, 17 each to three players, 3 face-up to the landlord.
    "doudizhu":       ("Dou Dizhu 3p", "multi", deal_bits(34, [17]),
                       "the two opponents' hands (one is public once the landlord takes the kitty)"),
    # 52 - my 10 - the face-up discard = 41 unseen. Must match the OpenSpiel gin_rummy row
    # exactly: the same game measured by two engines is the cheapest cross-check we have.
    "gin-rummy":      ("Gin Rummy (RLCard)", "card", deal_bits(41, [10]),
                       "opponent's 10-card hand out of the 41 cards I have not seen"),
    "limit-holdem":   ("Limit Texas hold'em 2p", "card", deal_bits(50, [2]),
                       "opponent's two hole cards"),
    "no-limit-holdem": ("No-limit Texas hold'em 2p", "card", deal_bits(50, [2]),
                        "opponent's two hole cards"),
    "leduc-holdem":   ("Leduc hold'em (RLCard)", "card", deal_bits(5, [1]),
                       "opponent's single card from a 6-card deck"),
    "bridge":         ("Contract bridge (RLCard)", "multi", deal_bits(39, [13, 13, 13]),
                       "the other three hands"),
    "blackjack":      ("Blackjack (RLCard)", "solo", None,
                       "dealer's hole card and the shoe order, no opponent to model"),
}


def playout_stats(env, n_games, seed=0, time_budget=25.0):
    """Random play through the RLCard API: episode length and legal-action counts."""
    rng = random.Random(seed)
    deadline = time.time() + time_budget
    lengths, branches = [], []
    for _ in range(n_games):
        if lengths and time.time() > deadline:
            break
        state, _ = env.reset()
        moves = 0
        while not env.is_over() and moves < 400:
            legal = list(state["legal_actions"].keys())
            if not legal:
                break
            branches.append(len(legal))
            state, _ = env.step(rng.choice(legal))
            moves += 1
        lengths.append(moves)
    return (sum(lengths) / max(1, len(lengths)),
            sum(branches) / max(1, len(branches)),
            max(branches) if branches else 0,
            len(lengths))


def obs_size(env):
    try:
        shp = list(env.state_shape[0])
        n = 1
        for d in shp:
            n *= d
        return shp, n
    except Exception:
        return None, None


def survey(name, n_games):
    t0 = time.time()
    label, family, hidden, note = GAMES[name]
    env = rlcard.make(name, config={"seed": 0})
    shp, n = obs_size(env)
    mean_len, mean_branch, max_branch, played = playout_stats(env, n_games)
    return {
        "game": f"rlcard:{name}",
        "label": label,
        "family": family,
        "library": "rlcard",
        "players": env.num_players,
        "information": "IMPERFECT_INFORMATION",
        "dynamics": "SEQUENTIAL",
        "chance": "EXPLICIT_STOCHASTIC",
        "actions_declared": env.num_actions,
        "branch_mean": round(mean_branch, 1),
        "branch_max": max_branch,
        "ist_shape": None, "ist_size": None,
        "obs_shape": shp, "obs_size": n,
        "input_size": n,
        "maxlen_declared": None,
        "len_mean": round(mean_len, 1),
        "infoset_bits_exact": None,
        "infoset_bits_max": None,
        "infoset_bits_resampled": None,
        "resample_censored_frac": None,
        "hand_bits_closed_form": None if hidden is None else round(hidden, 1),
        "hidden_note": note,
        "episodes_played": played,
        "tree_nodes_walked": 0,
        "survey_seconds": round(time.time() - t0, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--out", default=os.path.join(PROJECT_ROOT, "sweep", "game_survey_rlcard.json"))
    args = ap.parse_args()

    rows = []
    for name in GAMES:
        try:
            r = survey(name, args.games)
            rows.append(r)
            print(f"  ok   {r['label']:32s} {r['survey_seconds']:6.1f}s  "
                  f"{r['episodes_played']} episodes", flush=True)
        except Exception as e:
            rows.append({"game": f"rlcard:{name}", "label": GAMES[name][0],
                         "family": GAMES[name][1], "library": "rlcard",
                         "error": f"{type(e).__name__}: {e}"})
            print(f"  FAIL {GAMES[name][0]:32s} {type(e).__name__}: {str(e)[:80]}", flush=True)

    with open(args.out, "w") as f:
        json.dump({"library": "rlcard", "version": rlcard.__version__,
                   "playouts": args.games, "rows": rows}, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
