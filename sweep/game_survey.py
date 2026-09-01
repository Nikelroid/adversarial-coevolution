"""Survey candidate two-player games for the information-ladder study.

Everything here is measured, not assumed:

  * action space       -- declared size, plus the empirical branching factor (mean legal actions
                          at a decision node under random play)
  * observation space  -- information-state tensor and/or observation tensor size
  * episode length     -- declared maximum, plus the empirical mean over random playouts
  * hidden information -- up to three independent estimates, so they can cross-check:
      (a) infoset_bits_exact: full tree walk on small games. Group histories by the acting
          player's information-state string; average log2(|infoset|) over reachable decision
          nodes. This is the game-theoretic amount of hidden information: how many distinct
          world states the acting player cannot tell apart.
      (b) infoset_bits_resampled: for games too big to enumerate. At sampled decision points,
          resample worlds consistent with the player's information state and count distinct
          ones. Censored at log2(n_resamples), so treat it as a lower bound.
      (c) hand_bits: closed form for card games where the hidden part is the opponent's hand,
          log2 C(unseen, hand_size).

Usage:
  /scratch1/kelidari/envs/coev/bin/python sweep/game_survey.py [--nodes 40000] [--games 100]

Writes sweep/game_survey.json and prints a markdown table.
"""
import argparse
import json
import math
import os
import random
import signal
import sys
import time

import pyspiel

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _hand_bits(unseen, k):
    """Closed form: bits needed to pin down which k of `unseen` cards the opponent holds."""
    return math.log2(math.comb(unseen, k)) if 0 <= k <= unseen else float("nan")


def _deal_bits(unseen, hands):
    """Closed form for more than one hidden hand: log2 of the ways to deal `hands` from `unseen`.

    Multinomial rather than a single binomial, because at a four-player table it matters which
    opponent holds which cards. Same quantity as _hand_bits when there is only one hidden hand.
    """
    if sum(hands) > unseen:
        return float("nan")
    bits, pool = 0.0, unseen
    for k in hands:
        bits += math.log2(math.comb(pool, k))
        pool -= k
    return bits


# (game string, family, closed-form hand bits or None, human label)
CANDIDATES = [
    ("goofspiel(num_cards=5,imp_info=False,points_order=descending)",  "card", 0.0,
     "Goofspiel-5 (open)"),
    ("goofspiel(num_cards=13,imp_info=False,points_order=descending)", "card", 0.0,
     "Goofspiel-13 (open)"),
    ("goofspiel(num_cards=5,imp_info=True,points_order=descending)",   "card", None,
     "Goofspiel-5 (hidden bids)"),
    ("goofspiel(num_cards=13,imp_info=True,points_order=descending)",  "card", None,
     "Goofspiel-13 (hidden bids)"),
    ("kuhn_poker",                                                     "card", _hand_bits(2, 1),
     "Kuhn poker"),
    ("leduc_poker",                                                    "card", _hand_bits(5, 1),
     "Leduc poker"),
    ("tiny_bridge_2p",                                                 "card", _hand_bits(6, 2),
     "Tiny bridge 2p"),
    ("universal_poker",                                                "card", None,
     "Universal poker (default)"),
    ("bridge_uncontested_bidding",                                     "card", _hand_bits(39, 13),
     "Bridge uncontested bidding"),
    ("crazy_eights(players=2)",                                        "card", _hand_bits(46, 5),
     "Crazy eights 2p"),
    ("hanabi(players=2)",                                              "card", None,
     "Hanabi 2p (self-hidden)"),
    ("tiny_hanabi",                                                    "card", None,
     "Tiny Hanabi"),
    # the gin rummy dial: identical rules, only the deck and hand size change
    ("gin_rummy(num_ranks=7,num_suits=2,hand_size=5,knock_card=5)",    "dial", _hand_bits(8, 5),
     "Gin dial: deck 14, hand 5"),
    ("gin_rummy(num_ranks=8,num_suits=2,hand_size=6,knock_card=6)",    "dial", _hand_bits(9, 6),
     "Gin dial: deck 16, hand 6"),
    ("gin_rummy(num_ranks=9,num_suits=3,hand_size=7,knock_card=7)",    "dial", _hand_bits(19, 7),
     "Gin dial: deck 27, hand 7"),
    ("gin_rummy(num_ranks=10,num_suits=3,hand_size=8,knock_card=8)",   "dial", _hand_bits(21, 8),
     "Gin dial: deck 30, hand 8"),
    ("gin_rummy(num_ranks=11,num_suits=4,hand_size=9,knock_card=9)",   "dial", _hand_bits(34, 9),
     "Gin dial: deck 44, hand 9"),
    ("gin_rummy(num_ranks=13,num_suits=4,hand_size=10,knock_card=10)", "dial", _hand_bits(41, 10),
     "Gin rummy (standard)"),
    # non-card contrast rungs
    ("connect_four",                                                   "board", 0.0,
     "Connect Four"),
    ("oshi_zumo",                                                      "board", 0.0,
     "Oshi Zumo"),
    ("phantom_ttt",                                                    "board", None,
     "Phantom tic-tac-toe"),
    ("dark_hex_ir(board_size=3)",                                      "board", None,
     "Dark Hex 3x3"),
    ("liars_dice(numdice=1)",                                          "dice", 1 * math.log2(6),
     "Liar's dice, 1 die"),
    ("liars_dice(numdice=2)",                                          "dice", 2 * math.log2(6),
     "Liar's dice, 2 dice"),
    ("liars_dice(numdice=5)",                                          "dice", 5 * math.log2(6),
     "Liar's dice, 5 dice"),
    ("battleship",                                                     "board", None,
     "Battleship 10x10"),
    # ---- added in the wide pass: every remaining 2-player imperfect-information game
    # OpenSpiel registers, plus multiplayer card games and the two solo-vs-chance games.
    # phantom / fog-of-war board games
    ("phantom_ttt_ir",                                                 "board", None,
     "Phantom tic-tac-toe (imperfect recall)"),
    ("latent_ttt",                                                     "board", None,
     "Latent tic-tac-toe"),
    ("dark_hex(board_size=3)",                                         "board", None,
     "Dark Hex 3x3 (perfect recall)"),
    ("dark_hex_ir(board_size=4)",                                      "board", None,
     "Dark Hex 4x4"),
    ("dark_chess(board_size=4)",                                       "board", None,
     "Dark chess 4x4"),
    ("dark_chess",                                                     "board", None,
     "Dark chess 8x8"),
    ("kriegspiel(board_size=4)",                                       "board", None,
     "Kriegspiel 4x4"),
    ("phantom_go(board_size=5)",                                       "board", None,
     "Phantom Go 5x5"),
    ("rbc(board_size=4)",                                              "board", None,
     "Reconnaissance blind chess 4x4"),
    ("rbc",                                                            "board", None,
     "Reconnaissance blind chess 8x8"),
    # dice, imperfect-recall contrast
    ("liars_dice_ir(numdice=2)",                                       "dice", 2 * math.log2(6),
     "Liar's dice 2 dice (imperfect recall)"),
    # bargaining, signalling, auctions: hidden preferences rather than hidden cards
    ("bargaining",                                                     "comm", None,
     "Bargaining"),
    ("negotiation",                                                    "comm", None,
     "Negotiation"),
    ("trade_comm",                                                     "comm", None,
     "Trade and communicate"),
    ("sheriff",                                                        "comm", None,
     "Sheriff of Nottingham"),
    ("lewis_signaling",                                                "comm", None,
     "Lewis signaling"),
    ("coordinated_mp",                                                 "comm", None,
     "Coordinated matching pennies"),
    ("first_sealed_auction",                                           "comm", None,
     "First-price sealed auction"),
    ("coop_box_pushing",                                               "comm", None,
     "Cooperative box pushing"),
    # more 2-player card games
    ("cribbage(players=2)",                                            "card", _deal_bits(46, [6]),
     "Cribbage 2p"),
    ("repeated_leduc_poker",                                           "card", _hand_bits(5, 1),
     "Repeated Leduc poker"),
    ("crazy_eights(players=2,use_special_cards=True)",                 "card", _hand_bits(46, 5),
     "Crazy eights 2p (special cards)"),
    ("hanabi(players=2,hand_size=3)",                                  "card", None,
     "Hanabi 2p, hand 3"),
    # multiplayer card games: same instrument, different table size
    ("dou_dizhu",                                                      "multi", _deal_bits(34, [17]),
     "Dou Dizhu 3p"),
    ("skat",                                                           "multi", _deal_bits(22, [10, 10, 2]),
     "Skat 3p"),
    ("hearts",                                                         "multi", _deal_bits(39, [13, 13, 13]),
     "Hearts 4p"),
    ("spades",                                                         "multi", _deal_bits(39, [13, 13, 13]),
     "Spades 4p"),
    ("euchre",                                                         "multi", _deal_bits(19, [5, 5, 5]),
     "Euchre 4p"),
    ("oh_hell",                                                        "multi", None,
     "Oh Hell"),
    ("tiny_bridge_4p",                                                 "multi", None,
     "Tiny bridge 4p"),
    ("bridge",                                                         "multi", _deal_bits(39, [13, 13, 13]),
     "Contract bridge 4p"),
    ("tarok(players=3)",                                               "multi", None,
     "Tarok 3p"),
    ("colored_trails",                                                 "multi", None,
     "Colored trails 3p"),
    ("crazy_eights(players=4)",                                        "multi", _hand_bits(47, 5),
     "Crazy eights 4p"),
    # solo against chance: the Balatro shape. No opponent, so every bit hidden is shuffle
    # order rather than another player's private state.
    ("blackjack",                                                      "solo", None,
     "Blackjack (solo vs deck)"),
    ("solitaire",                                                      "solo", None,
     "Klondike solitaire"),
]


def _key(state, player):
    """Best available label for what `player` can distinguish at this node."""
    try:
        return state.information_state_string(player)
    except Exception:
        try:
            return state.observation_string(player)
        except Exception:
            return None


def exact_infoset_bits(game, max_nodes, time_budget=12.0, max_chance_fanout=400):
    """Average log2(|infoset|) over reachable decision nodes, or None if the tree is too big.

    World states are keyed by history_str(), a compact action list rather than a rendered board,
    so the walk stays cheap. Three independent bail-outs, because a single node cap is not enough:
    a node budget, a wall-clock budget, and a chance-fanout limit. The last one matters most --
    games that deal a full hand at once (bridge) enumerate millions of outcomes at a single chance
    node, so merely counting them costs more than the whole rest of the walk.
    """
    groups = {}
    nodes = [0]
    too_big = [False]
    deadline = time.time() + time_budget

    def rec(state):
        if too_big[0]:
            return
        nodes[0] += 1
        if nodes[0] > max_nodes or (nodes[0] % 512 == 0 and time.time() > deadline):
            too_big[0] = True
            return
        if state.is_terminal():
            return
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            if len(outcomes) > max_chance_fanout:
                too_big[0] = True
                return
            for a, _ in outcomes:
                rec(state.child(a))
            return
        if state.is_simultaneous_node():
            for p in range(game.num_players()):
                k = _key(state, p)
                if k is not None:
                    groups.setdefault((p, k), set()).add(state.history_str())
            for a in state.legal_actions(0):
                joint = [a] + [state.legal_actions(p)[0] for p in range(1, game.num_players())]
                child = state.clone()
                child.apply_actions(joint)
                rec(child)
            return
        p = state.current_player()
        k = _key(state, p)
        if k is not None:
            groups.setdefault((p, k), set()).add(state.history_str())
        for a in state.legal_actions():
            rec(state.child(a))

    try:
        rec(game.new_initial_state())
    except (RecursionError, MemoryError):
        return None, None, nodes[0]
    if too_big[0] or not groups:
        return None, None, nodes[0]
    sizes = [len(v) for v in groups.values()]
    return (sum(math.log2(s) for s in sizes) / len(sizes),
            math.log2(max(sizes)),
            nodes[0])


def resampled_infoset_bits(game, n_points=12, n_resamples=80, seed=0):
    """Lower bound on hidden information for games too big to enumerate.

    At sampled decision points, draw worlds consistent with the acting player's information
    state and count how many distinct ones appear. Censored at log2(n_resamples).
    """
    try:
        sampler = pyspiel.UniformProbabilitySampler(0.0, 1.0)
    except Exception:
        return None, None
    rng = random.Random(seed)
    deadline = time.time() + 10.0
    per_point, censored = [], 0
    for i in range(n_points):
        if time.time() > deadline:
            break
        s = game.new_initial_state()
        target = rng.randint(3, 12)
        steps = 0
        while not s.is_terminal() and steps < target:
            if s.is_chance_node():
                outcomes, probs = zip(*s.chance_outcomes())
                s.apply_action(rng.choices(outcomes, probs)[0])
            elif s.is_simultaneous_node():
                s.apply_actions([rng.choice(s.legal_actions(p)) for p in range(game.num_players())])
                steps += 1
            else:
                s.apply_action(rng.choice(s.legal_actions()))
                steps += 1
        if s.is_terminal() or s.is_chance_node():
            continue
        p = s.current_player()
        worlds = set()
        try:
            for _ in range(n_resamples):
                worlds.add(s.resample_from_infostate(p, sampler).history_str())
        except Exception:
            return None, None
        if worlds:
            per_point.append(math.log2(len(worlds)))
            if len(worlds) > 0.9 * n_resamples:
                censored += 1
    if not per_point:
        return None, None
    return sum(per_point) / len(per_point), censored / len(per_point)


def playout_stats(game, n_games, seed=0, time_budget=10.0):
    """Empirical length and branching under random play, time-boxed.

    The budget matters for games that deal a whole hand at one chance node: sampling there
    materializes the full outcome list, so a single playout can cost a second.
    """
    rng = random.Random(seed)
    deadline = time.time() + time_budget
    lengths, branches = [], []
    for _ in range(n_games):
        if lengths and time.time() > deadline:
            break
        s = game.new_initial_state()
        moves = 0
        while not s.is_terminal() and moves < 500:
            if s.is_chance_node():
                outcomes, probs = zip(*s.chance_outcomes())
                s.apply_action(rng.choices(outcomes, probs)[0])
            elif s.is_simultaneous_node():
                joint = []
                for p in range(game.num_players()):
                    la = s.legal_actions(p)
                    branches.append(len(la))
                    joint.append(rng.choice(la))
                s.apply_actions(joint)
                moves += 1
            else:
                la = s.legal_actions()
                branches.append(len(la))
                s.apply_action(rng.choice(la))
                moves += 1
        lengths.append(moves)
    return (sum(lengths) / len(lengths),
            sum(branches) / max(1, len(branches)),
            max(branches) if branches else 0)


def shape_size(fn):
    try:
        shp = list(fn())
        n = 1
        for d in shp:
            n *= d
        return shp, n
    except Exception:
        return None, None


class _Timeout(Exception):
    pass


def _alarm(seconds):
    """Hard per-game wall-clock guard.

    The phase budgets can only be checked between iterations, so a single pathological call
    (one chance node that materializes every possible deal) can still overrun. SIGALRM
    interrupts regardless of where we are.
    """
    def _fire(signum, frame):
        raise _Timeout()
    signal.signal(signal.SIGALRM, _fire)
    signal.alarm(seconds)


def survey(name, family, hand_bits, label, max_nodes, n_games):
    t0 = time.time()
    game = pyspiel.load_game(name)
    t = game.get_type()
    ist_shape, ist_n = shape_size(game.information_state_tensor_shape)
    obs_shape, obs_n = shape_size(game.observation_tensor_shape)
    try:
        _alarm(45)
        mean_len, mean_branch, max_branch = playout_stats(game, n_games)
    except _Timeout:
        mean_len, mean_branch, max_branch = float('nan'), float('nan'), 0
    finally:
        signal.alarm(0)

    # Only attempt full enumeration when the tree is plausibly small. The chance check must come
    # from the declared max_chance_outcomes(): calling chance_outcomes() on a game that deals a
    # whole hand at once materializes every possible deal, which hangs before any guard can fire.
    try:
        max_chance = game.max_chance_outcomes()
    except Exception:
        max_chance = 0
    est_tree = float('inf') if mean_branch != mean_branch else mean_branch ** min(mean_len, 30)
    avg_bits, max_bits, nodes = None, None, 0
    if est_tree <= 5e6 and max_chance <= 400:
        try:
            _alarm(45)
            avg_bits, max_bits, nodes = exact_infoset_bits(game, max_nodes)
        except _Timeout:
            pass
        finally:
            signal.alarm(0)
    rs_bits, rs_censored = None, None
    if avg_bits is None:
        try:
            _alarm(45)
            rs_bits, rs_censored = resampled_infoset_bits(game)
        except _Timeout:
            pass
        finally:
            signal.alarm(0)

    return {
        "game": name, "label": label, "family": family,
        "players": game.num_players(),
        "information": str(t.information).split(".")[-1],
        "dynamics": str(t.dynamics).split(".")[-1],
        "chance": str(t.chance_mode).split(".")[-1],
        "actions_declared": game.num_distinct_actions(),
        "branch_mean": None if mean_branch != mean_branch else round(mean_branch, 1), "branch_max": max_branch,
        "ist_shape": ist_shape, "ist_size": ist_n,
        "obs_shape": obs_shape, "obs_size": obs_n,
        "input_size": ist_n or obs_n,
        "maxlen_declared": game.max_game_length(),
        "len_mean": None if mean_len != mean_len else round(mean_len, 1),
        "infoset_bits_exact": None if avg_bits is None else round(avg_bits, 2),
        "infoset_bits_max": None if max_bits is None else round(max_bits, 2),
        "infoset_bits_resampled": None if rs_bits is None else round(rs_bits, 2),
        "resample_censored_frac": None if rs_censored is None else round(rs_censored, 2),
        "hand_bits_closed_form": None if hand_bits is None else round(hand_bits, 1),
        "tree_nodes_walked": nodes,
        "survey_seconds": round(time.time() - t0, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, default=40_000)
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--out", default=os.path.join(PROJECT_ROOT, "sweep", "game_survey.json"))
    args = ap.parse_args()

    rows = []
    for name, family, hand_bits, label in CANDIDATES:
        try:
            r = survey(name, family, hand_bits, label, args.nodes, args.games)
            rows.append(r)
            print(f"  ok   {label:32s} {r['survey_seconds']:6.1f}s", flush=True)
        except Exception as e:
            rows.append({"game": name, "label": label, "family": family,
                         "error": f"{type(e).__name__}: {e}"})
            print(f"  FAIL {label:32s} {type(e).__name__}", flush=True)

    with open(args.out, "w") as f:
        json.dump({"node_cap": args.nodes, "playouts": args.games, "rows": rows}, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
