#!/usr/bin/env python3
"""Local web app to play Gin Rummy against the trained PPO agent.

Why this is robust ("zero bug"):
  * The backend runs the REAL PettingZoo ``gin_rummy_v4`` env -- the exact env
    the model was trained on -- via the tested ``GinRummySB3Wrapper`` with the
    human as the agent and the model as the opponent. No re-implemented rules.
  * The server is authoritative. Every move the UI offers comes from the env's
    ``action_mask``, and the server re-checks the mask before stepping, so
    illegal moves and rule desyncs are structurally impossible.
  * No Flask: only the stdlib ``http.server`` is used, so the sole dependencies
    are the ones the model already needs (torch, stable-baselines3, pettingzoo).

Run from the repo root:
    python game/server.py
then open http://localhost:8000 in your browser.

Pick a different model with --model PATH or $GIN_MODEL_PATH.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import combinations

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

# Repo imports. ppo_train must be importable so PPO.load can unpickle the custom
# MaskedGinRummyPolicy the model was trained with.
from stable_baselines3 import PPO  # noqa: E402
import ppo_train  # noqa: E402,F401  (registers MaskedGinRummyPolicy)
from gym_wrapper import GinRummySB3Wrapper  # noqa: E402
from agents.ppo_agent import PPOAgent  # noqa: E402
from agents.random_agent import RandomAgent  # noqa: E402
from agents.gold_standard_agent import GoldStandardAgent  # noqa: E402

GAME_DIR = os.path.join(REPO_ROOT, "game")
WEB_DIR = os.path.join(GAME_DIR, "web")
DECK_DIR = os.path.join(GAME_DIR, "deck_images")
IMAGES_DIR = os.path.join(GAME_DIR, "images")
# The opponents ("heroes") you can play against, hardest first. Each says exactly
# what it is. type "ppo" loads the named .zip; "gold"/"random" are code agents.
#   gold     -> hand-coded optimal expert (benchmark yardstick, never trained on)
#   selfplay -> the strongest learned agent
#   llm      -> champion fine-tuned against a Qwen2.5-7B opponent
#   pool/winrate/reward -> Phase-1/2 PPO variants
#   random   -> uniformly random legal move (easiest)
OPPONENTS = {
    "gold": {"emoji": "🏆", "label": "Gold Standard", "type": "gold",
             "stat": "optimal expert · benchmark",
             "desc": "Hand-coded expert: always finds the best melds and knocks "
                     "the moment it can. The benchmark we measure against, not a "
                     "learned agent. Hardest."},
    "selfplay": {"emoji": "🤖", "label": "Self-Play Champion", "type": "ppo",
                 "file": "ppo_gin_rummy_selfplay.zip", "stat": "strongest RL agent",
                 "desc": "PPO trained against frozen copies of itself. Our "
                         "strongest learned agent."},
    "llm": {"emoji": "🧠", "label": "LLM-Tutored", "type": "ppo",
            "file": "ppo_gin_rummy_llm_full.zip", "stat": "fine-tuned vs Qwen2.5-7B",
            "desc": "The champion fine-tuned for 1.5M steps against a Qwen2.5-7B "
                    "language-model opponent."},
    "pool": {"emoji": "♟️", "label": "Pool Veteran", "type": "ppo",
             "file": "ppo_gin_rummy_pool.zip", "stat": "self-play pool (regressed)",
             "desc": "PPO trained against a growing pool of its own past versions "
                     "(AlphaZero-style); regressed after ~10M steps."},
    "winrate": {"emoji": "🎯", "label": "Win-Rate Specialist", "type": "ppo",
                "file": "ppo_gin_rummy_winrate.zip", "stat": "99.6% vs random",
                "desc": "Phase-1 PPO tuned to beat the random player as often as "
                        "possible (99.6%)."},
    "reward": {"emoji": "💰", "label": "Reward Maximizer", "type": "ppo",
               "file": "ppo_gin_rummy_reward.zip", "stat": "highest avg score",
               "desc": "Phase-1 PPO with the highest average score per game."},
    "curriculum": {"emoji": "🃏", "label": "Curriculum Champion", "type": "ppo",
                   "file": "gin_curriculum_champion.zip", "stat": "33% vs gold · Phase-6 best",
                   "desc": "The strongest agent from our Phase-6 sweep. Trained through a full "
                           "league of opponents (random, its own past selves, then the champion) "
                           "and rewarded for knocking early with low deadwood, the optimal style. "
                           "Wins ~33% vs the gold standard and ~50% vs the old champion."},
    "goldhunter": {"emoji": "🥇", "label": "Gold Hunter", "type": "ppo",
                   "file": "gin_gold_hunter.zip", "stat": "best win-rate vs gold (33%)",
                   "desc": "The curriculum-sweep agent with the single highest win-rate against "
                           "the gold standard (33%). Curious twist: its reward paid a big bonus "
                           "for ginning, yet it still learned almost never to gin, exactly like "
                           "the optimal player does."},
    "random": {"emoji": "🎲", "label": "Random", "type": "random",
               "stat": "random legal moves",
               "desc": "Plays a uniformly random legal move every turn. Easiest."},
}
DEFAULT_OPPONENT = os.environ.get("GIN_OPPONENT", "selfplay")


def opponent_path(key: str) -> str:
    return os.path.join(REPO_ROOT, "game", "model", OPPONENTS[key]["file"])


def opponent_factory(key: str, models: dict):
    """Build the opponent_policy for a given hero key (class the wrapper will
    instantiate with the env)."""
    kind = OPPONENTS[key]["type"]
    if kind == "gold":
        return GoldStandardAgent
    if kind == "random":
        return RandomAgent
    return functools.partial(PPOAgent, model=models[key])

# PettingZoo card index = suit*13 + rank, suits in this order, rank 0=Ace..12=King.
_SUITS_FILE = ["spades", "hearts", "diamonds", "clubs"]
_SUITS_DISP = ["Spades", "Hearts", "Diamonds", "Clubs"]
_RANKS_DISP = ["Ace", "2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King"]

# RLCard card encoding -> PettingZoo index. We read the opponent's true hand from
# the underlying RLCard state because env.observe() ignores its agent argument
# (it always returns the *current* player's view, i.e. the human's).
_RLCARD_SUITS = "SHDC"
_RLCARD_RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]


def card_image(idx: int) -> str:
    return f"{idx % 13 + 1}_of_{_SUITS_FILE[idx // 13]}.png"


def card_label(idx: int) -> str:
    return f"{_RANKS_DISP[idx % 13]} of {_SUITS_DISP[idx // 13]}"


def card_value(idx: int) -> int:
    r = idx % 13  # 0=Ace .. 12=King
    if r == 0:
        return 1
    if r >= 9:  # 10, J, Q, K
        return 10
    return r + 1


def card_obj(idx: int) -> dict:
    return {"idx": int(idx), "img": card_image(idx), "label": card_label(idx),
            "value": card_value(idx)}


def best_melds_deadwood(cards):
    """Optimal (min-deadwood) meld decomposition of a list of 52-card indices.
    Returns (deadwood_value, list_of_melds) where each meld is a list of indices.
    Exact search; hands are <=11 cards so this is cheap. Ace is low (A-2-3),
    no wrap, matching the RLCard/PettingZoo env."""
    cards = list(cards)
    melds = []
    # sets: 3+ cards of the same rank
    by_rank = {}
    for c in cards:
        by_rank.setdefault(c % 13, []).append(c)
    for group in by_rank.values():
        if len(group) >= 3:
            for k in range(3, len(group) + 1):
                for combo in combinations(sorted(group), k):
                    melds.append(frozenset(combo))
    # runs: 3+ consecutive ranks in the same suit
    by_suit = {}
    for c in cards:
        by_suit.setdefault(c // 13, []).append(c % 13)
    for suit, ranks in by_suit.items():
        ranks = sorted(set(ranks))
        for a in range(len(ranks)):
            run = [ranks[a]]
            for b in range(a + 1, len(ranks)):
                if ranks[b] == run[-1] + 1:
                    run.append(ranks[b])
                    if len(run) >= 3:
                        melds.append(frozenset(suit * 13 + r for r in run))
                else:
                    break

    memo = {}

    def solve(remaining):
        if remaining in memo:
            return memo[remaining]
        best_dw = sum(card_value(c) for c in remaining)
        best_sel = []
        for m in melds:
            if m <= remaining:
                dw, sel = solve(remaining - m)
                if dw < best_dw:
                    best_dw, best_sel = dw, [m] + sel
        memo[remaining] = (best_dw, best_sel)
        return memo[remaining]

    dw, sel = solve(frozenset(cards))
    return dw, [sorted(m) for m in sel]


class GameSession:
    """One human-vs-model game, driven through the tested SB3 wrapper."""

    def __init__(self, models, default_key):
        self.lock = threading.Lock()
        self.models = models               # {key: loaded PPO model}
        self.opponent_key = default_key
        self.wrapper = None
        self.new_game(default_key)

    def _build_wrapper(self, key):
        self.opponent_key = key
        self.wrapper = GinRummySB3Wrapper(
            opponent_policy=opponent_factory(key, self.models),
            randomize_position=True, turns_limit=200, curriculum_manager=None)

    def new_game(self, opponent_key=None):
        key = opponent_key if opponent_key in OPPONENTS else self.opponent_key
        self._build_wrapper(key)
        obs, _ = self.wrapper.reset()
        self.obs = obs
        self.done = False
        self.result = None
        self._opp_snapshot = []        # last opponent hand seen while in play
        self._human_snapshot = []      # last human hand (for end-of-game deadwood)
        self.last_reward = 0.0
        self.events = []               # opponent moves during the last step (for UI)
        self._auto_advance()
        if not self.done:
            self._set_turn_message()

    def act(self, action: int):
        mask = self.obs["action_mask"]
        if action < 0 or action >= len(mask) or not mask[action]:
            return False, "That move isn't legal right now."
        self.events = []
        try:
            _, opp_before = self._player_hands()
        except Exception:
            opp_before = []
        try:
            # the discard top BEFORE this move -- this is the card revealed
            # underneath if the opponent then takes the card we're discarding.
            under_top = self._top_idx()
        except Exception:
            under_top = None
        obs, reward, done, trunc, _ = self.wrapper.step(action)
        self.obs = obs
        self.last_reward = float(reward)
        try:
            if not (done or trunc):
                self.events = self._opponent_events(action, set(opp_before), under_top)
        except Exception:
            self.events = []
        if done or trunc:
            self.done = True
            self._finish(float(reward))
            return True, None
        self._auto_advance()
        if not self.done:
            self._set_turn_message()
        return True, None

    # The env occasionally hands the player a forced "score" decision (actions
    # 0/1) that carries no strategy; the trained model just predicts through it,
    # so we auto-advance the human past it to the next real choice / game end.
    def _auto_advance(self):
        SCORE = (0, 1)
        for _ in range(40):
            if self.done:
                return
            mask = self.obs["action_mask"]
            legal = [i for i in range(len(mask)) if mask[i]]
            if not legal or any(a not in SCORE for a in legal):
                return
            obs, reward, done, trunc, _ = self.wrapper.step(legal[0])
            self.obs = obs
            self.last_reward = float(reward)
            if done or trunc:
                self.done = True
                self._finish(float(reward))
                return

    def _set_turn_message(self):
        mask = self.obs["action_mask"]
        if mask[2] or mask[3]:
            self.message = "Your turn — draw a card."
        else:
            self.message = "Your turn — discard, knock, or go for gin."

    def _finish(self, reward: float):
        if reward > 1e-9:
            self.result, self.message = "win", "You win! 🎉"
        elif reward < -1e-9:
            self.result, self.message = "loss", "You lost this round."
        else:
            self.result, self.message = "draw", "Round over — a draw."
        # The reveal uses the last snapshot of the opponent's hand taken while it
        # was still in play (the env clears hands at termination).

    def _player_hands(self):
        """(human_hand, opponent_hand) as sorted PettingZoo card-index lists,
        read from the underlying RLCard game state. Empty lists if unavailable."""
        try:
            players = self.wrapper.env.unwrapped.env.game.round.players

            def pz(p):
                return sorted(_RLCARD_SUITS.index(c.suit) * 13
                              + _RLCARD_RANKS.index(c.rank) for c in p.hand)

            h_pid = int(self.wrapper.training_agent.split("_")[1])
            o_pid = int(self.wrapper.opponent_agent.split("_")[1])
            return pz(players[h_pid]), pz(players[o_pid])
        except Exception:
            return [], []

    def _top_idx(self):
        board = np.asarray(self.obs["observation"])
        t = np.where(board[1] == 1)[0]
        return int(t[0]) if len(t) else None

    def _opponent_events(self, action, opp_before, under_top=None):
        """Infer what the opponent did this turn (draw source + discard) by
        diffing its hand, so the UI can animate it. Only meaningful after a human
        discard/knock (the opponent doesn't move after a draw). under_top is the
        discard top before this move -- the card revealed if the opponent takes
        the card we just discarded."""
        is_discard = 6 <= action <= 57
        is_knock = 58 <= action <= 109
        if not (is_discard or is_knock):
            return []
        human_discard = (action - 6) if is_discard else (action - 58)
        _, opp_after = self._player_hands()
        opp_after = set(opp_after)
        drawn = list(opp_after - opp_before)
        top_after = self._top_idx()
        events = []
        if len(drawn) == 1:
            d = drawn[0]
            source = "discard" if d == human_discard else "stock"
            ev = {"type": "opp_draw", "source": source, "card": card_obj(d)}
            if source == "discard":
                # what the UI should show on the pile once the top is taken
                ev["under"] = card_obj(under_top) if under_top is not None else None
            events.append(ev)
        elif top_after is not None and top_after != human_discard:
            # hand unchanged -> the opponent drew a card from stock and discarded
            # that SAME card; surface it so the UI still animates the fly-in/out
            # (otherwise card=None and the draw animation is silently skipped).
            events.append({"type": "opp_draw", "source": "stock", "card": card_obj(top_after)})
        else:
            events.append({"type": "opp_draw", "source": "stock", "card": None})
        if top_after is not None and top_after != human_discard:
            events.append({"type": "opp_discard", "card": card_obj(top_after)})
        return events


def build_view(session: GameSession) -> dict:
    obs = session.obs
    mask = obs["action_mask"]
    board = np.asarray(obs["observation"])
    hand = sorted(int(i) for i in np.where(board[0] == 1)[0])
    top = np.where(board[1] == 1)[0]
    top_discard = int(top[0]) if len(top) else None
    known = sorted(int(i) for i in np.where(board[2] == 1)[0]) if board.shape[0] > 2 else []

    legal = {
        "draw_stock": bool(mask[2]),
        "take_discard": bool(mask[3]),
        "gin": bool(mask[5]),
        "declare_dead": bool(mask[4]),
        "discard": [i for i in range(52) if mask[6 + i]],
        "knock": [i for i in range(52) if mask[58 + i]],
    }
    if session.done:
        phase = "over"
    elif legal["draw_stock"] or legal["take_discard"]:
        phase = "draw"
    else:
        phase = "discard"

    # True hands from the RLCard state, snapshotted so we can still show them
    # after the env clears the live hands at termination.
    h_hand, o_hand = session._player_hands()
    if h_hand:
        session._human_snapshot = h_hand
    if o_hand:
        session._opp_snapshot = o_hand

    # During play use the live obs hand for the player; at the end the env has
    # cleared it, so fall back to the snapshot.
    player_hand = (session._human_snapshot or hand) if session.done else hand
    opp_hand = session._opp_snapshot if session.done else o_hand

    deadwood, melds = best_melds_deadwood(player_hand)
    opp_deadwood, opp_melds = best_melds_deadwood(opp_hand) if opp_hand else (None, [])

    return {
        "hand": [card_obj(i) for i in player_hand],
        "hand_count": len(player_hand),
        "top_discard": card_obj(top_discard) if top_discard is not None else None,
        "discard_known": [card_obj(i) for i in known],
        "phase": phase,
        "legal": legal,
        "deadwood": deadwood,
        "melds": melds,
        "done": session.done,
        "result": session.result,
        "message": session.message,
        "opponent_key": session.opponent_key,
        "opponent_label": OPPONENTS[session.opponent_key]["label"],
        "opponent_count": len(opp_hand) if opp_hand else 10,
        # debug "see opponent" view (live, exact); only while in play
        "opponent_hand_live": ([card_obj(i) for i in o_hand]
                               if (o_hand and not session.done) else None),
        "opponent_melds": opp_melds,   # melded (matched) opponent cards to highlight
        # end-of-game reveal + both deadwoods
        "opponent_reveal": ([card_obj(i) for i in session._opp_snapshot]
                            if session.done else None),
        "opponent_deadwood": opp_deadwood if session.done else None,
        "events": session.events,   # opponent moves during the last step (for UI)
    }


# --- HTTP layer -------------------------------------------------------------
_STATIC = {
    "/": (os.path.join(WEB_DIR, "index.html"), "text/html; charset=utf-8"),
    "/index.html": (os.path.join(WEB_DIR, "index.html"), "text/html; charset=utf-8"),
    "/style.css": (os.path.join(WEB_DIR, "style.css"), "text/css; charset=utf-8"),
    "/app.js": (os.path.join(WEB_DIR, "app.js"), "application/javascript; charset=utf-8"),
}


def _make_handler(session: GameSession):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # quieter console
            pass

        def _send_json(self, obj, code=200):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_file(self, path, ctype):
            if not os.path.isfile(path):
                self.send_error(404, "not found")
                return
            with open(path, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _serve_image(self, base_dir, name):
            # Path-traversal safe: only a bare filename, must be a .png in base_dir.
            name = os.path.basename(name)
            if not name.endswith(".png"):
                self.send_error(404, "not found")
                return
            self._send_file(os.path.join(base_dir, name), "image/png")

        def do_GET(self):
            try:
                path = self.path.split("?", 1)[0]
                if path in _STATIC:
                    self._send_file(*_STATIC[path])
                elif path.startswith("/deck_images/"):
                    self._serve_image(DECK_DIR, path[len("/deck_images/"):])
                elif path.startswith("/images/"):
                    self._serve_image(IMAGES_DIR, path[len("/images/"):])
                elif path == "/api/opponents":
                    self._send_json([
                        {"key": k, "label": v["label"], "stat": v["stat"],
                         "emoji": v.get("emoji", ""), "desc": v.get("desc", "")}
                        for k, v in OPPONENTS.items()
                    ])
                elif path == "/api/state":
                    with session.lock:
                        self._send_json(build_view(session))
                elif path == "/favicon.ico":
                    self.send_response(204)         # no icon; avoid a console 404
                    self.end_headers()
                else:
                    self.send_error(404, "not found")
            except Exception as exc:  # never 500-crash the loop
                self._send_json({"error": str(exc)}, code=500)

        def _read_json(self):
            length = int(self.headers.get("Content-Length", 0) or 0)
            if length == 0:
                return {}
            return json.loads(self.rfile.read(length) or b"{}")

        def do_POST(self):
            try:
                path = self.path.split("?", 1)[0]
                if path == "/api/new_game":
                    data = self._read_json()
                    with session.lock:
                        session.new_game(data.get("opponent"))
                        self._send_json(build_view(session))
                elif path == "/api/action":
                    data = self._read_json()
                    with session.lock:
                        if session.done:
                            self._send_json({"error": "game over; start a new game"},
                                            code=400)
                            return
                        ok, err = session.act(int(data.get("action", -1)))
                        if not ok:
                            self._send_json({"error": err}, code=400)
                            return
                        self._send_json(build_view(session))
                else:
                    self.send_error(404, "not found")
            except Exception as exc:
                self._send_json({"error": str(exc)}, code=500)

    return Handler


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    models = {}
    for key, spec in OPPONENTS.items():
        if spec["type"] != "ppo":
            print(f"Ready '{key}' opponent: {spec['label']} (code agent)")
            continue
        path = opponent_path(key)
        if not os.path.isfile(path):
            print(f"  [skip] '{key}': model not found at {path}")
            continue
        print(f"Loading '{key}' opponent: {path}")
        try:
            models[key] = PPO.load(path, device="cpu")
        except Exception:                       # curriculum winners may be TRPO-trained
            from sb3_contrib import TRPO
            models[key] = TRPO.load(path, device="cpu")
    # Drop any PPO opponents whose weights are missing so the UI never offers them.
    for key in [k for k, s in OPPONENTS.items()
                if s["type"] == "ppo" and k not in models]:
        del OPPONENTS[key]
    print("Models loaded. Starting a game session...")
    session = GameSession(models, DEFAULT_OPPONENT)

    httpd = ThreadingHTTPServer((args.host, args.port), _make_handler(session))
    url = f"http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}"
    print("=" * 60)
    print(f"  Gin Rummy vs PPO is running at:  {url}")
    print("  Open that URL in your browser. Ctrl-C to stop.")
    print("=" * 60)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        httpd.shutdown()


if __name__ == "__main__":
    main()
