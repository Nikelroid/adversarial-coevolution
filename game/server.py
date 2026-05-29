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

GAME_DIR = os.path.join(REPO_ROOT, "game")
WEB_DIR = os.path.join(GAME_DIR, "web")
DECK_DIR = os.path.join(GAME_DIR, "deck_images")
IMAGES_DIR = os.path.join(GAME_DIR, "images")
# Opponents the player can choose between, drawn from the Phase-1 sweep:
#   winrate -> run_5 (highest win rate, 99.6% vs random)
#   reward  -> run_2 (highest mean reward, 0.54)
# Both ship in game/model/ so a fresh clone has them.
OPPONENTS = {
    "winrate": {"label": "Highest win rate", "stat": "99.6% wins",
                "file": "ppo_gin_rummy_winrate.zip"},
    "reward": {"label": "Highest reward", "stat": "0.54 avg reward",
               "file": "ppo_gin_rummy_reward.zip"},
}
DEFAULT_OPPONENT = os.environ.get("GIN_OPPONENT", "winrate")


def opponent_path(key: str) -> str:
    return os.path.join(REPO_ROOT, "game", "model", OPPONENTS[key]["file"])

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
            opponent_policy=functools.partial(PPOAgent, model=self.models[key]),
            randomize_position=True, turns_limit=200, curriculum_manager=None)

    def new_game(self, opponent_key=None):
        key = opponent_key if opponent_key in self.models else self.opponent_key
        self._build_wrapper(key)
        obs, _ = self.wrapper.reset()
        self.obs = obs
        self.done = False
        self.result = None
        self._opp_snapshot = []        # last opponent hand seen while in play
        self._human_snapshot = []      # last human hand (for end-of-game deadwood)
        self.last_reward = 0.0
        self._auto_advance()
        if not self.done:
            self._set_turn_message()

    def act(self, action: int):
        mask = self.obs["action_mask"]
        if action < 0 or action >= len(mask) or not mask[action]:
            return False, "That move isn't legal right now."
        obs, reward, done, trunc, _ = self.wrapper.step(action)
        self.obs = obs
        self.last_reward = float(reward)
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
        # end-of-game reveal + both deadwoods
        "opponent_reveal": ([card_obj(i) for i in session._opp_snapshot]
                            if session.done else None),
        "opponent_deadwood": (best_melds_deadwood(session._opp_snapshot)[0]
                              if (session.done and session._opp_snapshot) else None),
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
                        {"key": k, "label": v["label"], "stat": v["stat"]}
                        for k, v in OPPONENTS.items()
                    ])
                elif path == "/api/state":
                    with session.lock:
                        self._send_json(build_view(session))
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
    for key in OPPONENTS:
        path = opponent_path(key)
        if not os.path.isfile(path):
            sys.exit(f"Model not found: {path}")
        print(f"Loading '{key}' opponent: {path}")
        models[key] = PPO.load(path, device="cpu")
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
