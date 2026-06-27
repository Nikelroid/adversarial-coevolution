"""Determinized Monte-Carlo (PIMC / ISMCTS-style) search agent for Gin Rummy -- a non-learned SEARCH
baseline in the spirit of Cowling et al. 2012 and Long et al. 2010.

At each decision it evaluates every legal action by cloning the underlying RLCard game, applying the
action, and rolling out knock-greedy legal play to terminal K times; it picks the action with the
best mean zero-sum payoff (my payoff minus the opponent's). It only ever returns a legal action
(intersected with the PettingZoo mask), and plugs into the same eval harness as GoldStandardAgent:
__init__(env), set_player(p), do_action().

HIDDEN INFORMATION -- two modes (this is the important honesty knob):
  * determinize=True  (DEFAULT, a fair imperfect-information baseline): before EACH rollout the agent
    re-deals the cards it cannot see -- the opponent's hand and the stock pile are reshuffled
    uniformly at random from the unseen pool (every card not in my hand and not in the public discard
    pile). The search therefore NEVER uses the opponent's true cards; it averages over many plausible
    worlds consistent only with what it has observed. (Slightly conservative: it does not track which
    discards the opponent is known to have picked up, so it can only be <= a perfect-tracking ISMCTS.)
  * determinize=False (oracle / perfect-information PIMC): rollouts run from the TRUE state, i.e. the
    search can see the opponent's hand. NOT a fair baseline -- report it only as an oracle UPPER BOUND.

Env knobs: ISMCTS_ROLLOUTS (default 20), ISMCTS_MAXDEPTH (default 400),
ISMCTS_DETERMINIZE (default 1; set 0 for the oracle upper bound).
"""
from __future__ import annotations

import copy
import os
import random

import numpy as np


class ISMCTSAgent:
    def __init__(self, env, rollouts=None, max_depth=None, seed=0, determinize=None):
        self.env = env
        self.player = None
        self.k = int(rollouts if rollouts is not None else os.environ.get("ISMCTS_ROLLOUTS", 20))
        self.max_depth = int(max_depth if max_depth is not None else os.environ.get("ISMCTS_MAXDEPTH", 400))
        self.determinize = (bool(int(os.environ.get("ISMCTS_DETERMINIZE", 1)))
                            if determinize is None else bool(determinize))
        self.rng = random.Random(seed)

    def set_player(self, p):
        self.player = p

    def _game(self):
        return self.env.unwrapped.env.game

    def _determinized_clone(self, g, me):
        """A deepcopy of g with the cards I cannot see (opponent hand + stock pile) re-dealt
        uniformly at random. My own hand, the public discard pile, and game phase are untouched, so
        every action that was legal for me stays legal. This is what makes the search respect the
        information set instead of peeking at the true opponent hand."""
        gc = copy.deepcopy(g)
        opp = 1 - me
        opp_hand = gc.round.players[opp].hand
        stock = gc.round.dealer.stock_pile
        pool = list(opp_hand) + list(stock)          # exactly the unseen cards
        self.rng.shuffle(pool)
        n = len(opp_hand)
        gc.round.players[opp].hand = pool[:n]         # re-deal: random hand of the same size
        gc.round.dealer.stock_pile = pool[n:]         # the rest, in random order
        return gc

    def _legal_mask(self):
        """Legal action ids from the live PettingZoo observation mask (the ground truth)."""
        obs = self.env.observe(self.player) if self.player is not None else None
        if isinstance(obs, dict) and "action_mask" in obs:
            return set(int(a) for a in np.flatnonzero(np.asarray(obs["action_mask"]).reshape(-1)))
        return None

    @staticmethod
    def _apply_id(game, action_id):
        for la in game.judge.get_legal_actions():
            if la.action_id == action_id:
                game.step(la)
                return True
        return False

    def _rollout_value(self, game, me):
        """Knock-greedy legal play to terminal; return zero-sum payoff for `me`. Going-out actions
        (gin id 5, knock ids 58-109) are only LEGAL when deadwood permits, so taking them when
        available is a sound fast rollout policy -- and crucially ends games (random rollouts almost
        never knock, so games drag to the turn limit and the payoff signal is noise)."""
        d = 0
        while not game.is_over() and d < self.max_depth:
            acts = game.judge.get_legal_actions()
            go = [a for a in acts if a.action_id == 5 or 58 <= a.action_id <= 109]
            nongo = [a for a in acts if a.action_id != 5 and not (58 <= a.action_id <= 109)]
            choice = self.rng.choice(go) if go else self.rng.choice(acts)
            try:
                game.step(choice)
            except Exception:
                # RLCard's going-out logic can be inconsistent on some re-dealt (determinized)
                # hands; fall back to a safe non-going-out move so the rollout still terminates.
                if not nongo:
                    break
                game.step(self.rng.choice(nongo))
            d += 1
        pay = game.judge.scorer.get_payoffs(game=game)
        opp = 1 - me
        return float(pay[me]) - float(pay[opp])

    def do_action(self):
        g = self._game()
        me = g.get_player_id()
        mask = self._legal_mask()
        legal_ids = [la.action_id for la in g.judge.get_legal_actions()]
        if mask is not None:
            legal_ids = [a for a in legal_ids if a in mask] or legal_ids
        if not legal_ids:
            raise ValueError("ISMCTSAgent: no legal actions")
        if len(legal_ids) == 1:
            return legal_ids[0]
        best, best_v = legal_ids[0], -1e18
        for aid in legal_ids:
            tot = 0.0
            for _ in range(self.k):
                gc = self._determinized_clone(g, me) if self.determinize else copy.deepcopy(g)
                if not self._apply_id(gc, aid):
                    tot += -1e9  # action not applicable in clone (shouldn't happen)
                    continue
                tot += self._rollout_value(gc, me)
            v = tot / self.k
            if v > best_v:
                best_v, best = v, aid
        return best
