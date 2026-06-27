"""Flat Monte-Carlo (PIMC-style) search agent for Gin Rummy -- a non-learned SEARCH baseline.

At each decision it evaluates every legal action by cloning the underlying RLCard game, applying the
action, and rolling out random legal play to terminal K times; it picks the action with the best
mean zero-sum payoff (my payoff minus the opponent's). This is a determinized / perfect-information
Monte-Carlo (PIMC) search agent in the spirit of Cowling et al. 2012 and Long et al. 2010 -- a
strong, classic baseline that complements the learned RL agents and the fixed expert. Rollouts run
from the true game state (an oracle upper bound on what determinized search achieves); we report it
as such. It only ever returns a legal action (intersected with the PettingZoo mask).

Plugs into the same eval harness as GoldStandardAgent: __init__(env), set_player(p), do_action().

Env knobs: ISMCTS_ROLLOUTS (default 20), ISMCTS_MAXDEPTH (default 400).
"""
from __future__ import annotations

import copy
import os
import random

import numpy as np


class ISMCTSAgent:
    def __init__(self, env, rollouts=None, max_depth=None, seed=0):
        self.env = env
        self.player = None
        self.k = int(rollouts if rollouts is not None else os.environ.get("ISMCTS_ROLLOUTS", 20))
        self.max_depth = int(max_depth if max_depth is not None else os.environ.get("ISMCTS_MAXDEPTH", 400))
        self.rng = random.Random(seed)

    def set_player(self, p):
        self.player = p

    def _game(self):
        return self.env.unwrapped.env.game

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
            game.step(self.rng.choice(go) if go else self.rng.choice(acts))
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
                gc = copy.deepcopy(g)
                if not self._apply_id(gc, aid):
                    tot += -1e9  # action not applicable in clone (shouldn't happen)
                    continue
                tot += self._rollout_value(gc, me)
            v = tot / self.k
            if v > best_v:
                best_v, best = v, aid
        return best
