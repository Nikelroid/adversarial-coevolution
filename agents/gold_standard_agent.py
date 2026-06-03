"""Gold-standard expert Gin Rummy agent.

A strong, deterministic baseline built on EXACT optimal meld decomposition
(RLCard's get_best_meld_clusters) plus a principled draw / discard / knock policy:

  * Draw  : take the up-card only if it joins a meld or strictly lowers the best
            achievable deadwood; otherwise draw from the stock.
  * Discard: among legal discards, drop the card that minimises the resulting
            deadwood (tie-break: shed the highest point card).
  * End    : go gin whenever possible; otherwise knock as soon as deadwood is
            within `knock_threshold` (default 10). `hold_for_gin` keeps playing
            for a gin instead of taking a small knock.

The agent reads the real hand from the observation planes (plane 0 = hand,
plane 1 = top of discard) and only ever returns a legal action. It is meant as a
gold-standard opponent / benchmark, and as a strong teacher for later phases.
"""
from functools import lru_cache

import numpy as np

from .agent import Agent
from rlcard.games.gin_rummy.utils import melding
from rlcard.games.gin_rummy.utils import utils as gu
from rlcard.games.gin_rummy.utils.utils import (
    card_from_card_id, get_card_id, get_deadwood_value,
)

# Action-id layout (RLCard / PettingZoo gin_rummy_v4), 110 actions total.
A_SCORE0, A_SCORE1 = 0, 1
A_DRAW, A_PICKUP, A_DEAD, A_GIN = 2, 3, 4, 5
A_DISCARD = 6      # discard card_id  ->  6  + card_id   (ids 6..57)
A_KNOCK = 58       # knock   card_id  ->  58 + card_id   (ids 58..109)


@lru_cache(maxsize=400_000)
def _deadwood_for_ids(ids):
    """Minimum deadwood of a 10-card hand (given as a sorted tuple of card ids)."""
    cards = [card_from_card_id(i) for i in ids]
    clusters = melding.get_best_meld_clusters(hand=cards)
    if not clusters:
        return gu.get_deadwood_count(hand=cards, meld_cluster=[])
    return min(gu.get_deadwood_count(hand=cards, meld_cluster=c) for c in clusters)


def _best_deadwood(card_ids):
    return _deadwood_for_ids(tuple(sorted(card_ids)))


class GoldStandardAgent(Agent):
    def __init__(self, env, knock_threshold: int = 10, hold_for_gin: bool = False):
        super().__init__(env)
        self.player = None
        self.knock_threshold = knock_threshold
        self.hold_for_gin = hold_for_gin

    # -- observation access (works for raw PettingZoo env or a state-API wrapper) --
    def _obs(self):
        if hasattr(self.env, "get_current_state"):
            o, _, _, _, _ = self.env.get_current_state()
        else:
            o, _, _, _, _ = self.env.last()
        return o

    def do_action(self):
        obs = self._obs()
        mask = np.asarray(obs["action_mask"]).reshape(-1)
        legal = set(int(a) for a in np.where(mask == 1)[0])
        if not legal:
            raise ValueError("GoldStandardAgent: no legal actions available")

        planes = np.asarray(obs["observation"])
        if planes.ndim == 2 and planes.shape[0] not in (4, 5):  # tolerate transpose
            planes = planes.T
        hand_ids = [int(c) for c in np.where(planes[0] == 1)[0]]
        top_ids = [int(c) for c in np.where(planes[1] == 1)[0]]

        # Terminal scoring move.
        if A_SCORE0 in legal:
            return A_SCORE0
        if A_SCORE1 in legal:
            return A_SCORE1

        # Draw phase.
        if A_DRAW in legal or A_PICKUP in legal:
            return self._draw_decision(hand_ids, top_ids, legal)

        # Discard / knock / gin phase.
        return self._discard_decision(hand_ids, legal)

    def _draw_decision(self, hand_ids, top_ids, legal):
        if A_PICKUP in legal and top_ids and len(hand_ids) == 10:
            up = top_ids[0]
            hand11 = hand_ids + [up]
            # Take the up-card iff keeping it (and discarding the worst other card)
            # strictly lowers our best achievable deadwood. This already captures
            # "the up-card completes/extends a meld" (melding only scores 10 cards).
            cur = _best_deadwood(hand_ids)
            best_take = min(
                (_best_deadwood([c for c in hand11 if c != d])
                 for d in hand11 if d != up),
                default=cur + 1,
            )
            if best_take < cur:
                return A_PICKUP
        if A_DRAW in legal:
            return A_DRAW
        if A_PICKUP in legal:
            return A_PICKUP
        return min(legal)

    def _discard_decision(self, hand_ids, legal):
        discards = {cid: A_DISCARD + cid for cid in range(52) if (A_DISCARD + cid) in legal}
        knocks = {cid: A_KNOCK + cid for cid in range(52) if (A_KNOCK + cid) in legal}

        # Evaluate the resulting 10-card deadwood for each legal discard.
        best_cid, best_dw, best_shed = None, 10 ** 9, -1
        for cid in discards:
            ten = [c for c in hand_ids if c != cid]
            if len(ten) != 10:
                continue
            dw = _best_deadwood(ten)
            shed = get_deadwood_value(card_from_card_id(cid))
            if dw < best_dw or (dw == best_dw and shed > best_shed):
                best_cid, best_dw, best_shed = cid, dw, shed

        # Gin if the engine offers it.
        if A_GIN in legal:
            return A_GIN
        # A discard that reaches 0 deadwood and can knock == gin.
        if best_cid is not None and best_dw == 0 and best_cid in knocks:
            return knocks[best_cid]

        # Knock as soon as we are within threshold (unless holding for gin).
        if best_cid is not None and best_dw <= self.knock_threshold and knocks:
            if not (self.hold_for_gin and best_dw > 0):
                kbest = min(
                    knocks,
                    key=lambda c: _best_deadwood([x for x in hand_ids if x != c]),
                )
                return knocks[kbest]

        # Otherwise discard the deadwood-minimising card.
        if best_cid is not None:
            return discards[best_cid]
        if A_DEAD in legal:
            return A_DEAD
        return min(legal)
