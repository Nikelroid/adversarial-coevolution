"""Tests for OllamaAPI suit-symmetry canonicalization (lossless cache key).

Verifies (a) suit-equivalent boards canonicalize to the SAME board, (b) actions
survive permute->invert round-trips, and (c) get_action returns a *real* action
after inverting the canonical choice. No network/LLM needed.

    python -m llm.canonical_test
"""
import numpy as np

from llm.api import OllamaAPI


def _board(hand, top=None, others=()):
    b = np.zeros((4, 52), dtype=np.int8)
    for c in hand:
        b[0, c] = 1
    if top is not None:
        b[1, top] = 1
    for c in others:
        b[2, c] = 1
    return b


def _relabel(board, sigma):
    """Move every card from suit s to suit sigma[s] (a suit relabeling)."""
    out = np.zeros_like(board)
    for c in range(52):
        s, r = c // 13, c % 13
        out[:, sigma[s] * 13 + r] = board[:, c]
    return out


def test_canonical_invariance():
    api = OllamaAPI(model="x")
    # spades A,2,3 ; a heart 7 ; a diamond 7 ; top = a club Jack ; one discard
    board = _board([0, 1, 2, 13 + 6, 26 + 6], top=39 + 10, others=[13 + 1])
    for sigma in ([2, 0, 3, 1], [1, 0, 2, 3], [3, 2, 1, 0]):
        b2 = _relabel(board, sigma)
        c1 = api._permute_board(board, api._canonical_suit_perm(board))
        c2 = api._permute_board(b2, api._canonical_suit_perm(b2))
        assert np.array_equal(c1, c2), f"not invariant under {sigma}"


def test_action_roundtrip():
    api = OllamaAPI(model="x")
    board = _board([0, 1, 2, 13 + 6, 26 + 6], top=39 + 10)
    perm = api._canonical_suit_perm(board)
    for a in [0, 1, 2, 3, 4, 5, 6, 6 + 13, 6 + 51, 58, 58 + 25, 109]:
        assert api._invert_action(api._permute_action(a, perm), perm) == a, a


def test_get_action_inverts():
    api = OllamaAPI(model="x")
    board = _board([0, 1, 2, 13 + 6, 26 + 6], top=39 + 10)
    valid = [2] + [6 + c for c in [0, 1, 2, 13 + 6, 26 + 6]]
    captured = {}

    def fake_generate(prompt, temperature=0.7, max_tokens=8192):
        captured["map"] = dict(api.current_action_map)
        for s, aid in api.current_action_map.items():
            if s.startswith("discard"):
                captured["chosen_canon"] = aid
                return "...reasoning...\n" + s
        return "draw from stock"

    api.generate = fake_generate
    obs = {"observation": board}
    real = api.get_action("State:\n{game_state}\nActions:\n{valid_actions}", obs, valid)
    assert real in valid, (real, valid)                       # legal real action
    perm = api._canonical_suit_perm(board)
    assert real == api._invert_action(captured["chosen_canon"], perm)


if __name__ == "__main__":
    test_canonical_invariance(); print("ok  invariance under suit relabeling")
    test_action_roundtrip(); print("ok  action permute/invert round-trip")
    test_get_action_inverts(); print("ok  get_action inverts to a real action")
    print("CANONICAL_TEST_PASS")
