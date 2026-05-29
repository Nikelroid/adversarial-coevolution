"""Client-side smoke test for a running master (run AFTER the master is up).

Fires a cold request (cache miss -> real vLLM generate), then the identical
request again (cache hit -> instant), then a couple more, and prints per-call
latency, the ``cached`` flag, and the server's /stats. This is the "1000-step
games in seconds after the cache warms" check from the architecture doc, scaled
down to a handful of calls.

    python -m llm.master_smoke --url http://localhost:11434
"""
from __future__ import annotations

import argparse
import time

import requests

# A couple of plausible Gin-Rummy-style prompts; the second is a repeat of the
# first to force a guaranteed cache hit.
PROMPTS = [
    "You are an expert Gin Rummy player.\nYour hand: Ace of Spades, 2 of Spades, "
    "3 of Spades, 7 of Hearts, 7 of Diamonds.\nTop of discard: 7 of Clubs.\n"
    "Valid actions:\n- draw from stock\n- pick top card from discard pile (7 of Clubs)\n"
    "Reply with ONLY the action string on the last line.",
    "You are an expert Gin Rummy player.\nYour hand: 5 of Clubs, 6 of Clubs, "
    "7 of Clubs, King of Hearts, 2 of Diamonds.\nValid actions:\n- discard King of Hearts\n"
    "- discard 2 of Diamonds\nReply with ONLY the action string on the last line.",
]


def call(url: str, model: str, prompt: str, max_tokens: int) -> dict:
    t0 = time.time()
    r = requests.post(
        f"{url}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.7, "num_predict": max_tokens},
        },
        timeout=600,
    )
    r.raise_for_status()
    body = r.json()
    body["_wall_ms"] = (time.time() - t0) * 1000.0
    return body


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:11434")
    p.add_argument("--model", default="opponent")
    p.add_argument("--max-tokens", type=int, default=256)
    args = p.parse_args()

    print(f"health: {requests.get(args.url + '/health', timeout=10).json()}")

    # Sequence: cold[0], warm[0] (repeat -> hit), cold[1], warm[1] (repeat -> hit)
    sequence = [PROMPTS[0], PROMPTS[0], PROMPTS[1], PROMPTS[1]]
    for i, prompt in enumerate(sequence):
        body = call(args.url, args.model, prompt, args.max_tokens)
        tag = "HIT " if body.get("cached") else "miss"
        print(f"[{i}] {tag} wall={body['_wall_ms']:7.1f}ms "
              f"eval={body.get('eval_ms', 0):7.1f}ms "
              f"resp={body['response'][:60]!r}")

    print(f"stats: {requests.get(args.url + '/stats', timeout=10).json()}")


if __name__ == "__main__":
    main()
