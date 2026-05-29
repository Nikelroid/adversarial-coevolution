"""Async unit test of WorkerPool (CPU-only): registry discovery + health,
least-outstanding routing, and the admission queue / backpressure when every
worker is at its concurrency cap. Uses an httpx MockTransport so no real worker
or GPU is needed.

    python -m llm.pool_test
"""
from __future__ import annotations

import asyncio
import os
import tempfile

import httpx

from llm import discovery
from llm.master import WorkerPool, _parse_args


def _mock_client() -> httpx.AsyncClient:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(200, json={"text": "draw from stock", "model": "fake"})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def _run() -> None:
    os.environ["GINLLM_RUNTIME_DIR"] = tempfile.mkdtemp()
    client = _mock_client()
    cfg = _parse_args(["--per-worker-concurrency", "2", "--scan-interval", "100"])
    pool = WorkerPool(client, cfg)

    # discovery + health
    discovery.write_worker("w0", "127.0.0.1", 9001, "fake")
    await pool._refresh()
    assert pool.snapshot()["n_healthy"] == 1, pool.snapshot()

    # cap=2: two acquires succeed, the third must block then time out (backpressure)
    w1 = await pool.acquire(5)
    w2 = await pool.acquire(5)
    assert w1 is not None and w2 is not None and w1.outstanding == 2
    w3 = await pool.acquire(0.3)
    assert w3 is None, "expected backpressure when all slots are taken"

    # freeing a slot lets a queued request through
    await pool.release(w1)
    w4 = await pool.acquire(2)
    assert w4 is not None, "should get a slot after release"

    # least-outstanding routing: a fresh idle worker should be preferred
    discovery.write_worker("w1", "127.0.0.1", 9002, "fake")
    await pool._refresh()                      # w0 has 2 outstanding, w1 has 0
    picked = pool._pick()
    assert picked is not None and picked.id == "w1", \
        (picked.id if picked else None, pool.snapshot())

    await client.aclose()
    print("POOL_TEST_PASS")


if __name__ == "__main__":
    asyncio.run(_run())
