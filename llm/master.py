"""Phase-2 master: CPU-only router with a cache + admission queue over a pool of
GPU workers.

This process holds NO model. It is the single point every RL env subprocess
talks to (Ollama ``/api/generate`` protocol, so ``llm.api.OllamaAPI`` is a
drop-in client), and it:

  1. checks the in-process :class:`llm.cache.PromptCache` -- a hit returns with
     zero GPU work, and because all env workers share this one process the cache
     deduplicates queries across the whole training run;
  2. on a miss, admits the request to a queue (per-worker concurrency cap =
     backpressure), picks the least-outstanding healthy worker, and forwards the
     generation over HTTP;
  3. caches the response and returns it.

Workers are discovered through the shared-FS registry (:mod:`llm.discovery`):
the pool scans ``runtime/workers/`` every few seconds, health-checks each entry,
and routes only to live ones -- so GPU array tasks can come and go freely.

Request<->response correlation needs no explicit tag table: each incoming
request is its own coroutine that awaits its own forwarded response. Every
request still gets a UUID for logging.

Run (on a CPU node):
    python -m llm.master --port 11434
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from llm import discovery
from llm.cache import build_cache

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("llm.master")

DEFAULT_MAX_OUTPUT_TOKENS = 1024


# --- Ollama-compatible wire types ------------------------------------------
class OllamaPayload(BaseModel):
    model: str = ""
    prompt: str
    stream: Optional[bool] = False
    options: Optional[Dict[str, Any]] = Field(default_factory=dict)


class OllamaResponse(BaseModel):
    model: str
    response: str
    done: bool = True
    cached: bool = False
    eval_ms: float = 0.0


# --- worker pool -----------------------------------------------------------
class WorkerState:
    __slots__ = ("id", "host", "port", "model", "outstanding", "healthy",
                 "fails", "cap")

    def __init__(self, id: str, host: str, port: int, model: str, cap: int):
        self.id = id
        self.host = host
        self.port = port
        self.model = model
        self.cap = cap
        self.outstanding = 0
        self.healthy = False
        self.fails = 0

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def free(self) -> bool:
        return self.healthy and self.outstanding < self.cap


class WorkerPool:
    """Tracks live workers (via the shared-FS registry + health checks) and hands
    out the least-loaded one. The per-worker concurrency cap is the queue: when
    every worker is full, :meth:`acquire` waits on a condition until capacity
    frees or the request times out."""

    def __init__(self, http: httpx.AsyncClient, cfg: argparse.Namespace):
        self.http = http
        self.cfg = cfg
        self.workers: Dict[str, WorkerState] = {}
        self.cond = asyncio.Condition()
        self.waiting = 0
        self._scan_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        self._scan_task = asyncio.create_task(self._scan_loop())

    async def stop(self) -> None:
        if self._scan_task is not None:
            self._scan_task.cancel()

    async def _scan_loop(self) -> None:
        while True:
            try:
                await self._refresh()
            except Exception as exc:  # noqa: BLE001
                log.warning("pool refresh failed: %s", exc)
            await asyncio.sleep(self.cfg.scan_interval)

    async def _refresh(self) -> None:
        entries = {w["id"]: w
                   for w in discovery.list_workers(max_age=self.cfg.stale_age)}
        # add newly-registered, drop vanished/stale
        for wid, info in entries.items():
            if wid not in self.workers:
                self.workers[wid] = WorkerState(
                    wid, info["host"], int(info["port"]),
                    info.get("model", "?"), self.cfg.per_worker_concurrency)
                log.info("worker joined: %s at %s:%s (%s)", wid, info["host"],
                         info["port"], info.get("model"))
        for wid in list(self.workers):
            if wid not in entries:
                log.info("worker left: %s", wid)
                del self.workers[wid]

        # health-check all in parallel
        async def _hc(w: WorkerState) -> None:
            try:
                r = await self.http.get(f"{w.url}/health",
                                        timeout=self.cfg.health_timeout)
                ok = r.status_code == 200
            except Exception:
                ok = False
            if ok:
                w.fails = 0
                w.healthy = True
            else:
                w.fails += 1
                if w.fails >= self.cfg.max_health_fails:
                    w.healthy = False

        if self.workers:
            await asyncio.gather(*(_hc(w) for w in self.workers.values()))

        # waiters re-evaluate after every refresh (cheap; correctness over churn)
        async with self.cond:
            self.cond.notify_all()

    def _pick(self) -> Optional[WorkerState]:
        free = [w for w in self.workers.values() if w.free]
        return min(free, key=lambda w: w.outstanding) if free else None

    def fleet_model(self) -> Optional[str]:
        for w in self.workers.values():
            if w.healthy:
                return w.model
        for w in self.workers.values():
            return w.model
        return None

    async def acquire(self, timeout: float) -> Optional[WorkerState]:
        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout
        async with self.cond:
            while True:
                w = self._pick()
                if w is not None:
                    w.outstanding += 1
                    return w
                remaining = deadline - loop.time()
                if remaining <= 0:
                    return None
                self.waiting += 1
                try:
                    await asyncio.wait_for(self.cond.wait(), timeout=remaining)
                except asyncio.TimeoutError:
                    return None
                finally:
                    self.waiting -= 1

    async def release(self, w: WorkerState) -> None:
        async with self.cond:
            w.outstanding = max(0, w.outstanding - 1)
            self.cond.notify_all()

    async def mark_failure(self, w: WorkerState) -> None:
        async with self.cond:
            w.fails += 1
            if w.fails >= self.cfg.max_health_fails:
                w.healthy = False
            self.cond.notify_all()

    def snapshot(self) -> Dict[str, Any]:
        return {
            "n_healthy": sum(1 for w in self.workers.values() if w.healthy),
            "n_total": len(self.workers),
            "waiting": self.waiting,
            "workers": [
                {"id": w.id, "host": w.host, "port": w.port, "model": w.model,
                 "healthy": w.healthy, "outstanding": w.outstanding,
                 "cap": w.cap, "fails": w.fails}
                for w in self.workers.values()
            ],
        }


def _build_app(cfg: argparse.Namespace) -> FastAPI:

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.http = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=cfg.max_connections,
                                max_keepalive_connections=cfg.max_connections),
        )
        app.state.cache = build_cache(backend=cfg.cache_backend, maxsize=cfg.cache_size)
        app.state.pool = WorkerPool(app.state.http, cfg)
        await app.state.pool.start()
        host = discovery.this_host()
        discovery.write_master(host, cfg.port, role="router")
        log.info("master up at %s:%d | runtime=%s | cache=%s", host, cfg.port,
                 discovery.runtime_dir(), app.state.cache.stats()["backend"])
        yield
        await app.state.pool.stop()
        await app.state.http.aclose()
        log.info("master down | final cache stats=%s", app.state.cache.stats())

    app = FastAPI(lifespan=lifespan)

    @app.post("/api/generate", response_model=OllamaResponse)
    async def generate(payload: OllamaPayload):
        pool: WorkerPool = app.state.pool
        cache = app.state.cache

        model = pool.fleet_model() or "unknown"
        hit = cache.get(model, payload.prompt)
        if hit is not None:
            return OllamaResponse(model=model, response=hit, cached=True)

        opts = payload.options or {}
        body = {
            "prompt": payload.prompt,
            "temperature": float(opts.get("temperature", 0.7)),
            "max_tokens": int(opts.get("num_predict", DEFAULT_MAX_OUTPUT_TOKENS)),
        }
        rid = uuid.uuid4().hex[:8]
        last_err: Optional[str] = None

        for attempt in range(cfg.max_retries + 1):
            w = await pool.acquire(cfg.request_timeout)
            if w is None:
                snap = pool.snapshot()
                log.warning("[%s] no worker available (healthy=%d waiting=%d)",
                            rid, snap["n_healthy"], snap["waiting"])
                return JSONResponse(status_code=503,
                                    content={"error": "no available workers"})
            try:
                t0 = time.time()
                r = await app.state.http.post(f"{w.url}/generate", json=body,
                                              timeout=cfg.gen_timeout)
                r.raise_for_status()
                text = r.json().get("text", "")
                cache.set(w.model, payload.prompt, text)
                return OllamaResponse(model=w.model, response=text, cached=False,
                                      eval_ms=(time.time() - t0) * 1000.0)
            except Exception as exc:  # noqa: BLE001
                last_err = str(exc)
                await pool.mark_failure(w)
                log.warning("[%s] worker %s failed (attempt %d/%d): %s", rid,
                            w.id, attempt + 1, cfg.max_retries + 1, last_err)
            finally:
                await pool.release(w)

        return JSONResponse(status_code=502,
                            content={"error": "all workers failed",
                                     "detail": last_err})

    @app.get("/api/tags")
    async def tags():
        return {"models": [{"name": app.state.pool.fleet_model() or "none"}]}

    @app.get("/health")
    async def health():
        snap = app.state.pool.snapshot()
        return {"status": "ok", "n_healthy": snap["n_healthy"],
                "n_total": snap["n_total"], "waiting": snap["waiting"]}

    @app.get("/stats")
    async def stats():
        return {"cache": app.state.cache.stats(), "pool": app.state.pool.snapshot()}

    @app.get("/workers")
    async def workers():
        return app.state.pool.snapshot()

    return app


def _parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=int(os.environ.get("MASTER_PORT", 11434)))
    p.add_argument("--scan-interval", type=float, default=5.0,
                   help="seconds between registry scans + health checks")
    p.add_argument("--stale-age", type=float, default=90.0,
                   help="drop workers whose heartbeat is older than this")
    p.add_argument("--health-timeout", type=float, default=3.0)
    p.add_argument("--gen-timeout", type=float, default=180.0,
                   help="timeout for a single worker /generate call")
    p.add_argument("--request-timeout", type=float, default=120.0,
                   help="max time to wait in the queue for a free worker")
    p.add_argument("--per-worker-concurrency", type=int, default=64,
                   help="max in-flight requests per worker (vLLM batches these)")
    p.add_argument("--max-health-fails", type=int, default=3)
    p.add_argument("--max-retries", type=int, default=1,
                   help="retries on a different worker when one fails")
    p.add_argument("--max-connections", type=int, default=512)
    p.add_argument("--cache-backend", default="auto", choices=["auto", "memory", "redis"])
    p.add_argument("--cache-size", type=int, default=200_000)
    return p.parse_args(argv)


def main(argv: Optional[list] = None) -> None:
    import uvicorn

    cfg = _parse_args(argv)
    log.info("master config: %s", vars(cfg))
    app = _build_app(cfg)
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="info")


if __name__ == "__main__":
    main()
