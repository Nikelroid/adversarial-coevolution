"""Model-less stand-in for llm.worker, to test the master/router on a GPU-less
node. It honours the same /generate + /health contract and registers in the
shared-FS registry, but returns a canned move instead of running a model. Used
only by the router integration check (llm/router_test.sh); never in production.
"""
from __future__ import annotations

import argparse
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from llm import discovery


class GenRequest(BaseModel):
    prompt: str
    temperature: float = 0.7
    max_tokens: int = 256


def build(args: argparse.Namespace) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if args.register:
            discovery.write_worker(args.worker_id, discovery.this_host(),
                                   args.port, args.model)
        yield
        if args.register:
            discovery.remove_worker(args.worker_id)

    app = FastAPI(lifespan=lifespan)

    @app.post("/generate")
    async def generate(req: GenRequest):
        # Deterministic canned reply so the cache hit/miss is observable.
        return {"text": "draw from stock", "model": args.model, "eval_ms": 1.0}

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": args.model, "outstanding": 0}

    return app


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8077)
    p.add_argument("--model", default="fake-olmoe")
    p.add_argument("--worker-id", default=None)
    p.add_argument("--no-register", dest="register", action="store_false")
    p.set_defaults(register=True)
    args = p.parse_args()
    args.worker_id = args.worker_id or discovery.default_worker_id(args.port)
    uvicorn.run(build(args), host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
