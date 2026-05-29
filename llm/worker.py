"""Per-GPU generation worker for the Phase-2 distributed LLM stack.

One of these runs per GPU (``slurm/worker.slurm`` is a V100 array job). On
startup it loads the model, registers itself in the shared-FS registry
(:mod:`llm.discovery`) so the master can find and route to it, and serves:

    POST /generate   {prompt, temperature, max_tokens} -> {text, eval_ms, model}
    GET  /health     {status, model, outstanding}
    GET  /info       full config

A background task refreshes the registry heartbeat; on shutdown the worker
removes its registry file. The master owns the cache and the queue -- the worker
is a stateless generator.

Two backends (``--backend``):

  hf    transformers ``model.generate`` (default). Works with the installed
        transformers 5.x; generation is serialized per worker (one at a time,
        run in a threadpool so the event loop stays free). Throughput comes from
        the worker fan-out + the master cache, not from in-worker batching.
  vllm  vLLM ``AsyncLLMEngine`` with continuous batching. Far higher throughput,
        but the installed vLLM 0.8.5 is incompatible with transformers 5.7.0
        (it calls the removed ``all_special_tokens_extended``); use once a
        compatible vLLM/transformers pair is available.

Run (on a GPU node):
    python -m llm.worker --preset olmoe --port 8001            # hf backend
    python -m llm.worker --preset olmoe --port 8001 --backend vllm

torch/transformers/vllm are imported lazily inside the lifespan so this module
stays importable (route/arg check) on a GPU-less node.
"""
from __future__ import annotations

import argparse
import asyncio
import itertools
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from llm import discovery

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("llm.worker")

# dtype="auto" resolves to fp16 on pre-Ampere (V100, CC 7.0, no bf16) and bf16 on
# Ampere+. qwen3-30b needs 2 GPUs (tp=2); the others fit one.
MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
    "olmoe": dict(model="allenai/OLMoE-1B-7B-0125-Instruct", tensor_parallel_size=1,
                  max_model_len=4096, dtype="auto", gpu_memory_utilization=0.90),
    # Strong, fast, ungated 7B instruct — the doc's "7B opponent" target. fp16 on
    # a V100 is ~15GB, fits one 32GB card; much stronger than OLMoE-1B as a teacher.
    "qwen2.5-7b": dict(model="Qwen/Qwen2.5-7B-Instruct", tensor_parallel_size=1,
                       max_model_len=8192, dtype="auto", gpu_memory_utilization=0.90),
    "qwen3-30b": dict(model="Qwen/Qwen3-30B-A3B", tensor_parallel_size=2,
                      max_model_len=8192, dtype="auto", gpu_memory_utilization=0.90),
    "gpt-oss-20b": dict(model="openai/gpt-oss-20b", tensor_parallel_size=1,
                        max_model_len=8192, dtype="auto", gpu_memory_utilization=0.92),
}

DEFAULT_MAX_OUTPUT_TOKENS = 1024
HEARTBEAT_SECONDS = 15


def _resolve_dtype(requested: str) -> str:
    """Pick a GPU-safe dtype. V100 (CC 7.0) has no bf16, so anything that would
    land on bf16 there is downgraded to fp16; Ampere+ keeps bf16."""
    import torch

    major = torch.cuda.get_device_capability(0)[0] if torch.cuda.is_available() else 0
    if requested in ("auto", "bfloat16"):
        resolved = "bfloat16" if major >= 8 else "float16"
        if requested == "bfloat16" and resolved != "bfloat16":
            log.warning("GPU CC %d.x lacks bf16; downgrading dtype to float16", major)
        return resolved
    return requested


class GenRequest(BaseModel):
    prompt: str
    temperature: float = 0.7
    max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS


class GenResponse(BaseModel):
    text: str
    model: str
    eval_ms: float = 0.0


# --- HF backend -------------------------------------------------------------
class _HFEngine:
    """transformers model.generate behind an asyncio.Lock (serialized, threaded)."""

    def __init__(self, model, tokenizer, has_chat_template: bool):
        import torch  # noqa: F401

        self.model = model
        self.tokenizer = tokenizer
        self.has_chat_template = has_chat_template
        self.device = next(model.parameters()).device
        self.lock = asyncio.Lock()

    def _render(self, prompt: str) -> str:
        if self.has_chat_template:
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True)
        return prompt

    def _generate_sync(self, text: str, temperature: float, max_tokens: int) -> str:
        import torch

        tok = self.tokenizer
        inputs = tok(text, return_tensors="pt").to(self.device)
        do_sample = temperature > 0.01
        kwargs = dict(
            max_new_tokens=max_tokens,
            do_sample=do_sample,
            pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
        )
        if do_sample:
            kwargs["temperature"] = temperature
        with torch.no_grad():
            out = self.model.generate(**inputs, **kwargs)
        new = out[0][inputs["input_ids"].shape[1]:]
        return tok.decode(new, skip_special_tokens=True).strip()

    async def generate(self, prompt: str, temperature: float, max_tokens: int) -> str:
        text = self._render(prompt)
        async with self.lock:  # serialize: HF generate is not concurrency-safe
            return await asyncio.to_thread(self._generate_sync, text, temperature,
                                           max_tokens)


# --- vLLM backend -----------------------------------------------------------
class _VLLMEngine:
    """vLLM AsyncLLMEngine with continuous batching across concurrent requests."""

    def __init__(self, engine, tokenizer, SamplingParams, has_chat_template: bool):
        self.engine = engine
        self.tokenizer = tokenizer
        self.SamplingParams = SamplingParams
        self.has_chat_template = has_chat_template
        self.counter = itertools.count()

    def _render(self, prompt: str) -> str:
        if self.has_chat_template:
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True)
        return prompt

    async def generate(self, prompt: str, temperature: float, max_tokens: int) -> str:
        sp = self.SamplingParams(temperature=max(temperature, 0.0), max_tokens=max_tokens)
        text = self._render(prompt)
        rid = f"req-{next(self.counter)}"
        final = None
        async for out in self.engine.generate(text, sp, rid):
            final = out
        return final.outputs[0].text.strip() if final else ""


def _load_hf_engine(cfg: argparse.Namespace) -> _HFEngine:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved = _resolve_dtype(cfg.dtype)
    torch_dtype = torch.float16 if resolved == "float16" else torch.bfloat16
    log.info("[hf] loading %s (dtype=%s)", cfg.model, resolved)
    tok = AutoTokenizer.from_pretrained(cfg.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model, torch_dtype=torch_dtype, device_map="auto",
        trust_remote_code=True)
    model.eval()
    return _HFEngine(model, tok, tok.chat_template is not None)


def _load_vllm_engine(cfg: argparse.Namespace) -> _VLLMEngine:
    from transformers import AutoTokenizer
    from vllm import AsyncLLMEngine, SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs

    resolved = _resolve_dtype(cfg.dtype)
    log.info("[vllm] loading %s (tp=%d, dtype=%s)", cfg.model,
             cfg.tensor_parallel_size, resolved)
    tok = AutoTokenizer.from_pretrained(cfg.model, trust_remote_code=True)
    engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
        model=cfg.model, tensor_parallel_size=cfg.tensor_parallel_size,
        dtype=resolved, max_model_len=cfg.max_model_len,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        trust_remote_code=True, disable_log_requests=True))
    return _VLLMEngine(engine, tok, SamplingParams, tok.chat_template is not None)


def _build_app(cfg: argparse.Namespace) -> FastAPI:

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        t0 = time.time()
        app.state.engine = (_load_vllm_engine(cfg) if cfg.backend == "vllm"
                            else _load_hf_engine(cfg))
        app.state.outstanding = 0
        log.info("model ready in %.1fs | backend=%s | chat_template=%s",
                 time.time() - t0, cfg.backend, app.state.engine.has_chat_template)

        host = discovery.this_host()
        if cfg.register:
            discovery.write_worker(cfg.worker_id, host, cfg.port, cfg.model,
                                   backend=cfg.backend)
            log.info("registered worker id=%s at %s:%d", cfg.worker_id, host, cfg.port)

        async def _heartbeat():
            while True:
                await asyncio.sleep(HEARTBEAT_SECONDS)
                try:
                    discovery.heartbeat_worker(cfg.worker_id)
                except Exception as exc:  # noqa: BLE001
                    log.warning("heartbeat failed: %s", exc)

        hb_task = asyncio.create_task(_heartbeat()) if cfg.register else None
        try:
            yield
        finally:
            if hb_task is not None:
                hb_task.cancel()
            if cfg.register:
                discovery.remove_worker(cfg.worker_id)
                log.info("deregistered worker id=%s", cfg.worker_id)

    app = FastAPI(lifespan=lifespan)

    @app.post("/generate", response_model=GenResponse)
    async def generate(req: GenRequest):
        max_tokens = min(req.max_tokens, cfg.max_output_tokens)
        app.state.outstanding += 1
        t0 = time.time()
        try:
            text = await app.state.engine.generate(req.prompt, req.temperature,
                                                    max_tokens)
        finally:
            app.state.outstanding -= 1
        return GenResponse(text=text, model=cfg.model,
                           eval_ms=(time.time() - t0) * 1000.0)

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": cfg.model, "backend": cfg.backend,
                "outstanding": app.state.outstanding, "worker_id": cfg.worker_id}

    @app.get("/info")
    async def info():
        return {"worker_id": cfg.worker_id, "model": cfg.model,
                "backend": cfg.backend, "host": discovery.this_host(),
                "port": cfg.port}

    return app


def _parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", default=os.environ.get("WORKER_BACKEND", "hf"),
                   choices=["hf", "vllm"])
    p.add_argument("--preset", default=os.environ.get("WORKER_PRESET", "olmoe"),
                   choices=list(MODEL_PRESETS))
    p.add_argument("--model", default=None)
    p.add_argument("--tensor-parallel", type=int, default=None)
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument("--dtype", default=None)
    p.add_argument("--gpu-memory-utilization", type=float, default=None)
    p.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=int(os.environ.get("WORKER_PORT", 8001)))
    p.add_argument("--worker-id", default=None)
    p.add_argument("--no-register", dest="register", action="store_false")
    p.set_defaults(register=True)
    args = p.parse_args(argv)

    preset = MODEL_PRESETS[args.preset]
    args.model = args.model or preset["model"]
    args.tensor_parallel_size = args.tensor_parallel or preset["tensor_parallel_size"]
    args.max_model_len = args.max_model_len or preset["max_model_len"]
    args.dtype = args.dtype or preset["dtype"]
    args.gpu_memory_utilization = (
        args.gpu_memory_utilization or preset["gpu_memory_utilization"]
    )
    args.worker_id = args.worker_id or discovery.default_worker_id(args.port)
    return args


def main(argv: Optional[list] = None) -> None:
    import uvicorn

    cfg = _parse_args(argv)
    log.info("worker config: %s", vars(cfg))
    app = _build_app(cfg)
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="info")


if __name__ == "__main__":
    main()
