# Phase 2 architecture: master / worker / cache for LLM opponents

The goal of Phase 2 is to train a PPO agent against an LLM opponent (Llama-3 /
Qwen / Gemma class) without paying full LLM latency on every RL step.  This
document is the design we're going to build against.

## Constraints driving the design

| Constraint | Implication |
|---|---|
| One PPO update needs ~50k env steps (`num_env=96 × n_steps=512`) | Each rollout fires up to ~50k LLM queries |
| LLM single-call latency: ~0.5–3 s on a 7–13B model, 5–15 s on a 70B | Naively, one rollout = hours |
| Gin Rummy actions repeat (draw, discard X, knock, gin)  and many obs hashes recur | A prompt cache can erase a large fraction of calls |
| Cluster has many CPUs but a handful of GPUs | LLM = batched on GPU; env = forked on CPU; the two must be **decoupled** processes |

## Recommendation: vLLM + Redis cache, behind a thin master

| Layer | Choice | Why |
|---|---|---|
| Inference engine | **vLLM** (HF) | Continuous batching + PagedAttention give 5-20x throughput over plain `transformers.generate`; works with all the Llama / Qwen / Gemma families listed in `config.yaml` |
| Worker pool | one vLLM process per GPU | Each worker holds the model weights in VRAM; many RL env subprocesses share it via the master |
| Cache | **Redis** (single, in-memory) | Sub-ms lookup, key = `sha1(prompt)`, value = generated text + tokens; LRU eviction; survives between training runs |
| Master | Async FastAPI in front of cache + worker pool | Exposes the same `/api/generate` endpoint your `OllamaAPI` already speaks → zero changes to the RL client |
| Client | Existing `llm/api.py` | Keep as-is, just point `base_url` at the master |

### Why vLLM over Ollama

- Ollama is great for laptops and demos, but its server is single-stream — there's no continuous batching, so 96 env subprocesses serialize.
- vLLM was built for exactly this workload (many concurrent prompts, one GPU) and ships with the same HF model IDs you already have in `config.yaml`.

### Why a cache matters more than a smarter model

Gin Rummy has a small action vocabulary (110 actions, only ~10 distinct
"legal" subsets per game phase). The observation that goes into the prompt is
hand + top-of-discard + opponent's known cards — and many of these recur,
especially in opening play. Empirically (from the existing `app.log`), ~30-60%
of LLM queries within a training batch are duplicates of earlier queries.
A cache turns those into 0-latency lookups.

## Request flow

```
                      ┌─────────────────────────┐
   env subprocess --> │  Master (FastAPI, port  │ ──► Redis (cache hit? -> return)
   (per-step query)   │  11434, OllamaAPI shim) │       │
                      └────────────┬────────────┘       └ miss
                                   │
                            (round-robin)
                                   │
                ┌──────────────────┼──────────────────┐
                ▼                  ▼                  ▼
          ┌──────────┐       ┌──────────┐       ┌──────────┐
          │ vLLM     │       │ vLLM     │       │ vLLM     │
          │ GPU 0    │       │ GPU 1    │       │ GPU N    │
          │ Llama-3B │       │ Llama-3B │       │ Llama-3B │
          └──────────┘       └──────────┘       └──────────┘
```

- The master keeps a token-bucket per worker so no single worker is overloaded.
- All responses written back to Redis with TTL = 24 h.
- Same prompt format as the existing Complex CoT prompt (it already achieves 100% valid-move rate per the midterm).

## What lives in this repo

| File | Status | Purpose |
|---|---|---|
| `model_server.py` | exists, HF native + 4-bit quantization | Becomes a vLLM-backed worker (1 model, many concurrent requests) |
| `llm/api.py` | exists, `OllamaAPI` client | Unchanged — the master speaks the Ollama protocol |
| `llm/master.py` | **NEW** | Async FastAPI that exposes `/api/generate`, hashes the prompt, checks Redis, fans out to workers |
| `llm/cache.py` | **NEW** | Redis adapter (with an in-process LRU fallback for local dev) |
| `slurm/master.slurm` | **NEW** | One-node, no GPU — runs master + Redis |
| `slurm/worker.slurm` | **NEW** | Per-GPU vLLM worker; array job sized to your GPU budget |

## Async opponent step (gym_wrapper)

Even with batched inference the LLM call dominates wall-time. The second
optimisation is to **decouple the opponent step from the agent step** inside
`gym_wrapper.py`:

1. After the agent acts, fire the LLM request for the opponent's turn asynchronously and yield the post-agent observation immediately.
2. The next time the wrapper is asked to step, it joins the async call.
3. With many env subprocesses already in flight, this hides per-call latency behind useful work.

This is a ~30-line patch — straightforward once the master is up.

## What to build first (smallest end-to-end slice)

1. **`llm/cache.py`** + a tiny test (mock prompt -> cache miss -> cache hit).
2. **`llm/master.py`** with a single in-process vLLM worker, talking to the cache, exposing the same protocol the existing `OllamaAPI` expects.
3. Run **1 env** of `gym_wrapper.py` against the master; check that 1000-step games complete in seconds, not minutes, after cache warms up.
4. Then horizontally scale: spin up more workers (slurm array), bump `num_env`, measure throughput.

The win condition for Phase 2 is: **one PPO rollout (49k steps) completes
in under 30 minutes wall-time against a 7B opponent.** Today, naive Ollama
would take many hours.
