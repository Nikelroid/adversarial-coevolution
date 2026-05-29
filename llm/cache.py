"""Prompt cache for the Phase-2 master/worker/cache LLM-opponent stack.

Design notes
------------
The master (``llm/master.py``) is a single FastAPI process that every RL env
subprocess talks to over HTTP. Because that process is the one shared point in
the system, an *in-process* LRU already deduplicates queries across all of the
``num_env`` env workers within a single training run -- no external service
required.

Redis would only add (a) persistence of a warm cache across separate training
runs and (b) sharing across multiple master processes. Neither is needed for the
smallest end-to-end slice, and this cluster currently has neither the ``redis``
python package nor a ``redis-server`` binary, so the default backend is the
in-process LRU and Redis is used only when it is importable *and* a server is
reachable.

Key = ``sha1(model \\x00 prompt)``. Sampling options (temperature, max_tokens)
are intentionally *not* part of the key: for the opponent we want a recurring
game state to map to a stable move, which also maximises the hit rate. Different
models never collide because the model id is part of the key.
"""
from __future__ import annotations

import hashlib
import logging
import threading
from collections import OrderedDict
from typing import Optional, Protocol


def cache_key(model: str, prompt: str) -> str:
    """Stable content hash for a (model, prompt) pair."""
    h = hashlib.sha1()
    h.update(model.encode("utf-8"))
    h.update(b"\x00")
    h.update(prompt.encode("utf-8"))
    return h.hexdigest()


class CacheBackend(Protocol):
    def get(self, key: str) -> Optional[str]: ...
    def set(self, key: str, value: str) -> None: ...
    def __len__(self) -> int: ...


class LRUBackend:
    """Thread-safe in-process LRU; evicts least-recently-used on overflow."""

    def __init__(self, maxsize: int = 100_000):
        if maxsize <= 0:
            raise ValueError("maxsize must be positive")
        self.maxsize = maxsize
        self._data: "OrderedDict[str, str]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            if key not in self._data:
                return None
            self._data.move_to_end(key)
            return self._data[key]

    def set(self, key: str, value: str) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


class RedisBackend:
    """Redis-backed cache. Raises on construction if redis is unimportable or the
    server is unreachable, so :func:`build_cache` can fall back to the LRU."""

    def __init__(self, url: str = "redis://localhost:6379/0",
                 namespace: str = "ginllm", ttl_seconds: int = 24 * 3600):
        import redis  # local import: optional dependency
        self._r = redis.Redis.from_url(url)
        self._r.ping()  # fail fast if no server is listening
        self.namespace = namespace
        self.ttl = ttl_seconds

    def _k(self, key: str) -> str:
        return f"{self.namespace}:{key}"

    def get(self, key: str) -> Optional[str]:
        v = self._r.get(self._k(key))
        return v.decode("utf-8") if v is not None else None

    def set(self, key: str, value: str) -> None:
        self._r.set(self._k(key), value, ex=self.ttl)

    def __len__(self) -> int:
        # SCAN-based count is O(n); used only for stats/debugging.
        return sum(1 for _ in self._r.scan_iter(f"{self.namespace}:*"))


class PromptCache:
    """Front-end over a :class:`CacheBackend` that hashes keys and tracks stats."""

    def __init__(self, backend: CacheBackend):
        self.backend = backend
        self._hits = 0
        self._misses = 0
        self._lock = threading.Lock()

    def get(self, model: str, prompt: str) -> Optional[str]:
        value = self.backend.get(cache_key(model, prompt))
        with self._lock:
            if value is None:
                self._misses += 1
            else:
                self._hits += 1
        return value

    def set(self, model: str, prompt: str, response: str) -> None:
        self.backend.set(cache_key(model, prompt), response)

    def stats(self) -> dict:
        with self._lock:
            total = self._hits + self._misses
            return {
                "hits": self._hits,
                "misses": self._misses,
                "total": total,
                "hit_rate": (self._hits / total) if total else 0.0,
                "size": len(self.backend),
                "backend": type(self.backend).__name__,
            }


def build_cache(backend: str = "auto", maxsize: int = 100_000,
                redis_url: str = "redis://localhost:6379/0") -> PromptCache:
    """Construct a :class:`PromptCache`.

    Args:
        backend: ``"auto"`` tries Redis then falls back to the in-process LRU;
            ``"memory"`` forces the LRU; ``"redis"`` forces Redis (raises if down).
        maxsize: LRU capacity (entries) for the in-process backend.
        redis_url: connection URL when the Redis backend is used.
    """
    backend = backend.lower()
    if backend in ("redis", "auto"):
        try:
            return PromptCache(RedisBackend(url=redis_url))
        except Exception as exc:  # noqa: BLE001 - any redis import/conn error
            if backend == "redis":
                raise
            logging.getLogger(__name__).info(
                "Redis unavailable (%s); using in-process LRU cache.", exc)
    return PromptCache(LRUBackend(maxsize=maxsize))
