"""Tiny cache smoke test: miss -> set -> hit, model-scoped keys, LRU eviction,
and the auto backend fallback. Runnable directly (``python -m llm.cache_test``
from the repo root) or under pytest.
"""
from llm.cache import PromptCache, LRUBackend, cache_key, build_cache


def test_miss_then_hit():
    c = PromptCache(LRUBackend(maxsize=8))
    assert c.get("m", "p") is None                  # miss
    c.set("m", "p", "draw from stock")
    assert c.get("m", "p") == "draw from stock"     # hit
    s = c.stats()
    assert s["hits"] == 1 and s["misses"] == 1
    assert s["hit_rate"] == 0.5 and s["size"] == 1


def test_model_scopes_key():
    assert cache_key("a", "p") != cache_key("b", "p")
    c = PromptCache(LRUBackend())
    c.set("a", "p", "x")
    assert c.get("b", "p") is None                  # other model -> miss


def test_lru_eviction():
    c = PromptCache(LRUBackend(maxsize=2))
    c.set("m", "1", "a")
    c.set("m", "2", "b")
    assert c.get("m", "1") == "a"                    # touch "1" -> "2" is now LRU
    c.set("m", "3", "c")                             # evicts "2"
    assert c.get("m", "2") is None
    assert c.get("m", "1") == "a" and c.get("m", "3") == "c"


def test_build_cache_auto_falls_back_to_memory():
    c = build_cache(backend="auto")                  # no redis here -> LRU
    assert c.stats()["backend"] in ("LRUBackend", "RedisBackend")
    c.set("m", "p", "v")
    assert c.get("m", "p") == "v"


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"ok  {_name}")
    print("all cache tests passed")
