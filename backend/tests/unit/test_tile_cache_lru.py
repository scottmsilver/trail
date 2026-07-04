"""Unit tests for the bounded LRU that caps in-memory cache growth.

Background: TiledDEMCache.memory_cache was an unbounded dict. Over a long-running
server it accumulated every ~1km tile ever queried (~2.2MB each), reaching ~14GB.
A locality study showed a single route touches 144-400 tiles and within-region
reuse saturates by ~768 tiles, so we bound the cache with an LRU.
"""

from app.services.lru import BoundedLRUCache
from app.services.tiled_dem_cache import TiledDEMCache


class TestBoundedLRUCache:
    def test_evicts_least_recently_used_on_overflow(self):
        cache = BoundedLRUCache(maxsize=3)
        for k in ("a", "b", "c"):
            cache[k] = k.upper()
        assert set(cache) == {"a", "b", "c"}

        cache["d"] = "D"  # overflow -> evict oldest ("a")
        assert "a" not in cache
        assert set(cache) == {"b", "c", "d"}

    def test_get_refreshes_recency(self):
        cache = BoundedLRUCache(maxsize=3)
        for k in ("a", "b", "c"):
            cache[k] = k.upper()

        _ = cache["a"]  # touch "a" -> now "b" is the oldest
        cache["d"] = "D"
        assert "b" not in cache
        assert "a" in cache and set(cache) == {"a", "c", "d"}

    def test_reassigning_existing_key_updates_recency_not_size(self):
        cache = BoundedLRUCache(maxsize=2)
        cache["a"] = 1
        cache["b"] = 2
        cache["a"] = 99  # refresh "a", "b" now oldest
        cache["c"] = 3  # evict "b"
        assert len(cache) == 2
        assert cache["a"] == 99
        assert "b" not in cache

    def test_maxsize_none_or_zero_is_unbounded(self):
        for unbounded in (0, None):
            cache = BoundedLRUCache(maxsize=unbounded)
            for i in range(1000):
                cache[i] = i
            assert len(cache) == 1000

    def test_get_method_respects_default_and_recency(self):
        cache = BoundedLRUCache(maxsize=2)
        cache["a"] = 1
        cache["b"] = 2
        assert cache.get("missing", "fallback") == "fallback"
        assert cache.get("a") == 1  # refresh "a"
        cache["c"] = 3  # evict "b", not "a"
        assert "a" in cache and "b" not in cache


class TestTiledDEMCacheMemoryBound:
    def _fake_compute(self, min_lat, max_lat, min_lon, max_lon):
        return {"cost_surface": [min_lat, max_lat]}

    def test_memory_cache_never_exceeds_cap(self, tmp_path):
        cache = TiledDEMCache(cache_dir=str(tmp_path), max_memory_tiles=768)
        # Touch far more tiles than the cap; each computes+stores one entry.
        for i in range(1200):
            cache.get_tile(i, 0, "cost", compute_func=self._fake_compute)
        assert len(cache.memory_cache) == 768
        # The most-recently-touched tiles must still be resident.
        assert cache._get_tile_key(1199, 0, "cost") in cache.memory_cache
        assert cache._get_tile_key(0, 0, "cost") not in cache.memory_cache

    def test_default_cap_is_768(self, tmp_path, monkeypatch):
        monkeypatch.delenv("TILE_CACHE_MAX_TILES", raising=False)
        cache = TiledDEMCache(cache_dir=str(tmp_path))
        assert cache.max_memory_tiles == 768

    def test_cap_overridable_via_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TILE_CACHE_MAX_TILES", "128")
        cache = TiledDEMCache(cache_dir=str(tmp_path))
        assert cache.max_memory_tiles == 128

    def test_invalid_env_falls_back_to_default(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TILE_CACHE_MAX_TILES", "not-a-number")
        cache = TiledDEMCache(cache_dir=str(tmp_path))
        assert cache.max_memory_tiles == 768

    def test_nonpositive_env_falls_back_to_default(self, tmp_path, monkeypatch):
        for bad in ("0", "-5"):
            monkeypatch.setenv("TILE_CACHE_MAX_TILES", bad)
            cache = TiledDEMCache(cache_dir=str(tmp_path))
            assert cache.max_memory_tiles == 768, f"{bad} should fall back, not disable the bound"

    def test_concurrent_get_tile_stays_bounded_and_safe(self, tmp_path):
        """Many threads hammering a shared cache must not raise (KeyError from a
        non-atomic check-then-get) and must respect the bound."""
        import threading

        cache = TiledDEMCache(cache_dir=str(tmp_path), max_memory_tiles=64)
        errors = []

        def worker(base):
            try:
                for i in range(400):
                    cache.get_tile((base + i) % 128, 0, "cost", compute_func=self._fake_compute)
            except Exception as e:  # pragma: no cover - only on a real race
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(b,)) for b in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, f"concurrent access raised: {errors[:3]}"
        assert len(cache.memory_cache) <= 64
