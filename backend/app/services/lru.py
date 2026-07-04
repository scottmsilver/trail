"""A small thread-safe bounded LRU mapping built on the stdlib OrderedDict recipe.

This is the same move-to-end / evict-oldest idiom already used inline in
app/engine_v2/elevation_fd_safe.py, packaged as a dict drop-in so callers with
many assignment sites (see dem_tile_cache / tiled_dem_cache) don't have to
duplicate the eviction logic. No third-party cache dependency is needed.

Reads refresh recency (move_to_end), so a single shared instance is mutated by
concurrent readers; all cache operations are guarded by a re-entrant lock and
callers should use ``get(key, MISSING)`` for an atomic present-or-default lookup
rather than a non-atomic ``key in cache`` / ``cache[key]`` sequence.
"""
import threading
from collections import OrderedDict

# Sentinel for atomic "present vs absent" lookups via get(); distinct from a
# stored value of None so callers can cache None safely.
MISSING = object()


class BoundedLRUCache(OrderedDict):
    """OrderedDict that keeps at most ``maxsize`` most-recently-used items.

    Recency is refreshed on both read (``__getitem__`` / ``get``) and write, so
    a plain ``cache[key]`` access pattern behaves as a true LRU. ``maxsize`` of
    0 or ``None`` disables the bound (behaves like an ordinary dict). All
    mutating/reordering operations hold a re-entrant lock so the cache can be
    shared across concurrent request threads without corrupting its order.
    """

    def __init__(self, maxsize, *args, **kwargs):
        self.maxsize = maxsize
        self._lock = threading.RLock()
        super().__init__(*args, **kwargs)
        self._evict()

    def __getitem__(self, key):
        with self._lock:
            value = super().__getitem__(key)
            self.move_to_end(key)
            return value

    def __setitem__(self, key, value):
        with self._lock:
            if key in self:
                self.move_to_end(key)
            super().__setitem__(key, value)
            self._evict()

    def get(self, key, default=None):
        """Atomic present-or-default lookup that also refreshes recency.

        Holding the lock across the membership test and read prevents another
        thread from evicting the key in between (which a bare ``key in cache``
        then ``cache[key]`` would race on).
        """
        with self._lock:
            if key in self:
                value = super().__getitem__(key)
                self.move_to_end(key)
                return value
            return default

    def clear(self):
        with self._lock:
            super().clear()

    def _evict(self):
        # Callers hold self._lock (or run during single-threaded __init__).
        if self.maxsize and self.maxsize > 0:
            while len(self) > self.maxsize:
                self.popitem(last=False)
