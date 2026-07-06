"""OSM data freshness: versioned tile cache (auto-invalidate on tag change),
per-area failure-retry cooldown, missing-data reporting (no hard fail), and a
force-refresh that bypasses the cache."""

import os
import time
from collections import namedtuple

import geopandas as gpd
from rasterio.transform import from_bounds

from app.engine_v2.path_layer import _FETCH_RETRY_COOLDOWN_S, CACHE_VERSION, PathLayer, PathType

_BBox = namedtuple("_BBox", ["south", "west", "north", "east"])
_BOUNDS = _BBox(40.60, -111.60, 40.61, -111.59)
_SHAPE = (8, 8)
_TRANSFORM = from_bounds(-111.60, 40.60, -111.59, 40.61, 8, 8)


def _empty_gdf():
    return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")


class _CountingFetch:
    """A fetch_fn that counts calls and can be made to fail."""

    def __init__(self, ok=True):
        self.calls = 0
        self.ok = ok

    def __call__(self, bounds, tags):
        self.calls += 1
        if not self.ok:
            raise RuntimeError("overpass down")
        return _empty_gdf()


def test_tile_cache_filename_encodes_version(tmp_path):
    layer = PathLayer(cache_dir=str(tmp_path), fetch_fn=lambda b, t: _empty_gdf())
    assert CACHE_VERSION in os.path.basename(layer._tile_cache(10, -20))


def test_old_unversioned_tile_is_ignored_and_refetched(tmp_path):
    # A pre-versioning cache file (old naming) must not be trusted: the tile is
    # treated as missing and refetched with the current tag config.
    fetch = _CountingFetch(ok=True)
    layer = PathLayer(cache_dir=str(tmp_path), fetch_fn=fetch)
    for ti, tj in layer._tile_indices(_BOUNDS):
        with open(os.path.join(str(tmp_path), f"osmtile_{ti}_{tj}.pkl"), "wb") as fh:
            fh.write(b"stale")
    layer.get_grid(_BOUNDS, _SHAPE, _TRANSFORM)
    assert fetch.calls > 0


def test_failed_fetch_not_retried_within_cooldown(tmp_path):
    fetch = _CountingFetch(ok=False)
    layer = PathLayer(cache_dir=str(tmp_path), fetch_fn=fetch)
    layer.get_grid(_BOUNDS, _SHAPE, _TRANSFORM)
    first = fetch.calls
    assert first > 0
    layer.get_grid(_BOUNDS, _SHAPE, _TRANSFORM)  # still within cooldown
    assert fetch.calls == first  # no new attempt
    for k in list(layer._fetch_failed_at):  # expire the cooldown
        layer._fetch_failed_at[k] = time.monotonic() - (_FETCH_RETRY_COOLDOWN_S + 1)
    layer.get_grid(_BOUNDS, _SHAPE, _TRANSFORM)
    assert fetch.calls > first  # retried after cooldown


def test_get_grid_reports_missing_via_missing_out_without_raising(tmp_path):
    fetch = _CountingFetch(ok=False)
    layer = PathLayer(cache_dir=str(tmp_path), fetch_fn=fetch, strict_obstacles=True)
    missing = []
    grid = layer.get_grid(_BOUNDS, _SHAPE, _TRANSFORM, missing_out=missing)
    assert grid.shape == _SHAPE
    assert (grid == PathType.UNKNOWN).all()  # degraded, not crashed
    assert len(missing) > 0  # but the gap is reported


def test_refresh_refetches_cached_tiles(tmp_path):
    fetch = _CountingFetch(ok=True)
    layer = PathLayer(cache_dir=str(tmp_path), fetch_fn=fetch)
    layer.get_grid(_BOUNDS, _SHAPE, _TRANSFORM)
    after_first = fetch.calls
    layer.get_grid(_BOUNDS, _SHAPE, _TRANSFORM)  # cached -> no fetch
    assert fetch.calls == after_first
    layer.get_grid(_BOUNDS, _SHAPE, _TRANSFORM, refresh=True)  # force redo
    assert fetch.calls > after_first


def test_failed_tile_map_is_pruned_after_cooldown(tmp_path):
    from app.engine_v2.path_layer import _MAX_FAILED_TILES

    layer = PathLayer(cache_dir=str(tmp_path), fetch_fn=_CountingFetch(ok=False))
    # A stale (expired) failure entry from an old area must be pruned when a new
    # failure is recorded, so the map tracks only tiles still in cooldown.
    layer._fetch_failed_at[(9999, 9999)] = time.monotonic() - (_FETCH_RETRY_COOLDOWN_S + 5)
    layer.get_grid(_BOUNDS, _SHAPE, _TRANSFORM)  # fresh failures recorded
    assert (9999, 9999) not in layer._fetch_failed_at
    assert 0 < len(layer._fetch_failed_at) <= _MAX_FAILED_TILES
