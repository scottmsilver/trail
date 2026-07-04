"""Tiled OSM cache: overlapping routes reuse cached tiles instead of re-fetching.

This is the fix for the v2 engine re-hitting Overpass for data it already had:
the OSM path layer now caches features on a FIXED geographic tile grid, so any
route overlapping already-fetched tiles reuses them.
"""

import geopandas as gpd
import numpy as np
from rasterio.transform import from_bounds
from shapely.geometry import LineString

from app.engine_v2.elevation import Bounds
from app.engine_v2.path_layer import PathLayer, PathType

TILE_DEG = 0.02


def _gdf(geoms_and_tags):
    if not geoms_and_tags:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    records = []
    for geom, tags in geoms_and_tags:
        rec = dict(tags)
        rec["geometry"] = geom
        records.append(rec)
    return gpd.GeoDataFrame(records, crs="EPSG:4326")


def _counting_fetch(calls):
    """A fetcher that records each call and returns a trail spanning the query."""

    def fetch(bounds, tags):
        calls.append((tags.get("highway") is not None, bounds))
        if "building" in tags:  # obstacle query -> nothing
            return _gdf([])
        # A horizontal trail at a latitude every test route below covers
        # (~40.6425), spanning the fetched bbox so it lands in each tile.
        lat = 40.6425
        trail = LineString([(bounds.west, lat), (bounds.east, lat)])
        return _gdf([(trail, {"highway": "path"})])

    return fetch


def _shape_transform(b):
    shape = (16, 16)
    return shape, from_bounds(b.west, b.south, b.east, b.north, shape[1], shape[0])


def test_route_fully_inside_cached_tile_does_not_refetch(tmp_path):
    calls = []
    layer = PathLayer(cache_dir=str(tmp_path), fetch_fn=_counting_fetch(calls), tile_deg=TILE_DEG)

    # Route A occupies a single 0.02 tile.
    a = Bounds(south=40.641, north=40.644, west=-111.509, east=-111.506)
    layer.get_grid(a, *_shape_transform(a))
    assert len(calls) == 2  # one union fetch: paths + obstacles

    # Route B sits fully inside the same tile -> served from cache, no new fetch.
    b = Bounds(south=40.6415, north=40.6435, west=-111.5085, east=-111.5065)
    grid_b = layer.get_grid(b, *_shape_transform(b))
    assert len(calls) == 2
    assert (grid_b == PathType.TRAIL).any()


def test_route_extending_into_new_tile_fetches_only_the_new_area(tmp_path):
    calls = []
    layer = PathLayer(cache_dir=str(tmp_path), fetch_fn=_counting_fetch(calls), tile_deg=TILE_DEG)

    a = Bounds(south=40.641, north=40.644, west=-111.509, east=-111.506)  # tile (2032, -5576)
    layer.get_grid(a, *_shape_transform(a))
    assert len(calls) == 2

    # Route C reaches east into the neighboring tile (-5575); only the new tile
    # triggers a fetch (one more union call = 2 more), not the whole route.
    c = Bounds(south=40.641, north=40.644, west=-111.508, east=-111.498)
    layer.get_grid(c, *_shape_transform(c))
    assert len(calls) == 4

    # And a repeat of C is now fully cached: no further fetches.
    layer.get_grid(c, *_shape_transform(c))
    assert len(calls) == 4


def test_tiles_persist_across_pathlayer_instances(tmp_path):
    """A fresh PathLayer over the same cache dir reuses tiles from disk."""
    calls1 = []
    a = Bounds(south=40.641, north=40.644, west=-111.509, east=-111.506)
    PathLayer(cache_dir=str(tmp_path), fetch_fn=_counting_fetch(calls1), tile_deg=TILE_DEG).get_grid(
        a, *_shape_transform(a)
    )
    assert len(calls1) == 2

    calls2 = []
    grid = PathLayer(cache_dir=str(tmp_path), fetch_fn=_counting_fetch(calls2), tile_deg=TILE_DEG).get_grid(
        a, *_shape_transform(a)
    )
    assert calls2 == []  # entirely from the on-disk tile cache
    assert grid.dtype == np.uint8
