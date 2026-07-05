"""Display-only terrain overlay: fetch notable terrain (water, glacier, cliffs,
...) as polygons so the UI can MARK what a route crosses — even passable terrain
like glaciers that isn't a routing obstacle."""

from collections import namedtuple

import geopandas as gpd
from shapely.geometry import LineString, Polygon

from app.engine_v2.path_layer import TERRAIN_DISPLAY_TAGS, PathLayer

_B = namedtuple("_B", ["south", "west", "north", "east"])
_BOUNDS = _B(40.64, -111.60, 40.66, -111.58)


def test_terrain_tags_include_glacier_and_water():
    # The whole point: glaciers (passable, but they read as ponds on the map) and
    # every water form must be visible in the overlay.
    assert "glacier" in TERRAIN_DISPLAY_TAGS["natural"]
    assert "water" in TERRAIN_DISPLAY_TAGS["natural"]
    assert TERRAIN_DISPLAY_TAGS.get("water") is True


def _layer(fetch, tmp_path):
    return PathLayer(cache_dir=str(tmp_path), fetch_fn=fetch)


def test_get_terrain_polygons_returns_kind_and_latlon_ring(tmp_path):
    poly = Polygon([(-111.588, 40.650), (-111.587, 40.650), (-111.587, 40.652), (-111.588, 40.652)])

    def fake(bounds, tags):
        return gpd.GeoDataFrame({"natural": ["glacier"]}, geometry=[poly], crs="EPSG:4326")

    out = _layer(fake, tmp_path).get_terrain_polygons(_BOUNDS)
    assert len(out) == 1
    assert out[0]["kind"] == "glacier"
    ring = out[0]["polygon"]
    assert len(ring) >= 4
    for lat, lon in ring:  # emitted as [lat, lon]
        assert 40 < lat < 41 and -112 < lon < -111


def test_get_terrain_polygons_skips_non_polygons(tmp_path):
    line = LineString([(-111.588, 40.650), (-111.587, 40.651)])

    def fake(bounds, tags):
        return gpd.GeoDataFrame({"waterway": ["stream"]}, geometry=[line], crs="EPSG:4326")

    assert _layer(fake, tmp_path).get_terrain_polygons(_BOUNDS) == []


def test_get_terrain_polygons_empty_on_fetch_failure(tmp_path):
    def broken(bounds, tags):
        raise RuntimeError("overpass down")

    assert _layer(broken, tmp_path).get_terrain_polygons(_BOUNDS) == []
