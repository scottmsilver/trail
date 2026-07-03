# RULE #0: MANDATORY FIRST ACTION FOR EVERY REQUEST
# 1. Read CLAUDE.md COMPLETELY before responding
# 2. Setup Python venv: [ -d "venv" ] || ./setup-venv.sh && source venv/bin/activate
# 3. Search for rules related to the request
# 4. Only proceed after confirming no violations
# Failure to follow Rule #0 has caused real harm. Check BEFORE acting, not AFTER making mistakes.
#
# GUARDS ARE SAFETY EQUIPMENT - WHEN THEY FIRE, FIX THE PROBLEM THEY FOUND
# NEVER weaken, disable, or bypass guards - they prevent real harm
"""Unit tests for the v2 path layer (OSM classification + rasterization)."""

from app.engine_v2.path_layer import PathType, classify_feature, get_path_type_name


class TestClassifyFeature:
    def test_hiking_path_is_trail(self):
        assert classify_feature({"highway": "path"}) == PathType.TRAIL

    def test_track_is_trail(self):
        assert classify_feature({"highway": "track"}) == PathType.TRAIL

    def test_piste_is_trail(self):
        assert classify_feature({"piste:type": "downhill"}) == PathType.TRAIL

    def test_hiking_route_is_trail(self):
        assert classify_feature({"route": "hiking"}) == PathType.TRAIL

    def test_steps_and_cycleway_are_path(self):
        assert classify_feature({"highway": "steps"}) == PathType.PATH
        assert classify_feature({"highway": "cycleway"}) == PathType.PATH

    def test_footway_and_pedestrian_are_footway(self):
        assert classify_feature({"highway": "footway"}) == PathType.FOOTWAY
        assert classify_feature({"highway": "pedestrian"}) == PathType.FOOTWAY

    def test_roads_are_residential(self):
        for hw in ["residential", "living_street", "service", "unclassified", "tertiary", "secondary", "primary"]:
            assert classify_feature({"highway": hw}) == PathType.RESIDENTIAL, hw

    def test_park_and_meadow_are_natural(self):
        assert classify_feature({"leisure": "park"}) == PathType.NATURAL
        assert classify_feature({"natural": "meadow"}) == PathType.NATURAL
        assert classify_feature({"landuse": "grass"}) == PathType.NATURAL

    def test_highway_takes_priority_over_landuse(self):
        # A road through a park is still a road
        assert classify_feature({"highway": "residential", "landuse": "grass"}) == PathType.RESIDENTIAL

    def test_unrecognized_is_unknown(self):
        assert classify_feature({"building": "yes"}) == PathType.UNKNOWN
        assert classify_feature({}) == PathType.UNKNOWN

    def test_none_values_ignored(self):
        # osmnx GeoDataFrames contain NaN/None for absent tags
        assert classify_feature({"highway": None, "leisure": "park"}) == PathType.NATURAL


def test_path_type_names():
    assert get_path_type_name(PathType.TRAIL) == "trail"
    assert get_path_type_name(PathType.UNKNOWN) == "off_path"
    assert get_path_type_name(255) == "invalid"


import geopandas as gpd
import numpy as np
from app.engine_v2.path_layer import rasterize_features
from rasterio.transform import from_bounds
from shapely.geometry import LineString, Polygon

# 10x10 grid covering a 0.01 x 0.01 degree box
SHAPE = (10, 10)
TRANSFORM = from_bounds(-111.51, 40.64, -111.50, 40.65, SHAPE[1], SHAPE[0])


def _gdf(geoms_and_tags):
    """Build a GeoDataFrame from [(geometry, {tag: value}), ...]."""
    if not geoms_and_tags:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    records = []
    for geom, tags in geoms_and_tags:
        rec = dict(tags)
        rec["geometry"] = geom
        records.append(rec)
    return gpd.GeoDataFrame(records, crs="EPSG:4326")


class TestRasterizeFeatures:
    def test_trail_line_marks_trail_cells(self):
        # Horizontal trail across the middle of the box
        trail = LineString([(-111.51, 40.645), (-111.50, 40.645)])
        grid = rasterize_features(_gdf([(trail, {"highway": "path"})]), _gdf([]), SHAPE, TRANSFORM)
        assert grid.dtype == np.uint8
        assert (grid == PathType.TRAIL).sum() >= SHAPE[1]  # touches every column
        assert (grid == PathType.UNKNOWN).sum() > 0  # rest is off-path

    def test_obstacle_wins_over_trail(self):
        trail = LineString([(-111.51, 40.645), (-111.50, 40.645)])
        water = Polygon([(-111.51, 40.64), (-111.505, 40.64), (-111.505, 40.65), (-111.51, 40.65)])  # left half
        grid = rasterize_features(
            _gdf([(trail, {"highway": "path"})]), _gdf([(water, {"natural": "water"})]), SHAPE, TRANSFORM
        )
        # Left half is obstacle even where the trail crosses it
        left = grid[:, : SHAPE[1] // 2 - 1]
        assert (left == PathType.OBSTACLE).all()
        # Trail still visible on the right half
        assert (grid[:, SHAPE[1] // 2 + 1 :] == PathType.TRAIL).any()

    def test_trail_wins_over_park(self):
        park = Polygon([(-111.51, 40.64), (-111.50, 40.64), (-111.50, 40.65), (-111.51, 40.65)])  # whole box
        trail = LineString([(-111.51, 40.645), (-111.50, 40.645)])
        grid = rasterize_features(
            _gdf([(park, {"leisure": "park"}), (trail, {"highway": "path"})]), _gdf([]), SHAPE, TRANSFORM
        )
        assert (grid == PathType.TRAIL).any()
        assert (grid == PathType.NATURAL).any()

    def test_empty_inputs_give_unknown_grid(self):
        grid = rasterize_features(_gdf([]), _gdf([]), SHAPE, TRANSFORM)
        assert grid.shape == SHAPE
        assert (grid == PathType.UNKNOWN).all()


from app.engine_v2.elevation import Bounds
from app.engine_v2.path_layer import PathLayer

BOUNDS = Bounds(south=40.64, north=40.65, west=-111.51, east=-111.50)


class TestPathLayer:
    def test_get_grid_uses_injected_fetcher_and_caches(self, tmp_path):
        calls = []

        def fake_fetch(bounds, tags):
            calls.append(tags)
            if "building" in tags:  # obstacle query
                return _gdf([])
            trail = LineString([(-111.51, 40.645), (-111.50, 40.645)])
            return _gdf([(trail, {"highway": "path"})])

        layer = PathLayer(cache_dir=str(tmp_path), fetch_fn=fake_fetch)
        grid1 = layer.get_grid(BOUNDS, SHAPE, TRANSFORM)
        assert (grid1 == PathType.TRAIL).any()
        assert len(calls) == 2  # one paths fetch + one obstacles fetch

        # Second call: served from disk cache, no new fetches
        grid2 = layer.get_grid(BOUNDS, SHAPE, TRANSFORM)
        assert len(calls) == 2
        assert np.array_equal(grid1, grid2)

    def test_fetch_failure_degrades_to_unknown_grid(self, tmp_path):
        def broken_fetch(bounds, tags):
            raise ConnectionError("overpass down")

        layer = PathLayer(cache_dir=str(tmp_path), fetch_fn=broken_fetch)
        grid = layer.get_grid(BOUNDS, SHAPE, TRANSFORM)
        assert (grid == PathType.UNKNOWN).all()
