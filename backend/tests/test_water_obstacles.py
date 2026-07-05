"""Water/obstacle hardening for engine v2.

Covers three gaps that let routes cross bodies of water:
  1. area-water tag coverage (reservoir/basin, water=*, riverbank, bay),
  2. line-geometry major waterways barely blocking (centerline → real width),
  3. silently routing with no obstacle data when the OSM fetch degrades.
"""

import geopandas as gpd
from rasterio.transform import from_bounds
from shapely.geometry import LineString, Polygon

from app.engine_v2.path_layer import OBSTACLE_TAGS, PathLayer, PathType, rasterize_features


# --- 1. tag coverage --------------------------------------------------------
def test_obstacle_tags_cover_area_water():
    """Lakes/ponds/reservoirs must be fetched however OSM tags them."""
    assert OBSTACLE_TAGS.get("water") is True  # water=pond/lake/reservoir/lagoon
    assert "bay" in OBSTACLE_TAGS["natural"]
    assert "water" in OBSTACLE_TAGS["natural"]
    for lu in ("reservoir", "basin"):
        assert lu in OBSTACLE_TAGS["landuse"]
    assert "riverbank" in OBSTACLE_TAGS["waterway"]


# --- 2. line-waterway buffering --------------------------------------------
def _grid_setup():
    # ~0.001deg box, 50x50 cells => ~2.2 m cells, far finer than the ~9 m buffer.
    west, south, east, north = -111.6000, 40.6000, -111.5990, 40.6010
    shape = (50, 50)
    transform = from_bounds(west, south, east, north, shape[1], shape[0])
    midlat = (south + north) / 2
    line = LineString([(west, midlat), (east, midlat)])  # horizontal across the box
    return shape, transform, line


def _obstacle_count(waterway_value):
    shape, transform, line = _grid_setup()
    obstacles = gpd.GeoDataFrame({"waterway": [waterway_value]}, geometry=[line], crs="EPSG:4326")
    grid = rasterize_features(None, obstacles, shape, transform)
    return int((grid == PathType.OBSTACLE).sum())


def test_major_waterway_line_is_buffered_wider_than_centerline():
    river = _obstacle_count("river")  # buffered to a real width
    stream = _obstacle_count("stream")  # fordable → left as a ~1-cell centerline
    assert stream > 0
    assert river > stream * 2  # the buffer adds several cell-rows of blockage


def test_area_water_polygon_still_fills_solidly():
    shape, transform, _ = _grid_setup()
    west, south, east, north = -111.6000, 40.6000, -111.5990, 40.6010
    pond = Polygon([(-111.5997, 40.6003), (-111.5993, 40.6003), (-111.5993, 40.6007), (-111.5997, 40.6007)])
    obstacles = gpd.GeoDataFrame({"natural": ["water"]}, geometry=[pond], crs="EPSG:4326")
    grid = rasterize_features(None, obstacles, shape, transform)
    assert (grid == PathType.OBSTACLE).sum() > 20  # a solid block, not a thin edge


# --- 3. strict fail-closed on missing obstacle data -------------------------
from collections import namedtuple  # noqa: E402

_BBox = namedtuple("_BBox", ["south", "west", "north", "east"])
_BOUNDS = _BBox(south=40.60, west=-111.60, north=40.61, east=-111.59)
_SHAPE = (8, 8)
_TRANSFORM = from_bounds(-111.60, 40.60, -111.59, 40.61, 8, 8)


def _broken_fetch(bounds, tags):
    raise RuntimeError("overpass unreachable")


def test_get_grid_strict_reports_missing_without_raising(tmp_path):
    # Strict mode no longer hard-fails: it reports missing tiles via missing_out
    # so the caller can flag "OSM data missing" and offer a force-refresh.
    layer = PathLayer(cache_dir=str(tmp_path), fetch_fn=_broken_fetch, strict_obstacles=True)
    missing = []
    grid = layer.get_grid(_BOUNDS, _SHAPE, _TRANSFORM, missing_out=missing)
    assert grid.shape == _SHAPE
    assert len(missing) > 0


def test_get_grid_nonstrict_degrades_gracefully(tmp_path):
    # Default (non-strict) keeps the legacy degrade-to-terrain-only behavior.
    layer = PathLayer(cache_dir=str(tmp_path), fetch_fn=_broken_fetch, strict_obstacles=False)
    grid = layer.get_grid(_BOUNDS, _SHAPE, _TRANSFORM)
    assert grid.shape == _SHAPE
    assert (grid == PathType.UNKNOWN).all()  # no data => nothing marked


def test_strict_does_not_raise_once_tiles_are_cached(tmp_path):
    # A successful fetch caches the tiles; a later strict call must not raise.
    def good_fetch(bounds, tags):
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")

    layer = PathLayer(cache_dir=str(tmp_path), fetch_fn=good_fetch, strict_obstacles=True)
    grid = layer.get_grid(_BOUNDS, _SHAPE, _TRANSFORM)  # caches empty (no obstacles) tiles
    assert grid.shape == _SHAPE
