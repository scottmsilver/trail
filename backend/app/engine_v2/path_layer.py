# RULE #0: MANDATORY FIRST ACTION FOR EVERY REQUEST
# 1. Read CLAUDE.md COMPLETELY before responding
# 2. Setup Python venv: [ -d "venv" ] || ./setup-venv.sh && source venv/bin/activate
# 3. Search for rules related to the request
# 4. Only proceed after confirming no violations
# Failure to follow Rule #0 has caused real harm. Check BEFORE acting, not AFTER making mistakes.
#
# GUARDS ARE SAFETY EQUIPMENT - WHEN THEY FIRE, FIX THE PROBLEM THEY FOUND
# NEVER weaken, disable, or bypass guards - they prevent real harm
"""
Path layer for engine v2: fetches OSM features and rasterizes them into a
terrain-type grid aligned with the elevation array.

This is the module the backend3 rewrite referenced but never implemented.
Tag vocabularies are ported from app/services/path_preferences.py and
app/services/obstacle_config.py (the proven v1 configuration).
"""
import logging
import math
import os
import pickle
from collections import namedtuple
from enum import IntEnum
from typing import Callable, List, Optional, Tuple

import numpy as np
from rasterio.features import rasterize as _rio_rasterize

logger = logging.getLogger(__name__)


class PathType(IntEnum):
    UNKNOWN = 0  # off-path terrain
    OBSTACLE = 1  # water, buildings, cliffs — impassable
    TRAIL = 2  # natural hiking trails / tracks / pistes
    PATH = 3  # steps, cycleways
    FOOTWAY = 4  # sidewalks, pedestrian areas
    RESIDENTIAL = 5  # streets
    NATURAL = 6  # parks, meadows, grass


_TYPE_NAMES = {
    PathType.UNKNOWN: "off_path",
    PathType.OBSTACLE: "obstacle",
    PathType.TRAIL: "trail",
    PathType.PATH: "path",
    PathType.FOOTWAY: "footway",
    PathType.RESIDENTIAL: "residential",
    PathType.NATURAL: "natural",
}

# Ported from PathPreferences.get_default_path_tags()
PATH_TAGS = {
    "highway": [
        "footway",
        "path",
        "track",
        "pedestrian",
        "steps",
        "cycleway",
        "bridleway",
        "trail",
        "residential",
        "living_street",
        "service",
        "unclassified",
    ],
    "leisure": ["park", "nature_reserve", "garden", "common", "playground"],
    "natural": ["grassland", "meadow", "heath", "scrub", "beach", "sand"],
    "landuse": ["grass", "meadow", "recreation_ground", "village_green"],
    "route": ["hiking", "foot", "walking"],
    "piste:type": ["downhill", "nordic", "sled", "hike", "skitour", "connection"],
}

# Ported from ObstacleConfig.get_default_osm_tags(), then broadened so bodies of
# water are caught however OSM tags them. Water is mapped inconsistently:
# lakes/ponds are usually `natural=water` but also carry `water=pond|lake|...`;
# reservoirs/basins are often only `landuse=reservoir|basin`; wide rivers appear
# as `waterway=riverbank` polygons. Missing any of these lets a route cross the
# water body. `water: True` fetches every `water=*` area regardless of `natural`.
OBSTACLE_TAGS = {
    "natural": ["water", "wetland", "bay", "cliff", "rock", "scree"],
    "water": True,
    "waterway": ["river", "stream", "canal", "riverbank", "dock"],
    "landuse": ["industrial", "commercial", "military", "reservoir", "basin"],
    "building": True,
    "leisure": ["golf_course", "swimming_pool"],
    "barrier": True,
}

# OSM often maps a river/canal as a bare centerline (LineString) with no width.
# Rasterized with all_touched that is only a ~1-cell strip, which barely blocks a
# real river. Buffer such major line-waterways to a modest real width before
# rasterizing so they block like the water they represent. Area water (polygons)
# already fills solidly and is left untouched. Streams stay a centerline: they
# are generally fordable and buffering them would wall off legitimate crossings.
_WATERWAY_BUFFER_DEG = 0.00008  # ~9 m at these latitudes
_BUFFERED_WATERWAYS = {"river", "canal", "riverbank"}
_MAX_WATERWAY_COORDS = 10000  # skip buffering absurdly complex lines (cost guard)

# Tags shown by the terrain DISPLAY overlay. A superset of obstacles plus
# passable-but-notable terrain — chiefly `natural=glacier` (permanent snowfields
# render pale blue and read as ponds on the map, yet aren't routing obstacles).
# Display-only: this never affects routing.
TERRAIN_DISPLAY_TAGS = {
    "natural": ["water", "glacier", "wetland", "bay", "cliff", "rock", "bare_rock", "scree"],
    "water": True,
    "landuse": ["reservoir", "basin"],
}
# Hard caps so a pathological/huge OSM response can't blow up the backend
# (memory/JSON) or the frontend (Leaflet polygon count). The overlay is a rough
# marker, not a survey — dropping the tail past these limits is fine.
_MAX_TERRAIN_FEATURES = 2000
_MAX_TERRAIN_VERTICES = 120000


def terrain_kind(tags: dict) -> str:
    """A short label for a terrain-display polygon, from its most specific tag."""
    for key in ("natural", "landuse"):
        value = _tag(tags, key)
        if value:
            return value
    if _tag(tags, "water"):
        return "water"
    return "terrain"


class ObstacleDataUnavailableError(ValueError):
    """Raised when obstacle/water data cannot be loaded for part of a route and
    the caller requires it (strict mode). Prevents silently routing across
    unmodeled water. Subclasses ValueError so API layers surface it as a 400."""


def osm_disabled() -> bool:
    """True when OSM fetching is explicitly turned off (OSM_DISABLE). This is an
    opt-in to terrain-only routing; when it's off we expect OSM to be reachable
    (e.g. a local Overpass) and treat a fetch failure as a hard error."""
    return os.environ.get("OSM_DISABLE") in ("1", "true", "True")


_TRAIL_HIGHWAYS = {"path", "track", "trail", "bridleway"}
_PATH_HIGHWAYS = {"steps", "cycleway"}
_FOOTWAY_HIGHWAYS = {"footway", "pedestrian"}
_ROAD_HIGHWAYS = {"residential", "living_street", "service", "unclassified", "tertiary", "secondary", "primary"}
_NATURAL_LEISURE = {"park", "nature_reserve", "garden", "common", "playground"}
_NATURAL_NATURAL = {"grassland", "meadow", "heath", "scrub", "beach", "sand"}
_NATURAL_LANDUSE = {"grass", "meadow", "recreation_ground", "village_green"}
_TRAIL_ROUTES = {"hiking", "foot", "walking"}


def _tag(tags: dict, key: str) -> Optional[str]:
    """Get a tag value, treating None/NaN as absent."""
    value = tags.get(key)
    if value is None:
        return None
    # pandas NaN
    if isinstance(value, float) and np.isnan(value):
        return None
    return str(value)


def classify_feature(tags: dict) -> PathType:
    """Map a feature's OSM tags to a PathType. Highway takes priority."""
    highway = _tag(tags, "highway")
    if highway in _TRAIL_HIGHWAYS:
        return PathType.TRAIL
    if highway in _PATH_HIGHWAYS:
        return PathType.PATH
    if highway in _FOOTWAY_HIGHWAYS:
        return PathType.FOOTWAY
    if highway in _ROAD_HIGHWAYS:
        return PathType.RESIDENTIAL
    if _tag(tags, "piste:type") is not None:
        return PathType.TRAIL
    if _tag(tags, "route") in _TRAIL_ROUTES:
        return PathType.TRAIL
    if _tag(tags, "leisure") in _NATURAL_LEISURE:
        return PathType.NATURAL
    if _tag(tags, "natural") in _NATURAL_NATURAL:
        return PathType.NATURAL
    if _tag(tags, "landuse") in _NATURAL_LANDUSE:
        return PathType.NATURAL
    return PathType.UNKNOWN


def get_path_type_name(code: int) -> str:
    try:
        return _TYPE_NAMES[PathType(code)]
    except ValueError:
        return "invalid"


# Rasterization precedence, low to high. Later entries overwrite earlier ones;
# obstacles are stamped last so they always win (safety).
_PRECEDENCE = [
    PathType.NATURAL,
    PathType.RESIDENTIAL,
    PathType.FOOTWAY,
    PathType.PATH,
    PathType.TRAIL,
    PathType.OBSTACLE,
]


def _buffer_line_waterway(geom, row):
    """Give a line-geometry major waterway (river/canal/riverbank centerline) a
    real width so it blocks like the water body it represents. Polygons, points
    and minor waterways (streams) are returned unchanged."""
    if getattr(geom, "geom_type", "") not in ("LineString", "MultiLineString"):
        return geom
    tags = row.to_dict() if hasattr(row, "to_dict") else dict(row)
    if _tag(tags, "waterway") not in _BUFFERED_WATERWAYS:
        return geom
    # Guard against a pathologically complex geometry making .buffer() expensive;
    # such a line already rasterizes as a barrier, so skip the buffer rather than
    # pay for it. Route bounds are distance-capped upstream, but this is cheap.
    if _num_coords(geom) > _MAX_WATERWAY_COORDS:
        return geom
    return geom.buffer(_WATERWAY_BUFFER_DEG)


def _num_coords(geom) -> int:
    """Total coordinate count of a (Multi)LineString, best-effort."""
    try:
        if geom.geom_type == "LineString":
            return len(geom.coords)
        return sum(len(g.coords) for g in geom.geoms)
    except Exception:
        return 0


def rasterize_features(paths_gdf, obstacles_gdf, shape, transform) -> np.ndarray:
    """
    Rasterize classified OSM features onto a grid of PathType codes.

    paths_gdf/obstacles_gdf: GeoDataFrames (EPSG:4326) as returned by
    osmnx.features_from_polygon, or empty. shape/transform must match the
    elevation array exactly so the grids align cell-for-cell.
    """
    grid = np.full(shape, PathType.UNKNOWN, dtype=np.uint8)

    shapes_by_type = {t: [] for t in _PRECEDENCE}

    if paths_gdf is not None and len(paths_gdf) > 0:
        for _, row in paths_gdf.iterrows():
            geom = row.get("geometry")
            if geom is None or geom.is_empty:
                continue
            ptype = classify_feature(row.to_dict())
            if ptype in shapes_by_type:
                shapes_by_type[ptype].append(geom)

    if obstacles_gdf is not None and len(obstacles_gdf) > 0:
        for _, row in obstacles_gdf.iterrows():
            geom = row.get("geometry")
            if geom is None or geom.is_empty:
                continue
            shapes_by_type[PathType.OBSTACLE].append(_buffer_line_waterway(geom, row))

    for ptype in _PRECEDENCE:
        geoms = shapes_by_type[ptype]
        if not geoms:
            continue
        layer = _rio_rasterize(
            [(g, 1) for g in geoms],
            out_shape=shape,
            transform=transform,
            fill=0,
            all_touched=True,
            dtype=np.uint8,
        )
        grid[layer == 1] = ptype

    return grid


def _default_fetch(bounds, tags):
    """Fetch OSM features via osmnx. Imported lazily so unit tests never need it.

    The Overpass endpoint(s) and request timeout come from the environment so we
    never hard-code a server URL: OVERPASS_URLS is a comma-separated list of
    mirrors tried in order; OVERPASS_TIMEOUT is the per-request seconds. With no
    OVERPASS_URLS set, osmnx's own default endpoint is used (unchanged behavior).
    """
    # Fast opt-out for environments with no OSM/Overpass reachability: raising
    # here (caught by _safe_fetch) degrades to terrain-only routing instantly,
    # instead of osmnx splitting the polygon into sub-queries and retrying each
    # against an unreachable host (a multi-minute stall per route).
    if osm_disabled():
        raise RuntimeError("OSM disabled via OSM_DISABLE")

    import osmnx as ox
    from app.services.osm_settings import apply_osm_settings, overpass_urls
    from shapely.geometry import box as shapely_box

    apply_osm_settings(ox)
    ox.settings.log_console = False
    ox.settings.requests_timeout = int(os.environ.get("OVERPASS_TIMEOUT", "30"))
    bbox = shapely_box(bounds.west, bounds.south, bounds.east, bounds.north)

    endpoints = overpass_urls()
    if not endpoints:
        return ox.features_from_polygon(bbox, tags)

    last_err = None
    for url in endpoints:
        ox.settings.overpass_url = url
        try:
            return ox.features_from_polygon(bbox, tags)
        except Exception as e:  # try the next mirror
            last_err = e
            logger.warning("Overpass endpoint failed (%s): %s", url, e)
    raise last_err


# OSM features are cached on a FIXED geographic tile grid (not per-route
# bounding boxes) so overlapping routes reuse tiles instead of re-fetching data
# we already have. Tile size is in degrees; override with OSM_TILE_DEG.
_DEFAULT_TILE_DEG = 0.02

_BBox = namedtuple("_BBox", ["south", "west", "north", "east"])


def _geom_to_latlon_lines(geom) -> List[List[Tuple[float, float]]]:
    """Convert a shapely LineString/MultiLineString to lists of (lat, lon).

    Shapely stores coordinates as (x=lon, y=lat); we emit (lat, lon). Non-line
    geometries (points, polygons) are skipped — snapping only makes sense to
    linear trail features.
    """
    gtype = getattr(geom, "geom_type", None)
    if gtype == "LineString":
        return [[(y, x) for x, y in geom.coords]]
    if gtype in ("MultiLineString", "GeometryCollection"):
        out: List[List[Tuple[float, float]]] = []
        for part in geom.geoms:
            out.extend(_geom_to_latlon_lines(part))
        return out
    return []


def _concat_features(gdfs):
    """Concatenate non-empty GeoDataFrames, or return None if there are none."""
    gdfs = [g for g in gdfs if g is not None and len(g) > 0]
    if not gdfs:
        return None
    if len(gdfs) == 1:
        return gdfs[0]
    import pandas as pd

    return pd.concat(gdfs, ignore_index=True)


class PathLayer:
    """Produces terrain-type grids aligned to elevation arrays.

    OSM features are fetched from Overpass and cached per FIXED tile, so
    overlapping routes reuse cached data instead of re-hitting Overpass. A
    route's grid is composed by loading every tile it overlaps (fetching only
    the tiles not already on disk) and rasterizing their features onto the
    route's exact shape/transform.
    """

    def __init__(
        self,
        cache_dir: str,
        fetch_fn: Optional[Callable] = None,
        tile_deg: Optional[float] = None,
        strict_obstacles: bool = False,
    ):
        self.cache_dir = cache_dir
        self.fetch_fn = fetch_fn or _default_fetch
        self.tile_deg = tile_deg if tile_deg is not None else float(os.environ.get("OSM_TILE_DEG", _DEFAULT_TILE_DEG))
        # When True, get_grid raises ObstacleDataUnavailableError instead of
        # silently degrading to an obstacle-free (water-crossing) grid if a tile's
        # OSM data can't be loaded. The service turns this on whenever OSM isn't
        # explicitly disabled, so a real Overpass outage fails loudly.
        self.strict_obstacles = strict_obstacles
        os.makedirs(cache_dir, exist_ok=True)

    @staticmethod
    def get_path_type_name(code: int) -> str:
        return get_path_type_name(code)

    # --- fixed tile grid -------------------------------------------------
    def _tile_indices(self, bounds):
        deg = self.tile_deg
        lat_lo, lat_hi = math.floor(bounds.south / deg), math.floor(bounds.north / deg)
        lon_lo, lon_hi = math.floor(bounds.west / deg), math.floor(bounds.east / deg)
        return [(ti, tj) for ti in range(lat_lo, lat_hi + 1) for tj in range(lon_lo, lon_hi + 1)]

    def _tile_bounds(self, ti, tj):
        deg = self.tile_deg
        return _BBox(south=ti * deg, west=tj * deg, north=(ti + 1) * deg, east=(tj + 1) * deg)

    def _tile_cache(self, ti, tj) -> str:
        return os.path.join(self.cache_dir, f"osmtile_{ti}_{tj}.pkl")

    def _safe_fetch(self, bounds, tags):
        try:
            return self.fetch_fn(bounds, tags)
        except Exception as e:
            logger.warning(f"OSM fetch failed ({e}); continuing without this layer")
            return None

    def get_terrain_polygons(self, bounds) -> List[dict]:
        """Fetch notable terrain features (water, glacier, cliffs, scree, ...) in
        ``bounds`` for a DISPLAY overlay: ``[{"kind": str, "polygon": [[lat,lon],
        ...]}]``. Display-only — fetched fresh (glaciers etc. aren't in the
        routing tile cache) and never affects routing. Returns [] on OSM failure
        or when disabled, so the overlay simply shows nothing."""
        gdf = self._safe_fetch(bounds, TERRAIN_DISPLAY_TAGS)
        if gdf is None or len(gdf) == 0:
            return []
        out: List[dict] = []
        total_vertices = 0
        for _, row in gdf.iterrows():
            if len(out) >= _MAX_TERRAIN_FEATURES or total_vertices >= _MAX_TERRAIN_VERTICES:
                break
            geom = row.get("geometry")
            if geom is None or geom.is_empty or geom.geom_type not in ("Polygon", "MultiPolygon"):
                continue
            kind = terrain_kind(row.to_dict())
            parts = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
            for part in parts:
                ring = [[lat, lon] for lon, lat in part.exterior.coords]
                if len(ring) >= 4:
                    out.append({"kind": kind, "polygon": ring})
                    total_vertices += len(ring)
                    if len(out) >= _MAX_TERRAIN_FEATURES or total_vertices >= _MAX_TERRAIN_VERTICES:
                        break
        return out

    def _fetch_and_cache_tiles(self, missing):
        """Fetch the bounding box of the missing tiles once, split the features
        into each tile, and cache them. Only caches when BOTH the path and
        obstacle queries succeed, so a partial outage never pins incomplete data
        to disk. A single union fetch covers every missing tile (their bbox
        contains each), keeping Overpass calls to two regardless of tile count."""
        south = min(ti for ti, _ in missing) * self.tile_deg
        north = (max(ti for ti, _ in missing) + 1) * self.tile_deg
        west = min(tj for _, tj in missing) * self.tile_deg
        east = (max(tj for _, tj in missing) + 1) * self.tile_deg
        union = _BBox(south=south, west=west, north=north, east=east)

        paths_gdf = self._safe_fetch(union, PATH_TAGS)
        obstacles_gdf = self._safe_fetch(union, OBSTACLE_TAGS)
        if paths_gdf is None or obstacles_gdf is None:
            return  # outage / partial failure: cache nothing, retry next time

        for ti, tj in missing:
            tb = self._tile_bounds(ti, tj)
            tile_paths = paths_gdf.cx[tb.west : tb.east, tb.south : tb.north]
            tile_obstacles = obstacles_gdf.cx[tb.west : tb.east, tb.south : tb.north]
            with open(self._tile_cache(ti, tj), "wb") as fh:
                pickle.dump({"paths": tile_paths, "obstacles": tile_obstacles}, fh)

    def get_grid(self, bounds, shape, transform) -> np.ndarray:
        tiles = self._tile_indices(bounds)
        missing = [(ti, tj) for ti, tj in tiles if not os.path.exists(self._tile_cache(ti, tj))]
        if missing:
            self._fetch_and_cache_tiles(missing)

        # A tile still missing after the fetch attempt means the OSM/obstacle
        # query failed (a successful fetch caches even an empty tile). In strict
        # mode refuse to route rather than silently drop that tile's water.
        still_missing = [(ti, tj) for ti, tj in tiles if not os.path.exists(self._tile_cache(ti, tj))]
        if still_missing and self.strict_obstacles:
            raise ObstacleDataUnavailableError(
                f"OSM obstacle/water data unavailable for {len(still_missing)} of {len(tiles)} tile(s); "
                "refusing to route to avoid crossing unmodeled water. "
                "Set OSM_DISABLE=1 to force terrain-only routing."
            )

        paths, obstacles = [], []
        for ti, tj in tiles:
            cache_file = self._tile_cache(ti, tj)
            if not os.path.exists(cache_file):
                continue  # still missing after an outage -> degrade gracefully
            with open(cache_file, "rb") as fh:
                data = pickle.load(fh)
            if data.get("paths") is not None:
                paths.append(data["paths"])
            if data.get("obstacles") is not None:
                obstacles.append(data["obstacles"])

        return rasterize_features(_concat_features(paths), _concat_features(obstacles), shape, transform)

    def get_trail_lines(self, bounds, cached_only: bool = False) -> List[List[Tuple[float, float]]]:
        """Return path/trail geometries in ``bounds`` as lists of (lat, lon)
        points, for snapping a drawn polyline. Reuses the same fixed-tile cache
        as get_grid; degrades to [] on an OSM outage (no snapping, not a crash).

        With ``cached_only=True`` (a passive display overlay), never fetch on a
        cache miss -- just return whatever tiles are already cached. This keeps
        an untrusted, high-frequency endpoint from driving Overpass load."""
        tiles = self._tile_indices(bounds)
        missing = [(ti, tj) for ti, tj in tiles if not os.path.exists(self._tile_cache(ti, tj))]
        if missing and not cached_only:
            self._fetch_and_cache_tiles(missing)

        lines: List[List[Tuple[float, float]]] = []
        for ti, tj in tiles:
            cache_file = self._tile_cache(ti, tj)
            if not os.path.exists(cache_file):
                continue
            with open(cache_file, "rb") as fh:
                data = pickle.load(fh)
            gdf = data.get("paths")
            if gdf is None or len(gdf) == 0:
                continue
            for geom in gdf.geometry:
                lines.extend(_geom_to_latlon_lines(geom))
        return lines
