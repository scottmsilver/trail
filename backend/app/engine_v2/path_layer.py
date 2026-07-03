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
from enum import IntEnum
from typing import Optional

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

# Ported from ObstacleConfig.get_default_osm_tags()
OBSTACLE_TAGS = {
    "natural": ["water", "wetland", "cliff", "rock", "scree"],
    "waterway": ["river", "stream", "canal"],
    "landuse": ["industrial", "commercial", "military"],
    "building": True,
    "leisure": ["golf_course", "swimming_pool"],
    "barrier": True,
}

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
            shapes_by_type[PathType.OBSTACLE].append(geom)

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
