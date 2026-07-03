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
Path expansion service.

Widens 1-pixel-wide rasterized OSM path lines into realistic corridors via
morphological dilation, with corridor width depending on path type category
(trail/path, road, or natural/area). This turns thin rasterized lines into
usable cost-surface corridors for the pathfinding engine.
"""
import math
from typing import Dict, Tuple

import numpy as np
from scipy import ndimage


def _round_half_up(value: float) -> int:
    """Round to nearest int, ties rounding away from zero.

    Python's builtin round() uses banker's rounding (round-half-to-even),
    which would round 2.5 -> 2 instead of 3. For corridor widths expressed
    in whole meters/pixels (e.g. 2.5m trail width at 1.0m pixels), we want
    the conventional "round half up" behavior so a .5 width still yields a
    corridor wider than the raw 1px line.
    """
    return int(math.floor(value + 0.5))


class PathExpander:
    """Expands rasterized path lines into realistic-width corridors."""

    # Corridor widths in meters, by category.
    TRAIL_WIDTH_M = 2.5
    ROAD_WIDTH_M = 5.0
    NATURAL_WIDTH_M = 2.0

    TRAIL_TYPES = {
        "trail",
        "path",
        "track",
        "bridleway",
        "footway",
        "steps",
        "cycleway",
        "pedestrian",
        "hike",
        "nordic",
        "downhill",
        "skitour",
        "connection",
    }

    ROAD_TYPES = {
        "residential",
        "living_street",
        "service",
        "unclassified",
        "tertiary",
        "secondary",
        "primary",
    }

    @classmethod
    def _category_for_type(cls, path_type: str) -> str:
        """Classify a path type string into 'trail', 'road', or 'natural'.

        Any type containing "piste" is treated as a trail. Anything not
        recognized as trail or road falls back to 'natural'.
        """
        if path_type in cls.TRAIL_TYPES or "piste" in path_type:
            return "trail"
        if path_type in cls.ROAD_TYPES:
            return "road"
        return "natural"

    @staticmethod
    def _disk_footprint(radius: int) -> np.ndarray:
        """Build a disk-shaped boolean footprint of the given radius."""
        y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
        return (x * x + y * y) <= (radius * radius)

    @classmethod
    def expand_paths(
        cls,
        path_raster: np.ndarray,
        path_types: Dict[int, str],
        path_raw_tags: Dict,
        pixel_size_m: float,
    ) -> Tuple[np.ndarray, Dict[int, str]]:
        """Widen rasterized path lines into realistic-width corridors.

        Args:
            path_raster: 2D uint16 array of path IDs (0 = no path).
            path_types: {path_id -> type_string}.
            path_raw_tags: {path_id -> osm tags}. Unused but preserved for
                interface compatibility.
            pixel_size_m: meters per pixel.

        Returns:
            (expanded_raster, path_types) -- path_types is returned unchanged;
            every nonzero cell in expanded_raster is guaranteed to be a key
            present in path_types.
        """
        if path_raster is None or path_types is None or len(path_types) == 0:
            return path_raster, path_types

        present_ids = np.unique(path_raster[path_raster != 0])
        if present_ids.size == 0:
            return path_raster, path_types

        # Group present ids by category, choosing a representative id for each.
        ids_by_category: Dict[str, list] = {"trail": [], "road": [], "natural": []}
        for path_id in present_ids.tolist():
            path_type = path_types.get(int(path_id))
            if path_type is None:
                # No known type for this id; skip categorization (leave as-is).
                continue
            category = cls._category_for_type(path_type)
            ids_by_category[category].append(int(path_id))

        width_m_by_category = {
            "trail": cls.TRAIL_WIDTH_M,
            "road": cls.ROAD_WIDTH_M,
            "natural": cls.NATURAL_WIDTH_M,
        }

        expanded = path_raster.copy()

        # Precedence order LOW -> HIGH: road, natural, trail (trail applied last
        # so trail corridors win over overlapping road/natural corridors).
        for category in ("road", "natural", "trail"):
            category_ids = ids_by_category[category]
            if not category_ids:
                continue

            representative_id = category_ids[0]
            width_m = width_m_by_category[category]
            width_px = max(1, _round_half_up(width_m / pixel_size_m))
            radius = max(0, (width_px - 1) // 2)

            mask = np.isin(path_raster, category_ids)

            if radius > 0:
                footprint = cls._disk_footprint(radius)
                dilated_mask = ndimage.binary_dilation(mask, structure=footprint)
            else:
                dilated_mask = mask

            expanded[dilated_mask] = representative_id

        return expanded.astype(np.uint16), path_types
