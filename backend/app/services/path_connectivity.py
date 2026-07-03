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
Path connectivity repair.

Rasterizing OSM path/trail geometries onto a grid can leave small gaps
between segments that are logically connected (e.g. where a trail bends
sharply, or where two ways share an endpoint that lands on the same pixel
but adjacent line segments don't touch). These gaps break 8-connectivity
in the path raster, which can make downstream pathfinding treat two halves
of the same trail as disconnected.

This module bridges SMALL gaps (a few pixels, controlled by ``max_gap_m``)
using morphological closing, while never bridging through obstacle cells.
Bridged cells inherit the path id of the nearest existing path pixel, so
every nonzero cell in the repaired raster remains a valid key into the
``path_types`` mapping.
"""
import numpy as np
from scipy import ndimage


def _make_disk(radius_px):
    """Build a boolean disk structuring element of the given pixel radius."""
    radius_px = max(1, int(radius_px))
    y, x = np.ogrid[-radius_px : radius_px + 1, -radius_px : radius_px + 1]
    return (x**2 + y**2) <= radius_px**2


class PathConnectivityRepair:
    """Bridges small rasterization gaps between nearby path segments."""

    MAX_GAP_M = 5.0

    @staticmethod
    def repair_path_connectivity(path_raster, path_types, path_raw_tags, pixel_size_m, obstacle_mask=None):
        """
        Bridge small gaps in the path raster without filling obstacle cells.

        Args:
            path_raster: 2D uint16 array of path ids (0 = no path).
            path_types: dict {path_id: type_string}.
            path_raw_tags: dict of raw OSM tags per path id (unused, kept for
                interface compatibility with callers).
            pixel_size_m: meters per pixel.
            obstacle_mask: 2D bool array (True = impassable) or None.

        Returns:
            (repaired_raster, path_types) -- path_types is returned unchanged.
            Every nonzero cell in repaired_raster is guaranteed to be a valid
            key in path_types.
        """
        mask = path_raster > 0

        if np.count_nonzero(mask) < 2:
            return path_raster, path_types

        max_gap_m = PathConnectivityRepair.MAX_GAP_M
        gap_px = max(1, round(max_gap_m / pixel_size_m))

        structure = _make_disk(gap_px)
        closed = ndimage.binary_closing(mask, structure=structure)

        bridge = closed & (~mask)
        if obstacle_mask is not None:
            bridge &= ~obstacle_mask

        repaired = path_raster.copy()

        if np.any(bridge):
            # For every cell, find the index of the nearest True cell in `mask`.
            # (return_distances=False -> a single (ndim, *shape) index array is
            # returned, NOT a (distances, indices) tuple -- do not unpack it.)
            nearest_indices = ndimage.distance_transform_edt(~mask, return_distances=False, return_indices=True)
            nearest_ids = path_raster[nearest_indices[0], nearest_indices[1]]
            repaired[bridge] = nearest_ids[bridge]

        return repaired.astype(np.uint16), path_types
