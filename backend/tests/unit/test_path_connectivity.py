# RULE #0: MANDATORY FIRST ACTION FOR EVERY REQUEST
# 1. Read CLAUDE.md COMPLETELY before responding
# 2. Setup Python venv: [ -d "venv" ] || ./setup-venv.sh && source venv/bin/activate
# 3. Search for rules related to the request
# 4. Only proceed after confirming no violations
# Failure to follow Rule #0 has caused real harm. Check BEFORE acting, not AFTER making mistakes.
#
# GUARDS ARE SAFETY EQUIPMENT - WHEN THEY FIRE, FIX THE PROBLEM THEY FOUND
# NEVER weaken, disable, or bypass guards - they prevent real harm
"""Unit tests for PathConnectivityRepair.repair_path_connectivity (gap-bridging)."""
import numpy as np
from app.services.path_connectivity import PathConnectivityRepair
from scipy import ndimage


def _make_gapped_horizontal_line(shape, row, col_start, col_end, gap_start, gap_len, path_id, width=3):
    """Build a horizontal corridor (width rows tall) with a gap of `gap_len`
    pixels starting at gap_start. A multi-row width mirrors realistic input:
    in production this repair runs AFTER corridor expansion (see
    dem_tile_cache.py), so path segments are rarely a mathematically-thin
    1px line. Morphological closing with a disk element -- by construction --
    cannot bridge a gap in a true 1px-wide line for ANY radius (the erosion
    step always finds an uncovered "shadow" pixel at the tangent point), so a
    small corridor width is required for the closing approach to work at all.
    """
    raster = np.zeros(shape, dtype=np.uint16)
    half = width // 2
    raster[row - half : row + half + 1, col_start:col_end] = path_id
    raster[row - half : row + half + 1, gap_start : gap_start + gap_len] = 0
    return raster


class TestPathConnectivityRepair:
    def test_small_gap_is_bridged_and_becomes_single_component(self):
        shape = (30, 30)
        row = 15
        # Two collinear segments: [5,14) and [16,25) with a 2px gap at [14,16)
        raster = _make_gapped_horizontal_line(shape, row, col_start=5, col_end=25, gap_start=14, gap_len=2, path_id=1)
        path_types = {1: "trail"}
        path_raw_tags = {1: {}}

        before_mask = raster > 0
        _, before_count = ndimage.label(before_mask, structure=np.ones((3, 3)))
        assert before_count == 2, "Sanity check: gap should split into two components before repair"

        repaired, types_out = PathConnectivityRepair.repair_path_connectivity(
            raster, path_types, path_raw_tags, pixel_size_m=1.0, obstacle_mask=None
        )

        assert types_out == path_types

        # Gap cells should now be nonzero and equal to id 1
        gap_cells = repaired[row, 14:16]
        assert np.all(gap_cells == 1), f"Expected gap cells to be bridged with id 1, got {gap_cells}"

        # Segments should now be a single 8-connected component
        after_mask = repaired > 0
        _, after_count = ndimage.label(after_mask, structure=np.ones((3, 3)))
        assert after_count == 1, "Expected bridged segments to form a single connected component"

    def test_gap_over_obstacle_is_not_bridged(self):
        shape = (30, 30)
        row = 15
        raster = _make_gapped_horizontal_line(shape, row, col_start=5, col_end=25, gap_start=14, gap_len=2, path_id=1)
        path_types = {1: "trail"}
        path_raw_tags = {1: {}}

        obstacle_mask = np.zeros(shape, dtype=bool)
        obstacle_mask[row - 1 : row + 2, 14:16] = True

        repaired, types_out = PathConnectivityRepair.repair_path_connectivity(
            raster, path_types, path_raw_tags, pixel_size_m=1.0, obstacle_mask=obstacle_mask
        )

        assert types_out == path_types
        gap_cells = repaired[row - 1 : row + 2, 14:16]
        assert np.all(gap_cells == 0), "Gap over obstacle must not be bridged"

    def test_bridged_cells_are_valid_path_types_keys(self):
        shape = (30, 30)
        row = 15
        raster = _make_gapped_horizontal_line(shape, row, col_start=5, col_end=25, gap_start=14, gap_len=2, path_id=1)
        path_types = {1: "trail"}
        path_raw_tags = {1: {}}

        repaired, types_out = PathConnectivityRepair.repair_path_connectivity(
            raster, path_types, path_raw_tags, pixel_size_m=1.0, obstacle_mask=None
        )

        nonzero_ids = set(np.unique(repaired[repaired != 0]).tolist())
        assert nonzero_ids.issubset(set(types_out.keys()))
        assert len(nonzero_ids) > 0

    def test_large_gap_is_not_fully_bridged(self):
        shape = (40, 40)
        row = 20
        # max_gap_m = 5.0, pixel_size_m = 1.0 -> gap_px = 5, closing bridges up to ~2*gap_px = 10px.
        # Use a gap much larger than that (20px) to ensure it stays disconnected.
        # Segments: [2,8) and [28,38) separated by a 20px gap at [8,28).
        raster = _make_gapped_horizontal_line(shape, row, col_start=2, col_end=38, gap_start=8, gap_len=20, path_id=1)
        path_types = {1: "trail"}
        path_raw_tags = {1: {}}

        repaired, types_out = PathConnectivityRepair.repair_path_connectivity(
            raster, path_types, path_raw_tags, pixel_size_m=1.0, obstacle_mask=None
        )

        assert types_out == path_types

        after_mask = repaired > 0
        _, after_count = ndimage.label(after_mask, structure=np.ones((3, 3)))
        assert after_count > 1, "Expected large gap to remain only partially bridged (still >1 component)"

    def test_empty_raster_returned_unchanged(self):
        shape = (10, 10)
        raster = np.zeros(shape, dtype=np.uint16)
        path_types = {}
        path_raw_tags = {}

        repaired, types_out = PathConnectivityRepair.repair_path_connectivity(
            raster, path_types, path_raw_tags, pixel_size_m=1.0, obstacle_mask=None
        )

        assert np.array_equal(repaired, raster)
        assert types_out == path_types

    def test_single_cell_raster_returned_unchanged(self):
        shape = (10, 10)
        raster = np.zeros(shape, dtype=np.uint16)
        raster[5, 5] = 1
        path_types = {1: "trail"}
        path_raw_tags = {1: {}}

        repaired, types_out = PathConnectivityRepair.repair_path_connectivity(
            raster, path_types, path_raw_tags, pixel_size_m=1.0, obstacle_mask=None
        )

        assert np.array_equal(repaired, raster)
        assert types_out == path_types
