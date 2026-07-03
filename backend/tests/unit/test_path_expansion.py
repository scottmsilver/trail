# RULE #0: MANDATORY FIRST ACTION FOR EVERY REQUEST
# 1. Read CLAUDE.md COMPLETELY before responding
# 2. Setup Python venv: [ -d "venv" ] || ./setup-venv.sh && source venv/bin/activate
# 3. Search for rules related to the request
# 4. Only proceed after confirming no violations
# Failure to follow Rule #0 has caused real harm. Check BEFORE acting, not AFTER making mistakes.
#
# GUARDS ARE SAFETY EQUIPMENT - WHEN THEY FIRE, FIX THE PROBLEM THEY FOUND
# NEVER weaken, disable, or bypass guards - they prevent real harm
"""Unit tests for PathExpander.expand_paths (corridor widening of rasterized paths)."""
import numpy as np
from app.services.path_expansion import PathExpander


def _make_horizontal_line_raster(shape, row, col_start, col_end, path_id):
    raster = np.zeros(shape, dtype=np.uint16)
    raster[row, col_start:col_end] = path_id
    return raster


class TestPathExpander:
    def test_single_trail_line_expands_into_corridor(self):
        shape = (40, 40)
        raster = _make_horizontal_line_raster(shape, row=20, col_start=5, col_end=35, path_id=1)
        path_types = {1: "trail"}
        path_raw_tags = {1: {}}

        expanded, types_out = PathExpander.expand_paths(raster, path_types, path_raw_tags, pixel_size_m=1.0)

        assert types_out == path_types

        # Original line row should still be nonzero
        assert np.count_nonzero(expanded[20, 5:35]) > 0

        # Corridor should now cover more than 1 row for at least one column
        col = 20
        nonzero_rows_in_col = np.count_nonzero(expanded[:, col])
        assert nonzero_rows_in_col >= 2, "Expected corridor to widen beyond a single pixel row"

        # Invariant: every nonzero cell must map to a valid path_types key
        nonzero_ids = set(np.unique(expanded[expanded != 0]).tolist())
        assert nonzero_ids.issubset(set(types_out.keys()))

    def test_road_expands_wider_than_trail(self):
        shape = (40, 40)

        trail_raster = _make_horizontal_line_raster(shape, row=20, col_start=5, col_end=35, path_id=1)
        trail_types = {1: "trail"}

        road_raster = _make_horizontal_line_raster(shape, row=20, col_start=5, col_end=35, path_id=2)
        road_types = {2: "residential"}

        expanded_trail, _ = PathExpander.expand_paths(trail_raster, trail_types, {1: {}}, pixel_size_m=1.0)
        expanded_road, _ = PathExpander.expand_paths(road_raster, road_types, {2: {}}, pixel_size_m=1.0)

        trail_count = np.count_nonzero(expanded_trail)
        road_count = np.count_nonzero(expanded_road)

        assert road_count > trail_count, "Road corridor should be wider (more nonzero cells) than trail corridor"

    def test_overlap_trail_wins_over_road(self):
        shape = (40, 40)
        raster = np.zeros(shape, dtype=np.uint16)

        # Horizontal road crossing a vertical trail at the same point
        raster[20, 5:35] = 2  # road, id=2
        raster[5:35, 20] = 1  # trail, id=1

        path_types = {1: "trail", 2: "residential"}
        path_raw_tags = {1: {}, 2: {}}

        expanded, types_out = PathExpander.expand_paths(raster, path_types, path_raw_tags, pixel_size_m=1.0)

        # At the crossing point, trail (id=1) should win due to precedence
        assert expanded[20, 20] == 1

        # Invariant holds
        nonzero_ids = set(np.unique(expanded[expanded != 0]).tolist())
        assert nonzero_ids.issubset(set(types_out.keys()))

    def test_empty_raster_returned_unchanged(self):
        shape = (10, 10)
        raster = np.zeros(shape, dtype=np.uint16)
        path_types = {}
        path_raw_tags = {}

        expanded, types_out = PathExpander.expand_paths(raster, path_types, path_raw_tags, pixel_size_m=1.0)

        assert np.array_equal(expanded, raster)
        assert types_out == path_types

    def test_output_ids_always_valid_keys_in_path_types(self):
        shape = (50, 50)
        raster = np.zeros(shape, dtype=np.uint16)
        raster[10, 5:45] = 1  # trail
        raster[30, 5:45] = 2  # residential road
        raster[20:25, 25] = 3  # footway (trail-category)

        path_types = {1: "trail", 2: "residential", 3: "footway"}
        path_raw_tags = {1: {}, 2: {}, 3: {}}

        expanded, types_out = PathExpander.expand_paths(raster, path_types, path_raw_tags, pixel_size_m=1.0)

        nonzero_ids = set(np.unique(expanded[expanded != 0]).tolist())
        assert nonzero_ids.issubset(set(types_out.keys()))
        assert len(nonzero_ids) > 0
