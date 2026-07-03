# RULE #0: MANDATORY FIRST ACTION FOR EVERY REQUEST
# 1. Read CLAUDE.md COMPLETELY before responding
# 2. Setup Python venv: [ -d "venv" ] || ./setup-venv.sh && source venv/bin/activate
# 3. Search for rules related to the request
# 4. Only proceed after confirming no violations
# Failure to follow Rule #0 has caused real harm. Check BEFORE acting, not AFTER making mistakes.
#
# GUARDS ARE SAFETY EQUIPMENT - WHEN THEY FIRE, FIX THE PROBLEM THEY FOUND
# NEVER weaken, disable, or bypass guards - they prevent real harm
"""Unit tests for the v2 terrain-aware pathfinder on synthetic terrain."""
import numpy as np
from app.engine_v2.path_layer import PathType
from app.engine_v2.pathfinder import TerrainAwarePathfinder
from rasterio.transform import from_bounds

SHAPE = (30, 30)
# 30x30 grid over a small box; ~11m per cell N-S
TRANSFORM = from_bounds(-111.51, 40.64, -111.507, 40.643, SHAPE[1], SHAPE[0])
FLAT = np.zeros(SHAPE, dtype=np.float32)
OPEN = np.full(SHAPE, PathType.UNKNOWN, dtype=np.uint8)

START = (40.6402, -111.5098)  # bottom-left area
END = (40.6428, -111.5072)  # top-right area


def make_pathfinder(elevation=FLAT, terrain=OPEN, **kwargs):
    return TerrainAwarePathfinder(elevation.copy(), TRANSFORM, terrain.copy(), resolution=10.0, **kwargs)


def test_finds_path_on_flat_open_terrain():
    pf = make_pathfinder()
    result = pf.find_path(*START, *END)
    assert result is not None
    path, stats = result
    assert len(path) > 5
    assert stats["distance_m"] > 0
    assert stats["nodes_explored"] > 0
    # Endpoints are where we asked
    assert abs(path[0][0] - START[0]) < 0.001
    assert abs(path[-1][0] - END[0]) < 0.001


def test_obstacle_wall_blocks_direct_route():
    terrain = OPEN.copy()
    terrain[10:20, :25] = PathType.OBSTACLE  # wall with a gap on the right
    pf = make_pathfinder(terrain=terrain)
    result = pf.find_path(*START, *END)
    assert result is not None
    path, _ = result
    # Path must detour through the gap: some point in the wall rows must be
    # east of the wall (col >= 25 → lon > left + 25/30 of width)
    gap_lon = -111.51 + 0.003 * (25 / 30)
    # Strictly inside the wall band: corner latitudes of adjacent open rows
    # (e.g. row 20) coincide exactly with the inclusive bounds.
    rows_in_wall = [p for p in path if 40.64 + 0.003 * (10 / 30) < p[0] < 40.64 + 0.003 * (20 / 30)]
    assert rows_in_wall, "path never crossed the wall band"
    assert all(p[1] >= gap_lon for p in rows_in_wall)


def test_full_obstacle_wall_means_no_path():
    terrain = OPEN.copy()
    terrain[15, :] = PathType.OBSTACLE  # full-width wall
    pf = make_pathfinder(terrain=terrain)
    assert pf.find_path(*START, *END) is None


def test_prefers_trail_over_off_path():
    terrain = OPEN.copy()
    terrain[:, 15] = PathType.TRAIL  # vertical trail down the middle
    pf = make_pathfinder(terrain=terrain)
    result = pf.find_path(40.6401, -111.5085, 40.6428, -111.5085)  # start/end on col 15
    assert result is not None
    path, stats = result
    assert stats["terrain_breakdown"].get("trail", 0) > 0.8


def test_max_slope_enforced():
    elevation = FLAT.copy()
    # A cliff: +100m from row 15 down — every N-S move across it exceeds 45°
    elevation[15:, :] += 100.0
    pf = make_pathfinder(elevation=elevation)
    pf.set_parameters(max_slope_degrees=45.0)
    assert pf.find_path(*START, *END) is None


class TestWeightedAStar:
    def test_weight_one_follows_cheap_corridor(self):
        # Optimal search must take the trail corridor even though it's longer
        terrain = OPEN.copy()
        terrain[:, 5] = PathType.TRAIL  # cheap vertical corridor at col 5
        pf = make_pathfinder(terrain=terrain, heuristic_weight=1.0)
        result = pf.find_path(40.6401, -111.5095, 40.6428, -111.5095)  # along col 5
        assert result is not None
        _, stats = result
        assert stats["terrain_breakdown"].get("trail", 0) > 0.8

    def test_higher_weight_explores_fewer_nodes(self):
        elevation = FLAT.copy()
        # Bumpy terrain so the search actually has work to do
        rng_rows, rng_cols = np.meshgrid(np.arange(SHAPE[0]), np.arange(SHAPE[1]), indexing="ij")
        elevation += 3.0 * np.sin(rng_rows / 3.0) * np.cos(rng_cols / 3.0)

        pf1 = make_pathfinder(elevation=elevation, heuristic_weight=1.0)
        r1 = pf1.find_path(*START, *END)
        pf2 = make_pathfinder(elevation=elevation, heuristic_weight=2.0)
        r2 = pf2.find_path(*START, *END)

        assert r1 is not None and r2 is not None
        _, stats1 = r1
        path2, stats2 = r2
        assert stats2["nodes_explored"] < stats1["nodes_explored"]
        # Bounded quality loss (spec: ~15%; allow slack on tiny grids)
        assert stats2["distance_m"] <= stats1["distance_m"] * 1.2
