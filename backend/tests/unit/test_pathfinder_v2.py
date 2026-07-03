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
