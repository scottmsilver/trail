"""Extent-aware passability gate on the v2 pathfinder.

The classic gate is memoryless: any single move steeper than ``max_slope_degrees``
is impassable. The extent-aware gate (opt-in via ``scramble_budget_m``) instead
blocks only when the *continuous* vertical climb on steep ground exceeds a budget
that stands in for hiker expertise (~4 m hiker .. ~15 m alpinist). A short
scramble-able step passes; a sustained wall does not; a benched staircase of the
same total gain passes because each run resets at the bench.

See docs/terrain-passability-extent-aware.md.
"""

import numpy as np
from rasterio.transform import from_bounds

from app.engine_v2.path_layer import PathType
from app.engine_v2.pathfinder import TerrainAwarePathfinder

# Fine grid so a "steep but only a few metres per cell" wall is expressible.
# At 2 m/cell, an orthogonal move needs >2 m rise to exceed 45°; a diagonal
# move needs >2.83 m. We use 3 m/cell rises so BOTH orthogonal (56°) and
# diagonal (47°) crossings count as steep — no gentle-diagonal escape.
SHAPE = (40, 12)
RES = 2.0
TRANSFORM = from_bounds(-111.60, 40.60, -111.599, 40.604, SHAPE[1], SHAPE[0])
OPEN = np.full(SHAPE, PathType.UNKNOWN, dtype=np.uint8)


def _pf(elevation, **kw):
    pf = TerrainAwarePathfinder(elevation.astype(np.float32), TRANSFORM, OPEN.copy(), resolution=RES)
    pf.set_parameters(**kw)
    return pf


def _elev_by_row(row_to_z):
    """Build an elevation grid constant across each row (a N-S profile)."""
    e = np.zeros(SHAPE, dtype=np.float32)
    for r in range(SHAPE[0]):
        e[r, :] = row_to_z(r)
    return e


def _continuous_wall():
    # Flat below row 23; a 3 m/row ramp up rows 22..18 (15 m continuous, no
    # bench); flat plateau at 15 m above. Every crossing of the ramp is steep.
    def z(r):
        if r >= 23:
            return 0.0
        if r <= 18:
            return 15.0
        return 3.0 * (23 - r)  # r=22->3 .. r=18->15

    return _elev_by_row(z)


def _benched_staircase():
    # Same 15 m total gain, but as alternating 3 m steps and flat benches, so no
    # continuous steep run exceeds 3 m.
    steps = {23: 0, 22: 3, 21: 3, 20: 6, 19: 6, 18: 9, 17: 9, 16: 12, 15: 12, 14: 15}

    def z(r):
        if r >= 23:
            return 0.0
        if r <= 14:
            return 15.0
        return float(steps[r])

    return _elev_by_row(z)


# start on the low flat, end on the high plateau — the crossing is unavoidable.
def _ends(pf):
    return (*pf.grid_to_lat_lon(30, 6), *pf.grid_to_lat_lon(8, 6))


def test_memoryless_gate_blocks_the_steep_wall():
    # Baseline: today's gate kills any route because every crossing exceeds 45°.
    pf = _pf(_continuous_wall(), max_slope_degrees=45.0)
    assert pf.find_path(*_ends(pf)) is None


def test_extent_aware_low_budget_blocks_sustained_wall():
    # A hiker (4 m budget) cannot climb a continuous 15 m steep wall.
    pf = _pf(_continuous_wall(), scramble_budget_m=4.0)
    assert pf.find_path(*_ends(pf)) is None


def test_extent_aware_high_budget_passes_wall_the_memoryless_gate_killed():
    # An alpinist (20 m budget) CAN take the 15 m wall the memoryless gate wrongly
    # blocked — the core thesis: expertise changes the verdict on the same ground.
    pf = _pf(_continuous_wall(), scramble_budget_m=20.0)
    result = pf.find_path(*_ends(pf))
    assert result is not None
    path, _ = result
    # It actually reaches the high plateau.
    assert path[-1][2] >= 14.0


def test_extent_aware_low_budget_passes_benched_staircase():
    # Same 15 m total gain, but benched into 3 m steps — a hiker passes because
    # each steep run resets at a bench. Proves the gate measures continuous
    # climb (extent), not total gain.
    pf = _pf(_benched_staircase(), scramble_budget_m=4.0)
    result = pf.find_path(*_ends(pf))
    assert result is not None
    path, _ = result
    assert path[-1][2] >= 14.0


def test_budget_none_is_classic_behavior_untouched():
    # With no budget set, extent-aware is inert: identical to the memoryless gate.
    wall = _continuous_wall()
    assert _pf(wall, max_slope_degrees=45.0).find_path(*_ends(_pf(wall))) is None
    # ...and a gentle grid still routes fine.
    flat = np.zeros(SHAPE, dtype=np.float32)
    pf = _pf(flat)
    assert pf.find_path(*_ends(pf)) is not None
