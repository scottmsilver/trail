"""Cost-overlay (corridor tax) on the v2 pathfinder.

``cost_overlay`` multiplies per-cell terrain cost; the alternative-routes
search uses it to tax a corridor around already-found routes so the next
search prefers a genuinely different line. None => behavior unchanged.
"""

import numpy as np
from rasterio.transform import from_bounds

from app.engine_v2.path_layer import PathType
from app.engine_v2.pathfinder import TerrainAwarePathfinder

SHAPE = (30, 30)
TRANSFORM = from_bounds(-111.51, 40.64, -111.507, 40.643, SHAPE[1], SHAPE[0])
FLAT = np.zeros(SHAPE, dtype=np.float32)
OPEN = np.full(SHAPE, PathType.UNKNOWN, dtype=np.uint8)

# Vertical crossing: start near the south edge, end near the north edge of the
# same column, so every path must pass through the middle rows.
START = (40.6402, -111.5085)
END = (40.6428, -111.5085)


def _pf(overlay=None):
    pf = TerrainAwarePathfinder(FLAT.copy(), TRANSFORM, OPEN.copy(), resolution=10.0)
    if overlay is not None:
        pf.set_parameters(cost_overlay=overlay)
    return pf


def _mid_cols(path, pf):
    """Columns used while crossing the middle rows."""
    cols = []
    for lat, lon, _z in path:
        r, c = pf.lat_lon_to_grid(lat, lon)
        if 12 <= r <= 18:
            cols.append(c)
    return cols


def test_no_overlay_goes_straight():
    pf = _pf()
    path, _ = pf.find_path(*START, *END)
    mid = _mid_cols(path, pf)
    assert mid and all(abs(c - 15) <= 1 for c in mid)  # stays on its column


def test_overlay_taxes_corridor_and_diverts_path():
    # Tax a wide vertical corridor around the straight line through the middle
    # rows; the path should detour around it even though the ground is flat.
    overlay = np.ones(SHAPE, dtype=np.float32)
    overlay[10:20, 10:21] = 8.0
    pf = _pf(overlay)
    path, _ = pf.find_path(*START, *END)
    assert path is not None
    mid = _mid_cols(path, pf)
    assert mid and all(c < 10 or c > 20 for c in mid), f"path did not divert: cols {sorted(set(mid))}"


def test_overlay_is_a_tax_not_a_wall():
    # If the whole width is taxed, the route still crosses (pays the tax).
    overlay = np.ones(SHAPE, dtype=np.float32)
    overlay[14:16, :] = 8.0
    pf = _pf(overlay)
    assert pf.find_path(*START, *END) is not None
