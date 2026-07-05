"""Service-level tests for score_path + the optimal<=drawn invariant.

Uses the same injected-fake elevation/path harness as test_service_v2 so it
runs offline and deterministically (no DEM download, honoring the fail-loud
rule). The detour waypoint is kept inside the straight line's bounding box so
both paths are scored on an identical grid — apples-to-apples.
"""

import numpy as np
import pytest
from rasterio.transform import from_bounds

from app.engine_v2.path_layer import PathType
from app.engine_v2.service import TrailFinderServiceV2
from app.models.route import Coordinate

SHAPE = (40, 40)


class FakeElevationLib:
    def __init__(self, hill=False):
        self.hill = hill
        self.loaded = []

    def load_area(self, bounds):
        self.loaded.append(bounds)
        return {"total_tiles": 1}

    def get_elevation_array(self, bounds):
        transform = from_bounds(bounds.west, bounds.south, bounds.east, bounds.north, SHAPE[1], SHAPE[0])
        elev = np.zeros(SHAPE, dtype=np.float32)
        if self.hill:
            # A ridge across the middle rows so crossing it costs slope.
            elev[SHAPE[0] // 2, :] = 40.0
        return elev, {"transform": transform, "resolution": 10}


class FakePathLayer:
    def get_grid(self, bounds, shape, transform):
        return np.full(shape, PathType.UNKNOWN, dtype=np.uint8)


def make_service(hill=False):
    return TrailFinderServiceV2(elevation_lib=FakeElevationLib(hill=hill), path_layer=FakePathLayer())


START = Coordinate(lat=40.6410, lon=-111.5090)
END = Coordinate(lat=40.6420, lon=-111.5080)
# Opposite corner of the START/END bounding box: a detour that does NOT enlarge
# the bbox, so both paths are scored on the same grid.
CORNER = Coordinate(lat=40.6420, lon=-111.5090)
OPTS = {"buffer": 0.001}


@pytest.mark.asyncio
async def test_score_path_returns_segments_and_total():
    service = make_service()
    scored = await service.score_path([START, END], OPTS)
    assert scored.totalCost > 0
    assert scored.distanceM > 0
    assert len(scored.segments) == 1
    assert set(scored.segments[0].factors) == {"base", "terrain", "slope", "sustained", "deviation"}
    # factors sum to the segment cost
    seg = scored.segments[0]
    assert abs(sum(seg.factors.values()) - seg.cost) < 1e-6


@pytest.mark.asyncio
async def test_detour_costs_at_least_as_much_as_direct():
    """Core of the optimal<=drawn invariant: a detour through a corner of the
    same bounding box cannot be cheaper than the direct line."""
    service = make_service()
    direct = await service.score_path([START, END], OPTS)
    detour = await service.score_path([START, CORNER, END], OPTS)
    assert direct.totalCost <= detour.totalCost + 1e-6
    assert len(detour.segments) == 2


@pytest.mark.asyncio
async def test_find_route_output_is_scorable():
    service = make_service()
    path, stats = await service.find_route(START, END, OPTS)
    assert len(path) >= 2
    scored = await service.score_path(path, OPTS)
    assert scored.totalCost > 0
    assert len(scored.segments) == len(path) - 1


@pytest.mark.asyncio
async def test_crossing_a_ridge_attributes_slope_cost():
    service = make_service(hill=True)
    # A path that climbs across the ridge should carry slope cost.
    scored = await service.score_path([START, END], OPTS)
    total_slope = sum(s.factors["slope"] for s in scored.segments)
    assert total_slope > 0
