"""Tests for the trails-overlay service method (viewport -> cached OSM trail
geometry), including the viewport-size guard. Offline via a fake path layer."""

import numpy as np
import pytest

from app.engine_v2.path_layer import PathType
from app.engine_v2.service import MAX_TRAILS_SPAN_DEG, TrailFinderServiceV2
from tests.test_score_path_service import FakeElevationLib

TRAIL = [(40.640, -111.510), (40.641, -111.509), (40.642, -111.508)]


class FakeTrailPathLayer:
    tile_deg = 0.02

    def __init__(self):
        self.bounds_seen = None
        self.cached_only_seen = None

    def get_grid(self, bounds, shape, transform):
        return np.full(shape, PathType.UNKNOWN, dtype=np.uint8)

    def get_trail_lines(self, bounds, cached_only=False):
        self.bounds_seen = bounds
        self.cached_only_seen = cached_only
        return [list(TRAIL)]


def make_service():
    pl = FakeTrailPathLayer()
    return TrailFinderServiceV2(elevation_lib=FakeElevationLib(), path_layer=pl), pl


@pytest.mark.asyncio
async def test_trail_lines_returns_geometry_for_viewport():
    service, pl = make_service()
    lines = await service.trail_lines_in_bounds(south=40.63, west=-111.52, north=40.65, east=-111.50)
    assert lines == [TRAIL]
    # Bounds forwarded to the path layer verbatim (south/west/north/east).
    assert pl.bounds_seen.south == 40.63 and pl.bounds_seen.north == 40.65
    assert pl.bounds_seen.west == -111.52 and pl.bounds_seen.east == -111.50
    # A passive overlay must read cached tiles only, never trigger a fetch.
    assert pl.cached_only_seen is True


@pytest.mark.asyncio
async def test_rejects_inverted_bounds():
    service, _ = make_service()
    with pytest.raises(ValueError):
        await service.trail_lines_in_bounds(south=40.65, west=-111.50, north=40.63, east=-111.52)


@pytest.mark.asyncio
async def test_rejects_out_of_range_and_nonfinite_bounds():
    service, _ = make_service()
    with pytest.raises(ValueError):  # lat > 90
        await service.trail_lines_in_bounds(south=89.9, west=-111.5, north=91.0, east=-111.4)
    with pytest.raises(ValueError):  # non-finite
        await service.trail_lines_in_bounds(south=float("inf"), west=-111.5, north=40.6, east=-111.4)


@pytest.mark.asyncio
async def test_rejects_oversized_viewport():
    service, _ = make_service()
    big = MAX_TRAILS_SPAN_DEG + 0.5
    with pytest.raises(ValueError):
        await service.trail_lines_in_bounds(south=40.0, west=-111.0, north=40.0 + big, east=-111.0 + 0.1)
