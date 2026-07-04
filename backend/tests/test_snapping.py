"""Offline tests for the pure snap-to-trail geometry."""

from app.engine_v2.snapping import snap_polyline_to_lines
from app.models.route import Coordinate

# One straight trail running east-west along latitude 37.7400.
TRAIL = [[(37.7400, -111.5350), (37.7400, -111.5300)]]


def test_snaps_nearby_point_onto_trail():
    # ~22 m north of the trail (0.0002 deg lat ≈ 22 m).
    pts = [Coordinate(lat=37.7402, lon=-111.5320), Coordinate(lat=37.7402, lon=-111.5310)]
    snapped, did = snap_polyline_to_lines(pts, TRAIL, threshold_m=50)
    assert did is True
    assert abs(snapped[0].lat - 37.7400) < 1e-4  # pulled onto the trail's latitude
    assert abs(snapped[0].lon - (-111.5320)) < 1e-4  # longitude ~unchanged


def test_far_point_left_alone():
    far = Coordinate(lat=37.7600, lon=-111.5000)
    snapped, did = snap_polyline_to_lines([far, far], TRAIL, threshold_m=50)
    assert did is False
    assert snapped[0].lat == far.lat and snapped[0].lon == far.lon


def test_idempotent():
    pts = [Coordinate(lat=37.7402, lon=-111.5320), Coordinate(lat=37.7401, lon=-111.5310)]
    once, _ = snap_polyline_to_lines(pts, TRAIL, threshold_m=50)
    twice, did2 = snap_polyline_to_lines(once, TRAIL, threshold_m=50)
    assert did2 is False  # already on the trail → nothing moves
    for a, b in zip(once, twice):
        assert abs(a.lat - b.lat) < 1e-9 and abs(a.lon - b.lon) < 1e-9


def test_no_lines_returns_unchanged():
    pts = [Coordinate(lat=37.74, lon=-111.53), Coordinate(lat=37.75, lon=-111.52)]
    snapped, did = snap_polyline_to_lines(pts, [], threshold_m=50)
    assert did is False
    assert [(p.lat, p.lon) for p in snapped] == [(37.74, -111.53), (37.75, -111.52)]


# --- service-level wiring (async, fake path layer) ---------------------------
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from rasterio.transform import from_bounds  # noqa: E402

from app.engine_v2.path_layer import PathType  # noqa: E402
from app.engine_v2.service import TrailFinderServiceV2  # noqa: E402

_SHAPE = (40, 40)


class _FakeElev:
    def load_area(self, bounds):
        return {"total_tiles": 1}

    def get_elevation_array(self, bounds):
        t = from_bounds(bounds.west, bounds.south, bounds.east, bounds.north, _SHAPE[1], _SHAPE[0])
        return np.zeros(_SHAPE, dtype=np.float32), {"transform": t, "resolution": 10}


class _FakePathLayerWithTrail:
    def get_grid(self, bounds, shape, transform):
        return np.full(shape, PathType.UNKNOWN, dtype=np.uint8)

    def get_trail_lines(self, bounds):
        return TRAIL


@pytest.mark.asyncio
async def test_service_snaps_to_trail_lines():
    service = TrailFinderServiceV2(elevation_lib=_FakeElev(), path_layer=_FakePathLayerWithTrail())
    pts = [Coordinate(lat=37.7402, lon=-111.5320), Coordinate(lat=37.7402, lon=-111.5310)]
    snapped, did = await service.snap_to_trails(pts, {"buffer": 0.01}, threshold_m=50)
    assert did is True
    assert abs(snapped[0].lat - 37.7400) < 1e-4
