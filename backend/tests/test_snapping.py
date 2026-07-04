"""Offline tests for the pure snap-to-trail geometry."""

from app.engine_v2.snapping import densify_polyline, snap_polyline_to_lines
from app.models.route import Coordinate

# One straight trail running east-west along latitude 37.7400.
TRAIL = [[(37.7400, -111.5350), (37.7400, -111.5300)]]


def test_densify_inserts_points_and_keeps_endpoints():
    # ~445 m east-west segment; step 15 m -> ~29 interior points.
    pts = [Coordinate(lat=37.7400, lon=-111.5350), Coordinate(lat=37.7400, lon=-111.5300)]
    dense = densify_polyline(pts, step_m=15, max_points=1500)
    assert len(dense) > 20
    assert dense[0].lat == pts[0].lat and dense[0].lon == pts[0].lon  # endpoints preserved
    assert dense[-1].lat == pts[1].lat and dense[-1].lon == pts[1].lon
    # interior points lie on the segment (constant latitude) and are ordered
    lons = [p.lon for p in dense]
    assert all(abs(p.lat - 37.7400) < 1e-9 for p in dense)
    assert lons == sorted(lons)


def test_densify_caps_at_max_points():
    pts = [Coordinate(lat=37.0, lon=-112.0), Coordinate(lat=37.0, lon=-111.0)]  # ~88 km
    dense = densify_polyline(pts, step_m=1, max_points=500)  # 1 m step would be ~88k pts
    assert len(dense) <= 500


def test_densify_noop_on_degenerate_input():
    p = Coordinate(lat=37.74, lon=-111.53)
    assert densify_polyline([p], step_m=15, max_points=100) == [p]  # <2 points
    assert densify_polyline([p, p], step_m=15, max_points=100) == [p, p]  # zero length


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


@pytest.mark.asyncio
async def test_service_densifies_so_whole_line_hugs_trail():
    """A sparse 2-click path near the trail should come back as MANY points, all
    pulled onto the trail — not just its two endpoints nudged."""
    service = TrailFinderServiceV2(elevation_lib=_FakeElev(), path_layer=_FakePathLayerWithTrail())
    # Two clicks ~40 m north of the trail, ~440 m apart along it.
    pts = [Coordinate(lat=37.7404, lon=-111.5350), Coordinate(lat=37.7404, lon=-111.5306)]
    snapped, did = await service.snap_to_trails(pts, {"buffer": 0.01})  # default 60 m threshold
    assert did is True
    assert len(snapped) > 10  # densified, not just the 2 original clicks
    assert all(abs(p.lat - 37.7400) < 1e-4 for p in snapped)  # the entire line hugs the trail
