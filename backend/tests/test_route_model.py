"""RouteRequest gains an optional `points` list for multi-waypoint routing while
staying back-compatible with the legacy {start, end} shape. Both normalize to a
single ordered points list via normalized_points()."""

import pytest
from app.models.route import RouteRequest
from pydantic import ValidationError

A = {"lat": 40.10, "lon": -111.10}
B = {"lat": 40.20, "lon": -111.20}
C = {"lat": 40.30, "lon": -111.30}


def test_legacy_start_end_normalizes_to_two_points():
    req = RouteRequest(start=A, end=B)
    pts = req.normalized_points()
    assert [(p.lat, p.lon) for p in pts] == [(40.10, -111.10), (40.20, -111.20)]


def test_points_list_normalizes_to_itself():
    req = RouteRequest(points=[A, B, C])
    assert [(p.lat, p.lon) for p in req.normalized_points()] == [
        (40.10, -111.10),
        (40.20, -111.20),
        (40.30, -111.30),
    ]


def test_points_takes_precedence_over_start_end():
    req = RouteRequest(start=A, end=B, points=[A, B, C])
    assert len(req.normalized_points()) == 3


def test_points_requires_at_least_two():
    with pytest.raises(ValidationError):
        RouteRequest(points=[A])


def test_missing_everything_is_rejected():
    with pytest.raises(ValidationError):
        RouteRequest()


def test_start_without_end_is_rejected():
    with pytest.raises(ValidationError):
        RouteRequest(start=A)


def test_too_many_points_rejected():
    with pytest.raises(ValidationError):
        RouteRequest(points=[A] * 26)


def test_max_points_allowed():
    # 25 is the cap and must be accepted.
    req = RouteRequest(points=[A] * 25)
    assert len(req.normalized_points()) == 25


def test_variants_request_requires_start_end():
    from app.models.route import RouteVariantsRequest

    with pytest.raises(ValidationError):
        RouteVariantsRequest(points=[A, B])  # points-only, no start/end
    # start+end is accepted
    req = RouteVariantsRequest(start=A, end=B)
    assert req.start.lat == A["lat"]
