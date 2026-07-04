"""Regression tests for the security/robustness fixes from the codex audit:
extent + vertex-count guards (DoS), grid-index clamping (OOB), and JSON-safe
finite costs for impassable/NaN moves.
"""

import math

import pytest

from app.engine_v2.service import MAX_PATH_POINTS
from app.models.route import Coordinate
from tests.test_score_path_service import END, START, make_service


@pytest.mark.asyncio
async def test_rejects_oversized_extent():
    service = make_service()  # default max_distance_km = 50
    far = [Coordinate(lat=37.0, lon=-119.0), Coordinate(lat=38.7, lon=-117.0)]  # ~230 km
    with pytest.raises(ValueError):
        await service.score_path(far, {})


@pytest.mark.asyncio
async def test_rejects_too_many_points():
    service = make_service()
    pts = [Coordinate(lat=37.7400 + i * 1e-7, lon=-111.5300) for i in range(MAX_PATH_POINTS + 2)]
    with pytest.raises(ValueError):
        await service.score_path(pts, {"buffer": 0.001})


@pytest.mark.asyncio
async def test_edge_points_do_not_index_error():
    """A vertex on the bbox max edge truncates to index == shape; the clamp must
    keep it in-grid instead of IndexError / negative wraparound."""
    service = make_service()
    pts = [Coordinate(lat=37.7400, lon=-111.5350), Coordinate(lat=37.7450, lon=-111.5300)]
    scored = await service.score_path(pts, {"buffer": 0.0})  # buffer 0 → points ON the edges
    assert scored.totalCost >= 0
    assert len(scored.segments) == 1


@pytest.mark.asyncio
async def test_impassable_cost_is_json_safe_finite():
    service = make_service(hill=True)  # ridge across the middle
    scored = await service.score_path([START, END], {"buffer": 0.001, "maxSlope": 0})
    assert math.isfinite(scored.totalCost)
    assert scored.totalCost > 0
    for seg in scored.segments:
        assert all(math.isfinite(v) for v in seg.factors.values())
