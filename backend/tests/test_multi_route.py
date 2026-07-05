"""find_multi_route stitches N-1 single-pair A* legs into one path + aggregated
stats. Reuses the offline injected-fake harness so it runs deterministically
without DEM downloads."""

import pytest
from tests.test_score_path_service import CORNER, END, START, make_service

# Distinct waypoints inside a small bbox so every leg routes on flat terrain.
P1 = START
P2 = CORNER
P3 = END
OPTS = {"buffer": 0.001}


@pytest.mark.asyncio
async def test_three_points_stitch_without_duplicated_junction():
    service = make_service()
    path, stats = await service.find_multi_route([P1, P2, P3], dict(OPTS))
    assert len(path) > 2
    # No consecutive duplicate coordinates anywhere in the stitched path.
    for a, b in zip(path, path[1:]):
        assert (a.lat, a.lon) != (b.lat, b.lon)


@pytest.mark.asyncio
async def test_stitched_stats_sum_legs():
    service = make_service()
    _, s12 = await service.find_route(P1, P2, dict(OPTS))
    _, s23 = await service.find_route(P2, P3, dict(OPTS))
    _, multi = await service.find_multi_route([P1, P2, P3], dict(OPTS))
    assert multi["distance_m"] == pytest.approx(s12["distance_m"] + s23["distance_m"], rel=1e-6)
    assert multi["elevation_gain_m"] == pytest.approx(s12["elevation_gain_m"] + s23["elevation_gain_m"], rel=1e-6)
    assert multi["waypoints"] == len(multi["path_with_slopes"])
    assert [leg["from"] for leg in multi["legs"]] == [0, 1]


@pytest.mark.asyncio
async def test_two_points_matches_single_find_route_length():
    service = make_service()
    single_path, _ = await service.find_route(P1, P3, dict(OPTS))
    multi_path, _ = await service.find_multi_route([P1, P3], dict(OPTS))
    assert len(multi_path) == len(single_path)


@pytest.mark.asyncio
async def test_consecutive_duplicate_points_are_collapsed():
    service = make_service()
    path_dup, _ = await service.find_multi_route([P1, P1, P3], dict(OPTS))
    path_plain, _ = await service.find_multi_route([P1, P3], dict(OPTS))
    assert len(path_dup) == len(path_plain)


@pytest.mark.asyncio
async def test_all_duplicate_points_error():
    service = make_service()
    path, stats = await service.find_multi_route([P1, P1], dict(OPTS))
    assert path == []
    assert "error" in stats


@pytest.mark.asyncio
async def test_failing_leg_names_the_leg(monkeypatch):
    service = make_service()
    real_find_route = service.find_route

    async def fake_find_route(a, b, options):
        # Fail only the second leg (P2 -> P3).
        if (a.lat, a.lon) == (P2.lat, P2.lon):
            return [], {"error": "No route found", "engine": "v2"}
        return await real_find_route(a, b, options)

    monkeypatch.setattr(service, "find_route", fake_find_route)
    path, stats = await service.find_multi_route([P1, P2, P3], dict(OPTS))
    assert path == []
    assert "leg 2" in stats["error"]
