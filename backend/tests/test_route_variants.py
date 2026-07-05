"""find_route_variants: one call routes the same start->end at several
expertise levels (DEM loaded once), returning a family of route options with
duplicates marked. Uses the injected-fake harness so it runs offline.
"""

import pytest

from app.engine_v2.service import EXPERTISE_LEVELS
from tests.test_score_path_service import END, OPTS, START, make_service


@pytest.mark.asyncio
async def test_variants_cover_all_levels_in_order():
    service = make_service()
    variants = await service.find_route_variants(START, END, dict(OPTS))
    assert [v["level"] for v in variants] == list(EXPERTISE_LEVELS)
    for v in variants:
        assert v["scrambleBudgetM"] == EXPERTISE_LEVELS[v["level"]]["scramble_budget_m"]
        assert len(v["path"]) > 1, f"{v['level']} found no route on flat terrain"
        assert v["stats"].get("error") is None


@pytest.mark.asyncio
async def test_identical_lines_are_marked_duplicates():
    # Flat terrain: every level takes the same straight line, so the first
    # level owns it and the rest carry duplicateOf.
    service = make_service()
    variants = await service.find_route_variants(START, END, dict(OPTS))
    assert "duplicateOf" not in variants[0]
    first = variants[0]["level"]
    assert all(v.get("duplicateOf") == first for v in variants[1:])


@pytest.mark.asyncio
async def test_custom_level_subset_and_order_respected():
    service = make_service()
    variants = await service.find_route_variants(START, END, dict(OPTS), levels=["alpinist", "casual"])
    assert [v["level"] for v in variants] == ["alpinist", "casual"]


@pytest.mark.asyncio
async def test_duplicate_levels_collapse_to_one_search():
    # ["hiker"] * 10000 must NOT run 10000 A* searches (codex audit: DoS).
    service = make_service()
    variants = await service.find_route_variants(START, END, dict(OPTS), levels=["hiker"] * 10000)
    assert [v["level"] for v in variants] == ["hiker"]


@pytest.mark.asyncio
async def test_unknown_level_rejected():
    service = make_service()
    with pytest.raises(ValueError):
        await service.find_route_variants(START, END, dict(OPTS), levels=["hiker", "ninja"])


@pytest.mark.asyncio
async def test_terrain_loaded_once_for_the_whole_family():
    service = make_service()
    await service.find_route_variants(START, END, dict(OPTS))
    assert len(service.elevation_lib.loaded) == 1


@pytest.mark.asyncio
async def test_invalid_request_raises():
    service = make_service()  # max_distance_km default
    far_end = type(END)(lat=END.lat + 5.0, lon=END.lon + 5.0)
    with pytest.raises(ValueError):
        await service.find_route_variants(START, far_end, dict(OPTS))
