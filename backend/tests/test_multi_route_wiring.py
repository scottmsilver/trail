"""process_route dispatches multi-point v2 requests to find_multi_route, single
pairs to find_route, and rejects multi-point v1 requests. Uses a fake finder so
it runs offline (no DEM)."""

import app.main as main
import pytest
from app.models.route import RouteOptions, RouteRequest

A = {"lat": 40.10, "lon": -111.10}
B = {"lat": 40.20, "lon": -111.20}
C = {"lat": 40.30, "lon": -111.30}


class FakeFinder:
    def __init__(self):
        self.multi_calls = []
        self.pair_calls = []

    async def find_route(self, start, end, options):
        self.pair_calls.append((start, end))
        return [start, end], {"engine": "v2", "distance_km": 1.0}

    async def find_multi_route(self, points, options):
        self.multi_calls.append(list(points))
        return list(points), {"engine": "v2", "distance_km": 2.0, "waypoints": len(points)}


@pytest.mark.asyncio
async def test_v2_multipoint_dispatches_to_find_multi_route(monkeypatch):
    fake = FakeFinder()
    monkeypatch.setattr(main, "trail_finder_v2", fake)
    rid = "r1"
    main.routes_storage[rid] = {"id": rid, "status": None, "progress": 0}
    req = RouteRequest(points=[A, B, C], options=RouteOptions(engine="v2"))

    await main.process_route(rid, req)

    assert len(fake.multi_calls) == 1
    assert len(fake.multi_calls[0]) == 3
    assert main.routes_storage[rid]["status"] == main.RouteStatus.COMPLETED
    assert len(main.routes_storage[rid]["path"]) == 3


@pytest.mark.asyncio
async def test_v2_two_point_still_uses_find_route(monkeypatch):
    fake = FakeFinder()
    monkeypatch.setattr(main, "trail_finder_v2", fake)
    rid = "r2"
    main.routes_storage[rid] = {"id": rid, "status": None, "progress": 0}
    req = RouteRequest(start=A, end=B, options=RouteOptions(engine="v2"))

    await main.process_route(rid, req)

    assert len(fake.pair_calls) == 1
    assert fake.multi_calls == []
    assert main.routes_storage[rid]["status"] == main.RouteStatus.COMPLETED


@pytest.mark.asyncio
async def test_v1_multipoint_is_rejected(monkeypatch):
    rid = "r3"
    main.routes_storage[rid] = {"id": rid, "status": None, "progress": 0}
    req = RouteRequest(points=[A, B, C], options=RouteOptions(engine="v1"))

    await main.process_route(rid, req)

    assert main.routes_storage[rid]["status"] == main.RouteStatus.FAILED
    assert "v1" in main.routes_storage[rid]["message"]


def test_v1_multipoint_gpx_export_rejected():
    from fastapi.testclient import TestClient

    client = TestClient(main.app)
    resp = client.post("/api/routes/export/gpx", json={"points": [A, B, C], "options": {"engine": "v1"}})
    assert resp.status_code == 400
    assert "v2" in resp.json()["detail"]
