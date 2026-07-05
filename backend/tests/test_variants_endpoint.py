"""HTTP-layer tests for POST /api/routes/variants.

The variants logic is covered in test_route_variants.py; here we swap in the
fake-backed service so the endpoint's validation and serialization are tested
without DEM data.
"""

from fastapi.testclient import TestClient

import app.main as main
from tests.test_score_path_service import make_service

client = TestClient(main.app)

BODY = {
    "start": {"lat": 40.6410, "lon": -111.5090},
    "end": {"lat": 40.6420, "lon": -111.5080},
    "options": {"buffer": 0.001},
}


def test_variants_endpoint_returns_family(monkeypatch):
    monkeypatch.setattr(main, "trail_finder_v2", make_service())
    r = client.post("/api/routes/variants", json=BODY)
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 4
    levels = [v["level"] for v in body["variants"]]
    assert levels == ["casual", "hiker", "scrambler", "alpinist"]
    for v in body["variants"]:
        assert v["scrambleBudgetM"] > 0
        assert len(v["path"]) > 1
        assert {"lat", "lon"} <= set(v["path"][0])


def test_variants_endpoint_level_subset(monkeypatch):
    monkeypatch.setattr(main, "trail_finder_v2", make_service())
    r = client.post("/api/routes/variants", json={**BODY, "levels": ["alpinist"]})
    assert r.status_code == 200
    assert [v["level"] for v in r.json()["variants"]] == ["alpinist"]


def test_variants_endpoint_rejects_unknown_level(monkeypatch):
    monkeypatch.setattr(main, "trail_finder_v2", make_service())
    r = client.post("/api/routes/variants", json={**BODY, "levels": ["ninja"]})
    assert r.status_code == 400
    assert "ninja" in r.json()["detail"]
