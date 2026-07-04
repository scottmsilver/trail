"""HTTP-layer tests for POST /api/eval/score-path.

The scorer itself is covered in test_score_path_service.py; here we monkeypatch
the service so the endpoint's validation and response serialization (notably the
`from` field alias) are tested without needing DEM data.
"""

from fastapi.testclient import TestClient

import app.main as main
from app.models.eval import ScoredPath, ScoredSegment
from app.models.route import Coordinate

client = TestClient(main.app)


def _canned_scored_path():
    return ScoredPath(
        path=[Coordinate(lat=1, lon=2), Coordinate(lat=1, lon=3)],
        snapped=False,
        totalCost=42.0,
        distanceM=10.0,
        elevationGainM=0.0,
        segments=[
            ScoredSegment(
                **{
                    "from": Coordinate(lat=1, lon=2),
                    "to": Coordinate(lat=1, lon=3),
                    "cost": 42.0,
                    "factors": {"base": 40.0, "terrain": 2.0, "slope": 0.0, "sustained": 0.0, "deviation": 0.0},
                    "dominantFactor": "terrain",
                }
            )
        ],
    )


def test_rejects_short_path():
    r = client.post("/api/eval/score-path", json={"path": [{"lat": 0, "lon": 0}], "options": {}})
    assert r.status_code == 400


def test_returns_scored_path_with_from_alias(monkeypatch):
    async def fake_score_path(path, options):
        return _canned_scored_path()

    monkeypatch.setattr(main.trail_finder_v2, "score_path", fake_score_path)
    r = client.post(
        "/api/eval/score-path",
        json={"path": [{"lat": 1, "lon": 2}, {"lat": 1, "lon": 3}], "options": {"engine": "v2"}},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["totalCost"] == 42.0
    seg = body["segments"][0]
    assert "from" in seg and seg["from"]["lat"] == 1
    assert seg["dominantFactor"] == "terrain"
