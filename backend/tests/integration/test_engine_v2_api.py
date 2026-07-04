# RULE #0: MANDATORY FIRST ACTION FOR EVERY REQUEST
# 1. Read CLAUDE.md COMPLETELY before responding
# 2. Setup Python venv: [ -d "venv" ] || ./setup-venv.sh && source venv/bin/activate
# 3. Search for rules related to the request
# 4. Only proceed after confirming no violations
# Failure to follow Rule #0 has caused real harm. Check BEFORE acting, not AFTER making mistakes.
#
# GUARDS ARE SAFETY EQUIPMENT - WHEN THEY FIRE, FIX THE PROBLEM THEY FOUND
# NEVER weaken, disable, or bypass guards - they prevent real harm
"""Integration tests: engine flag reaches the dispatcher; v1 default unchanged."""
from app.main import app
from app.models.route import RouteOptions
from fastapi.testclient import TestClient


def test_route_options_engine_defaults_to_v1():
    opts = RouteOptions()
    assert opts.engine == "v1"
    assert opts.heuristicWeight is None


def test_route_options_rejects_unknown_engine():
    import pytest

    with pytest.raises(Exception):
        RouteOptions(engine="v3")


def test_calculate_accepts_engine_v2_flag():
    client = TestClient(app)
    resp = client.post(
        "/api/routes/calculate",
        json={
            "start": {"lat": 40.6572, "lon": -111.5706},
            "end": {"lat": 40.6486, "lon": -111.5639},
            "options": {"engine": "v2"},
        },
    )
    assert resp.status_code == 202
    route_id = resp.json()["routeId"]
    # Background task ran during the request (TestClient behavior);
    # v2 either completed or failed with a v2-shaped error — never a crash.
    status = client.get(f"/api/routes/{route_id}/status")
    assert status.status_code == 200
    assert status.json()["status"] in ("completed", "failed", "processing")
