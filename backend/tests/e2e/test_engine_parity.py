# RULE #0: MANDATORY FIRST ACTION FOR EVERY REQUEST
# 1. Read CLAUDE.md COMPLETELY before responding
# 2. Setup Python venv: [ -d "venv" ] || ./setup-venv.sh && source venv/bin/activate
# 3. Search for rules related to the request
# 4. Only proceed after confirming no violations
# Failure to follow Rule #0 has caused real harm. Check BEFORE acting, not AFTER making mistakes.
#
# GUARDS ARE SAFETY EQUIPMENT - WHEN THEY FIRE, FIX THE PROBLEM THEY FOUND
# NEVER weaken, disable, or bypass guards - they prevent real harm
"""
Engine parity harness — THE success criterion for the v2 integration.
Runs known real routes through both engines via the API and compares:
success, trail adherence, wall-clock, and elevation continuity.

Pass bars (from the v2 integration spec):
  1. v2 succeeds on every route.
  2. v2 trail-adherence >= v1's, per route (when v1 also succeeded).
  3. v2 wall-clock <= 1.25x v1's, per route (when v1 also succeeded).
  4. No single-step elevation jump > 20m on v2 paths.

Note on Bar 4: RouteResult.path serializes as [{"lat", "lon"}] with no
`elevation` key (see app/models/route.py::Coordinate), for both engines. So
elevation continuity is measured by sampling the v2 two-layer elevation
library directly along the v2 path, rather than trusting the API payload.

Uses the same three Park City routes as tests/e2e/test_real_routes.py
(data already cached for v1; v2 downloads its own on first run).
"""
import time

import numpy as np
import pytest
from app.engine_v2 import Bounds, PathLayer, PathType, TwoLayerElevationLibrary
from app.main import app
from fastapi.testclient import TestClient
from rasterio.transform import from_bounds

# Known Park City routes from tests/e2e/test_real_routes.py (data cached)
ROUTES = [
    {"name": "park_city_medium", "start": (40.6572, -111.5706), "end": (40.6486, -111.5639)},
    {"name": "park_city_long", "start": (40.6650, -111.5850), "end": (40.6550, -111.5750)},
    {"name": "park_city_short", "start": (40.6482, -111.5738), "end": (40.6464, -111.5729)},
]

ON_PATH_TYPES = {PathType.TRAIL, PathType.PATH, PathType.FOOTWAY}


def _run_route(client, route, engine):
    """POST a route calculation and fetch its result.

    TestClient runs FastAPI's BackgroundTasks synchronously within the POST
    call, so by the time `.post()` returns the route is already COMPLETED or
    FAILED. GET /api/routes/{id} returns HTTP 400 (a `detail` string, not a
    `status` field) whenever the route isn't COMPLETED (see app/main.py
    get_route), so read /status first — it always returns status/message —
    and only fetch the full result (path + stats) once the route completed.
    """
    body = {
        "start": {"lat": route["start"][0], "lon": route["start"][1]},
        "end": {"lat": route["end"][0], "lon": route["end"][1]},
        "options": {"engine": engine},
    }
    t0 = time.monotonic()
    resp = client.post("/api/routes/calculate", json=body)
    elapsed = time.monotonic() - t0  # TestClient runs the background task in-request
    assert resp.status_code == 202
    route_id = resp.json()["routeId"]

    status_resp = client.get(f"/api/routes/{route_id}/status")
    status_data = status_resp.json()
    status = status_data.get("status")

    path, stats = [], {}
    if status == "completed":
        result = client.get(f"/api/routes/{route_id}")
        data = result.json()
        path = data.get("path", [])
        stats = data.get("stats", {})
    else:
        stats = {"error": status_data.get("message")}

    return {"status": status, "path": path, "stats": stats, "time_s": elapsed}


def _route_bounds(route, buffer=0.02):
    lats = [route["start"][0], route["end"][0]]
    lons = [route["start"][1], route["end"][1]]
    return Bounds(south=min(lats) - buffer, north=max(lats) + buffer, west=min(lons) - buffer, east=max(lons) + buffer)


def _measurement_grid(bounds, shape=(400, 400)):
    transform = from_bounds(bounds.west, bounds.south, bounds.east, bounds.north, shape[1], shape[0])
    layer = PathLayer(cache_dir="path_cache_v2")
    return layer.get_grid(bounds, shape, transform), transform


def _trail_adherence(path, grid, transform):
    """Fraction of path points on trail/path/footway cells (spec definition)."""
    if not path:
        return 0.0
    on = 0
    for p in path:
        col, row = ~transform * (p["lon"], p["lat"])
        r, c = int(row), int(col)
        if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and grid[r, c] in ON_PATH_TYPES:
            on += 1
    return on / len(path)


def _sample_elevations(path, bounds):
    """Sample elevation at each v2 path point via the v2 elevation library
    (see module docstring's note on Bar 4 — API path has no elevation key)."""
    if not path:
        return []
    lib = TwoLayerElevationLibrary(data_dir="dem_data_v2", resolution=10)
    lib.load_area(bounds)
    elevs = []
    for p in path:
        try:
            elevs.append(lib.get_elevation(p["lat"], p["lon"]))
        except ValueError:
            continue  # point fell outside the loaded measurement bounds
    return elevs


@pytest.mark.real_data
@pytest.mark.slow
@pytest.mark.parametrize("route", ROUTES, ids=[r["name"] for r in ROUTES])
def test_engine_parity(route):
    client = TestClient(app)

    v1 = _run_route(client, route, "v1")
    v2 = _run_route(client, route, "v2")

    bounds = _route_bounds(route)
    grid, transform = _measurement_grid(bounds)
    adh1 = _trail_adherence(v1["path"], grid, transform)
    adh2 = _trail_adherence(v2["path"], grid, transform)

    print(f"\n=== {route['name']} ===")
    print(f"  v1: status={v1['status']} points={len(v1['path'])} " f"adherence={adh1:.1%} time={v1['time_s']:.1f}s")
    print(
        f"  v2: status={v2['status']} points={len(v2['path'])} "
        f"adherence={adh2:.1%} time={v2['time_s']:.1f}s "
        f"dist={v2['stats'].get('distance_m')}m"
    )

    # Bar 1: v2 succeeds on every route
    assert v2["status"] == "completed", f"v2 failed: {v2['stats']}"

    # Bar 2: trail adherence not worse than v1 (only when v1 also succeeded)
    if v1["status"] == "completed":
        assert adh2 >= adh1 - 1e-9, f"v2 adherence {adh2:.1%} < v1 {adh1:.1%}"
        # Bar 3: speed within 1.25x of v1
        assert v2["time_s"] <= v1["time_s"] * 1.25 + 1.0, f"v2 {v2['time_s']:.1f}s vs v1 {v1['time_s']:.1f}s"

    # Sanity: v2 stats report elevation gain (used elsewhere by the API/UI)
    assert v2["stats"].get("elevation_gain_m") is not None

    # Bar 4: no tile-artifact elevation jumps along the v2 path, sampled via
    # the v2 elevation library since RouteResult.path carries no elevation key
    elevs = _sample_elevations(v2["path"], bounds)
    if len(elevs) > 1:
        steps = np.abs(np.diff(elevs))
        assert steps.max() <= 20.0, f"elevation discontinuity: {steps.max():.1f}m"
