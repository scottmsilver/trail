# Multi-Waypoint Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the user drop N points on the map and get one route that passes through all of them, in drop order (point 1 = start, point N = end).

**Architecture:** No routing-engine changes. A multi-point route is N−1 existing single-pair A\* legs stitched server-side into one concatenated path with aggregated stats. The request model gains an optional `points` list (start/end become the legacy 2-point shape). Feature is **v2 only**; v1 stays two-point and rejects multi-point requests.

**Tech Stack:** FastAPI + Pydantic v2 (2.5.3) backend, `trail_env` venv; React 19 + react-leaflet + Vite + Vitest frontend; pytest (async) backend tests.

## Global Constraints

- **Python venv is `trail_env`.** Run backend code/tests as `cd backend && PYTHONPATH=. ../trail_env/bin/python -m pytest ...`. Never system/anaconda python.
- **Pydantic v2 (2.5.3).** Use `model_validator`, not v1 `root_validator`.
- **v2 only.** Multi-waypoint applies to `engine == "v2"`. A multi-point request with `engine == "v1"` is rejected with a clear error. The v1 code path is otherwise untouched.
- **No caching changes.** Honor the existing "fail loudly when data absent" rule; add no caches.
- **Commits are gated.** Per repo owner policy (CLAUDE.md), do NOT run `git commit` until the user explicitly authorizes it (password "1234"). "Commit" steps below mean *stage and prepare the commit message*; actually run the commit only once authorized. Never push.
- **Back-compat is a hard requirement.** Existing `{start, end}` requests must keep working unchanged.

---

## File Structure

**Backend**
- `backend/app/models/route.py` — Modify `RouteRequest`: add `points`, relax `start`/`end`, add validator + `normalized_points()`.
- `backend/app/engine_v2/service.py` — Add `find_multi_route()` to `TrailFinderServiceV2`.
- `backend/app/main.py` — Wire `process_route` and `export_route_as_gpx` to route through `points`; reject multi-point on v1.
- `backend/tests/test_route_model.py` — **New.** Model validation + normalization.
- `backend/tests/test_multi_route.py` — **New.** Stitching service tests (offline fake harness).
- `backend/tests/test_multi_route_wiring.py` — **New.** `process_route` dispatch + v1 rejection.

**Frontend**
- `frontend/src/services/api.ts` — `calculateRoute(points, options)` + `exportRouteAsGPX(points, options)` send `points`.
- `frontend/src/App.tsx` — `points: Coordinate[]` state; append-on-click; derived start/end; points list panel with delete.
- `frontend/src/components/Map/Map.tsx` — numbered draggable markers with per-marker delete; `points`, `onPointDrag`, `onPointDelete` props.
- `frontend/src/services/api.test.ts` — Update for new signatures.
- `frontend/src/components/Map/Map.test.tsx` — Numbered markers, drag, delete.

---

## Task 1: Request model — points support + normalization

**Files:**
- Modify: `backend/app/models/route.py:87-90` (the `RouteRequest` class)
- Test: `backend/tests/test_route_model.py` (create)

**Interfaces:**
- Produces: `RouteRequest.points: Optional[List[Coordinate]]`; `RouteRequest.start`/`.end: Optional[Coordinate]`; `RouteRequest.normalized_points() -> List[Coordinate]` returning `points` if set else `[start, end]`.

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_route_model.py`:

```python
"""RouteRequest gains an optional `points` list for multi-waypoint routing while
staying back-compatible with the legacy {start, end} shape. Both normalize to a
single ordered points list via normalized_points()."""

import pytest
from pydantic import ValidationError

from app.models.route import Coordinate, RouteRequest

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && PYTHONPATH=. ../trail_env/bin/python -m pytest tests/test_route_model.py -v`
Expected: FAIL — `RouteRequest` rejects `points` (unexpected kwarg) / has no `normalized_points`.

- [ ] **Step 3: Modify `RouteRequest`**

In `backend/app/models/route.py`, update the imports at the top (line 4) to include `model_validator`:

```python
from pydantic import BaseModel, Field, model_validator, validator
```

Replace the `RouteRequest` class (currently lines 87-90):

```python
class RouteRequest(BaseModel):
    start: Optional[Coordinate] = None
    end: Optional[Coordinate] = None
    points: Optional[List[Coordinate]] = Field(
        None, description="Ordered waypoints (>=2). Route passes through each in order. Takes precedence over start/end."
    )
    options: Optional[RouteOptions] = RouteOptions()

    @model_validator(mode="after")
    def _require_points_or_endpoints(self):
        if self.points is not None:
            if len(self.points) < 2:
                raise ValueError("points must contain at least 2 coordinates")
        elif self.start is None or self.end is None:
            raise ValueError("Provide either 'points' (>=2) or both 'start' and 'end'")
        return self

    def normalized_points(self) -> List[Coordinate]:
        """Single ordered points list: `points` if given, else [start, end]."""
        if self.points is not None:
            return self.points
        return [self.start, self.end]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && PYTHONPATH=. ../trail_env/bin/python -m pytest tests/test_route_model.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Guard against regressions in existing route tests**

Run: `cd backend && PYTHONPATH=. ../trail_env/bin/python -m pytest tests/test_route_calculations_fast.py tests/test_services_fast.py -q`
Expected: PASS (existing 2-point behavior unchanged).

- [ ] **Step 6: Commit** *(stage; run commit only once authorized per Global Constraints)*

```bash
git add backend/app/models/route.py backend/tests/test_route_model.py
git commit -m "feat(models): RouteRequest accepts ordered points list (back-compat with start/end)"
```

---

## Task 2: Multi-route stitching in the v2 service

**Files:**
- Modify: `backend/app/engine_v2/service.py` — add method to `TrailFinderServiceV2` (place it right after `find_route`, which ends at line 264).
- Test: `backend/tests/test_multi_route.py` (create)

**Interfaces:**
- Consumes: existing `TrailFinderServiceV2.find_route(start, end, options) -> (List[Coordinate], dict)`; `self._difficulty(distance_km, max_slope)`.
- Produces: `async TrailFinderServiceV2.find_multi_route(points: List[Coordinate], options: dict) -> Tuple[List[Coordinate], dict]`. On success `stats` carries `engine, distance_m, distance_km, elevation_gain_m, estimated_time_min, max_slope, difficulty, waypoints, path_with_slopes, legs`. On failure returns `([], {"error": ..., "engine": "v2"})`.

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_multi_route.py`:

```python
"""find_multi_route stitches N-1 single-pair A* legs into one path + aggregated
stats. Reuses the offline injected-fake harness so it runs deterministically
without DEM downloads."""

import pytest

from app.models.route import Coordinate
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
    assert multi["elevation_gain_m"] == pytest.approx(
        s12["elevation_gain_m"] + s23["elevation_gain_m"], rel=1e-6
    )
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && PYTHONPATH=. ../trail_env/bin/python -m pytest tests/test_multi_route.py -v`
Expected: FAIL — `TrailFinderServiceV2` has no attribute `find_multi_route`.

- [ ] **Step 3: Implement `find_multi_route`**

In `backend/app/engine_v2/service.py`, insert this method immediately after `find_route` (after line 264, before `_load_pathfinder`):

```python
    async def find_multi_route(self, points, options: dict):
        """Route through `points` in order by stitching N-1 single-pair legs.

        Point 0 is the start, point N-1 the end. Consecutive identical points are
        collapsed. Any failing leg fails the whole route with a message naming the
        leg. Returns (concatenated_path, aggregated_stats) with the same client
        stat keys find_route produces, plus a per-leg `legs` breakdown.
        """
        options = options or {}

        # Collapse consecutive identical points (no zero-length legs).
        pts = [points[0]]
        for p in points[1:]:
            if (p.lat, p.lon) != (pts[-1].lat, pts[-1].lon):
                pts.append(p)
        if len(pts) < 2:
            return [], {"error": "Multi-point route needs at least 2 distinct points", "engine": "v2"}

        combined_path = []
        combined_slopes = []
        total_distance_m = 0.0
        total_gain_m = 0.0
        max_slope = 0.0
        legs = []

        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            leg_path, leg_stats = await self.find_route(a, b, options)
            if not leg_path:
                err = leg_stats.get("error", "No route found")
                return [], {
                    "error": f"{err} for leg {i + 1} (point {i + 1} -> point {i + 2})",
                    "engine": "v2",
                }
            leg_slopes = leg_stats.get("path_with_slopes", [])
            if i == 0:
                combined_path.extend(leg_path)
                combined_slopes.extend(leg_slopes)
            else:
                # Drop the duplicated junction vertex (leg start == previous leg end).
                combined_path.extend(leg_path[1:])
                combined_slopes.extend(leg_slopes[1:])
            total_distance_m += leg_stats.get("distance_m", 0.0)
            total_gain_m += leg_stats.get("elevation_gain_m", 0.0)
            max_slope = max(max_slope, leg_stats.get("max_slope", 0.0))
            legs.append(
                {
                    "from": i,
                    "to": i + 1,
                    "distance_m": leg_stats.get("distance_m", 0.0),
                    "elevation_gain_m": leg_stats.get("elevation_gain_m", 0.0),
                }
            )

        distance_km = round(total_distance_m / 1000.0, 2)
        stats = {
            "engine": "v2",
            "distance_m": round(total_distance_m, 2),
            "distance_km": distance_km,
            "elevation_gain_m": round(total_gain_m, 2),
            "estimated_time_min": int(distance_km * 15),
            "max_slope": round(max_slope, 1),
            "difficulty": self._difficulty(distance_km, max_slope),
            "waypoints": len(combined_path),
            "path_with_slopes": combined_slopes,
            "legs": legs,
        }
        return combined_path, stats
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && PYTHONPATH=. ../trail_env/bin/python -m pytest tests/test_multi_route.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit** *(stage; run commit only once authorized)*

```bash
git add backend/app/engine_v2/service.py backend/tests/test_multi_route.py
git commit -m "feat(engine-v2): find_multi_route stitches N-1 legs into one route"
```

---

## Task 3: Wire process_route + GPX export; reject multi-point on v1

**Files:**
- Modify: `backend/app/main.py:243-294` (`process_route`), `backend/app/main.py:425-443` (`export_route_as_gpx`)
- Test: `backend/tests/test_multi_route_wiring.py` (create)

**Interfaces:**
- Consumes: `RouteRequest.normalized_points()` (Task 1); `trail_finder_v2.find_multi_route` / `.find_route` (Task 2).
- Produces: `process_route` stores a stitched path for v2 multi-point requests; fails a v1 multi-point request with a message containing "v1"; export uses the normalized points for the first/last GPX naming.

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_multi_route_wiring.py`:

```python
"""process_route dispatches multi-point v2 requests to find_multi_route, single
pairs to find_route, and rejects multi-point v1 requests. Uses a fake finder so
it runs offline (no DEM)."""

import pytest

import app.main as main
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && PYTHONPATH=. ../trail_env/bin/python -m pytest tests/test_multi_route_wiring.py -v`
Expected: FAIL — `process_route` still calls `find_route(request.start, request.end, ...)` and doesn't branch on points count.

- [ ] **Step 3: Update `process_route`**

In `backend/app/main.py`, replace the body from line 249 (`# Get configurations...`) through the end of the v2/else block (line 278), i.e. replace lines 249-278 with:

```python
        # Get configurations based on user profile and custom options
        profile = request.options.userProfile if request.options else "default"
        engine = request.options.engine if request.options else "v2"
        points = request.normalized_points()
        options_dict = request.options.model_dump() if request.options else {}

        if engine == "v2":
            routes_storage[route_id]["progress"] = 30
            if len(points) > 2:
                path, stats = await trail_finder_v2.find_multi_route(points, options_dict)
            else:
                path, stats = await trail_finder_v2.find_route(points[0], points[-1], options_dict)
        else:
            # Multi-waypoint routing is a v2-only feature.
            if len(points) > 2:
                routes_storage[route_id]["status"] = RouteStatus.FAILED
                routes_storage[route_id]["message"] = (
                    "Multi-waypoint routing requires engine v2; v1 supports start/end only."
                )
                return

            obstacle_config, path_preferences = get_configs_for_profile(profile, request.options)

            # Create trail finder with user's configurations and shared cache
            profile_trail_finder = TrailFinderService(
                obstacle_config=obstacle_config, path_preferences=path_preferences, dem_cache=shared_dem_cache
            )

            # Validate request
            if not profile_trail_finder.validate_route_request(points[0], points[-1]):
                routes_storage[route_id]["status"] = RouteStatus.FAILED
                routes_storage[route_id]["message"] = "Invalid route request"
                return

            routes_storage[route_id]["progress"] = 30

            # Find the route
            path, stats = await profile_trail_finder.find_route(points[0], points[-1], options_dict)
```

- [ ] **Step 4: Update `export_route_as_gpx` to use normalized points**

In `backend/app/main.py`, replace lines 440-451 (from `# Find the route` through the `route_name` assignment) with:

```python
    # Find the route (v2 stitches multi-point; v1/legacy uses first & last).
    points = request.normalized_points()
    options_dict = request.options.model_dump() if request.options else {}
    engine = request.options.engine if request.options else "v2"
    if engine == "v2" and len(points) > 2:
        path, stats = await trail_finder_v2.find_multi_route(points, options_dict)
    else:
        path, stats = await profile_trail_finder.find_route(points[0], points[-1], options_dict)

    if not path:
        raise HTTPException(status_code=404, detail=stats.get("error", "No route found"))

    # Generate route name
    route_name = (
        f"Trail Route {points[0].lat:.4f},{points[0].lon:.4f} to {points[-1].lat:.4f},{points[-1].lon:.4f}"
    )
```

Note: delete the now-duplicated original "Find the route" block (old lines 440-446) and the old `route_name` (old lines 448-451) — the replacement above supersedes both. Leave the `if not path:` check present exactly once.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd backend && PYTHONPATH=. ../trail_env/bin/python -m pytest tests/test_multi_route_wiring.py -v`
Expected: PASS (3 passed).

- [ ] **Step 6: Run the fast smoke suite**

Run: `cd backend && PYTHONPATH=. ../trail_env/bin/python -m pytest tests/test_route_calculations_fast.py tests/test_services_fast.py tests/test_route_model.py tests/test_multi_route.py -q`
Expected: PASS.

- [ ] **Step 7: Commit** *(stage; run commit only once authorized)*

```bash
git add backend/app/main.py backend/tests/test_multi_route_wiring.py
git commit -m "feat(api): route through points server-side (v2 multi-waypoint); reject multi-point on v1"
```

---

## Task 4: Frontend API + types send `points`

**Files:**
- Modify: `frontend/src/services/api.ts:124-135` (`calculateRoute`), `184-197` (`exportRouteAsGPX`)
- Test: `frontend/src/services/api.test.ts:28-70` (update)

**Interfaces:**
- Produces: `TrailAPI.calculateRoute(points: Coordinate[], options?) -> Promise<RouteResponse>` posting `{ points, options }`; `TrailAPI.exportRouteAsGPX(points: Coordinate[], options?) -> Promise<Blob>` posting `{ points, options }`.

- [ ] **Step 1: Update the failing tests**

In `frontend/src/services/api.test.ts`, replace the `calculateRoute` describe block's body so it calls the new signature and asserts the payload. Use this test:

```ts
  describe('calculateRoute', () => {
    it('sends an ordered points list', async () => {
      const mockPost = vi.fn().mockResolvedValue({ data: { routeId: 'abc', status: 'processing' } })
      ;(api as any).client = { post: mockPost }

      const points = [
        { lat: 40.630, lon: -111.580 },
        { lat: 40.640, lon: -111.570 },
        { lat: 40.650, lon: -111.560 },
      ]
      const result = await api.calculateRoute(points, { engine: 'v2' })

      expect(mockPost).toHaveBeenCalledWith('/api/routes/calculate', {
        points,
        options: { engine: 'v2' },
      })
      expect(result.routeId).toBe('abc')
    })
  })
```

(If the existing test constructs `api` differently, mirror that construction; the assertion on the `{ points, options }` payload is the point.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd frontend && npx vitest run src/services/api.test.ts`
Expected: FAIL — `calculateRoute` still expects `(start, end, options)` and posts `{ start, end, options }`.

- [ ] **Step 3: Update `calculateRoute` and `exportRouteAsGPX`**

In `frontend/src/services/api.ts`, replace `calculateRoute` (lines 124-135):

```ts
  async calculateRoute(
    points: Coordinate[],
    options: RouteOptions = {}
  ): Promise<RouteResponse> {
    const response = await this.client.post<RouteResponse>('/api/routes/calculate', {
      points,
      options,
    })
    return response.data
  }
```

Replace `exportRouteAsGPX` (lines 184-197):

```ts
  async exportRouteAsGPX(
    points: Coordinate[],
    options: RouteOptions = {}
  ): Promise<Blob> {
    const response = await this.client.post('/api/routes/export/gpx', {
      points,
      options,
    }, {
      responseType: 'blob',
    })
    return response.data
  }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd frontend && npx vitest run src/services/api.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit** *(stage; run commit only once authorized)*

```bash
git add frontend/src/services/api.ts frontend/src/services/api.test.ts
git commit -m "feat(frontend): api sends ordered points list for routing + GPX export"
```

---

## Task 5: App state → points[] (append, delete, list panel)

**Files:**
- Modify: `frontend/src/App.tsx` — state (lines 15-16), `useAsStart`/`useAsEnd` (50-64), `handleMapClick` (66-88), `findRoute` (90-99), `reset` (178-186), `exportGPX` (215-220), `currentPoint` (48), `<Map>` render (548-551).

**Interfaces:**
- Consumes: `api.calculateRoute(points, options)`, `api.exportRouteAsGPX(points, options)` (Task 4); `Map` props `points`, `onPointDrag`, `onPointDelete` (Task 6).
- Produces: nothing consumed by later tasks (App is the top-level composition).

This task has no unit test of its own (App is exercised by manual QA + the Map/api tests). Verify by `tsc`/build and a manual smoke.

- [ ] **Step 1: Replace start/end state with a points array + derived endpoints**

In `frontend/src/App.tsx`, replace lines 15-16:

```tsx
  const [points, setPoints] = useState<Coordinate[]>([])
  // Legacy consumers still think in start/end; derive them from the list.
  const start = points.length > 0 ? points[0] : null
  const end = points.length > 1 ? points[points.length - 1] : null
```

- [ ] **Step 2: Update click-to-append and saved-location helpers**

Replace `currentPoint` (line 48) — it already works via derived `end`/`start`, no change needed. Replace `useAsStart`/`useAsEnd` (lines 50-64):

```tsx
  const useAsStart = (loc: SavedLocation) => {
    const coord = { lat: loc.lat, lon: loc.lon }
    setPoints(prev => (prev.length === 0 ? [coord] : [coord, ...prev.slice(1)]))
    setMapCenter(coord)
    recordRecent(loc.lat, loc.lon, loc.name)
    setStatus(`Start set to "${loc.name}".`)
  }

  const useAsEnd = (loc: SavedLocation) => {
    const coord = { lat: loc.lat, lon: loc.lon }
    setPoints(prev => (prev.length === 0 ? [coord] : [...prev, coord]))
    setMapCenter(coord)
    recordRecent(loc.lat, loc.lon, loc.name)
    setStatus(`End set to "${loc.name}".`)
  }
```

Replace `handleMapClick` (lines 66-88):

```tsx
  const handleMapClick = (coord: Coordinate) => {
    // Once a route is computed, the first new click starts a fresh route.
    if (route) {
      setRoute(null)
      setDebugData(null)
      setShowDebug(false)
      setPoints([coord])
      recordRecent(coord.lat, coord.lon)
      setStatus('Start point set. Click to add more points.')
      return
    }
    setPoints(prev => [...prev, coord])
    recordRecent(coord.lat, coord.lon)
    setStatus(
      points.length === 0
        ? 'Start point set. Click to add more points.'
        : `Point ${points.length + 1} added. Click "Find Route" to calculate.`
    )
  }

  const deletePoint = (index: number) => {
    setPoints(prev => prev.filter((_, i) => i !== index))
    setRoute(null)
    setStatus('Point removed.')
  }

  const movePoint = (index: number, coord: Coordinate) => {
    setPoints(prev => prev.map((p, i) => (i === index ? coord : p)))
    setRoute(null)
  }
```

- [ ] **Step 2b: Guard `findRoute` on point count and send the list**

In `frontend/src/App.tsx`, replace the guard + call inside `findRoute` (lines 91 and 99):

Line 91 `if (!start || !end) return` becomes:

```tsx
    if (points.length < 2) return
```

Line 99 `const response = await api.calculateRoute(start, end, routeOptions)` becomes:

```tsx
      const response = await api.calculateRoute(points, routeOptions)
```

Also change the `debugRoute` guard (line 140) `if (!start || !end) return` to `if (!start || !end) return` — **leave as-is**; debug stays two-point (it derives start/end). No further change.

- [ ] **Step 3: Update reset and exportGPX**

Replace `reset` (lines 178-186):

```tsx
  const reset = () => {
    setPoints([])
    setRoute(null)
    setDebugData(null)
    setShowDebug(false)
    setRouteTime(null)
    setStatus('Click on the map to set start point.')
  }
```

In `exportGPX`, replace the fallback branch guard + call (lines 215-220):

```tsx
        // Export directly from current route data
        if (points.length < 2) {
          setStatus('No route data to export')
          return
        }

        const blob = await api.exportRouteAsGPX(points, routeOptions)
```

- [ ] **Step 4: Add a points list panel and pass new props to `<Map>`**

Replace the `<Map ... />` opening props (lines 548-551) to pass points and handlers:

```tsx
          <Map
            points={points}
            path={route?.path}
```

(Remove the `start={...}` / `end={...}` lines — `Map` now takes `points`. Keep the remaining `<Map>` props as they were: `pathWithSlopes`, `center`, `onMapClick`, etc. Add `onPointDrag={movePoint}` and `onPointDelete={deletePoint}` to the prop list.)

Add a points list panel near the existing coordinate display. Find the block that renders `{(start || end) && ( ... )}` (around line 446 in the original) and replace its inner coordinate rows with a list driven by `points`:

```tsx
            {points.length > 0 && (
              <div className="points-list">
                <div className="points-list-header">Waypoints ({points.length})</div>
                {points.map((p, i) => (
                  <div className="point-row" key={i}>
                    <span className="point-badge">{i + 1}</span>
                    <span className="coord-value">{p.lat.toFixed(4)}, {p.lon.toFixed(4)}</span>
                    <button
                      className="point-delete"
                      onClick={() => deletePoint(i)}
                      title="Remove this point"
                    >×</button>
                  </div>
                ))}
              </div>
            )}
```

- [ ] **Step 5: Update the "Find Route" button disabled guard**

Find the Find Route button (original line 485) `disabled={!start || !end || loading || debugLoading}` and change to:

```tsx
              disabled={points.length < 2 || loading || debugLoading}
```

- [ ] **Step 6: Typecheck + build**

Run: `cd frontend && npx tsc --noEmit && npm run build`
Expected: no type errors; build succeeds. (Fix any remaining `start`/`end` references the compiler flags — they should all resolve via the derived consts.)

- [ ] **Step 7: Commit** *(stage; run commit only once authorized)*

```bash
git add frontend/src/App.tsx
git commit -m "feat(frontend): multi-waypoint state — append on click, delete, waypoint list"
```

---

## Task 6: Map — numbered draggable markers with delete

**Files:**
- Modify: `frontend/src/components/Map/Map.tsx` — props (31-45), signature (112), marker render (192-202), cost-surface start/end derivation (214-215)
- Test: `frontend/src/components/Map/Map.test.tsx` (update/extend)

**Interfaces:**
- Consumes: `points: Coordinate[]`, `onPointDrag?: (index, coord) => void`, `onPointDelete?: (index) => void` from App (Task 5).
- Produces: numbered markers; drag fires `onPointDrag`; popup Delete fires `onPointDelete`.

- [ ] **Step 1: Write the failing tests**

In `frontend/src/components/Map/Map.test.tsx`, update the marker test and add drag/delete coverage. The existing mock renders `Marker` as a div; extend the mock to expose `eventHandlers` and children. Replace the react-leaflet mock's `Marker` and add tests:

```tsx
  Marker: ({ children, eventHandlers }: any) => (
    <div data-testid="marker" onClick={() => eventHandlers?.dragend?.({ target: { getLatLng: () => ({ lat: 1, lng: 2 }) } })}>
      {children}
    </div>
  ),
```

```tsx
  it('renders one numbered marker per point', () => {
    const points = [
      { lat: 40.630, lon: -111.580 },
      { lat: 40.640, lon: -111.570 },
      { lat: 40.650, lon: -111.560 },
    ]
    render(<Map points={points} />)
    expect(screen.getAllByTestId('marker')).toHaveLength(3)
  })

  it('fires onPointDrag with index and new coord on dragend', () => {
    const onPointDrag = vi.fn()
    const points = [{ lat: 40.630, lon: -111.580 }, { lat: 40.640, lon: -111.570 }]
    render(<Map points={points} onPointDrag={onPointDrag} />)
    screen.getAllByTestId('marker')[1].click()
    expect(onPointDrag).toHaveBeenCalledWith(1, { lat: 1, lon: 2 })
  })
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd frontend && npx vitest run src/components/Map/Map.test.tsx`
Expected: FAIL — `Map` doesn't accept `points`/`onPointDrag`; only two hard-coded markers render.

- [ ] **Step 3: Update `MapProps` and the signature**

In `frontend/src/components/Map/Map.tsx`, replace `start`/`end` in `MapProps` (lines 32-33) with:

```tsx
  points?: Coordinate[]
  onPointDrag?: (index: number, coord: Coordinate) => void
  onPointDelete?: (index: number) => void
```

Update the destructured signature (line 112) — replace `start, end,` with `points, onPointDrag, onPointDelete,`. Then derive start/end for the cost-surface explorer just inside the function body (after line 116):

```tsx
  const pts = points ?? []
  const start = pts.length > 0 ? pts[0] : undefined
  const end = pts.length > 1 ? pts[pts.length - 1] : undefined
```

- [ ] **Step 4: Render numbered draggable markers**

Replace the two marker blocks (lines 192-202) with:

```tsx
      {pts.map((p, i) => {
        const isTerminal = i === 0 || i === pts.length - 1
        const label = i === 0 ? 'Start' : i === pts.length - 1 ? 'End' : `Point ${i + 1}`
        const icon = L.divIcon({
          className: 'waypoint-marker',
          html: `<div class="wp-pin ${isTerminal ? 'wp-terminal' : 'wp-via'}">${i + 1}</div>`,
          iconSize: [26, 26],
          iconAnchor: [13, 13],
        })
        return (
          <Marker
            key={i}
            position={[p.lat, p.lon]}
            icon={icon}
            draggable={true}
            eventHandlers={{
              dragend: (e: any) => {
                const ll = e.target.getLatLng()
                onPointDrag?.(i, { lat: ll.lat, lon: ll.lng })
              },
            }}
          >
            <Popup>
              <div>{label}</div>
              <button onClick={() => onPointDelete?.(i)}>Delete point</button>
            </Popup>
          </Marker>
        )
      })}
```

Ensure `import L from 'leaflet'` is present at the top of `Map.tsx` (it already imports `L` at line 11 area — verify; if not, add `import L from 'leaflet'`).

- [ ] **Step 5: Add marker styles**

Append to `frontend/src/App.css` (or `Map`'s stylesheet) so the numbered pins are visible:

```css
.waypoint-marker { background: transparent; border: none; }
.wp-pin {
  width: 26px; height: 26px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font: 600 13px/1 system-ui, sans-serif; color: #fff;
  box-shadow: 0 1px 4px rgba(0,0,0,.4);
}
.wp-terminal { background: #C2571A; }
.wp-via { background: #3E5C4B; }
.points-list { margin-top: .5rem; display: flex; flex-direction: column; gap: .25rem; }
.point-row { display: flex; align-items: center; gap: .5rem; }
.point-badge {
  width: 20px; height: 20px; border-radius: 50%; background: #3E5C4B; color: #fff;
  display: inline-flex; align-items: center; justify-content: center; font-size: 12px;
}
.point-delete { margin-left: auto; border: none; background: none; cursor: pointer; font-size: 16px; color: #A33326; }
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd frontend && npx vitest run src/components/Map/Map.test.tsx`
Expected: PASS.

- [ ] **Step 7: Full frontend test + build**

Run: `cd frontend && npx vitest run && npx tsc --noEmit && npm run build`
Expected: all tests pass; no type errors; build succeeds.

- [ ] **Step 8: Commit** *(stage; run commit only once authorized)*

```bash
git add frontend/src/components/Map/Map.tsx frontend/src/components/Map/Map.test.tsx frontend/src/App.css
git commit -m "feat(frontend): numbered draggable waypoint markers with delete"
```

---

## Task 7: End-to-end verification

**Files:** none (verification only)

- [ ] **Step 1: Backend fast suite**

Run: `cd backend && PYTHONPATH=. ../trail_env/bin/python -m pytest tests/test_route_model.py tests/test_multi_route.py tests/test_multi_route_wiring.py tests/test_route_calculations_fast.py tests/test_services_fast.py tests/test_v2_client_stats.py -q`
Expected: all pass.

- [ ] **Step 2: Frontend suite + build**

Run: `cd frontend && npx vitest run && npm run build`
Expected: all pass; build clean.

- [ ] **Step 3: Manual smoke (real terrain)**

Start backend + frontend (per `RUN_INSTRUCTIONS.md`). On the map: click 3+ points, confirm numbered markers, drag a middle point (route clears), delete a point (list + marker update), click **Find Route**, confirm a single connected path through all points and aggregated distance in the status line. Export GPX and confirm the file spans all waypoints.

- [ ] **Step 4: Back-compat smoke**

Two-point route still works exactly as before (click start, click end, Find Route). Switch engine to v1 with 2 points — routes. With 3+ points on v1 — status shows the "requires engine v2" message.

- [ ] **Step 5: Security audit (per CLAUDE.md)**

Run the two-track audit: `pip-audit` / `npm audit --production` for deps (no new deps expected — note that), and a `codex exec --sandbox read-only` code review over this session's diff (models/route.py, engine_v2/service.py, main.py wiring, api.ts, App.tsx, Map.tsx), focused on untrusted-input handling of the `points` list (unbounded length → DoS on N A\* searches), leg-count bounds, and error-message construction. Report findings before declaring done.

---

## Self-Review

- **Spec coverage:** points model + validator (T1) ✓; server-side stitch, drop junction, sum stats, name failing leg, collapse duplicates (T2) ✓; v2-only wiring + v1 rejection (T3) ✓; frontend points payload (T4) ✓; append-click + derived start/end + delete + list (T5) ✓; numbered draggable markers + delete (T6) ✓; back-compat (T1/T3 tests + T7 smoke) ✓; testing section (each task TDD + T7) ✓; non-goals respected — no TSP, no reorder/insert, one RouteOptions, no caching. ✓
- **Placeholder scan:** every code step carries real code; no TBD/TODO. ✓
- **Type consistency:** `find_multi_route(points, options) -> (List[Coordinate], dict)` used identically in T2/T3; `normalized_points()` defined T1, used T3; `calculateRoute(points, options)` defined T4, used T5; `points`/`onPointDrag`/`onPointDelete` defined T6, passed T5. Stat keys (`distance_m`, `elevation_gain_m`, `path_with_slopes`) verified against `pathfinder.py` and `_augment_client_stats`. ✓

## Bounds note (security)

The `points` list is untrusted input; N points ⇒ N−1 A\* searches, each already bounded by `validate_route_request`'s `max_distance_km`. Consider a hard cap (e.g. reject > 25 points) in T1's validator if the T7 audit flags unbounded work; left out of the base plan as YAGNI pending the audit.
