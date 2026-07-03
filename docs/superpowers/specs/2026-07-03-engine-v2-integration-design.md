<!--
RULE #0: MANDATORY FIRST ACTION FOR EVERY REQUEST
1. Read CLAUDE.md COMPLETELY before responding
2. Setup Python venv: [ -d "venv" ] || ./setup-venv.sh && source venv/bin/activate
3. Search for rules related to the request
4. Only proceed after confirming no violations
Failure to follow Rule #0 has caused real harm. Check BEFORE acting, not AFTER making mistakes.

GUARDS ARE SAFETY EQUIPMENT - WHEN THEY FIRE, FIX THE PROBLEM THEY FOUND
NEVER weaken, disable, or bypass guards - they prevent real harm
-->

# Engine v2 Integration — Design Spec

**Date:** 2026-07-03
**Status:** Approved by user (design review complete)

## Background

The trail finder app (`backend/` FastAPI + `frontend/` React) runs on the original
pathfinding engine — the 2,738-line `DEMTileCache` — which has two chronic,
documented problems: tile-boundary artifacts in elevation data and slow routing
(~4s for a 5km route). A rewrite in `backend3/` (July 2025) produced a working
two-layer elevation library that eliminates tile artifacts, plus a cleaner
terrain-aware A* pathfinder, but the work was never integrated into the app and
stopped mid-optimization.

**Critical gap discovered:** `backend3/elevation_pathfinder_terrain.py` imports
`path_layer.py` (OSM trail/road awareness), which does not exist anywhere in the
repo — it was never written. Without it, backend3 can only route over raw
terrain; it cannot prefer trails or avoid roads.

## Goal

Integrate the backend3 engine into the FastAPI backend as a selectable "v2"
engine, including a newly written `path_layer` module, with weighted A* folded
in. Success = parity on real routes (see Testing).

## Decisions made during design review

1. **OSM layer:** Rebuild the missing `path_layer.py` for the v2 engine, porting
   the proven OSM fetch/rasterize logic from the old engine.
2. **Migration strategy:** Side-by-side behind a per-request flag. Old engine
   remains the default until v2 proves out.
3. **Performance scope:** Include weighted A* only. Defer all other
   optimizations (tuple nodes, bidirectional search, etc.).
4. **Success bar:** Parity on 3–5 known real routes via the API, viewable in the
   frontend; no tile artifacts; similar-or-better speed.
5. **Packaging (Approach A):** v2 engine lives inside the backend app as a
   package. Rejected: standalone installable package (two projects to maintain),
   sidecar service (deployment complexity, no benefit at this scale).

## Architecture

New package: `backend/app/engine_v2/`

| Module | Origin | Responsibility |
|---|---|---|
| `elevation.py` | Moved from `backend3/elevation.py` (+ `elevation_fd_safe.py` wrapper) | Two-layer elevation data: Layer 1 large seamless py3dep regions, Layer 2 query tiles. Explicit `load_area()`; no artifacts at tile boundaries. |
| `path_layer.py` | **New.** Ports OSM logic from `dem_tile_cache.py` (`fetch_obstacles`, `rasterize_paths`, `get_obstacle_mask`) and tag configs from `path_preferences.py` / `obstacle_config.py` | Fetch OSM features via `osmnx.features_from_polygon`, rasterize trails / streets / obstacles into a terrain-type grid aligned to the elevation array's shape and transform. Disk-cached per bounds+resolution. |
| `pathfinder.py` | From `backend3/elevation_pathfinder_terrain.py` (`TerrainAwarePathfinder`) | A* over elevation + terrain-type costs, sustained-slope fatigue, max-slope enforcement. **Adds weighted A*:** `heuristic_weight` parameter, default 1.0 (optimal), configurable up to ~2.0. |
| `service.py` | New, thin adapter | `TrailFinderServiceV2` exposing `async find_route(start: Coordinate, end: Coordinate, options: dict) -> Tuple[List[Coordinate], dict]` — the exact contract v1's `TrailFinderService` provides. Maps API options to pathfinder parameters, orchestrates data loading. |

### Engine selection

- Add `engine: Literal["v1", "v2"] = "v1"` to `RouteOptions` in
  `backend/app/models/route.py`.
- `main.py` dispatches `process_route` to the matching service instance.
- v1 behavior is byte-for-byte unchanged when the flag is absent.
- No hardcoded URLs anywhere (existing rule); OSM/py3dep endpoints come from the
  libraries' defaults as today.

### Terrain-type granularity

The API's `customPathCosts` exposes five categories (footway, path, trail,
residential, off_path). backend3's coarse `PathType` enum (UNKNOWN, OBSTACLE,
HIKING_PATH, STREET, NATURAL) cannot honor that. The new `path_layer.py` defines
a type grid with these categories:

- `OBSTACLE` — water, buildings, cliffs (from `obstacle_config.py` tags)
- `TRAIL` — hiking trails (OSM `highway=path` with trail tags, `sac_scale`, etc.)
- `PATH` — generic paths
- `FOOTWAY` — sidewalks/footways
- `RESIDENTIAL` — residential streets
- `NATURAL` — natural surface (grass, meadow, etc.)
- `UNKNOWN` — off-path terrain

Option mapping in `service.py`:

- `customPathCosts.{footway,path,trail,residential,off_path}` → per-type cost multipliers
- `trailPreference` → scales trail/natural multipliers (1.0 = defaults)
- `gradientPreference` → `elevation_weight` / `elevation_exponent`
- `maxSlope` → `max_slope_degrees`
- `userProfile` presets → same preset tables v1 uses (reuse
  `PathPreferencePresets` mappings translated to v2 cost dicts)

## Data flow

1. Route request arrives; bounds computed from start/end + buffer (reuse v1's
   `calculate_bounding_box` logic).
2. `elevation_lib.load_area(bounds)` — downloads via py3dep only when not
   already cached (existing ~1.2GB HTTP cache makes known areas fast).
3. `path_layer.get_grid(bounds, shape, transform)` — loads from disk cache or
   fetches OSM and rasterizes to match the elevation grid exactly.
4. `TerrainAwarePathfinder` constructed with elevation array + terrain grid;
   parameters set from options.
5. `find_path` → list of (lat, lon, elevation) → converted to `Coordinate` list
   + stats dict (distance, elevation gain/loss, estimated time, nodes explored,
   engine name, timing breakdown).

## Error handling

Same contract as v1: failures return `([], {"error": <message>})`; the endpoint
already maps that to a FAILED route with the message.

- OSM fetch failure → degrade gracefully to terrain-only routing, add
  `"warnings": ["OSM data unavailable — terrain-only routing"]` to stats
  (matches old engine's empty-GeoDataFrame behavior).
- Elevation download failure → hard error with a clear message naming the
  bounds that failed.
- Validation (identical points, distance cap) → reuse v1's checks before any
  data loading.

## Testing

- **Unit — path_layer:** synthetic OSM geometries (a trail line, a water
  polygon, a residential road) rasterized onto a small grid produce the expected
  type codes at expected cells; grid aligns with a known transform.
- **Unit — weighted A*:** on a synthetic terrain, `heuristic_weight=1.0`
  reproduces the optimal path; `heuristic_weight=2.0` explores measurably fewer
  nodes and produces a path no more than ~15% longer.
- **Parity harness (the success criterion):** 3–5 known routes from existing
  e2e tests (Park City / Salt Lake area — data already cached). For each route,
  run both engines through the API and compare: success, route length,
  trail-adherence % (fraction of path points landing on trail/path/footway
  cells in the terrain grid), wall-clock time, and absence of tile-boundary
  artifacts (no single-step elevation jump > 20m along the path). Pass bar: v2
  succeeds on all routes, v2 trail-adherence ≥ v1's on each route, and v2
  wall-clock ≤ 1.25× v1 on each route.
- **API:** `engine` flag dispatches correctly; omitted flag → v1, unchanged
  responses on existing tests.

## Implementation notes

- Implementation will be executed by a team of parallel agents (per user
  request): independent workstreams for `path_layer.py`, pathfinder port +
  weighted A*, service adapter + engine flag, and the parity harness, followed
  by an integration pass.
- `backend3/` is left untouched as reference; retiring it (and pointing its CLI
  at the new package) is explicitly out of scope for this project.
- Tests run with `backend/venv` (has all geo deps); root `venv` lacks fastapi.

## Out of scope

- Frontend changes beyond optionally passing `engine` (not required for parity
  testing via API).
- Deeper performance work (tuple nodes, bidirectional A*, hierarchical search).
- Migrating slope-overlay / cost-surface / debug endpoints to v2.
- Deleting the v1 engine or backend3.
