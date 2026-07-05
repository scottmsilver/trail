# Eval UI — Design

**Date:** 2026-07-04
**Status:** Approved (pending spec review)
**Repo:** `development/trail`

## Purpose

Give the route author a workbench to (1) run a route and see the engine's path,
(2) draw a candidate "better" path on the map, and (3) understand *why* the engine
did not choose it — concretely, with per-segment cost attribution — and then tune
the cost weights to see what *would* make the candidate win. Saved cases accumulate
into a manually-labeled regression suite.

This maps to three user asks:

1. **Run a route, see the path** — already exists (`POST /api/routes/calculate`).
2. **Draw a candidate path** — new: a lightweight click-to-draw mode on the existing
   Leaflet map.
3. **Understand why it wasn't chosen** — new: score the drawn path with the *exact*
   cost function A\* minimizes, compare totals, attribute the excess cost per segment
   by factor, and offer a calibration loop.

## Core idea: one scorer, two paths

The engine's A\* minimizes a sum of per-cell move costs from
`calculate_move_cost()` (distance × terrain multiplier × (1 + slope penalty +
sustained-fatigue penalty + deviation penalty)). A path scorer walks *any* polyline
across the same DEM grid and sums that identical function cell-by-cell. Run it on:

- the **engine's optimal path** (from `calculate`), and
- the **user's drawn path**,

under **identical `RouteOptions`**, and the comparison is honest by construction.
"Why wasn't mine chosen" is then the per-segment cost difference, decomposed by factor.

### Correctness invariant (the tool's whole point)

Under identical weights, the engine's optimal cost **must be ≤** the drawn path's
cost. If a drawn path ever scores *lower* than the engine's optimal, either:

- the engine is suboptimal (a real routing bug the eval tool just caught), or
- the scorer is inconsistent with the engine (a scorer bug).

Either way it is a signal, not noise. This becomes a property test and is a headline
feature, not just a check.

## Scope decisions (from brainstorming)

- **Depth of "why":** full loop — cost breakdown + per-segment attribution **plus**
  calibration/what-if (tune weights, re-score, see if the engine would prefer the
  drawn path). Reuses the existing `CalibrationToolbar`.
- **Score semantics:** snap the drawn line to the nearest trail, then score the
  snapped path — *default*. Where no trail is near, fall back to scoring exactly what
  was drawn (off-trail is a legitimate "better path" in a terrain-aware router).
  Exact-drawn scoring is the foundation; snap is a preprocessing layer on top.
- **Persistence:** full eval harness — cases saved to disk, re-runnable.
- **Pass criterion:** manual labeling (✓ good / ✗ wrong / ? unsure) per re-run.
  Auto-verdicts (corridor-match, cost-competitive) are explicitly deferred.

## Where it lives

A new `/eval` page in the existing frontend, reusing `Map`, `CalibrationToolbar`,
and the cost-explorer components. Not a separate app. No new map dependency:
click-to-draw is implemented with native Leaflet map events (the app already sets
start/end by clicking), avoiding `leaflet-draw` / React-19 compatibility friction.

## The frozen API contract

This is the linchpin of parallel execution. Every track codes to these shapes.
Both the optimal and drawn paths are returned as the **same `ScoredPath`** shape, so
the frontend has a single renderer.

### `POST /api/eval/score-path`

Request:

```jsonc
{
  "path": [{ "lat": 0, "lon": 0 }],   // >= 2 points
  "options": { /* RouteOptions, identical shape to RouteRequest.options */ },
  "snap": "none" | "trail"            // default "none" until snap (T2) lands
}
```

Response — `ScoredPath`:

```jsonc
{
  "path": [{ "lat": 0, "lon": 0 }],   // geometry actually scored (snapped if snap="trail")
  "snapped": false,
  "totalCost": 0.0,
  "distanceM": 0.0,
  "elevationGainM": 0.0,
  "segments": [
    {
      "from": { "lat": 0, "lon": 0 },
      "to":   { "lat": 0, "lon": 0 },
      "cost": 0.0,
      "factors": { "base": 0, "terrain": 0, "slope": 0, "sustained": 0, "deviation": 0 },
      "dominantFactor": "slope"
    }
  ]
}
```

A "segment" is one drawn polyline edge (from vertex i to vertex i+1). Its `cost` is
the sum of the per-cell move costs along the rasterized edge; `factors` is the
decomposition of that sum; `dominantFactor` is the largest non-base contributor.

### Eval cases

- `GET  /api/eval/cases` → `EvalCase[]`
- `POST /api/eval/cases` → save/replace an `EvalCase` (returns it with an id)
- `DELETE /api/eval/cases/{id}`

`EvalCase`:

```jsonc
{
  "id": "half-dome-via-panorama",
  "name": "Half Dome via Panorama",
  "notes": "engine takes the ridge; Panorama trail is the real route",
  "start": { "lat": 0, "lon": 0 },
  "end":   { "lat": 0, "lon": 0 },
  "options": { /* RouteOptions */ },
  "referencePath": [{ "lat": 0, "lon": 0 }],
  "labels": [{ "ts": "2026-07-04T00:00:00Z", "verdict": "fail", "note": "" }]
}
```

Cases are JSON files under `evals/` (one file per case). Re-run = call `calculate`
(optimal) + `score-path` (reference) with the stored `options`, show both as
`ScoredPath`, user appends a label.

## Backend design

### Scorer (T1)

- **Refactor** `calculate_move_cost()` to *optionally* return a factor breakdown
  `{base, terrain, slope, sustained, deviation}` alongside the scalar it returns
  today. Additive and non-breaking — the hot A\* path keeps using the scalar; the
  breakdown is computed only when requested.
- **New service** `score_path(polyline, options) -> ScoredPath`:
  1. Load the DEM/cost grid for the polyline's bounding box (same loader the engine
     uses; fail loudly if data absent, per the no-implicit-caching rule).
  2. For each polyline edge, rasterize it into the sequence of grid cells it crosses
     (supercover line), then sum `calculate_move_cost` over adjacent cell pairs,
     accumulating both total and per-factor breakdown.
  3. Aggregate distance and elevation gain from the same cell walk.
- **New endpoint** `POST /api/eval/score-path` wraps the service and reuses
  `RouteOptions` validation.

### Snap-to-trail (T2)

Polyline preprocessing applied when `snap == "trail"`: fetch OSM way geometry for the
bbox via the existing `/api/terrain/osm-data` machinery, snap each drawn vertex/edge
to the nearest trail-class way within a threshold, and return the snapped polyline as
the scored geometry (`snapped: true`). Vertices with no trail within threshold are
left as-drawn. Integrated as a step before scoring; the scorer is unaware of snapping.

### Eval cases (T3)

CRUD over JSON files in `evals/`, plus a re-run helper that orchestrates
`calculate` + `score_path`. No auto-verdict logic — labels are user-supplied.

## Frontend design

### Eval page shell + draw (T4)

- New `/eval` route hosting the `Map`.
- Set start/end (existing interaction) → **Run** → optimal path rendered as a
  `ScoredPath`.
- **Draw mode** toggle: click to drop vertices, Enter / double-click to finish; the
  drawn polyline is sent to `/score-path`.

### Compare + attribution (T5)

- **Compare panel:** optimal vs drawn — distance, elevation gain, total cost, and %
  difference.
- **Attribution:** per-segment cost bars colored by `dominantFactor`; hovering a
  segment highlights it on the map and vice versa. This is the visible "why."
- Built against a mocked `ScoredPath` fixture so it proceeds in parallel with T1.

### Calibration integration (T6)

Embed `CalibrationToolbar`; on any weight change, re-score both paths and update a
live verdict line ("engine now prefers yours: 1,590 ≤ 1,610").

### Cases UI (T7)

Save the current setup as an `EvalCase` (name + notes), list/load/re-run cases, and
apply a manual ✓/✗/? label with an optional note.

## Parallel execution plan

Freezing the contract above is what makes this safe to fan out.

- **Wave 0 — contract seed (serial, thin):** Pydantic models + TS types for
  `ScoredPath` and `EvalCase`; endpoint stubs returning fixtures; `/eval` page route
  stub.
- **Wave 1 — 5 parallel tracks:** T1 scorer, T2 snap, T3 cases (backend); T4 page +
  draw, T5 compare + attribution (frontend, against the mock).
- **Wave 2 — join (parallel):** T6 calibration + T7 cases UI; swap mocks for live
  endpoints.
- **Wave 3:** end-to-end QA (`/qa`) + the optimal-≤-drawn invariant on real terrain.

Dependency notes: T5 renders the contract and is mockable, so it runs fully parallel
to T1. T6 and T7 need T4's page shell to exist, so they land in Wave 2. T2 and T3
depend only on the contract, not on each other or on T1's internals.

## Testing

- **Backend (TDD):** hand-computed costs on tiny known grids; the optimal-≤-drawn
  invariant as a property test; snap idempotence (snapping an already-snapped path is
  a no-op); factor-decomposition sums to the scalar cost.
- **Frontend:** vitest for panel rendering and draw-mode state against `ScoredPath`
  fixtures; manual QA per wave via `/qa`.

## Explicit non-goals (YAGNI)

- No automated pass/fail verdicts yet (manual labels only).
- No new map/drawing dependency.
- No caching changes — honor the existing "fail when data absent" rule.
- No cross-case aggregate dashboards beyond a simple list + per-case labels.
