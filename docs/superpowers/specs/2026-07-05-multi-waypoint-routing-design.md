# Multi-Waypoint Routing — Design

**Date:** 2026-07-05
**Status:** Approved (pending spec review)
**Repo:** `development/trail`

## Purpose

Let the route author drop **multiple points** on the map and get a single route that
passes **through all of them, in the order dropped**. Today the whole stack is
hard-wired to exactly two points (start + end). This generalizes it to
`start -> wp1 -> wp2 -> ... -> end` without changing the routing engine itself.

Maps to one user ask: "route building where I drop multiple points and the route
must go through those points."

## Core idea: one A* per leg, stitched server-side

The v2 engine already solves one terrain-aware A* from A->B over a bounding box
(`trail_finder_v2.find_route(start, end, options)`). A multi-point route is just
**N-1 of those legs concatenated**. No engine changes — we orchestrate the existing
single-pair solver once per consecutive pair and join the results.

**v2 only.** Multi-waypoint routing is a v2 feature. The v1 path is left exactly as
it is today (two-point start/end); a multi-point request against engine v1 is
rejected with a clear error rather than silently stitched.

Because order is fixed (drop order), there is no traveling-salesman problem: point 1
is the start, point N is the end, and each intermediate point is visited in sequence.

## Scope decisions (from brainstorming)

- **Order:** fixed drop order. Point 1 = start, point N = end. No order optimization / TSP.
- **Location:** extend the **existing main map**, not a separate page. Waypoints are a
  natural generalization of the current two-point flow; options, GPX export, and cost
  overlays already live there.
- **Placement:** sequential clicks — each map click **appends** a point.
- **Editing:** delete any point, drag any marker to move it, append at the end.
  **No** reorder and **no** insert-between in v1 (delete + re-add covers it).
- **Stitching:** server-side. One job id, one concatenated path, aggregated stats.
  The frontend does **not** fire N separate calls.

## Backend design

### Request model (`backend/app/models/route.py`)

- Add `points: Optional[List[Coordinate]]` to `RouteRequest` (must have >= 2 entries
  when present).
- Make `start` and `end` **optional**.
- A model validator requires **either** both `start` + `end` **or** `points` (>= 2).
- Normalize internally to a single points list: legacy `{start, end}` becomes
  `[start, end]`. Everything downstream operates on the list, so **existing 2-point
  callers keep working unchanged** (back-compat is a headline requirement).

### Stitching helper

`stitch_route(finder, points, options) -> (path, stats)`:

1. Collapse consecutive **identical** points (zero-length legs are skipped).
2. For each consecutive pair `(points[i], points[i+1])`, call the existing
   `trail_finder_v2.find_route(a, b, options)` (async).
3. Concatenate leg paths, **dropping the duplicated junction vertex** between legs
   (leg k's last point equals leg k+1's first point).
4. Aggregate stats: `distance_m` and `elevation_gain_m` are summed across legs; a
   per-leg breakdown is included in `stats` (e.g. `stats["legs"] = [...]`).

### Failure semantics

If any leg fails (no route found, points too far apart, invalid), the **whole job
fails** with a message naming the leg index — e.g.
`"No route found for leg 2 (point 2 -> point 3)"`. No partial routes are returned.

### Wiring (`backend/app/main.py`)

In the **v2 branch**, `process_route` (and the debug/export paths) build the
normalized points list from the request and call `stitch_route` against
`trail_finder_v2`. The v1 branch is untouched and continues to route two points
only. The async job model, status polling, and GPX generation are unchanged — they
still see one concatenated path.

## Frontend design

### State (`frontend/src/App.tsx`)

- Replace `start` / `end` with `points: Coordinate[]`.
- Consumers that still think in start/end — GPX export, "save current point",
  cost-surface `startCoord` / `endCoord`, coordinate display — derive
  `start = points[0]` and `end = points[points.length - 1]`, keeping their churn
  minimal.
- Each map click appends a point. "Clear" resets to `[]`.

### Map (`frontend/src/components/Map/Map.tsx`)

- Render **numbered markers** `1...N`. Point 1 and point N keep the distinct
  start/end styling; intermediate points get a "via" style.
- Markers are **draggable**: `dragend` updates that point's coordinate.
- Each marker popup has a **Delete** button that removes that point.
- The route polyline is the single concatenated path, rendered exactly as today.

### Points list panel

A small ordered list mirroring the markers (point index + coordinate + delete
control), so points can be managed without hunting on the map.

### Request (`frontend/src/services/api.ts`)

`calculateRoute` sends `{ points, options }`. Types updated so `points: Coordinate[]`.

## Testing (TDD)

### Backend

- A 3-point stitch equals the concatenation of its two legs, with **no duplicated
  junction** vertex.
- Aggregated `distance_m` / `elevation_gain_m` equal the sum of the per-leg values.
- A failing **middle** leg fails the whole route, with a message naming the leg.
- **Back-compat:** a legacy `{start, end}` request still routes (normalizes to a
  2-point list).
- Order is preserved (points visited in drop order).
- Consecutive **duplicate** points are collapsed (no zero-length leg).
- A **multi-point** request with `engine: "v1"` is **rejected** with a clear error.

### Frontend

- Placing N points renders N numbered markers.
- The request payload carries `points`.
- Delete removes a point and its marker; the list shrinks.
- Dragging a marker updates that point's coordinate.
- Deriving `start` / `end` for legacy consumers returns first / last point.

## Explicit non-goals (YAGNI)

- No order optimization / TSP — order is exactly the drop order.
- No reorder or insert-between in v1 (delete + re-add covers it).
- No per-leg option overrides (all legs share one `RouteOptions`).
- No caching changes — honor the existing "fail when data absent" rule.
