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

# Engine v2 Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate the backend3 engine (two-layer elevation + terrain-aware A*) into the FastAPI backend as a selectable "v2" engine, including the missing `path_layer` OSM module and weighted A*.

**Architecture:** New package `backend/app/engine_v2/` with four modules (elevation library moved from backend3, new path_layer, ported pathfinder with weighted A*, thin service adapter matching v1's `find_route` contract). Engine selected per-request via `options.engine`; v1 untouched and remains the default.

**Tech Stack:** Python 3, FastAPI, numpy, rasterio, osmnx, geopandas, shapely, py3dep, pytest.

**Spec:** `docs/superpowers/specs/2026-07-03-engine-v2-integration-design.md`

## Global Constraints

- **Rule #0 header:** EVERY new file starts with this comment block (adapt comment syntax to file type):

```python
# RULE #0: MANDATORY FIRST ACTION FOR EVERY REQUEST
# 1. Read CLAUDE.md COMPLETELY before responding
# 2. Setup Python venv: [ -d "venv" ] || ./setup-venv.sh && source venv/bin/activate
# 3. Search for rules related to the request
# 4. Only proceed after confirming no violations
# Failure to follow Rule #0 has caused real harm. Check BEFORE acting, not AFTER making mistakes.
#
# GUARDS ARE SAFETY EQUIPMENT - WHEN THEY FIRE, FIX THE PROBLEM THEY FOUND
# NEVER weaken, disable, or bypass guards - they prevent real harm
```

  In the task code blocks below this header is abbreviated as `# <RULE-0 HEADER>` — paste the full block verbatim.
- **Python interpreter:** ALWAYS `backend/venv` (has fastapi + geo stack). Run tests from `backend/`: `cd /home/ssilver/development/trail/backend && ./venv/bin/python -m pytest <path> -v`. The repo-root `venv` does NOT work.
- **No mocking frameworks.** Unit tests use dependency injection and synthetic in-memory data (GeoDataFrames, numpy arrays). Only the e2e parity test touches network/cached real data. (User approved this test design in the spec review.)
- **No hardcoded server URLs** (osmnx/py3dep library defaults are fine).
- **Commits:** commit after each task (user authorized commits this session). NEVER push. NEVER `--no-verify`. After each commit run `git log --oneline -1` to verify it went through.
- **v1 must remain byte-for-byte unchanged in behavior** when `options.engine` is absent or `"v1"`.
- Engine defaults locked in the spec: `heuristic_weight` default 1.0; terrain cost defaults per Task 5; parity pass bars per Task 9.

## File Structure

```
backend/app/engine_v2/
    __init__.py          # exports: TwoLayerElevationLibrary, Bounds, PathLayer, PathType,
                         #          TerrainAwarePathfinder, TrailFinderServiceV2
    elevation.py         # copied from backend3/elevation.py (proven code, minimal edits)
    elevation_fd_safe.py # copied from backend3/elevation_fd_safe.py
    path_layer.py        # NEW: PathType enum, classify_feature(), rasterize_features(), PathLayer
    pathfinder.py        # ported TerrainAwarePathfinder + heuristic_weight, returns (path, stats)
    service.py           # TrailFinderServiceV2 - same find_route contract as v1
backend/app/models/route.py   # MODIFY: add engine + heuristicWeight to RouteOptions
backend/app/main.py           # MODIFY: instantiate v2 service, dispatch in process_route
backend/tests/unit/test_path_layer.py        # NEW
backend/tests/unit/test_pathfinder_v2.py     # NEW
backend/tests/integration/test_engine_v2_api.py  # NEW
backend/tests/e2e/test_engine_parity.py      # NEW (parity harness = success criterion)
```

---

### Task 1: Package scaffold + move elevation library

**Files:**
- Create: `backend/app/engine_v2/__init__.py`
- Create: `backend/app/engine_v2/elevation.py` (copy of `backend3/elevation.py`)
- Create: `backend/app/engine_v2/elevation_fd_safe.py` (copy of `backend3/elevation_fd_safe.py`)
- Test: `backend/tests/unit/test_engine_v2_imports.py`

**Interfaces:**
- Consumes: nothing (first task)
- Produces: `from app.engine_v2.elevation import TwoLayerElevationLibrary, Bounds` and `from app.engine_v2.elevation_fd_safe import FDManagedElevationLibrary`. `Bounds(south, north, west, east)` dataclass. `lib.load_area(bounds) -> dict`, `lib.get_elevation_array(bounds) -> Tuple[np.ndarray, dict]`, `lib.get_elevation(lat, lon) -> float`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/unit/test_engine_v2_imports.py
# <RULE-0 HEADER>
"""Smoke test: engine_v2 package imports cleanly."""


def test_elevation_library_imports():
    from app.engine_v2.elevation import TwoLayerElevationLibrary, Bounds
    b = Bounds(south=40.0, north=40.1, west=-111.6, east=-111.5)
    assert b.north > b.south


def test_fd_safe_wrapper_imports():
    from app.engine_v2.elevation_fd_safe import FDManagedElevationLibrary
    assert hasattr(FDManagedElevationLibrary, "close_all")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ssilver/development/trail/backend && ./venv/bin/python -m pytest tests/unit/test_engine_v2_imports.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.engine_v2'`

- [ ] **Step 3: Create the package**

```bash
mkdir -p /home/ssilver/development/trail/backend/app/engine_v2
cp /home/ssilver/development/trail/backend3/elevation.py /home/ssilver/development/trail/backend/app/engine_v2/elevation.py
cp /home/ssilver/development/trail/backend3/elevation_fd_safe.py /home/ssilver/development/trail/backend/app/engine_v2/elevation_fd_safe.py
```

Then:
1. Add the Rule #0 header comment to the top of both copied files (below the shebang/docstring if present).
2. In `elevation_fd_safe.py`, fix its import of the elevation module if it has one: any `from elevation import ...` becomes `from app.engine_v2.elevation import ...`.
3. Create `__init__.py`:

```python
# backend/app/engine_v2/__init__.py
# <RULE-0 HEADER>
"""Engine v2: two-layer elevation + terrain-aware weighted A* pathfinding."""
from app.engine_v2.elevation import TwoLayerElevationLibrary, Bounds
from app.engine_v2.elevation_fd_safe import FDManagedElevationLibrary

__all__ = [
    "TwoLayerElevationLibrary",
    "Bounds",
    "FDManagedElevationLibrary",
]
```

(`PathLayer`, `PathType`, `TerrainAwarePathfinder`, `TrailFinderServiceV2` get added to this file by Tasks 2, 5, 7.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/ssilver/development/trail/backend && ./venv/bin/python -m pytest tests/unit/test_engine_v2_imports.py -v`
Expected: 2 PASSED

- [ ] **Step 5: Commit**

```bash
cd /home/ssilver/development/trail
git add backend/app/engine_v2/ backend/tests/unit/test_engine_v2_imports.py
git commit -m "feat(engine-v2): scaffold package, move two-layer elevation library from backend3"
git log --oneline -1
```

---

### Task 2: path_layer — PathType enum + OSM feature classification

**Files:**
- Create: `backend/app/engine_v2/path_layer.py`
- Modify: `backend/app/engine_v2/__init__.py` (add exports)
- Test: `backend/tests/unit/test_path_layer.py`

**Interfaces:**
- Consumes: nothing new
- Produces: `PathType(IntEnum)` with members `UNKNOWN=0, OBSTACLE=1, TRAIL=2, PATH=3, FOOTWAY=4, RESIDENTIAL=5, NATURAL=6`; `classify_feature(tags: dict) -> PathType`; `PATH_TAGS: dict`, `OBSTACLE_TAGS: dict` module constants; `get_path_type_name(code: int) -> str`.

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/unit/test_path_layer.py
# <RULE-0 HEADER>
"""Unit tests for the v2 path layer (OSM classification + rasterization)."""
import numpy as np
import pytest

from app.engine_v2.path_layer import PathType, classify_feature, get_path_type_name


class TestClassifyFeature:
    def test_hiking_path_is_trail(self):
        assert classify_feature({"highway": "path"}) == PathType.TRAIL

    def test_track_is_trail(self):
        assert classify_feature({"highway": "track"}) == PathType.TRAIL

    def test_piste_is_trail(self):
        assert classify_feature({"piste:type": "downhill"}) == PathType.TRAIL

    def test_hiking_route_is_trail(self):
        assert classify_feature({"route": "hiking"}) == PathType.TRAIL

    def test_steps_and_cycleway_are_path(self):
        assert classify_feature({"highway": "steps"}) == PathType.PATH
        assert classify_feature({"highway": "cycleway"}) == PathType.PATH

    def test_footway_and_pedestrian_are_footway(self):
        assert classify_feature({"highway": "footway"}) == PathType.FOOTWAY
        assert classify_feature({"highway": "pedestrian"}) == PathType.FOOTWAY

    def test_roads_are_residential(self):
        for hw in ["residential", "living_street", "service", "unclassified",
                   "tertiary", "secondary", "primary"]:
            assert classify_feature({"highway": hw}) == PathType.RESIDENTIAL, hw

    def test_park_and_meadow_are_natural(self):
        assert classify_feature({"leisure": "park"}) == PathType.NATURAL
        assert classify_feature({"natural": "meadow"}) == PathType.NATURAL
        assert classify_feature({"landuse": "grass"}) == PathType.NATURAL

    def test_highway_takes_priority_over_landuse(self):
        # A road through a park is still a road
        assert classify_feature({"highway": "residential", "landuse": "grass"}) == PathType.RESIDENTIAL

    def test_unrecognized_is_unknown(self):
        assert classify_feature({"building": "yes"}) == PathType.UNKNOWN
        assert classify_feature({}) == PathType.UNKNOWN

    def test_none_values_ignored(self):
        # osmnx GeoDataFrames contain NaN/None for absent tags
        assert classify_feature({"highway": None, "leisure": "park"}) == PathType.NATURAL


def test_path_type_names():
    assert get_path_type_name(PathType.TRAIL) == "trail"
    assert get_path_type_name(PathType.UNKNOWN) == "off_path"
    assert get_path_type_name(255) == "invalid"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ssilver/development/trail/backend && ./venv/bin/python -m pytest tests/unit/test_path_layer.py -v`
Expected: FAIL with `ImportError` / `ModuleNotFoundError` for `app.engine_v2.path_layer`

- [ ] **Step 3: Implement classification**

```python
# backend/app/engine_v2/path_layer.py
# <RULE-0 HEADER>
"""
Path layer for engine v2: fetches OSM features and rasterizes them into a
terrain-type grid aligned with the elevation array.

This is the module the backend3 rewrite referenced but never implemented.
Tag vocabularies are ported from app/services/path_preferences.py and
app/services/obstacle_config.py (the proven v1 configuration).
"""
import hashlib
import logging
import os
from enum import IntEnum
from typing import Callable, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class PathType(IntEnum):
    UNKNOWN = 0      # off-path terrain
    OBSTACLE = 1     # water, buildings, cliffs — impassable
    TRAIL = 2        # natural hiking trails / tracks / pistes
    PATH = 3         # steps, cycleways
    FOOTWAY = 4      # sidewalks, pedestrian areas
    RESIDENTIAL = 5  # streets
    NATURAL = 6      # parks, meadows, grass


_TYPE_NAMES = {
    PathType.UNKNOWN: "off_path",
    PathType.OBSTACLE: "obstacle",
    PathType.TRAIL: "trail",
    PathType.PATH: "path",
    PathType.FOOTWAY: "footway",
    PathType.RESIDENTIAL: "residential",
    PathType.NATURAL: "natural",
}

# Ported from PathPreferences.get_default_path_tags()
PATH_TAGS = {
    "highway": ["footway", "path", "track", "pedestrian", "steps",
                "cycleway", "bridleway", "trail", "residential",
                "living_street", "service", "unclassified"],
    "leisure": ["park", "nature_reserve", "garden", "common", "playground"],
    "natural": ["grassland", "meadow", "heath", "scrub", "beach", "sand"],
    "landuse": ["grass", "meadow", "recreation_ground", "village_green"],
    "route": ["hiking", "foot", "walking"],
    "piste:type": ["downhill", "nordic", "sled", "hike", "skitour", "connection"],
}

# Ported from ObstacleConfig.get_default_osm_tags()
OBSTACLE_TAGS = {
    "natural": ["water", "wetland", "cliff", "rock", "scree"],
    "waterway": ["river", "stream", "canal"],
    "landuse": ["industrial", "commercial", "military"],
    "building": True,
    "leisure": ["golf_course", "swimming_pool"],
    "barrier": True,
}

_TRAIL_HIGHWAYS = {"path", "track", "trail", "bridleway"}
_PATH_HIGHWAYS = {"steps", "cycleway"}
_FOOTWAY_HIGHWAYS = {"footway", "pedestrian"}
_ROAD_HIGHWAYS = {"residential", "living_street", "service", "unclassified",
                  "tertiary", "secondary", "primary"}
_NATURAL_LEISURE = {"park", "nature_reserve", "garden", "common", "playground"}
_NATURAL_NATURAL = {"grassland", "meadow", "heath", "scrub", "beach", "sand"}
_NATURAL_LANDUSE = {"grass", "meadow", "recreation_ground", "village_green"}
_TRAIL_ROUTES = {"hiking", "foot", "walking"}


def _tag(tags: dict, key: str) -> Optional[str]:
    """Get a tag value, treating None/NaN as absent."""
    value = tags.get(key)
    if value is None:
        return None
    # pandas NaN
    if isinstance(value, float) and np.isnan(value):
        return None
    return str(value)


def classify_feature(tags: dict) -> PathType:
    """Map a feature's OSM tags to a PathType. Highway takes priority."""
    highway = _tag(tags, "highway")
    if highway in _TRAIL_HIGHWAYS:
        return PathType.TRAIL
    if highway in _PATH_HIGHWAYS:
        return PathType.PATH
    if highway in _FOOTWAY_HIGHWAYS:
        return PathType.FOOTWAY
    if highway in _ROAD_HIGHWAYS:
        return PathType.RESIDENTIAL
    if _tag(tags, "piste:type") is not None:
        return PathType.TRAIL
    if _tag(tags, "route") in _TRAIL_ROUTES:
        return PathType.TRAIL
    if _tag(tags, "leisure") in _NATURAL_LEISURE:
        return PathType.NATURAL
    if _tag(tags, "natural") in _NATURAL_NATURAL:
        return PathType.NATURAL
    if _tag(tags, "landuse") in _NATURAL_LANDUSE:
        return PathType.NATURAL
    return PathType.UNKNOWN


def get_path_type_name(code: int) -> str:
    try:
        return _TYPE_NAMES[PathType(code)]
    except ValueError:
        return "invalid"
```

Also add to `backend/app/engine_v2/__init__.py`:

```python
from app.engine_v2.path_layer import PathLayer, PathType  # PathLayer added in Task 4
```

(For this task, export only `PathType`; add `PathLayer` to the import in Task 4.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/ssilver/development/trail/backend && ./venv/bin/python -m pytest tests/unit/test_path_layer.py -v`
Expected: all PASSED

- [ ] **Step 5: Commit**

```bash
cd /home/ssilver/development/trail
git add backend/app/engine_v2/path_layer.py backend/app/engine_v2/__init__.py backend/tests/unit/test_path_layer.py
git commit -m "feat(engine-v2): add PathType enum and OSM feature classification"
git log --oneline -1
```

---

### Task 3: path_layer — rasterization with precedence

**Files:**
- Modify: `backend/app/engine_v2/path_layer.py` (add `rasterize_features`)
- Test: `backend/tests/unit/test_path_layer.py` (append)

**Interfaces:**
- Consumes: `classify_feature`, `PathType` (Task 2)
- Produces: `rasterize_features(paths_gdf, obstacles_gdf, shape: Tuple[int, int], transform) -> np.ndarray` — uint8 grid of PathType codes. `paths_gdf`/`obstacles_gdf` are GeoDataFrames in EPSG:4326 (or empty). Precedence low→high: NATURAL < RESIDENTIAL < FOOTWAY < PATH < TRAIL < OBSTACLE (obstacles always win — safety).

- [ ] **Step 1: Write the failing tests** (append to `tests/unit/test_path_layer.py`)

```python
import geopandas as gpd
from rasterio.transform import from_bounds
from shapely.geometry import LineString, Polygon

from app.engine_v2.path_layer import rasterize_features

# 10x10 grid covering a 0.01 x 0.01 degree box
SHAPE = (10, 10)
TRANSFORM = from_bounds(-111.51, 40.64, -111.50, 40.65, SHAPE[1], SHAPE[0])


def _gdf(geoms_and_tags):
    """Build a GeoDataFrame from [(geometry, {tag: value}), ...]."""
    if not geoms_and_tags:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    records = []
    for geom, tags in geoms_and_tags:
        rec = dict(tags)
        rec["geometry"] = geom
        records.append(rec)
    return gpd.GeoDataFrame(records, crs="EPSG:4326")


class TestRasterizeFeatures:
    def test_trail_line_marks_trail_cells(self):
        # Horizontal trail across the middle of the box
        trail = LineString([(-111.51, 40.645), (-111.50, 40.645)])
        grid = rasterize_features(_gdf([(trail, {"highway": "path"})]),
                                  _gdf([]), SHAPE, TRANSFORM)
        assert grid.dtype == np.uint8
        assert (grid == PathType.TRAIL).sum() >= SHAPE[1]  # touches every column
        assert (grid == PathType.UNKNOWN).sum() > 0        # rest is off-path

    def test_obstacle_wins_over_trail(self):
        trail = LineString([(-111.51, 40.645), (-111.50, 40.645)])
        water = Polygon([(-111.51, 40.64), (-111.505, 40.64),
                         (-111.505, 40.65), (-111.51, 40.65)])  # left half
        grid = rasterize_features(
            _gdf([(trail, {"highway": "path"})]),
            _gdf([(water, {"natural": "water"})]),
            SHAPE, TRANSFORM)
        # Left half is obstacle even where the trail crosses it
        left = grid[:, : SHAPE[1] // 2 - 1]
        assert (left == PathType.OBSTACLE).all()
        # Trail still visible on the right half
        assert (grid[:, SHAPE[1] // 2 + 1:] == PathType.TRAIL).any()

    def test_trail_wins_over_park(self):
        park = Polygon([(-111.51, 40.64), (-111.50, 40.64),
                        (-111.50, 40.65), (-111.51, 40.65)])  # whole box
        trail = LineString([(-111.51, 40.645), (-111.50, 40.645)])
        grid = rasterize_features(
            _gdf([(park, {"leisure": "park"}), (trail, {"highway": "path"})]),
            _gdf([]), SHAPE, TRANSFORM)
        assert (grid == PathType.TRAIL).any()
        assert (grid == PathType.NATURAL).any()

    def test_empty_inputs_give_unknown_grid(self):
        grid = rasterize_features(_gdf([]), _gdf([]), SHAPE, TRANSFORM)
        assert grid.shape == SHAPE
        assert (grid == PathType.UNKNOWN).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ssilver/development/trail/backend && ./venv/bin/python -m pytest tests/unit/test_path_layer.py -v -k Rasterize`
Expected: FAIL with `ImportError: cannot import name 'rasterize_features'`

- [ ] **Step 3: Implement rasterization** (append to `path_layer.py`)

```python
from rasterio.features import rasterize as _rio_rasterize

# Rasterization precedence, low to high. Later entries overwrite earlier ones;
# obstacles are stamped last so they always win (safety).
_PRECEDENCE = [PathType.NATURAL, PathType.RESIDENTIAL, PathType.FOOTWAY,
               PathType.PATH, PathType.TRAIL, PathType.OBSTACLE]


def rasterize_features(paths_gdf, obstacles_gdf, shape, transform) -> np.ndarray:
    """
    Rasterize classified OSM features onto a grid of PathType codes.

    paths_gdf/obstacles_gdf: GeoDataFrames (EPSG:4326) as returned by
    osmnx.features_from_polygon, or empty. shape/transform must match the
    elevation array exactly so the grids align cell-for-cell.
    """
    grid = np.full(shape, PathType.UNKNOWN, dtype=np.uint8)

    shapes_by_type = {t: [] for t in _PRECEDENCE}

    if paths_gdf is not None and len(paths_gdf) > 0:
        for _, row in paths_gdf.iterrows():
            geom = row.get("geometry")
            if geom is None or geom.is_empty:
                continue
            ptype = classify_feature(row.to_dict())
            if ptype in shapes_by_type:
                shapes_by_type[ptype].append(geom)

    if obstacles_gdf is not None and len(obstacles_gdf) > 0:
        for _, row in obstacles_gdf.iterrows():
            geom = row.get("geometry")
            if geom is None or geom.is_empty:
                continue
            shapes_by_type[PathType.OBSTACLE].append(geom)

    for ptype in _PRECEDENCE:
        geoms = shapes_by_type[ptype]
        if not geoms:
            continue
        layer = _rio_rasterize(
            [(g, 1) for g in geoms],
            out_shape=shape,
            transform=transform,
            fill=0,
            all_touched=True,
            dtype=np.uint8,
        )
        grid[layer == 1] = ptype

    return grid
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/ssilver/development/trail/backend && ./venv/bin/python -m pytest tests/unit/test_path_layer.py -v`
Expected: all PASSED (classification tests still green)

- [ ] **Step 5: Commit**

```bash
cd /home/ssilver/development/trail
git add backend/app/engine_v2/path_layer.py backend/tests/unit/test_path_layer.py
git commit -m "feat(engine-v2): rasterize classified OSM features with obstacle-wins precedence"
git log --oneline -1
```

---

### Task 4: path_layer — PathLayer class (fetch + disk cache)

**Files:**
- Modify: `backend/app/engine_v2/path_layer.py` (add `PathLayer`)
- Modify: `backend/app/engine_v2/__init__.py` (export `PathLayer`)
- Test: `backend/tests/unit/test_path_layer.py` (append)

**Interfaces:**
- Consumes: `rasterize_features`, `PATH_TAGS`, `OBSTACLE_TAGS`, `Bounds` (from `app.engine_v2.elevation`)
- Produces: `PathLayer(cache_dir: str, fetch_fn: Optional[Callable] = None)`. `fetch_fn(bounds, tags) -> GeoDataFrame` is injectable for tests; default uses osmnx. `get_grid(bounds: Bounds, shape, transform) -> np.ndarray` (uint8 PathType codes, disk-cached). `PathLayer.get_path_type_name(code)` (staticmethod delegating to module function). OSM fetch failure degrades to empty GeoDataFrame (grid still produced), matching v1 behavior.

- [ ] **Step 1: Write the failing tests** (append to `tests/unit/test_path_layer.py`)

```python
from app.engine_v2.elevation import Bounds
from app.engine_v2.path_layer import PathLayer

BOUNDS = Bounds(south=40.64, north=40.65, west=-111.51, east=-111.50)


class TestPathLayer:
    def test_get_grid_uses_injected_fetcher_and_caches(self, tmp_path):
        calls = []

        def fake_fetch(bounds, tags):
            calls.append(tags)
            if "building" in tags:  # obstacle query
                return _gdf([])
            trail = LineString([(-111.51, 40.645), (-111.50, 40.645)])
            return _gdf([(trail, {"highway": "path"})])

        layer = PathLayer(cache_dir=str(tmp_path), fetch_fn=fake_fetch)
        grid1 = layer.get_grid(BOUNDS, SHAPE, TRANSFORM)
        assert (grid1 == PathType.TRAIL).any()
        assert len(calls) == 2  # one paths fetch + one obstacles fetch

        # Second call: served from disk cache, no new fetches
        grid2 = layer.get_grid(BOUNDS, SHAPE, TRANSFORM)
        assert len(calls) == 2
        assert np.array_equal(grid1, grid2)

    def test_fetch_failure_degrades_to_unknown_grid(self, tmp_path):
        def broken_fetch(bounds, tags):
            raise ConnectionError("overpass down")

        layer = PathLayer(cache_dir=str(tmp_path), fetch_fn=broken_fetch)
        grid = layer.get_grid(BOUNDS, SHAPE, TRANSFORM)
        assert (grid == PathType.UNKNOWN).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ssilver/development/trail/backend && ./venv/bin/python -m pytest tests/unit/test_path_layer.py -v -k PathLayer`
Expected: FAIL with `ImportError: cannot import name 'PathLayer'`

- [ ] **Step 3: Implement PathLayer** (append to `path_layer.py`)

```python
def _default_fetch(bounds, tags):
    """Fetch OSM features via osmnx. Imported lazily so unit tests never need it."""
    import geopandas as gpd
    import osmnx as ox
    from shapely.geometry import box as shapely_box

    bbox = shapely_box(bounds.west, bounds.south, bounds.east, bounds.north)
    ox.settings.log_console = False
    return ox.features_from_polygon(bbox, tags)


class PathLayer:
    """Produces terrain-type grids aligned to elevation arrays, cached on disk."""

    def __init__(self, cache_dir: str, fetch_fn: Optional[Callable] = None):
        self.cache_dir = cache_dir
        self.fetch_fn = fetch_fn or _default_fetch
        os.makedirs(cache_dir, exist_ok=True)

    @staticmethod
    def get_path_type_name(code: int) -> str:
        return get_path_type_name(code)

    def _cache_path(self, bounds, shape) -> str:
        key = (f"{bounds.south:.6f}_{bounds.west:.6f}_"
               f"{bounds.north:.6f}_{bounds.east:.6f}_{shape[0]}x{shape[1]}")
        digest = hashlib.sha1(key.encode()).hexdigest()[:16]
        return os.path.join(self.cache_dir, f"pathgrid_{digest}.npy")

    def _safe_fetch(self, bounds, tags):
        try:
            return self.fetch_fn(bounds, tags)
        except Exception as e:
            logger.warning(f"OSM fetch failed ({e}); continuing without this layer")
            return None

    def get_grid(self, bounds, shape, transform) -> np.ndarray:
        cache_file = self._cache_path(bounds, shape)
        if os.path.exists(cache_file):
            grid = np.load(cache_file)
            if grid.shape == tuple(shape):
                return grid
        paths_gdf = self._safe_fetch(bounds, PATH_TAGS)
        obstacles_gdf = self._safe_fetch(bounds, OBSTACLE_TAGS)
        grid = rasterize_features(paths_gdf, obstacles_gdf, shape, transform)
        # Only cache real data — a grid built during an OSM outage would
        # otherwise pin the outage to disk.
        if paths_gdf is not None or obstacles_gdf is not None:
            np.save(cache_file, grid)
        return grid
```

Update `__init__.py`'s path_layer import line to:

```python
from app.engine_v2.path_layer import PathLayer, PathType
```

and add `"PathLayer", "PathType"` to `__all__`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/ssilver/development/trail/backend && ./venv/bin/python -m pytest tests/unit/test_path_layer.py -v`
Expected: all PASSED

- [ ] **Step 5: Commit**

```bash
cd /home/ssilver/development/trail
git add backend/app/engine_v2/path_layer.py backend/app/engine_v2/__init__.py backend/tests/unit/test_path_layer.py
git commit -m "feat(engine-v2): PathLayer with injectable OSM fetch, graceful degradation, disk cache"
git log --oneline -1
```

---

### Task 5: pathfinder port — grid-in, stats-out

**Files:**
- Create: `backend/app/engine_v2/pathfinder.py` (port of `backend3/elevation_pathfinder_terrain.py`)
- Modify: `backend/app/engine_v2/__init__.py` (export `TerrainAwarePathfinder`)
- Test: `backend/tests/unit/test_pathfinder_v2.py`

**Interfaces:**
- Consumes: `PathType` (Task 2)
- Produces: `TerrainAwarePathfinder(elevation: np.ndarray, transform, terrain_types: np.ndarray, resolution: float = 10.0, heuristic_weight: float = 1.0)`. `set_parameters(**kwargs)`. `find_path(start_lat, start_lon, end_lat, end_lon) -> Optional[Tuple[List[Tuple[float, float, float]], dict]]` — `(path, stats)` or `None`; stats keys: `distance_m, elevation_gain_m, nodes_explored, time_s, terrain_breakdown` (dict name→fraction). `terrain_costs: Dict[PathType, float]` attribute (mutable by service).

**Port changes vs backend3 original (everything else is kept verbatim):**
1. Constructor takes arrays directly (no elevation_lib/path_layer objects, no tile loading/zoom logic) — the service aligns the grids.
2. All `print()` → `logger.info/debug`.
3. `reconstruct_path` returns `(path, stats)` instead of printing statistics; `find_path` returns that tuple.
4. New `PathType` vocabulary with these default costs (from the spec / v1 `path_costs` table):

```python
DEFAULT_TERRAIN_COSTS = {
    PathType.UNKNOWN: 0.5,       # off_path
    PathType.OBSTACLE: 10000.0,
    PathType.TRAIL: 0.2,
    PathType.PATH: 0.3,
    PathType.FOOTWAY: 0.6,
    PathType.RESIDENTIAL: 0.85,
    PathType.NATURAL: 0.4,
}
```

5. Trail-transition penalty updated to the new vocabulary: leaving `TRAIL` to anything not in `{TRAIL, PATH, NATURAL}` multiplies terrain cost by 1.5.
6. `heuristic_weight` (Task 6 tests it; wire it now): `heuristic()` returns `self.heuristic_weight * self._euclidean(...)`; the deviation penalty's `straight_line_distance` uses unweighted `_euclidean()` so weighting never distorts the deviation ratio.

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/unit/test_pathfinder_v2.py
# <RULE-0 HEADER>
"""Unit tests for the v2 terrain-aware pathfinder on synthetic terrain."""
import numpy as np
from rasterio.transform import from_bounds

from app.engine_v2.path_layer import PathType
from app.engine_v2.pathfinder import TerrainAwarePathfinder

SHAPE = (30, 30)
# 30x30 grid over a small box; ~11m per cell N-S
TRANSFORM = from_bounds(-111.51, 40.64, -111.507, 40.643, SHAPE[1], SHAPE[0])
FLAT = np.zeros(SHAPE, dtype=np.float32)
OPEN = np.full(SHAPE, PathType.UNKNOWN, dtype=np.uint8)

START = (40.6402, -111.5098)  # bottom-left area
END = (40.6428, -111.5072)    # top-right area


def make_pathfinder(elevation=FLAT, terrain=OPEN, **kwargs):
    return TerrainAwarePathfinder(elevation.copy(), TRANSFORM, terrain.copy(),
                                  resolution=10.0, **kwargs)


def test_finds_path_on_flat_open_terrain():
    pf = make_pathfinder()
    result = pf.find_path(*START, *END)
    assert result is not None
    path, stats = result
    assert len(path) > 5
    assert stats["distance_m"] > 0
    assert stats["nodes_explored"] > 0
    # Endpoints are where we asked
    assert abs(path[0][0] - START[0]) < 0.001
    assert abs(path[-1][0] - END[0]) < 0.001


def test_obstacle_wall_blocks_direct_route():
    terrain = OPEN.copy()
    terrain[10:20, :25] = PathType.OBSTACLE  # wall with a gap on the right
    pf = make_pathfinder(terrain=terrain)
    result = pf.find_path(*START, *END)
    assert result is not None
    path, _ = result
    # Path must detour through the gap: some point in the wall rows must be
    # east of the wall (col >= 25 → lon > left + 25/30 of width)
    gap_lon = -111.51 + 0.003 * (25 / 30)
    rows_in_wall = [p for p in path
                    if 40.64 + 0.003 * (10 / 30) <= p[0] <= 40.64 + 0.003 * (20 / 30)]
    assert rows_in_wall, "path never crossed the wall band"
    assert all(p[1] >= gap_lon for p in rows_in_wall)


def test_full_obstacle_wall_means_no_path():
    terrain = OPEN.copy()
    terrain[15, :] = PathType.OBSTACLE  # full-width wall
    pf = make_pathfinder(terrain=terrain)
    assert pf.find_path(*START, *END) is None


def test_prefers_trail_over_off_path():
    terrain = OPEN.copy()
    terrain[:, 15] = PathType.TRAIL  # vertical trail down the middle
    pf = make_pathfinder(terrain=terrain)
    result = pf.find_path(40.6401, -111.5085, 40.6428, -111.5085)  # start/end on col 15
    assert result is not None
    path, stats = result
    assert stats["terrain_breakdown"].get("trail", 0) > 0.8


def test_max_slope_enforced():
    elevation = FLAT.copy()
    # A cliff: +100m from row 15 down — every N-S move across it exceeds 45°
    elevation[15:, :] += 100.0
    pf = make_pathfinder(elevation=elevation)
    pf.set_parameters(max_slope_degrees=45.0)
    assert pf.find_path(*START, *END) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ssilver/development/trail/backend && ./venv/bin/python -m pytest tests/unit/test_pathfinder_v2.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.engine_v2.pathfinder'`

- [ ] **Step 3: Implement the port**

Start from `backend3/elevation_pathfinder_terrain.py` (copy it to `backend/app/engine_v2/pathfinder.py`) and apply the six port changes listed in the Interfaces block above. The resulting key sections:

```python
# backend/app/engine_v2/pathfinder.py
# <RULE-0 HEADER>
"""
Terrain-aware A* pathfinder for engine v2.
Ported from backend3/elevation_pathfinder_terrain.py with: direct array inputs,
logging instead of prints, (path, stats) return, new PathType vocabulary, and
weighted A* via heuristic_weight.
"""
import heapq
import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.engine_v2.path_layer import PathType, get_path_type_name

logger = logging.getLogger(__name__)

DEFAULT_TERRAIN_COSTS = {
    PathType.UNKNOWN: 0.5,
    PathType.OBSTACLE: 10000.0,
    PathType.TRAIL: 0.2,
    PathType.PATH: 0.3,
    PathType.FOOTWAY: 0.6,
    PathType.RESIDENTIAL: 0.85,
    PathType.NATURAL: 0.4,
}


@dataclass
class TerrainNode:
    row: int
    col: int
    g_cost: float
    h_cost: float
    parent: Optional["TerrainNode"] = None
    elevation: float = 0
    terrain_type: int = PathType.UNKNOWN
    consecutive_steep_distance: float = 0

    @property
    def f_cost(self) -> float:
        return self.g_cost + self.h_cost

    def __lt__(self, other):
        return self.f_cost < other.f_cost


class TerrainAwarePathfinder:
    def __init__(self, elevation: np.ndarray, transform, terrain_types: np.ndarray,
                 resolution: float = 10.0, heuristic_weight: float = 1.0):
        assert elevation.shape == terrain_types.shape, \
            f"elevation {elevation.shape} != terrain {terrain_types.shape}"
        self.elevation = elevation
        self.transform = transform
        self.terrain_types = terrain_types
        self.resolution = resolution
        self.heuristic_weight = heuristic_weight
        self.rows, self.cols = elevation.shape

        # Cost parameters (same defaults as backend3 original)
        self.elevation_weight = 1.0
        self.elevation_exponent = 2.0
        self.obstacle_cost = 10000.0
        self.max_slope_degrees = 45.0
        self.terrain_costs = dict(DEFAULT_TERRAIN_COSTS)
        self.steep_threshold = 15.0
        self.fatigue_distance = 100.0
        self.fatigue_exponent = 2.0
        self.sustained_slope_weight = 0.5

    def _euclidean(self, row, col, goal_row, goal_col) -> float:
        return self.resolution * np.sqrt((row - goal_row) ** 2 + (col - goal_col) ** 2)

    def heuristic(self, row, col, goal_row, goal_col) -> float:
        return self.heuristic_weight * self._euclidean(row, col, goal_row, goal_col)
```

`set_parameters`, `lat_lon_to_grid`, `grid_to_lat_lon`, `get_neighbors`, `calculate_move_cost`, and the `find_path` A* loop are kept from the original with these deltas:

- `calculate_move_cost`: the transition-penalty condition becomes

```python
        if (terrain_from == PathType.TRAIL and
                terrain_to not in (PathType.TRAIL, PathType.PATH, PathType.NATURAL)):
            terrain_multiplier *= 1.5
```

- `find_path`: `straight_line_distance = self._euclidean(start_row, start_col, end_row, end_col)` (unweighted); all `print` → `logger.debug`; on success `return self.reconstruct_path(current, nodes_explored, time.time() - start_time)`; on failure `return None`.
- `reconstruct_path` builds the same path list, then returns stats instead of printing:

```python
    def reconstruct_path(self, end_node, nodes_explored, elapsed_s):
        path = []
        current = end_node
        total_distance = 0.0
        total_elevation_gain = 0.0
        terrain_counts: Dict[int, int] = {}

        while current:
            lat, lon = self.grid_to_lat_lon(current.row, current.col)
            path.append((lat, lon, float(current.elevation)))
            terrain_counts[current.terrain_type] = terrain_counts.get(current.terrain_type, 0) + 1
            if current.parent:
                row_diff = current.row - current.parent.row
                col_diff = current.col - current.parent.col
                total_distance += self.resolution * np.sqrt(row_diff ** 2 + col_diff ** 2)
                elev_diff = current.elevation - current.parent.elevation
                if elev_diff > 0:
                    total_elevation_gain += elev_diff
            current = current.parent

        path.reverse()
        total_points = max(sum(terrain_counts.values()), 1)
        stats = {
            "distance_m": round(total_distance, 1),
            "elevation_gain_m": round(float(total_elevation_gain), 1),
            "nodes_explored": nodes_explored,
            "time_s": round(elapsed_s, 3),
            "terrain_breakdown": {
                get_path_type_name(t): round(c / total_points, 3)
                for t, c in terrain_counts.items()
            },
        }
        return path, stats
```

Add to `__init__.py`:

```python
from app.engine_v2.pathfinder import TerrainAwarePathfinder
```

and `"TerrainAwarePathfinder"` to `__all__`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/ssilver/development/trail/backend && ./venv/bin/python -m pytest tests/unit/test_pathfinder_v2.py -v`
Expected: all PASSED. If `test_obstacle_wall_blocks_direct_route` is flaky on coordinates, check the transform row order (row 0 is the NORTH edge with `from_bounds`) before touching the assertions — the wall indices assume row 0 = north.

- [ ] **Step 5: Commit**

```bash
cd /home/ssilver/development/trail
git add backend/app/engine_v2/pathfinder.py backend/app/engine_v2/__init__.py backend/tests/unit/test_pathfinder_v2.py
git commit -m "feat(engine-v2): port terrain-aware A* pathfinder with stats output"
git log --oneline -1
```

---

### Task 6: Weighted A* behavior tests

**Files:**
- Test: `backend/tests/unit/test_pathfinder_v2.py` (append; implementation was wired in Task 5)

**Interfaces:**
- Consumes: `TerrainAwarePathfinder` with `heuristic_weight` (Task 5)
- Produces: verified guarantee later tasks rely on: `heuristic_weight=1.0` is quality-neutral; `heuristic_weight=2.0` explores fewer nodes at bounded path-length cost.

- [ ] **Step 1: Write the failing-or-passing tests** (append to `tests/unit/test_pathfinder_v2.py`)

```python
class TestWeightedAStar:
    def test_weight_one_follows_cheap_corridor(self):
        # Optimal search must take the trail corridor even though it's longer
        terrain = OPEN.copy()
        terrain[:, 5] = PathType.TRAIL          # cheap vertical corridor at col 5
        pf = make_pathfinder(terrain=terrain, heuristic_weight=1.0)
        result = pf.find_path(40.6401, -111.5095, 40.6428, -111.5095)  # along col 5
        assert result is not None
        _, stats = result
        assert stats["terrain_breakdown"].get("trail", 0) > 0.8

    def test_higher_weight_explores_fewer_nodes(self):
        elevation = FLAT.copy()
        # Bumpy terrain so the search actually has work to do
        rng_rows, rng_cols = np.meshgrid(np.arange(SHAPE[0]), np.arange(SHAPE[1]),
                                         indexing="ij")
        elevation += 3.0 * np.sin(rng_rows / 3.0) * np.cos(rng_cols / 3.0)

        pf1 = make_pathfinder(elevation=elevation, heuristic_weight=1.0)
        r1 = pf1.find_path(*START, *END)
        pf2 = make_pathfinder(elevation=elevation, heuristic_weight=2.0)
        r2 = pf2.find_path(*START, *END)

        assert r1 is not None and r2 is not None
        _, stats1 = r1
        path2, stats2 = r2
        assert stats2["nodes_explored"] < stats1["nodes_explored"]
        # Bounded quality loss (spec: ~15%; allow slack on tiny grids)
        assert stats2["distance_m"] <= stats1["distance_m"] * 1.2
```

- [ ] **Step 2: Run tests**

Run: `cd /home/ssilver/development/trail/backend && ./venv/bin/python -m pytest tests/unit/test_pathfinder_v2.py -v -k Weighted`
Expected: PASS (Task 5 wired the mechanism). If `test_higher_weight_explores_fewer_nodes` fails, the weight isn't reaching `h_cost` — verify every `TerrainNode(... h_cost=self.heuristic(...))` call site uses `heuristic()` (weighted), and `straight_line_distance` uses `_euclidean()` (unweighted). Fix the implementation, not the test.

- [ ] **Step 3: Run the full unit suite**

Run: `cd /home/ssilver/development/trail/backend && ./venv/bin/python -m pytest tests/unit/test_path_layer.py tests/unit/test_pathfinder_v2.py tests/unit/test_engine_v2_imports.py -v`
Expected: all PASSED

- [ ] **Step 4: Commit**

```bash
cd /home/ssilver/development/trail
git add backend/tests/unit/test_pathfinder_v2.py
git commit -m "test(engine-v2): verify weighted A* explores fewer nodes with bounded quality loss"
git log --oneline -1
```

---

### Task 7: TrailFinderServiceV2 adapter

**Files:**
- Create: `backend/app/engine_v2/service.py`
- Modify: `backend/app/engine_v2/__init__.py` (export `TrailFinderServiceV2`)
- Test: `backend/tests/unit/test_service_v2.py`

**Interfaces:**
- Consumes: `TwoLayerElevationLibrary`, `Bounds` (Task 1), `PathLayer`, `PathType` (Task 4), `TerrainAwarePathfinder` (Task 5), `Coordinate` from `app.models.route`
- Produces: `TrailFinderServiceV2(data_dir="dem_data_v2", cache_dir="path_cache_v2", resolution=10, buffer=0.02, max_distance_km=50.0, elevation_lib=None, path_layer=None)` (last two injectable for tests). `async find_route(start: Coordinate, end: Coordinate, options: dict) -> Tuple[List[Coordinate], dict]`; `validate_route_request(start, end) -> bool` — the exact contract `main.py` uses for v1. Failure returns `([], {"error": msg})`. Success stats include `"engine": "v2"` plus pathfinder stats; `"warnings"` list when OSM data was absent.

**Option mapping (options dict = `RouteOptions.model_dump()`):**

| API option | v2 effect |
|---|---|
| `userProfile` | preset terrain-cost table applied first (see `_PROFILE_COSTS` below), then overridden by the options after it |
| `buffer` | bounding-box padding in degrees |
| `maxSlope` | `set_parameters(max_slope_degrees=...)` |
| `gradientPreference` (g) | `set_parameters(elevation_weight=g)` |
| `trailPreference` (t) | divide TRAIL/PATH/NATURAL/UNKNOWN costs by t; multiply FOOTWAY/RESIDENTIAL by t |
| `customPathCosts.trail/path/footway/residential/off_path` | override `terrain_costs[TRAIL/PATH/FOOTWAY/RESIDENTIAL/UNKNOWN]` |
| `heuristicWeight` | pathfinder `heuristic_weight` (default 1.0) |

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/unit/test_service_v2.py
# <RULE-0 HEADER>
"""Unit tests for TrailFinderServiceV2 using injected fakes (no network)."""
import numpy as np
import pytest
from rasterio.transform import from_bounds

from app.engine_v2.path_layer import PathType
from app.engine_v2.service import TrailFinderServiceV2
from app.models.route import Coordinate

SHAPE = (30, 30)


class FakeElevationLib:
    """In-memory stand-in for TwoLayerElevationLibrary (flat terrain)."""

    def __init__(self):
        self.loaded = []

    def load_area(self, bounds):
        self.loaded.append(bounds)
        return {"total_tiles": 1}

    def get_elevation_array(self, bounds):
        transform = from_bounds(bounds.west, bounds.south, bounds.east,
                                bounds.north, SHAPE[1], SHAPE[0])
        return np.zeros(SHAPE, dtype=np.float32), {"transform": transform,
                                                   "resolution": 10}


class FakePathLayer:
    def __init__(self, fill=PathType.UNKNOWN):
        self.fill = fill

    def get_grid(self, bounds, shape, transform):
        return np.full(shape, self.fill, dtype=np.uint8)


def make_service(**kwargs):
    return TrailFinderServiceV2(elevation_lib=FakeElevationLib(),
                                path_layer=FakePathLayer(), **kwargs)


START = Coordinate(lat=40.6410, lon=-111.5090)
END = Coordinate(lat=40.6420, lon=-111.5080)


@pytest.mark.asyncio
async def test_find_route_returns_path_and_v2_stats():
    service = make_service()
    path, stats = await service.find_route(START, END, {})
    assert len(path) > 2
    assert stats["engine"] == "v2"
    assert stats["distance_m"] > 0
    assert all(isinstance(c, Coordinate) for c in path)


@pytest.mark.asyncio
async def test_identical_points_rejected():
    service = make_service()
    path, stats = await service.find_route(START, START, {})
    assert path == []
    assert "error" in stats


@pytest.mark.asyncio
async def test_too_far_apart_rejected():
    service = make_service(max_distance_km=50.0)
    far = Coordinate(lat=41.5, lon=-111.5090)  # ~95km north
    path, stats = await service.find_route(START, far, {})
    assert path == []
    assert "error" in stats


@pytest.mark.asyncio
async def test_no_osm_data_adds_warning():
    service = make_service()  # FakePathLayer returns all-UNKNOWN grid
    _, stats = await service.find_route(START, END, {})
    assert any("OSM" in w for w in stats.get("warnings", []))


@pytest.mark.asyncio
async def test_custom_path_costs_applied():
    captured = {}

    class SpyService(TrailFinderServiceV2):
        def _make_pathfinder(self, *args, **kwargs):
            pf = super()._make_pathfinder(*args, **kwargs)
            captured["costs"] = dict(pf.terrain_costs)
            captured["weight"] = pf.heuristic_weight
            return pf

    service = SpyService(elevation_lib=FakeElevationLib(), path_layer=FakePathLayer())
    options = {"customPathCosts": {"trail": 0.11, "off_path": 1.4},
               "heuristicWeight": 1.7}
    await service.find_route(START, END, options)
    assert captured["costs"][PathType.TRAIL] == pytest.approx(0.11)
    assert captured["costs"][PathType.UNKNOWN] == pytest.approx(1.4)
    assert captured["weight"] == pytest.approx(1.7)


@pytest.mark.asyncio
async def test_user_profile_preset_applied():
    captured = {}

    class SpyService(TrailFinderServiceV2):
        def _make_pathfinder(self, *args, **kwargs):
            pf = super()._make_pathfinder(*args, **kwargs)
            captured["costs"] = dict(pf.terrain_costs)
            return pf

    service = SpyService(elevation_lib=FakeElevationLib(), path_layer=FakePathLayer())
    await service.find_route(START, END, {"userProfile": "experienced"})
    assert captured["costs"][PathType.TRAIL] == pytest.approx(0.15)
    assert captured["costs"][PathType.UNKNOWN] == pytest.approx(0.4)
```

Note: if `pytest-asyncio` is not installed in `backend/venv` (check with `./venv/bin/python -m pytest --co tests/integration/test_api_routes.py -q` — the existing suite will reveal the convention), follow whatever pattern the existing async tests use (e.g. `asyncio.get_event_loop().run_until_complete(...)` helpers) instead of `@pytest.mark.asyncio`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ssilver/development/trail/backend && ./venv/bin/python -m pytest tests/unit/test_service_v2.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.engine_v2.service'`

- [ ] **Step 3: Implement the service**

```python
# backend/app/engine_v2/service.py
# <RULE-0 HEADER>
"""TrailFinderServiceV2: v1-compatible find_route contract over the v2 engine."""
import asyncio
import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from rasterio.transform import from_bounds

from app.engine_v2.elevation import Bounds, TwoLayerElevationLibrary
from app.engine_v2.path_layer import PathLayer, PathType
from app.engine_v2.pathfinder import TerrainAwarePathfinder
from app.models.route import Coordinate

logger = logging.getLogger(__name__)


class TrailFinderServiceV2:
    def __init__(self, data_dir: str = "dem_data_v2", cache_dir: str = "path_cache_v2",
                 resolution: int = 10, buffer: float = 0.02,
                 max_distance_km: float = 50.0,
                 elevation_lib=None, path_layer=None):
        self.buffer = buffer
        self.resolution = resolution
        self.max_distance_km = max_distance_km
        self.elevation_lib = elevation_lib or TwoLayerElevationLibrary(
            data_dir=data_dir, resolution=resolution)
        self.path_layer = path_layer or PathLayer(cache_dir=cache_dir)

    # --- validation (same semantics as v1) -------------------------------
    @staticmethod
    def _haversine_km(a: Coordinate, b: Coordinate) -> float:
        r = 6371.0
        dlat = math.radians(b.lat - a.lat)
        dlon = math.radians(b.lon - a.lon)
        h = (math.sin(dlat / 2) ** 2 +
             math.cos(math.radians(a.lat)) * math.cos(math.radians(b.lat)) *
             math.sin(dlon / 2) ** 2)
        return 2 * r * math.asin(math.sqrt(h))

    def validate_route_request(self, start: Coordinate, end: Coordinate) -> bool:
        if start.lat == end.lat and start.lon == end.lon:
            return False
        return self._haversine_km(start, end) <= self.max_distance_km

    def calculate_bounding_box(self, start: Coordinate, end: Coordinate,
                               buffer: Optional[float] = None) -> Bounds:
        b = self.buffer if buffer is None else buffer
        return Bounds(south=min(start.lat, end.lat) - b,
                      north=max(start.lat, end.lat) + b,
                      west=min(start.lon, end.lon) - b,
                      east=max(start.lon, end.lon) + b)

    # --- pathfinder construction (seam for tests) -------------------------
    # Profile presets translated from v1's PathPreferencePresets
    # (urban_walker / trail_seeker / flexible_hiker path_costs tables).
    _PROFILE_COSTS = {
        "easy": {          # urban_walker: prefer built paths, avoid off-path
            PathType.TRAIL: 0.25, PathType.PATH: 0.3, PathType.FOOTWAY: 0.4,
            PathType.RESIDENTIAL: 0.7, PathType.NATURAL: 0.3, PathType.UNKNOWN: 1.5},
        "experienced": {   # trail_seeker: strong natural preference
            PathType.TRAIL: 0.15, PathType.PATH: 0.2, PathType.FOOTWAY: 0.7,
            PathType.RESIDENTIAL: 0.9, PathType.NATURAL: 0.3, PathType.UNKNOWN: 0.4},
        "trail_runner": {  # flexible_hiker: mild preference, off-path ok
            PathType.TRAIL: 0.5, PathType.PATH: 0.6, PathType.FOOTWAY: 0.6,
            PathType.RESIDENTIAL: 0.8, PathType.NATURAL: 0.8, PathType.UNKNOWN: 1.0},
        "accessibility": { # urban_walker, stricter off-path avoidance
            PathType.TRAIL: 0.6, PathType.PATH: 0.4, PathType.FOOTWAY: 0.2,
            PathType.RESIDENTIAL: 0.6, PathType.NATURAL: 0.8, PathType.UNKNOWN: 3.0},
    }  # "default" and unknown profiles keep DEFAULT_TERRAIN_COSTS

    def _make_pathfinder(self, elevation, transform, terrain_grid,
                         options: dict) -> TerrainAwarePathfinder:
        pf = TerrainAwarePathfinder(
            elevation, transform, terrain_grid,
            resolution=self.resolution,
            heuristic_weight=options.get("heuristicWeight") or 1.0)

        profile_costs = self._PROFILE_COSTS.get(options.get("userProfile"))
        if profile_costs:
            pf.terrain_costs.update(profile_costs)

        if options.get("maxSlope") is not None:
            pf.set_parameters(max_slope_degrees=options["maxSlope"])
        if options.get("gradientPreference") is not None:
            pf.set_parameters(elevation_weight=options["gradientPreference"])

        t = options.get("trailPreference") or 1.0
        if t != 1.0:
            for pt in (PathType.TRAIL, PathType.PATH, PathType.NATURAL, PathType.UNKNOWN):
                pf.terrain_costs[pt] /= t
            for pt in (PathType.FOOTWAY, PathType.RESIDENTIAL):
                pf.terrain_costs[pt] *= t

        custom = options.get("customPathCosts") or {}
        mapping = {"trail": PathType.TRAIL, "path": PathType.PATH,
                   "footway": PathType.FOOTWAY, "residential": PathType.RESIDENTIAL,
                   "off_path": PathType.UNKNOWN}
        for key, ptype in mapping.items():
            if custom.get(key) is not None:
                pf.terrain_costs[ptype] = custom[key]
        return pf

    # --- main entry --------------------------------------------------------
    async def find_route(self, start: Coordinate, end: Coordinate,
                         options: dict) -> Tuple[List[Coordinate], dict]:
        options = options or {}
        if not self.validate_route_request(start, end):
            return [], {"error": "Invalid route request: coordinates too far apart or identical"}
        try:
            return await asyncio.to_thread(self._find_route_sync, start, end, options)
        except Exception as e:
            logger.exception("v2 route failed")
            return [], {"error": f"v2 engine error: {e}"}

    def _find_route_sync(self, start, end, options):
        warnings: List[str] = []
        bounds = self.calculate_bounding_box(start, end, options.get("buffer"))

        self.elevation_lib.load_area(bounds)
        elevation, meta = self.elevation_lib.get_elevation_array(bounds)
        transform = meta.get("transform") if isinstance(meta, dict) else None
        if transform is None:
            transform = from_bounds(bounds.west, bounds.south, bounds.east,
                                    bounds.north, elevation.shape[1], elevation.shape[0])
        if hasattr(self.elevation_lib, "close_all"):
            self.elevation_lib.close_all()

        terrain_grid = self.path_layer.get_grid(bounds, elevation.shape, transform)
        if not (terrain_grid != PathType.UNKNOWN).any():
            warnings.append("OSM data unavailable — terrain-only routing")

        pf = self._make_pathfinder(elevation, transform, terrain_grid, options)
        result = pf.find_path(start.lat, start.lon, end.lat, end.lon)
        if result is None:
            stats = {"error": "No route found", "engine": "v2"}
            if warnings:
                stats["warnings"] = warnings
            return [], stats

        raw_path, stats = result
        stats["engine"] = "v2"
        if warnings:
            stats["warnings"] = warnings
        path = [Coordinate(lat=lat, lon=lon) for (lat, lon, _elev) in raw_path]
        return path, stats
```

Add to `__init__.py`:

```python
from app.engine_v2.service import TrailFinderServiceV2
```

and `"TrailFinderServiceV2"` to `__all__`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/ssilver/development/trail/backend && ./venv/bin/python -m pytest tests/unit/test_service_v2.py -v`
Expected: all PASSED

- [ ] **Step 5: Commit**

```bash
cd /home/ssilver/development/trail
git add backend/app/engine_v2/service.py backend/app/engine_v2/__init__.py backend/tests/unit/test_service_v2.py
git commit -m "feat(engine-v2): TrailFinderServiceV2 adapter with v1-compatible contract"
git log --oneline -1
```

---

### Task 8: API wiring — engine flag + dispatch

**Files:**
- Modify: `backend/app/models/route.py` (RouteOptions, after the `trailPreference` field around line 75)
- Modify: `backend/app/main.py` (service instantiation near line 60; dispatch inside `process_route` around lines 236-262)
- Test: `backend/tests/integration/test_engine_v2_api.py`

**Interfaces:**
- Consumes: `TrailFinderServiceV2` (Task 7)
- Produces: `RouteOptions.engine: str` (`"v1"`|`"v2"`, default `"v1"`), `RouteOptions.heuristicWeight: Optional[float]`; `main.py` module-level `trail_finder_v2` instance; `process_route` dispatches on `request.options.engine`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/integration/test_engine_v2_api.py
# <RULE-0 HEADER>
"""Integration tests: engine flag reaches the dispatcher; v1 default unchanged."""
from fastapi.testclient import TestClient

from app.main import app
from app.models.route import RouteOptions


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
    resp = client.post("/api/routes/calculate", json={
        "start": {"lat": 40.6572, "lon": -111.5706},
        "end": {"lat": 40.6486, "lon": -111.5639},
        "options": {"engine": "v2"},
    })
    assert resp.status_code == 202
    route_id = resp.json()["routeId"]
    # Background task ran during the request (TestClient behavior);
    # v2 either completed or failed with a v2-shaped error — never a crash.
    status = client.get(f"/api/routes/{route_id}/status")
    assert status.status_code == 200
    assert status.json()["status"] in ("completed", "failed", "processing")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ssilver/development/trail/backend && ./venv/bin/python -m pytest tests/integration/test_engine_v2_api.py -v`
Expected: `test_route_options_engine_defaults_to_v1` FAILS with `AttributeError: 'RouteOptions' object has no attribute 'engine'`

- [ ] **Step 3: Add the model fields**

In `backend/app/models/route.py`, inside `RouteOptions` directly after the `trailPreference` field:

```python
    engine: str = Field(
        "v1",
        pattern="^(v1|v2)$",
        description="Routing engine: v1 (legacy DEMTileCache) or v2 (two-layer + terrain A*)"
    )
    heuristicWeight: Optional[float] = Field(
        None,
        ge=1.0,
        le=3.0,
        description="v2 only: A* heuristic weight. 1.0 = optimal, higher = faster/greedier"
    )
```

(If the installed pydantic is v1.x, `pattern=` raises at import — use `regex=` instead. Check: `./venv/bin/python -c "import pydantic; print(pydantic.VERSION)"`. `main.py` already calls `.model_dump()`, so v2 is expected.)

- [ ] **Step 4: Wire the dispatch in main.py**

Near line 60, after the v1 service instances:

```python
from app.engine_v2.service import TrailFinderServiceV2

trail_finder_v2 = TrailFinderServiceV2()
```

In `process_route`, wrap the existing v1 block. The current code (lines ~236-262) creates `profile_trail_finder`, validates, then calls `find_route`. Change it to:

```python
        engine = request.options.engine if request.options else "v1"

        if engine == "v2":
            routes_storage[route_id]["progress"] = 30
            path, stats = await trail_finder_v2.find_route(
                request.start,
                request.end,
                request.options.model_dump() if request.options else {}
            )
        else:
            # --- existing v1 block, unchanged ---
            obstacle_config, path_preferences = get_configs_for_profile(profile, request.options)
            profile_trail_finder = TrailFinderService(
                obstacle_config=obstacle_config,
                path_preferences=path_preferences,
                dem_cache=shared_dem_cache
            )
            if not profile_trail_finder.validate_route_request(request.start, request.end):
                routes_storage[route_id]["status"] = RouteStatus.FAILED
                routes_storage[route_id]["message"] = "Invalid route request"
                return
            routes_storage[route_id]["progress"] = 30
            path, stats = await profile_trail_finder.find_route(
                request.start,
                request.end,
                request.options.model_dump() if request.options else {}
            )
```

Keep the rest of `process_route` (progress 90, success/failure handling) shared — it already works off `path`/`stats`. Note `get_configs_for_profile` stays inside the v1 branch (v2 does its own option mapping); the `profile` variable read stays where it is if other code uses it.

- [ ] **Step 5: Run the new tests + v1 regression check**

Run: `cd /home/ssilver/development/trail/backend && ./venv/bin/python -m pytest tests/integration/test_engine_v2_api.py -v`
Expected: all PASSED

Run v1 regression (existing integration suite):
`cd /home/ssilver/development/trail/backend && ./venv/bin/python -m pytest tests/integration/test_api_routes.py -v`
Expected: same results as before the change (run it on `git stash` baseline first if unsure). v1 must be unaffected.

- [ ] **Step 6: Commit**

```bash
cd /home/ssilver/development/trail
git add backend/app/models/route.py backend/app/main.py backend/tests/integration/test_engine_v2_api.py
git commit -m "feat(engine-v2): engine selection flag in API with v2 dispatch"
git log --oneline -1
```

---

### Task 9: Parity harness (success criterion)

**Files:**
- Create: `backend/tests/e2e/test_engine_parity.py`

**Interfaces:**
- Consumes: the full v2 stack via the API (Task 8), v2 `PathLayer` + elevation lib for measurement
- Produces: the project's pass/fail verdict. Pass bars (from spec): v2 succeeds on all routes; v2 trail-adherence ≥ v1's per route; v2 wall-clock ≤ 1.25× v1 per route; no single-step elevation jump > 20m on v2 paths.

**Note:** This test needs elevation + OSM data (network or the existing caches). Mark it `@pytest.mark.e2e` matching the existing e2e convention (check `pytest.ini` / existing e2e files for the exact marker; if none, no marker). The three routes are taken from `backend/tests/e2e/test_real_routes.py` (Park City area — data already cached).

- [ ] **Step 1: Write the harness**

```python
# backend/tests/e2e/test_engine_parity.py
# <RULE-0 HEADER>
"""
Engine parity harness — THE success criterion for the v2 integration.
Runs known real routes through both engines via the API and compares:
success, trail adherence, wall-clock, and elevation continuity.
"""
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from rasterio.transform import from_bounds

from app.engine_v2.elevation import Bounds
from app.engine_v2.path_layer import PathLayer, PathType
from app.main import app

# Known Park City routes from tests/e2e/test_real_routes.py (data cached)
ROUTES = [
    {"name": "park_city_medium", "start": (40.6572, -111.5706), "end": (40.6486, -111.5639)},
    {"name": "park_city_long", "start": (40.6650, -111.5850), "end": (40.6550, -111.5750)},
    {"name": "park_city_short", "start": (40.6482, -111.5738), "end": (40.6464, -111.5729)},
]

ON_PATH_TYPES = {PathType.TRAIL, PathType.PATH, PathType.FOOTWAY}


def _run_route(client, route, engine):
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
    result = client.get(f"/api/routes/{route_id}")
    data = result.json()
    return {"status": data.get("status"), "path": data.get("path", []),
            "stats": data.get("stats", {}), "time_s": elapsed}


def _measurement_grid(route, shape=(400, 400), buffer=0.02):
    lats = [route["start"][0], route["end"][0]]
    lons = [route["start"][1], route["end"][1]]
    bounds = Bounds(south=min(lats) - buffer, north=max(lats) + buffer,
                    west=min(lons) - buffer, east=max(lons) + buffer)
    transform = from_bounds(bounds.west, bounds.south, bounds.east, bounds.north,
                            shape[1], shape[0])
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


@pytest.mark.parametrize("route", ROUTES, ids=[r["name"] for r in ROUTES])
def test_engine_parity(route):
    client = TestClient(app)

    v1 = _run_route(client, route, "v1")
    v2 = _run_route(client, route, "v2")

    grid, transform = _measurement_grid(route)
    adh1 = _trail_adherence(v1["path"], grid, transform)
    adh2 = _trail_adherence(v2["path"], grid, transform)

    print(f"\n=== {route['name']} ===")
    print(f"  v1: status={v1['status']} points={len(v1['path'])} "
          f"adherence={adh1:.1%} time={v1['time_s']:.1f}s")
    print(f"  v2: status={v2['status']} points={len(v2['path'])} "
          f"adherence={adh2:.1%} time={v2['time_s']:.1f}s "
          f"dist={v2['stats'].get('distance_m')}m")

    # Bar 1: v2 succeeds on every route
    assert v2["status"] == "completed", f"v2 failed: {v2['stats']}"

    # Bar 2: trail adherence not worse than v1 (only when v1 also succeeded)
    if v1["status"] == "completed":
        assert adh2 >= adh1 - 1e-9, f"v2 adherence {adh2:.1%} < v1 {adh1:.1%}"
        # Bar 3: speed within 1.25x of v1
        assert v2["time_s"] <= v1["time_s"] * 1.25 + 1.0, \
            f"v2 {v2['time_s']:.1f}s vs v1 {v1['time_s']:.1f}s"

    # Bar 4: no tile-artifact elevation jumps along the v2 path
    gains = v2["stats"].get("elevation_gain_m")
    assert gains is not None
    elevs = [p.get("elevation") for p in v2["path"] if p.get("elevation") is not None]
    if len(elevs) > 1:
        steps = np.abs(np.diff(elevs))
        assert steps.max() <= 20.0, f"elevation discontinuity: {steps.max():.1f}m"
```

Implementation notes for the executor:
- If v1's `RouteResult.path` Coordinates carry no `elevation` key, the Bar 4 elevation check should sample elevations for v2's path points via the v2 elevation library (`lib.get_elevation(lat, lon)`) after a `load_area` for the measurement bounds — adjust accordingly rather than skipping the bar.
- The `+ 1.0` second slack on Bar 3 absorbs cold-cache noise on short routes; do not raise it further without user sign-off.
- First run downloads OSM/elevation data for any uncached areas — expect minutes, not seconds.

- [ ] **Step 2: Run the harness**

Run: `cd /home/ssilver/development/trail/backend && ./venv/bin/python -m pytest tests/e2e/test_engine_parity.py -v -s`
Expected: 3 PASSED with a printed comparison table per route. **If a bar fails, that is the project's real finding — investigate and fix the engine (cost tuning, path_layer coverage), never loosen the bar without user sign-off.**

- [ ] **Step 3: Run the complete new-code test suite**

Run: `cd /home/ssilver/development/trail/backend && ./venv/bin/python -m pytest tests/unit/test_path_layer.py tests/unit/test_pathfinder_v2.py tests/unit/test_service_v2.py tests/unit/test_engine_v2_imports.py tests/integration/test_engine_v2_api.py -v`
Expected: all PASSED

- [ ] **Step 4: Commit**

```bash
cd /home/ssilver/development/trail
git add backend/tests/e2e/test_engine_parity.py
git commit -m "test(engine-v2): parity harness comparing v1/v2 on real Park City routes"
git log --oneline -1
```

---

## Final verification (after all tasks)

- [ ] Run all engine-v2 tests (Task 9 Step 3 command) — all green.
- [ ] Run v1 regression: `./venv/bin/python -m pytest tests/integration/test_api_routes.py -v` — unchanged from baseline.
- [ ] Security audit per user's global CLAUDE.md: `pip-audit` on backend venv (no new deps expected — flag any) + codex read-only review of every file this plan created/modified, focusing on: the OSM fetch inputs (bounds/tags), `asyncio.to_thread` concurrency on shared service state, path traversal in `PathLayer._cache_path`, and untrusted `options` dict handling in `service.py`.
- [ ] Report parity table + audit findings to the user. Do NOT push.
