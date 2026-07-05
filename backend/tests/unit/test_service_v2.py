# RULE #0: MANDATORY FIRST ACTION FOR EVERY REQUEST
# 1. Read CLAUDE.md COMPLETELY before responding
# 2. Setup Python venv: [ -d "venv" ] || ./setup-venv.sh && source venv/bin/activate
# 3. Search for rules related to the request
# 4. Only proceed after confirming no violations
# Failure to follow Rule #0 has caused real harm. Check BEFORE acting, not AFTER making mistakes.
#
# GUARDS ARE SAFETY EQUIPMENT - WHEN THEY FIRE, FIX THE PROBLEM THEY FOUND
# NEVER weaken, disable, or bypass guards - they prevent real harm
"""Unit tests for TrailFinderServiceV2 using injected fakes (no network)."""
import numpy as np
import pytest
from rasterio.transform import from_bounds

from app.engine_v2.elevation import Bounds
from app.engine_v2.elevation_fd_safe import FDManagedElevationLibrary
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
        transform = from_bounds(bounds.west, bounds.south, bounds.east, bounds.north, SHAPE[1], SHAPE[0])
        return np.zeros(SHAPE, dtype=np.float32), {"transform": transform, "resolution": 10}


class FakePathLayer:
    def __init__(self, fill=PathType.UNKNOWN):
        self.fill = fill

    def get_grid(self, bounds, shape, transform, refresh=False, missing_out=None):
        return np.full(shape, self.fill, dtype=np.uint8)


def make_service(**kwargs):
    return TrailFinderServiceV2(elevation_lib=FakeElevationLib(), path_layer=FakePathLayer(), **kwargs)


START = Coordinate(lat=40.6410, lon=-111.5090)
END = Coordinate(lat=40.6420, lon=-111.5080)


@pytest.mark.asyncio
async def test_find_route_returns_path_and_v2_stats():
    service = make_service()
    # NOTE (deviation from brief): default buffer=0.02 against the fixed
    # 30x30 FakeElevationLib grid makes START/END (~110m apart) fall into
    # diagonally-adjacent cells, so the shortest path is mathematically
    # exactly 2 points on flat terrain -- `len(path) > 2` can never hold.
    # Passing a tighter buffer here (exercising the documented `buffer`
    # option) gives the grid enough resolution between START/END to
    # produce a genuine multi-hop path, matching the test's intent.
    path, stats = await service.find_route(START, END, {"buffer": 0.001})
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
    options = {"customPathCosts": {"trail": 0.11, "off_path": 1.4}, "heuristicWeight": 1.7}
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


def test_default_elevation_lib_is_fd_managed(tmp_path):
    """Default wiring must bound open file descriptors via the managed wrapper."""
    (tmp_path / "dem").mkdir()  # TwoLayerElevationLibrary requires an existing dir
    service = TrailFinderServiceV2(data_dir=str(tmp_path / "dem"), cache_dir=str(tmp_path / "cache"))
    assert isinstance(service.elevation_lib, FDManagedElevationLibrary)


class StubBase:
    """Stand-in for TwoLayerElevationLibrary behind the FD-managed wrapper."""

    def __init__(self):
        self.loaded = []
        self.close_all_called = False

    def load_area(self, bounds):
        self.loaded.append(bounds)
        return {"total_tiles": 1}

    def get_elevation_array(self, bounds):
        transform = from_bounds(bounds.west, bounds.south, bounds.east, bounds.north, SHAPE[1], SHAPE[0])
        return np.zeros(SHAPE, dtype=np.float32), {"transform": transform, "resolution": 10}

    def close_all(self):
        self.close_all_called = True


def test_fd_wrapper_delegates_get_elevation_array_and_close_all():
    """Wrapper must expose get_elevation_array (delegation) and its close_all
    must reach the base library's dataset cache."""
    stub = StubBase()
    wrapper = FDManagedElevationLibrary(stub)
    bounds = Bounds(south=40.63, north=40.66, west=-111.53, east=-111.49)

    data, meta = wrapper.get_elevation_array(bounds)
    assert data.shape == SHAPE
    assert "transform" in meta

    wrapper.close_all()
    assert stub.close_all_called


@pytest.mark.asyncio
async def test_find_route_through_fd_wrapper():
    """Regression test: default wiring routes through FDManagedElevationLibrary;
    find_route must work end-to-end through the wrapper (no AttributeError)."""
    service = TrailFinderServiceV2(elevation_lib=FDManagedElevationLibrary(StubBase()), path_layer=FakePathLayer())
    path, stats = await service.find_route(START, END, {"buffer": 0.001})
    assert len(path) > 0
    assert stats["engine"] == "v2"


def test_cache_dirs_default_when_no_env_or_arg(monkeypatch):
    """No constructor arg, no env var -> falls back to legacy CWD-relative defaults."""
    monkeypatch.delenv("TRAIL_V2_DEM_DIR", raising=False)
    monkeypatch.delenv("TRAIL_V2_PATH_CACHE_DIR", raising=False)
    service = TrailFinderServiceV2(elevation_lib=FakeElevationLib(), path_layer=FakePathLayer())
    assert service._data_dir == "dem_data_v2"
    assert service._cache_dir == "path_cache_v2"


def test_cache_dirs_follow_env_vars(monkeypatch):
    """Env vars override the legacy defaults when no explicit constructor arg is given."""
    monkeypatch.setenv("TRAIL_V2_DEM_DIR", "/shared/dem_data_v2")
    monkeypatch.setenv("TRAIL_V2_PATH_CACHE_DIR", "/shared/path_cache_v2")
    service = TrailFinderServiceV2(elevation_lib=FakeElevationLib(), path_layer=FakePathLayer())
    assert service._data_dir == "/shared/dem_data_v2"
    assert service._cache_dir == "/shared/path_cache_v2"


def test_explicit_cache_dir_args_beat_env_vars(monkeypatch):
    """Explicit constructor args take precedence over env vars."""
    monkeypatch.setenv("TRAIL_V2_DEM_DIR", "/shared/dem_data_v2")
    monkeypatch.setenv("TRAIL_V2_PATH_CACHE_DIR", "/shared/path_cache_v2")
    service = TrailFinderServiceV2(
        data_dir="/explicit/dem",
        cache_dir="/explicit/cache",
        elevation_lib=FakeElevationLib(),
        path_layer=FakePathLayer(),
    )
    assert service._data_dir == "/explicit/dem"
    assert service._cache_dir == "/explicit/cache"


@pytest.mark.asyncio
async def test_coefficient_dict_transform_converted():
    """TwoLayerElevationLibrary's metadata carries the transform as an
    {a,b,c,d,e,f} dict -- the service must convert it to a real Affine."""

    class DictTransformBase(StubBase):
        def get_elevation_array(self, bounds):
            data, meta = super().get_elevation_array(bounds)
            t = meta["transform"]
            meta["transform"] = {"a": t.a, "b": t.b, "c": t.c, "d": t.d, "e": t.e, "f": t.f}
            return data, meta

    service = TrailFinderServiceV2(
        elevation_lib=FDManagedElevationLibrary(DictTransformBase()), path_layer=FakePathLayer()
    )
    path, stats = await service.find_route(START, END, {"buffer": 0.001})
    assert len(path) > 0
    assert stats["engine"] == "v2"
    assert stats["distance_m"] > 0
