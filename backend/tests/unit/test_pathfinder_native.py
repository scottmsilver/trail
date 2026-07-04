# RULE #0: MANDATORY FIRST ACTION FOR EVERY REQUEST
# GUARDS ARE SAFETY EQUIPMENT - WHEN THEY FIRE, FIX THE PROBLEM THEY FOUND
"""The native C A* kernel must be byte-identical to the pure-Python pathfinder.

If gcc is unavailable the whole module is skipped (native is optional and the
engine falls back to pure Python). When it is available, every scenario must
produce the same path points and the same distance / elevation-gain /
nodes-explored as the pure-Python implementation.
"""
import numpy as np
import pytest
from rasterio.transform import from_bounds

from app.engine_v2 import pathfinder_native as native
from app.engine_v2.path_layer import PathType
from app.engine_v2.pathfinder import TerrainAwarePathfinder

pytestmark = pytest.mark.skipif(not native.available(), reason="native kernel (gcc) unavailable")

SHAPE = (30, 30)
TRANSFORM = from_bounds(-111.51, 40.64, -111.507, 40.643, SHAPE[1], SHAPE[0])
OPEN = np.full(SHAPE, PathType.UNKNOWN, dtype=np.uint8)
START = (40.6402, -111.5098)
END = (40.6428, -111.5072)


def _elev_ramp():
    # Deterministic, non-trivial elevation so slope/steep/fatigue branches run.
    r = np.arange(SHAPE[0])[:, None]
    c = np.arange(SHAPE[1])[None, :]
    return (0.7 * r + 0.4 * c + 3.0 * np.sin(0.5 * r) * np.cos(0.3 * c)).astype(np.float32)


def _py(pf, monkeypatch):
    monkeypatch.setenv("TRAIL_V2_DISABLE_NATIVE", "1")
    return pf.find_path(*START, *END)


def _assert_same(py_res, nat_res):
    if py_res is None or nat_res is None:
        assert py_res is None and nat_res is None
        return
    (pp, ps), (npth, ns) = py_res, nat_res
    assert len(pp) == len(npth)
    for a, b in zip(pp, npth):
        assert round(a[0], 9) == round(b[0], 9)
        assert round(a[1], 9) == round(b[1], 9)
        assert round(a[2], 4) == round(b[2], 4)
    assert ps["distance_m"] == ns["distance_m"]
    assert ps["elevation_gain_m"] == ns["elevation_gain_m"]
    assert ps["nodes_explored"] == ns["nodes_explored"]


def _make(elevation, terrain, **kwargs):
    return TerrainAwarePathfinder(elevation.copy(), TRANSFORM, terrain.copy(), resolution=10.0, **kwargs)


def _run(elevation, terrain, monkeypatch, **kwargs):
    py = _py(_make(elevation, terrain, **kwargs), monkeypatch)
    nat = native.find_path_native(_make(elevation, terrain, **kwargs), *START, *END)
    _assert_same(py, nat)


def test_flat_open(monkeypatch):
    _run(np.zeros(SHAPE, dtype=np.float32), OPEN, monkeypatch)


def test_elevation_ramp(monkeypatch):
    _run(_elev_ramp(), OPEN, monkeypatch)


def test_obstacle_wall_detour(monkeypatch):
    terrain = OPEN.copy()
    terrain[10:20, :25] = PathType.OBSTACLE
    _run(_elev_ramp(), terrain, monkeypatch)


def test_no_path(monkeypatch):
    terrain = OPEN.copy()
    terrain[15, :] = PathType.OBSTACLE
    _run(np.zeros(SHAPE, dtype=np.float32), terrain, monkeypatch)


def test_mixed_terrain(monkeypatch):
    terrain = OPEN.copy()
    terrain[:, 15] = PathType.TRAIL
    terrain[5, :] = PathType.FOOTWAY
    terrain[:5, :5] = PathType.RESIDENTIAL
    terrain[20:, 20:] = PathType.PATH
    _run(_elev_ramp(), terrain, monkeypatch)


def test_weighted_heuristic(monkeypatch):
    _run(_elev_ramp(), OPEN, monkeypatch, heuristic_weight=1.7)


def test_steep_elevation(monkeypatch):
    # Large vertical relief to trigger max-slope rejection + fatigue penalties.
    r = np.arange(SHAPE[0])[:, None]
    elev = (r * 8.0).astype(np.float32) * np.ones(SHAPE, dtype=np.float32)
    _run(elev, OPEN, monkeypatch)
