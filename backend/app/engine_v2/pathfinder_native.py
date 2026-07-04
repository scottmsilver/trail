# RULE #0: MANDATORY FIRST ACTION FOR EVERY REQUEST
# 1. Read CLAUDE.md COMPLETELY before responding
# GUARDS ARE SAFETY EQUIPMENT - WHEN THEY FIRE, FIX THE PROBLEM THEY FOUND
"""Optional native (C) A* kernel for the v2 terrain-aware pathfinder.

The C source (`_astar_kernel.c`) is a faithful port of
TerrainAwarePathfinder.find_path: identical cost math and a byte-for-byte port
of CPython's heapq, so it produces the *same* path as the pure-Python engine.
It is compiled on demand with gcc (no build step, no new pip dependency) and
loaded via ctypes. If gcc is unavailable or compilation fails, callers fall
back to the pure-Python implementation.

Compile flags deliberately avoid -ffast-math and force -ffp-contract=off so
float results are bit-identical to Python (no FMA contraction, no unsafe
reassociation).
"""
import ctypes
import hashlib
import logging
import math
import os
import subprocess
import threading

import numpy as np

from app.engine_v2.path_layer import get_path_type_name

logger = logging.getLogger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "_astar_kernel.c")
_LOCK = threading.Lock()
_LIB = None
_LOAD_FAILED = False

# Highest PathType int we build a cost lookup array for.
_NCOST = 16


def _build_dir() -> str:
    d = os.environ.get("TRAIL_V2_NATIVE_BUILD_DIR", os.path.join(_HERE, "_native_build"))
    os.makedirs(d, exist_ok=True)
    return d


def _compile() -> str:
    """Compile _astar_kernel.c to a cached .so keyed by source hash. Returns path."""
    with open(_SRC, "rb") as f:
        src = f.read()
    tag = hashlib.sha256(src).hexdigest()[:16]
    so_path = os.path.join(_build_dir(), f"astar_{tag}.so")
    if os.path.exists(so_path):
        return so_path
    cc = os.environ.get("CC", "gcc")
    cmd = [
        cc,
        "-O3",
        "-ffp-contract=off",
        "-fPIC",
        "-shared",
        "-o",
        so_path,
        _SRC,
        "-lm",
    ]
    logger.info("Compiling native A* kernel: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, capture_output=True)  # nosec B603 - fixed argv, no shell
    return so_path


def _load():
    global _LIB, _LOAD_FAILED
    if _LIB is not None or _LOAD_FAILED:
        return _LIB
    with _LOCK:
        if _LIB is not None or _LOAD_FAILED:
            return _LIB
        try:
            lib = ctypes.CDLL(_compile())
            lib.astar.restype = ctypes.c_int
            lib.astar.argtypes = [
                ctypes.POINTER(ctypes.c_double),  # elevation
                ctypes.POINTER(ctypes.c_int),  # terrain
                ctypes.c_int,
                ctypes.c_int,  # rows, cols
                ctypes.c_int,
                ctypes.c_int,  # start_idx, end_idx
                ctypes.c_double,
                ctypes.c_double,  # resolution, heuristic_weight
                ctypes.c_double,
                ctypes.c_double,  # elevation_weight, elevation_exponent
                ctypes.c_double,
                ctypes.c_double,  # max_slope, steep_threshold
                ctypes.c_double,
                ctypes.c_double,  # fatigue_distance, fatigue_exponent
                ctypes.c_double,  # sustained_slope_weight
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_int,  # terrain_cost, ncost
                ctypes.POINTER(ctypes.c_int),  # out_path
                ctypes.POINTER(ctypes.c_int),  # out_nodes_explored
            ]
            _LIB = lib
        except Exception as e:  # gcc missing, compile error, load error
            logger.warning("Native A* kernel unavailable (%s); using pure Python", e)
            _LOAD_FAILED = True
    return _LIB


def available() -> bool:
    return _load() is not None


def _cost_array(terrain_costs: dict) -> np.ndarray:
    """Build cost[t] with NaN for 'not present' (C uses 1.0 in that case)."""
    arr = np.full(_NCOST, np.nan, dtype=np.float64)
    for k, v in terrain_costs.items():
        ik = int(k)
        if 0 <= ik < _NCOST:
            arr[ik] = float(v)
    return arr


def find_path_native(pf, start_lat, start_lon, end_lat, end_lon):
    """Native equivalent of TerrainAwarePathfinder.find_path. Returns (path, stats)
    or None (no path / out of bounds). Raises if the kernel is unavailable."""
    lib = _load()
    if lib is None:
        raise RuntimeError("native kernel unavailable")

    start_row, start_col = pf.lat_lon_to_grid(start_lat, start_lon)
    end_row, end_col = pf.lat_lon_to_grid(end_lat, end_lon)
    rows, cols = pf.rows, pf.cols
    if not (0 <= start_row < rows and 0 <= start_col < cols):
        return None
    if not (0 <= end_row < rows and 0 <= end_col < cols):
        return None
    # Keep flat indices + node-pool counter within int range in the C kernel.
    # Real DEM tiles are far below this; oversized grids raise so find_path
    # falls back to pure Python rather than risking overflow.
    if rows * cols > (2**31 - 1) // 8:
        raise RuntimeError(f"grid too large for native kernel: {rows}x{cols}")

    elevation = np.ascontiguousarray(pf.elevation, dtype=np.float64)
    terrain = np.ascontiguousarray(pf.terrain_types, dtype=np.int32)
    cost = _cost_array(pf.terrain_costs)
    out_path = np.empty(rows * cols, dtype=np.int32)
    nodes_explored = ctypes.c_int(0)

    import time as _time

    t0 = _time.time()
    n = lib.astar(
        elevation.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        terrain.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        rows,
        cols,
        start_row * cols + start_col,
        end_row * cols + end_col,
        float(pf.resolution),
        float(pf.heuristic_weight),
        float(pf.elevation_weight),
        float(pf.elevation_exponent),
        float(pf.max_slope_degrees),
        float(pf.steep_threshold),
        float(pf.fatigue_distance),
        float(pf.fatigue_exponent),
        float(pf.sustained_slope_weight),
        cost.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        _NCOST,
        out_path.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ctypes.byref(nodes_explored),
    )
    elapsed = _time.time() - t0
    if n < 0:
        raise RuntimeError("native astar kernel returned error")
    if n == 0:
        return None  # no path found

    cells = out_path[:n]
    transform = pf.transform
    res = pf.resolution

    # Reconstruct (lat, lon, elev) + stats exactly as reconstruct_path does.
    path = []
    total_distance = 0.0
    total_elevation_gain = 0.0
    terrain_counts = {}
    prev_row = prev_col = None
    prev_elev = None
    for cell in cells:
        r = int(cell) // cols
        c = int(cell) % cols
        lon, lat = transform * (c, r)
        elev = float(elevation[r, c])
        path.append((lat, lon, elev))
        t = int(terrain[r, c])
        terrain_counts[t] = terrain_counts.get(t, 0) + 1
        if prev_row is not None:
            dr = r - prev_row
            dc = c - prev_col
            total_distance += res * math.sqrt(dr * dr + dc * dc)
            if elev - prev_elev > 0:
                total_elevation_gain += elev - prev_elev
        prev_row, prev_col, prev_elev = r, c, elev

    total_points = max(sum(terrain_counts.values()), 1)
    stats = {
        "distance_m": round(total_distance, 1),
        "elevation_gain_m": round(float(total_elevation_gain), 1),
        "nodes_explored": int(nodes_explored.value),
        "time_s": round(elapsed, 3),
        "terrain_breakdown": {get_path_type_name(t): round(cnt / total_points, 3) for t, cnt in terrain_counts.items()},
    }
    return path, stats
