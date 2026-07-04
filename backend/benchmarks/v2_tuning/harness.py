"""Benchmark + correctness harness for the v2 terrain-aware A* pathfinder.

Design goals
------------
* Isolate the pathfinder. Route data (DEM, transform, terrain grid) is loaded
  ONCE and cached to disk as .npz, so iterating on pathfinder.py never pays the
  DEM-load / OSM-retry cost again. We benchmark only TerrainAwarePathfinder.
* Faithful offline inputs. Offline, the service degrades to an all-UNKNOWN
  terrain grid; we reproduce that directly (no network). We ALSO build a
  synthetic mixed-terrain grid per route so the cost function's terrain,
  obstacle-skip and trail-transition branches are exercised.
* Correctness oracle. A candidate is only "faster" if it returns the EXACT
  same path (lat/lon/elev tuples) and the same distance_m / elevation_gain_m /
  nodes_explored as the golden baseline, for every route x terrain variant.

Run:
    PYTHONPATH=. ../trail_env/bin/python -m benchmarks.v2_tuning.harness build
    PYTHONPATH=. ../trail_env/bin/python -m benchmarks.v2_tuning.harness bench
"""

import json
import os
import sys
import time

import numpy as np
from rasterio.transform import Affine, from_bounds

# Point at the shared, already-downloaded caches in the main checkout so this
# runs from a worktree (which has no git-ignored data dirs of its own).
_MAIN = "/home/ssilver/development/trail/backend"
os.environ.setdefault("TRAIL_V2_DEM_DIR", f"{_MAIN}/dem_data_v2")
os.environ.setdefault("TRAIL_V2_PATH_CACHE_DIR", f"{_MAIN}/path_cache_v2")

from app.engine_v2.elevation import Bounds, TwoLayerElevationLibrary  # noqa: E402
from app.engine_v2.path_layer import PathType  # noqa: E402
from app.engine_v2.pathfinder import TerrainAwarePathfinder  # noqa: E402
from benchmarks.v2_tuning.routes import ROUTES  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
INPUT_CACHE = os.path.join(_HERE, "input_cache")
GOLDEN_PATH = os.path.join(_HERE, "golden.json")
RESULT_PATH = os.path.join(_HERE, "last_bench.json")
BUFFER = 0.02
RESOLUTION = 10


def _build_transform(bounds: Bounds, elevation, meta):
    raw = meta.get("transform") if isinstance(meta, dict) else None
    if isinstance(raw, Affine):
        return raw
    if isinstance(raw, dict) and all(k in raw for k in "abcdef"):
        return Affine(raw["a"], raw["b"], raw["c"], raw["d"], raw["e"], raw["f"])
    return from_bounds(bounds.west, bounds.south, bounds.east, bounds.north, elevation.shape[1], elevation.shape[0])


def _synthetic_terrain(shape) -> np.ndarray:
    """Deterministic mixed terrain so obstacle/terrain/trail branches run.

    A diagonal TRAIL band, a FOOTWAY stripe, scattered OBSTACLE blocks, and a
    RESIDENTIAL corner -- all index-derived (no RNG), so it is reproducible.
    """
    rows, cols = shape
    g = np.full(shape, PathType.UNKNOWN, dtype=np.int16)
    rr, cc = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    # Diagonal trail band.
    g[np.abs(rr - cc) <= 2] = PathType.TRAIL
    # A path parallel to it.
    g[np.abs(rr - cc - 6) <= 1] = PathType.PATH
    # Horizontal footway.
    g[np.abs(rr - rows // 3) <= 1] = PathType.FOOTWAY
    # Residential corner.
    g[(rr < rows // 5) & (cc < cols // 5)] = PathType.RESIDENTIAL
    # Scattered obstacle blocks (sparse, never fully walling off).
    g[((rr % 11 == 0) & (cc % 13 == 0))] = PathType.OBSTACLE
    return g


def build_inputs():
    """Load DEM once per route, build terrain variants, cache to .npz."""
    os.makedirs(INPUT_CACHE, exist_ok=True)
    lib = TwoLayerElevationLibrary(data_dir=os.environ["TRAIL_V2_DEM_DIR"], resolution=RESOLUTION)
    for name, s_lat, s_lon, e_lat, e_lon in ROUTES:
        out = os.path.join(INPUT_CACHE, f"{name}.npz")
        if os.path.exists(out):
            print(f"  [skip] {name} (cached)")
            continue
        bounds = Bounds(
            south=min(s_lat, e_lat) - BUFFER,
            north=max(s_lat, e_lat) + BUFFER,
            west=min(s_lon, e_lon) - BUFFER,
            east=max(s_lon, e_lon) + BUFFER,
        )
        lib.load_area(bounds)
        elevation, meta = lib.get_elevation_array(bounds)
        if hasattr(lib, "close_all"):
            lib.close_all()
        transform = _build_transform(bounds, elevation, meta)
        tf = np.array([transform.a, transform.b, transform.c, transform.d, transform.e, transform.f])
        unknown = np.full(elevation.shape, PathType.UNKNOWN, dtype=np.int16)
        synth = _synthetic_terrain(elevation.shape)
        np.savez(
            out,
            elevation=elevation.astype(np.float64),
            transform=tf,
            terrain_unknown=unknown,
            terrain_synth=synth,
            endpoints=np.array([s_lat, s_lon, e_lat, e_lon]),
        )
        print(f"  [built] {name} shape={elevation.shape}")
    print("build_inputs done.")


def _load(name):
    d = np.load(os.path.join(INPUT_CACHE, f"{name}.npz"))
    tf = d["transform"]
    transform = Affine(*tf)
    return d, transform


def _run_one(elevation, transform, terrain, endpoints, repeats):
    """Run find_path; return (path, stats, median_time_s over `repeats`)."""
    s_lat, s_lon, e_lat, e_lon = endpoints
    times = []
    result = None
    for _ in range(repeats):
        pf = TerrainAwarePathfinder(elevation, transform, terrain, resolution=RESOLUTION)
        t0 = time.perf_counter()
        result = pf.find_path(s_lat, s_lon, e_lat, e_lon)
        times.append(time.perf_counter() - t0)
    times.sort()
    median = times[len(times) // 2]
    return result, median


def _path_signature(result):
    """Compact, comparable signature of a route result."""
    if result is None:
        return {"found": False}
    path, stats = result
    # Round coords to 1e-9 deg (~0.1mm) to avoid float-repr noise; elevation to 1e-4.
    pts = [(round(la, 9), round(lo, 9), round(el, 4)) for (la, lo, el) in path]
    return {
        "found": True,
        "n": len(path),
        "distance_m": stats["distance_m"],
        "elevation_gain_m": stats["elevation_gain_m"],
        "nodes_explored": stats["nodes_explored"],
        "pts": pts,
    }


def bench(repeats=3, save=True):
    variants = [("unknown", "terrain_unknown"), ("synth", "terrain_synth")]
    results = {}
    total_time = 0.0
    for name, *_ in ROUTES:
        d, transform = _load(name)
        elevation = d["elevation"]
        endpoints = d["endpoints"]
        for vlabel, vkey in variants:
            key = f"{name}:{vlabel}"
            terrain = d[vkey]
            result, median = _run_one(elevation, transform, terrain, endpoints, repeats)
            sig = _path_signature(result)
            sig["time_s"] = median
            results[key] = sig
            total_time += median
            print(
                f"  {key:24s} n={sig.get('n', '-'):>4} "
                f"nodes={sig.get('nodes_explored', '-'):>6} t={median * 1000:8.2f}ms"
            )
    results["_total_time_s"] = total_time
    print(f"TOTAL pathfinder time: {total_time * 1000:.1f} ms  (repeats={repeats}, best-of-median)")
    if save:
        with open(RESULT_PATH, "w") as f:
            json.dump(results, f, indent=2)
    return results


def _sigs_equal(a, b):
    """Correctness: identical path + stats (time excluded)."""
    keys = ("found", "n", "distance_m", "elevation_gain_m", "nodes_explored", "pts")
    return all(a.get(k) == b.get(k) for k in keys)


def save_golden():
    res = bench(repeats=1, save=False)
    with open(GOLDEN_PATH, "w") as f:
        json.dump(res, f)
    print(f"Golden saved: {GOLDEN_PATH}")


def compare_to_golden(repeats=3):
    if not os.path.exists(GOLDEN_PATH):
        print("NO GOLDEN. Run `save_golden` first.")
        return None
    with open(GOLDEN_PATH) as f:
        golden = json.load(f)
    res = bench(repeats=repeats, save=True)
    ok = True
    route_keys = [k for k in golden if not k.startswith("_")]
    for k in route_keys:
        if not _sigs_equal(golden[k], res.get(k, {})):
            ok = False
            g, r = golden[k], res.get(k, {})
            print(
                f"  MISMATCH {k}: golden(n={g.get('n')},dist={g.get('distance_m')},"
                f"nodes={g.get('nodes_explored')}) vs cand(n={r.get('n')},dist={r.get('distance_m')},"
                f"nodes={r.get('nodes_explored')})"
            )
    gt = golden["_total_time_s"]
    rt = res["_total_time_s"]
    speedup = gt / rt if rt else float("inf")
    print(f"\nCORRECT={ok}  golden_total={gt * 1000:.1f}ms  cand_total={rt * 1000:.1f}ms  speedup={speedup:.2f}x")
    return {"correct": ok, "golden_total_s": gt, "cand_total_s": rt, "speedup": speedup}


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "bench"
    if cmd == "build":
        build_inputs()
    elif cmd == "golden":
        save_golden()
    elif cmd == "compare":
        r = compare_to_golden(repeats=int(sys.argv[2]) if len(sys.argv) > 2 else 3)
        sys.exit(0 if (r and r["correct"]) else 1)
    else:
        bench(repeats=int(sys.argv[2]) if len(sys.argv) > 2 else 3)
