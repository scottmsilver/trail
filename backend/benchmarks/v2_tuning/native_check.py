"""Verify + benchmark the native C A* kernel against the golden paths.

    PYTHONPATH=. ../trail_env/bin/python -m benchmarks.v2_tuning.native_check
"""

import json
import os
import sys
import time

_MAIN = "/home/ssilver/development/trail/backend"
os.environ.setdefault("TRAIL_V2_DEM_DIR", f"{_MAIN}/dem_data_v2")
os.environ.setdefault("TRAIL_V2_PATH_CACHE_DIR", f"{_MAIN}/path_cache_v2")

from app.engine_v2 import pathfinder_native as native  # noqa: E402
from app.engine_v2.pathfinder import TerrainAwarePathfinder  # noqa: E402
from benchmarks.v2_tuning.harness import (  # noqa: E402
    GOLDEN_PATH,
    RESOLUTION,
    ROUTES,
    _load,
    _norm_pts,
    _path_signature,
    _sigs_equal,
)


def run(repeats=3):
    if not native.available():
        print("NATIVE KERNEL UNAVAILABLE (gcc missing or compile failed).")
        return 2
    with open(GOLDEN_PATH) as f:
        golden = json.load(f)

    variants = [("unknown", "terrain_unknown"), ("synth", "terrain_synth")]
    total_native = 0.0
    total_golden = golden["_total_time_s"]
    ok = True
    for name, *_ in ROUTES:
        d, transform = _load(name)
        elevation = d["elevation"]
        s_lat, s_lon, e_lat, e_lon = d["endpoints"]
        for vlabel, vkey in variants:
            key = f"{name}:{vlabel}"
            terrain = d[vkey]
            best = None
            tmin = None
            for _ in range(repeats):
                pf = TerrainAwarePathfinder(elevation, transform, terrain, resolution=RESOLUTION)
                t0 = time.perf_counter()
                result = native.find_path_native(pf, s_lat, s_lon, e_lat, e_lon)
                dt = time.perf_counter() - t0
                if tmin is None or dt < tmin:
                    tmin = dt
                best = result
            total_native += tmin
            sig = _path_signature(best)
            g = golden.get(key, {})
            match = _sigs_equal(g, sig)
            if not match:
                ok = False
                print(
                    f"  MISMATCH {key}: golden(n={g.get('n')},dist={g.get('distance_m')},"
                    f"nodes={g.get('nodes_explored')}) vs native(n={sig.get('n')},"
                    f"dist={sig.get('distance_m')},nodes={sig.get('nodes_explored')})"
                )
                # show first differing point for diagnosis
                gp, sp = _norm_pts(g.get("pts")), _norm_pts(sig.get("pts"))
                for i in range(min(len(gp), len(sp))):
                    if gp[i] != sp[i]:
                        print(f"     first pt diff @ {i}: golden={gp[i]} native={sp[i]}")
                        break
            print(
                f"  {key:24s} n={sig.get('n', '-'):>4} nodes={sig.get('nodes_explored', '-'):>6} "
                f"t={tmin * 1000:8.2f}ms {'OK' if match else 'DIFF'}"
            )

    speedup = total_golden / total_native if total_native else float("inf")
    print(
        f"\nNATIVE CORRECT={ok}  golden_total={total_golden * 1000:.1f}ms  "
        f"native_total={total_native * 1000:.1f}ms  speedup={speedup:.2f}x"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(run(int(sys.argv[1]) if len(sys.argv) > 1 else 3))
