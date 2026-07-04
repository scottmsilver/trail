# v2 Pathfinder Speed-Tuning Log

Overnight optimization loop for `app/engine_v2/pathfinder.py` (terrain-aware A*).

## Method
- Harness: `benchmarks/v2_tuning/harness.py`. Loads DEM/terrain inputs once
  (cached to `input_cache/`), then benchmarks **only** `TerrainAwarePathfinder.find_path`
  over 10 Park City routes × 2 terrain variants (all-UNKNOWN + synthetic mixed).
- Correctness oracle (strict): a change is only accepted if every route returns
  the **exact** same path points + `distance_m` + `elevation_gain_m` +
  `nodes_explored` as `golden.json`. Plus `test_pathfinder_v2.py` must pass.
- Metric: total median pathfinder wall time across all 20 runs.

## Baseline (golden)
- Total: **79,087 ms** (repeats=1). Dominated by long routes (long_ne ~20s @ 390k nodes).
- ~51 µs/node. Bottleneck: per-edge scalar numpy calls in `calculate_move_cost`.

## Generations

Clean baseline @ repeats=3 median = **78,307 ms** (measured by stashing changes to HEAD).

## Result: ~104x faster, byte-identical output

Total v2 pathfinder time over 10 routes × 2 terrain variants: **78.3 s → 0.76 s**.
The worst single route (`long_ne`, 390k nodes) went **20.0 s → 0.20 s**. Every
generation produces *byte-identical* paths + stats to the original engine
(strict oracle), so this is a pure speed win with zero behavior change.

Extra confidence: `verify_opts` ran 84 native-vs-Python comparisons across
weighted heuristics, max-slope, elevation-weight, steep/fatigue params, and
`experienced` profile costs on real cached data — all identical.

Why this is the exact-preserving ceiling: the *set* of nodes A* explores is
fixed by the algorithm, so preserving output means preserving exploration —
the only lever left is per-node cost, now optimized C++. Going faster would
require algorithmic changes (weighted/hierarchical A*, corridor precompute)
that change which path is returned — a product decision, not a free win.


| Gen | Change | Correct? | Total ms | Speedup vs baseline | Committed |
|-----|--------|----------|----------|---------------------|-----------|
| 0 | baseline | — | 78307 | 1.00x | (golden) |
| 1 | `math` module instead of scalar `np.sqrt/arctan2/degrees` in hot path | ✅ exact | 57086 | **1.39x** | yes |
| 2 | `TerrainNode` → `__slots__` class, precompute `f_cost` (was `@property`) | ✅ exact | 51147 | **1.55x** | yes |
| 3 | inline get_neighbors+calculate_move_cost into loop; hoist locals; `.tolist()` native array access; precompute per-offset horiz dist; hoist deviation penalty per-node | ✅ exact | 17341 | **4.56x** | yes |
| 4 | flat integer keys (`row*cols+col`) for closed_set/best_g_cost; drop dead INF check | ✅ exact | 12552 | **6.30x** | yes |
| 5 | **native C kernel** (`_astar_kernel.c`) compiled on-demand via gcc+ctypes; faithful port incl. byte-for-byte CPython heapq sift → identical tie-breaking; default-on with pure-Python fallback + `TRAIL_V2_DISABLE_NATIVE` kill-switch | ✅ exact | 761 | **103.9x** | yes |

### Gen5 native notes
- Compiled with `-O3 -ffp-contract=off` (no `-ffast-math`, no FMA contraction) so
  double results are bit-identical to Python's `math`/`**`.
- Verified byte-identical (paths + distance + gain + nodes_explored) on all 20
  route×terrain runs AND on 7 synthetic unit-test scenarios
  (`tests/unit/test_pathfinder_native.py`), including no-path, obstacle detour,
  weighted heuristic, and steep/fatigue cases.
- `long_ne` (390k nodes): 20.0s → 0.198s.
- No new pip dependency (stdlib ctypes/subprocess + system compiler only).
- Gen6: ported the kernel from C to C++ (`_astar_kernel.cpp`, compiled with g++)
  purely for readability/safety — `std::vector` replaces all manual
  malloc/realloc/free (the memory surface the security review had to reason
  about). Still NOT using std::priority_queue (its heap layout differs from
  heapq and would change tie-breaking); the CPython heapq port is kept explicit.
  Re-verified byte-identical + 14 tests pass. Same ~104x speed.
