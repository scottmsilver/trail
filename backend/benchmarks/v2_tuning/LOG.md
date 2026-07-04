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

| Gen | Change | Correct? | Total ms | Speedup vs baseline | Committed |
|-----|--------|----------|----------|---------------------|-----------|
| 0 | baseline | — | 78307 | 1.00x | (golden) |
| 1 | `math` module instead of scalar `np.sqrt/arctan2/degrees` in hot path | ✅ exact | 57086 | **1.39x** | yes |
| 2 | `TerrainNode` → `__slots__` class, precompute `f_cost` (was `@property`) | ✅ exact | 51147 | **1.55x** | yes |
| 3 | inline get_neighbors+calculate_move_cost into loop; hoist locals; `.tolist()` native array access; precompute per-offset horiz dist; hoist deviation penalty per-node | ✅ exact | 17341 | **4.56x** | yes |
| 4 | flat integer keys (`row*cols+col`) for closed_set/best_g_cost; drop dead INF check | ✅ exact | 12552 | **6.30x** | yes |
