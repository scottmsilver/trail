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

| Gen | Change | Correct? | Total ms | Speedup vs baseline | Committed |
|-----|--------|----------|----------|---------------------|-----------|
| 0 | baseline | — | 79088 | 1.00x | (golden) |
