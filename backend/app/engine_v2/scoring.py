"""Score an arbitrary polyline with the engine's own move-cost function.

This is the heart of the Eval UI: instead of A* choosing a path, we walk a
*given* polyline across the same DEM/terrain grid and sum the identical
``TerrainAwarePathfinder.calculate_move_cost`` the pathfinder minimizes. Scoring
the engine's optimal path and a user-drawn path the same way makes their costs
directly comparable, and the per-factor breakdown explains *why* one is dearer.
"""

import math
from typing import Dict, List, Tuple

_FACTOR_KEYS = ("base", "terrain", "slope", "sustained", "deviation")

# Impassable moves (slope > max) cost +inf, and NaN DEM cells would poison the
# sum. Neither survives JSON (Infinity/NaN aren't valid JSON and break JS
# parsers), so we map them to a large finite sentinel that still ranks an
# impassable path as the worst option.
_IMPASSABLE = 1e18


def _finite(x: float) -> float:
    return x if math.isfinite(x) else _IMPASSABLE


def rasterize_segment(r0: int, c0: int, r1: int, c1: int) -> List[Tuple[int, int]]:
    """All grid cells a segment passes through, contiguous with 8-connected steps.

    Bresenham-style supercover; endpoints inclusive. Consecutive cells always
    differ by at most 1 in each axis, so ``calculate_move_cost`` (which expects
    adjacent cells) applies to every step.
    """
    cells = [(r0, c0)]
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r1 > r0 else -1
    sc = 1 if c1 > c0 else -1
    r, c = r0, c0
    err = dr - dc
    # Guard against pathological inputs; the grid is bounded so this is ample.
    for _ in range((dr + dc) * 2 + 2):
        if (r, c) == (r1, c1):
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc
        cells.append((r, c))
    return cells


def score_polyline_cells(
    pf,
    cells: List[Tuple[int, int]],
    straight_line_distance: float,
    running_distance: float = 0.0,
    running_steep: float = 0.0,
) -> Dict:
    """Walk contiguous grid cells, summing per-cell move-cost breakdowns.

    ``running_distance``/``running_steep`` carry cumulative state across earlier
    segments so the deviation and sustained-fatigue penalties match what an
    end-to-end traversal would see. Returns total/factors/distance/egain/steep.
    """
    total = 0.0
    factors = {k: 0.0 for k in _FACTOR_KEYS}
    dist = 0.0
    egain = 0.0
    steep = running_steep
    for (r0, c0), (r1, c1) in zip(cells, cells[1:]):
        current_distance = running_distance + dist
        bd = pf.calculate_move_cost(
            r0, c0, r1, c1, straight_line_distance, current_distance, steep, return_breakdown=True
        )
        total += bd["cost"]
        for k, v in bd["factors"].items():
            factors[k] = factors.get(k, 0.0) + v
        steep = bd["new_steep_distance"]
        dist += pf.resolution * math.sqrt((r1 - r0) ** 2 + (c1 - c0) ** 2)
        delta_elev = float(pf.elevation[r1, c1]) - float(pf.elevation[r0, c0])
        if delta_elev > 0:
            egain += delta_elev
    # Keep the result JSON-serializable even for impassable/NaN moves.
    total = _finite(total)
    factors = {k: _finite(v) for k, v in factors.items()}
    return {"total": total, "factors": factors, "distance": _finite(dist), "egain": _finite(egain), "steep": steep}


def dominant_factor(factors: Dict[str, float]) -> str:
    """Largest non-base contributor; falls back to 'base' when nothing else bites."""
    non_base = {k: v for k, v in factors.items() if k != "base"}
    if not non_base or max(non_base.values()) <= 0:
        return "base"
    return max(non_base, key=non_base.get)
