import numpy as np

from app.engine_v2.pathfinder import TerrainAwarePathfinder
from app.engine_v2.scoring import dominant_factor, rasterize_segment, score_polyline_cells


def _flat_pf(resolution=10.0):
    """A tiny flat, uniform-terrain pathfinder — no transform needed for
    cost math (only lat_lon_to_grid uses the transform)."""
    elev = np.zeros((6, 6), dtype=float)
    terrain = np.ones((6, 6), dtype=int)
    return TerrainAwarePathfinder(elev, transform=None, terrain_types=terrain, resolution=resolution)


def test_breakdown_sums_to_scalar():
    pf = _flat_pf()
    scalar, _ = pf.calculate_move_cost(0, 0, 0, 1, straight_line_distance=40, current_distance=10)
    bd = pf.calculate_move_cost(0, 0, 0, 1, straight_line_distance=40, current_distance=10, return_breakdown=True)
    assert set(bd["factors"]) == {"base", "terrain", "slope", "sustained", "deviation"}
    assert abs(bd["cost"] - scalar) < 1e-9
    assert abs(sum(bd["factors"].values()) - bd["cost"]) < 1e-9


def test_breakdown_attributes_slope_on_a_climb():
    pf = _flat_pf()
    pf.elevation[0, 1] = 8.0  # ~38.6 degrees over a 10 m horizontal step
    bd = pf.calculate_move_cost(0, 0, 0, 1, straight_line_distance=40, current_distance=10, return_breakdown=True)
    assert bd["factors"]["slope"] > 0
    assert abs(sum(bd["factors"].values()) - bd["cost"]) < 1e-9


def test_rasterize_segment_is_contiguous_and_inclusive():
    cells = rasterize_segment(0, 0, 3, 5)
    assert cells[0] == (0, 0)
    assert cells[-1] == (3, 5)
    for (r0, c0), (r1, c1) in zip(cells, cells[1:]):
        assert max(abs(r1 - r0), abs(c1 - c0)) == 1


def test_score_polyline_cells_accumulates():
    pf = _flat_pf()
    cells = rasterize_segment(0, 0, 0, 4)
    out = score_polyline_cells(pf, cells, straight_line_distance=40)
    assert out["total"] > 0
    assert abs(out["distance"] - 40.0) < 1e-6  # 4 steps * 10 m
    assert abs(sum(out["factors"].values()) - out["total"]) < 1e-6


def test_dominant_factor():
    assert dominant_factor({"base": 100, "slope": 30, "terrain": 5}) == "slope"
    assert dominant_factor({"base": 100}) == "base"
