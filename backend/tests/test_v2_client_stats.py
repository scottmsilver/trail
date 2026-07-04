"""Unit tests for v2 client-schema stat normalization.

The React frontend and the GPX generator read the v1 stats schema
(distance_km, estimated_time_min, difficulty, path_with_slopes). The v2
engine natively emits a different schema (distance_m, terrain_breakdown, ...),
so v2 routes rendered NaN/blank in the UI. TrailFinderServiceV2._augment_client_stats
bridges the two. These tests pin that bridge.
"""

from app.engine_v2.service import TrailFinderServiceV2


def test_augment_adds_v1_keys_without_dropping_v2_keys():
    svc = TrailFinderServiceV2()
    # Two points ~ (elevation +10m over a short horizontal hop)
    raw_path = [
        (40.6482, -111.5738, 2000.0),
        (40.6483, -111.5738, 2010.0),
    ]
    stats = {"distance_m": 1500.0, "terrain_breakdown": {"trail": 1.0}, "engine": "v2"}

    out = svc._augment_client_stats(raw_path, stats)

    # v1-contract keys the frontend/GPX read
    assert out["distance_km"] == 1.5
    assert out["estimated_time_min"] == int(1.5 * 15)
    assert out["difficulty"] in {"easy", "moderate", "hard"}
    assert out["waypoints"] == 2
    assert "path_with_slopes" in out and len(out["path_with_slopes"]) == 2
    # per-point overlay data present
    pt = out["path_with_slopes"][1]
    assert {"lat", "lon", "elevation", "slope"} <= set(pt)
    # native v2 keys preserved
    assert out["distance_m"] == 1500.0
    assert out["terrain_breakdown"] == {"trail": 1.0}
    assert out["engine"] == "v2"


def test_first_point_slope_is_zero_and_uphill_slope_positive():
    svc = TrailFinderServiceV2()
    raw_path = [
        (40.6482, -111.5738, 2000.0),
        (40.6483, -111.5738, 2010.0),  # gains elevation -> positive slope
    ]
    out = svc._augment_client_stats(raw_path, {"distance_m": 100.0})
    assert out["path_with_slopes"][0]["slope"] == 0.0
    assert out["path_with_slopes"][1]["slope"] > 0
    assert out["max_slope"] >= out["path_with_slopes"][1]["slope"]


def test_difficulty_thresholds_match_v1():
    # distance-driven
    assert TrailFinderServiceV2._difficulty(2.0, 0.0) == "easy"
    assert TrailFinderServiceV2._difficulty(5.0, 0.0) == "moderate"
    assert TrailFinderServiceV2._difficulty(12.0, 0.0) == "hard"
    # slope can escalate a short route
    assert TrailFinderServiceV2._difficulty(1.0, 25.0) == "hard"
    assert TrailFinderServiceV2._difficulty(1.0, 12.0) == "moderate"


def test_missing_distance_m_defaults_to_zero_km():
    svc = TrailFinderServiceV2()
    out = svc._augment_client_stats([(40.0, -111.0, 100.0)], {})
    assert out["distance_km"] == 0.0
    assert out["estimated_time_min"] == 0
    assert out["waypoints"] == 1
