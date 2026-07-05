"""The service maps a named expertise level (or explicit scrambleBudgetM) onto
the pathfinder's extent-aware gate. See docs/terrain-passability-extent-aware.md.
"""

import numpy as np
import pytest
from rasterio.transform import from_bounds

from app.engine_v2.path_layer import PathType
from app.engine_v2.service import EXPERTISE_BUDGETS_M
from tests.test_score_path_service import make_service

_EL = np.zeros((5, 5), dtype=np.float32)
_TG = np.full((5, 5), PathType.UNKNOWN, dtype=np.uint8)
_TR = from_bounds(-111.60, 40.60, -111.599, 40.601, 5, 5)


def _pf(options):
    return make_service()._make_pathfinder(_EL, _TR, _TG, options)


def test_named_levels_map_to_budgets():
    assert _pf({"expertiseLevel": "hiker"}).scramble_budget_m == 4.0
    assert _pf({"expertiseLevel": "alpinist"}).scramble_budget_m == 15.0
    # case-insensitive
    assert _pf({"expertiseLevel": "Casual"}).scramble_budget_m == 1.5


def test_explicit_budget_wins_over_level():
    pf = _pf({"expertiseLevel": "hiker", "scrambleBudgetM": 6.5})
    assert pf.scramble_budget_m == 6.5


def test_no_expertise_is_classic_memoryless_gate():
    assert _pf({}).scramble_budget_m is None


def test_unknown_level_raises():
    with pytest.raises(ValueError):
        _pf({"expertiseLevel": "ninja"})


def test_all_presets_ascend():
    vals = [EXPERTISE_BUDGETS_M[k] for k in ("casual", "hiker", "scrambler", "alpinist")]
    assert vals == sorted(vals)


def test_expertise_also_sets_steep_aversion():
    # The lever that actually smooths the route: expert => low elevation_weight.
    assert _pf({"expertiseLevel": "casual"}).elevation_weight == 3.0
    assert _pf({"expertiseLevel": "alpinist"}).elevation_weight == 0.15
    # steep-aversion descends as budget ascends
    ews = [_pf({"expertiseLevel": k}).elevation_weight for k in ("casual", "hiker", "scrambler", "alpinist")]
    assert ews == sorted(ews, reverse=True)


def test_explicit_gradient_pref_overrides_level_aversion():
    pf = _pf({"expertiseLevel": "alpinist", "gradientPreference": 2.0})
    assert pf.elevation_weight == 2.0  # explicit wins
    assert pf.scramble_budget_m == 15.0  # but the gate budget still comes from the level


def test_pathological_budget_rejected():
    # NaN/inf comparisons are always False, which would silently disable the
    # gate ("everything passable") — must raise instead (codex audit).
    for bad in (float("nan"), float("inf"), -1.0, 1e9):
        with pytest.raises(ValueError):
            _pf({"scrambleBudgetM": bad})
    for bad in (float("nan"), 0.0, 90.0):
        with pytest.raises(ValueError):
            _pf({"scrambleBudgetM": 4.0, "extentThresholdDeg": bad})
    # sane values still fine
    assert _pf({"scrambleBudgetM": 0.0}).scramble_budget_m == 0.0
    assert _pf({"scrambleBudgetM": 4.0, "extentThresholdDeg": 40.0}).extent_threshold_degrees == 40.0
