from app.models.eval import EvalCase, ScoredSegment, ScorePathRequest
from app.models.route import RouteOptions


def test_scored_segment_from_alias():
    seg = ScoredSegment(
        **{
            "from": {"lat": 1, "lon": 2},
            "to": {"lat": 1, "lon": 3},
            "cost": 5.0,
            "factors": {"base": 5.0},
            "dominantFactor": "base",
        }
    )
    assert seg.from_.lat == 1
    assert seg.model_dump(by_alias=True)["from"]["lon"] == 2


def test_score_request_defaults_snap_none():
    req = ScorePathRequest(path=[{"lat": 0, "lon": 0}, {"lat": 0, "lon": 1}], options=RouteOptions())
    assert req.snap == "none"


def test_eval_case_roundtrips():
    case = EvalCase(
        id="c1",
        name="n",
        start={"lat": 0, "lon": 0},
        end={"lat": 1, "lon": 1},
        options=RouteOptions(),
        referencePath=[{"lat": 0, "lon": 0}],
    )
    assert case.labels == []
    assert EvalCase(**case.model_dump()).id == "c1"
