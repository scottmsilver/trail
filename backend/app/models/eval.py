"""Pydantic contract models for the Eval UI: path scoring and eval cases.

This module is the single source of truth for the `/api/eval/*` request and
response shapes. The frontend mirrors these in `frontend/src/services/evalApi.ts`.
Both the engine's optimal path and the user's drawn path are returned as the
same `ScoredPath`, so the UI has one renderer.
"""

from typing import Dict, List, Literal

from app.models.route import Coordinate, RouteOptions
from pydantic import BaseModel, ConfigDict, Field


class ScorePathRequest(BaseModel):
    """Request to score an arbitrary polyline with the engine's cost function."""

    path: List[Coordinate]
    options: RouteOptions = RouteOptions()
    snap: Literal["none", "trail"] = "none"


class ScoredSegment(BaseModel):
    """One drawn polyline edge (vertex i -> i+1), with its cost decomposed.

    `factors` sum to `cost`; keys are base/terrain/slope/sustained/deviation.
    `dominantFactor` is the largest non-base contributor.
    """

    model_config = ConfigDict(populate_by_name=True)

    from_: Coordinate = Field(..., alias="from")
    to: Coordinate
    cost: float
    factors: Dict[str, float]
    dominantFactor: str


class ScoredPath(BaseModel):
    """A polyline scored by `score_path` — used for both optimal and drawn paths."""

    path: List[Coordinate]
    snapped: bool = False
    totalCost: float
    distanceM: float
    elevationGainM: float
    segments: List[ScoredSegment]


class EvalLabel(BaseModel):
    """A manual verdict recorded on a case at re-run time."""

    ts: str
    verdict: Literal["pass", "fail", "unsure"]
    note: str = ""


class EvalCase(BaseModel):
    """A saved regression case: endpoints + weights + the drawn reference path."""

    id: str
    name: str
    notes: str = ""
    start: Coordinate
    end: Coordinate
    options: RouteOptions = RouteOptions()
    referencePath: List[Coordinate]
    labels: List[EvalLabel] = Field(default_factory=list)
