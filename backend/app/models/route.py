from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator, validator


class Coordinate(BaseModel):
    lat: float = Field(..., ge=-90, le=90, description="Latitude")
    lon: float = Field(..., ge=-180, le=180, description="Longitude")


class SlopeConfig(BaseModel):
    """Custom slope configuration"""

    slope_degrees: float = Field(..., ge=0, le=90, description="Slope in degrees")
    cost_multiplier: float = Field(..., gt=0, description="Cost multiplier for this slope")

    class Config:
        schema_extra = {"example": {"slope_degrees": 15, "cost_multiplier": 2.0}}


class CustomPathCosts(BaseModel):
    """Custom path type costs"""

    footway: Optional[float] = Field(None, gt=0, description="Cost multiplier for footways/sidewalks")
    path: Optional[float] = Field(None, gt=0, description="Cost multiplier for paths")
    trail: Optional[float] = Field(None, gt=0, description="Cost multiplier for trails")
    residential: Optional[float] = Field(None, gt=0, description="Cost multiplier for residential streets")
    off_path: Optional[float] = Field(None, gt=0, description="Cost multiplier for off-path terrain")

    class Config:
        schema_extra = {"example": {"trail": 0.3, "residential": 0.7, "off_path": 1.5}}


class RouteOptions(BaseModel):
    avoidSteep: bool = Field(True, description="Avoid steep slopes")
    buffer: float = Field(0.05, gt=0, le=0.5, description="Buffer size in degrees")
    slopeThreshold: float = Field(5.71, gt=0, le=90, description="Slope threshold in degrees")
    userProfile: str = Field(
        "default", description="User profile: default, easy, experienced, trail_runner, accessibility"
    )

    # Custom configuration options
    customSlopeCosts: Optional[List[SlopeConfig]] = Field(
        None, description="Custom slope cost configuration. List must be in ascending order by slope_degrees."
    )
    customPathCosts: Optional[CustomPathCosts] = Field(
        None, description="Custom path cost multipliers. Overrides profile defaults."
    )
    maxSlope: Optional[float] = Field(
        None,
        ge=0,
        le=90,
        description="Maximum allowed slope in degrees. Routes with steeper slopes will be penalized heavily.",
    )
    gradientPreference: float = Field(
        1.0,
        ge=0.1,
        le=5.0,
        description="Gradient preference: 1.0=normal, >1=prefer gradual slopes, <1=accept steep slopes",
    )
    trailPreference: float = Field(
        1.0, ge=0.1, le=5.0, description="Trail preference: 1.0=normal, >1=prefer natural trails, <1=prefer urban paths"
    )
    engine: str = Field(
        "v2",
        pattern="^(v1|v2)$",
        description="Routing engine: v1 (legacy DEMTileCache) or v2 (two-layer + terrain A*)",
    )
    heuristicWeight: Optional[float] = Field(
        None,
        ge=1.0,
        le=3.0,
        description="v2 only: A* heuristic weight. 1.0 = optimal, higher = faster/greedier",
    )

    @validator("customSlopeCosts")
    def validate_slope_order(cls, v):
        if v is not None and len(v) > 1:
            # Check that slopes are in ascending order
            for i in range(1, len(v)):
                if v[i].slope_degrees <= v[i - 1].slope_degrees:
                    raise ValueError("Slope configurations must be in ascending order by slope_degrees")
        return v


class RouteRequest(BaseModel):
    start: Optional[Coordinate] = None
    end: Optional[Coordinate] = None
    points: Optional[List[Coordinate]] = Field(
        None,
        description="Ordered waypoints (>=2). Route passes through each in order. Takes precedence over start/end.",
    )
    options: Optional[RouteOptions] = RouteOptions()

    @model_validator(mode="after")
    def _require_points_or_endpoints(self):
        if self.points is not None:
            if len(self.points) < 2:
                raise ValueError("points must contain at least 2 coordinates")
        elif self.start is None or self.end is None:
            raise ValueError("Provide either 'points' (>=2) or both 'start' and 'end'")
        return self

    def normalized_points(self) -> List[Coordinate]:
        """Single ordered points list: `points` if given, else [start, end]."""
        if self.points is not None:
            return self.points
        return [self.start, self.end]


class RouteVariantsRequest(RouteRequest):
    """Route the same start/end at several expertise levels in one call.

    ``levels`` selects/orders the expertise levels (default: all, easiest
    first). See engine_v2.service.EXPERTISE_LEVELS.
    """

    levels: Optional[List[str]] = None


class RouteStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class RouteResponse(BaseModel):
    routeId: str
    status: RouteStatus


class RouteStatusResponse(BaseModel):
    status: RouteStatus
    progress: int = Field(..., ge=0, le=100)
    message: Optional[str] = None


class RouteResult(BaseModel):
    routeId: str
    status: RouteStatus
    path: List[Coordinate]
    stats: dict
    createdAt: str
