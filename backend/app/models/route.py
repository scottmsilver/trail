from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict
from enum import Enum


class Coordinate(BaseModel):
    lat: float = Field(..., ge=-90, le=90, description="Latitude")
    lon: float = Field(..., ge=-180, le=180, description="Longitude")


class SlopeConfig(BaseModel):
    """Custom slope configuration"""
    slope_degrees: float = Field(..., ge=0, le=90, description="Slope in degrees")
    cost_multiplier: float = Field(..., gt=0, description="Cost multiplier for this slope")
    
    class Config:
        schema_extra = {
            "example": {
                "slope_degrees": 15,
                "cost_multiplier": 2.0
            }
        }


class CustomPathCosts(BaseModel):
    """Custom path type costs"""
    footway: Optional[float] = Field(None, gt=0, description="Cost multiplier for footways/sidewalks")
    path: Optional[float] = Field(None, gt=0, description="Cost multiplier for paths")
    trail: Optional[float] = Field(None, gt=0, description="Cost multiplier for trails")
    residential: Optional[float] = Field(None, gt=0, description="Cost multiplier for residential streets")
    off_path: Optional[float] = Field(None, gt=0, description="Cost multiplier for off-path terrain")
    
    class Config:
        schema_extra = {
            "example": {
                "trail": 0.3,
                "residential": 0.7,
                "off_path": 1.5
            }
        }


class RouteOptions(BaseModel):
    avoidSteep: bool = Field(True, description="Avoid steep slopes")
    buffer: float = Field(0.05, gt=0, le=0.5, description="Buffer size in degrees")
    slopeThreshold: float = Field(5.71, gt=0, le=90, description="Slope threshold in degrees")
    userProfile: str = Field("default", description="User profile: default, easy, experienced, trail_runner, accessibility")
    
    # Custom configuration options
    customSlopeCosts: Optional[List[SlopeConfig]] = Field(
        None, 
        description="Custom slope cost configuration. List must be in ascending order by slope_degrees."
    )
    customPathCosts: Optional[CustomPathCosts] = Field(
        None,
        description="Custom path cost multipliers. Overrides profile defaults."
    )
    maxSlope: Optional[float] = Field(
        None,
        ge=0,
        le=90,
        description="Maximum allowed slope in degrees. Routes with steeper slopes will be penalized heavily."
    )
    
    @validator('customSlopeCosts')
    def validate_slope_order(cls, v):
        if v is not None and len(v) > 1:
            # Check that slopes are in ascending order
            for i in range(1, len(v)):
                if v[i].slope_degrees <= v[i-1].slope_degrees:
                    raise ValueError("Slope configurations must be in ascending order by slope_degrees")
        return v


class RouteRequest(BaseModel):
    start: Coordinate
    end: Coordinate
    options: Optional[RouteOptions] = RouteOptions()


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