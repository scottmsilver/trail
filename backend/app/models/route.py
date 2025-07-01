from pydantic import BaseModel, Field, validator
from typing import Optional, List
from enum import Enum


class Coordinate(BaseModel):
    lat: float = Field(..., ge=-90, le=90, description="Latitude")
    lon: float = Field(..., ge=-180, le=180, description="Longitude")


class RouteOptions(BaseModel):
    avoidSteep: bool = Field(True, description="Avoid steep slopes")
    buffer: float = Field(0.05, gt=0, le=0.5, description="Buffer size in degrees")
    slopeThreshold: float = Field(5.71, gt=0, le=90, description="Slope threshold in degrees")


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