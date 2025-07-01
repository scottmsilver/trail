from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import uuid
from datetime import datetime, timezone
from typing import Dict

from app.models.route import (
    RouteRequest, RouteResponse, RouteStatus, 
    RouteStatusResponse, RouteResult, Coordinate
)

app = FastAPI(
    title="Trail Finder API",
    description="API for finding optimal hiking trails",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for now (will be replaced with Redis)
routes_storage: Dict[str, dict] = {}


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "trail-finder-api"}


@app.post("/api/routes/calculate", response_model=RouteResponse, status_code=status.HTTP_202_ACCEPTED)
async def calculate_route(request: RouteRequest):
    """Start route calculation"""
    route_id = str(uuid.uuid4())
    
    # Store route request
    routes_storage[route_id] = {
        "id": route_id,
        "status": RouteStatus.PROCESSING,
        "progress": 0,
        "request": request.model_dump(),
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    
    # TODO: Start async processing with Celery
    
    return RouteResponse(routeId=route_id, status=RouteStatus.PROCESSING)


@app.get("/api/routes/{route_id}/status", response_model=RouteStatusResponse)
async def get_route_status(route_id: str):
    """Get route calculation status"""
    if route_id not in routes_storage:
        raise HTTPException(status_code=404, detail="Route not found")
    
    route = routes_storage[route_id]
    return RouteStatusResponse(
        status=route["status"],
        progress=route["progress"],
        message=route.get("message")
    )


@app.get("/api/routes/{route_id}", response_model=RouteResult)
async def get_route(route_id: str):
    """Get calculated route"""
    if route_id not in routes_storage:
        raise HTTPException(status_code=404, detail="Route not found")
    
    route = routes_storage[route_id]
    
    # For now, return a mock completed route
    if route["status"] == RouteStatus.PROCESSING:
        # TODO: Check actual processing status
        route["status"] = RouteStatus.COMPLETED
        route["path"] = [
            Coordinate(lat=40.630, lon=-111.580),
            Coordinate(lat=40.640, lon=-111.570),
            Coordinate(lat=40.650, lon=-111.560)
        ]
        route["stats"] = {
            "distance_km": 3.2,
            "elevation_gain_m": 150,
            "estimated_time_min": 45,
            "difficulty": "moderate"
        }
    
    return RouteResult(
        routeId=route_id,
        status=route["status"],
        path=route.get("path", []),
        stats=route.get("stats", {}),
        createdAt=route["created_at"]
    )


@app.get("/api/routes/{route_id}/gpx")
async def download_gpx(route_id: str):
    """Download route as GPX file"""
    if route_id not in routes_storage:
        raise HTTPException(status_code=404, detail="Route not found")
    
    # TODO: Generate GPX file
    raise HTTPException(status_code=501, detail="Not implemented yet")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)