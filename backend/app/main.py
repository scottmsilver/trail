from fastapi import FastAPI, HTTPException, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uuid
from datetime import datetime, timezone
from typing import Dict
import asyncio
import logging

from app.models.route import (
    RouteRequest, RouteResponse, RouteStatus, 
    RouteStatusResponse, RouteResult, Coordinate
)
from app.services.trail_finder import TrailFinderService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Initialize trail finder service
trail_finder = TrailFinderService()


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "trail-finder-api"}


async def process_route(route_id: str, request: RouteRequest):
    """Background task to process route calculation"""
    try:
        # Update progress
        routes_storage[route_id]["progress"] = 10
        
        # Validate request
        if not trail_finder.validate_route_request(request.start, request.end):
            routes_storage[route_id]["status"] = RouteStatus.FAILED
            routes_storage[route_id]["message"] = "Invalid route request"
            return
        
        routes_storage[route_id]["progress"] = 30
        
        # Find the route
        path, stats = await trail_finder.find_route(
            request.start, 
            request.end,
            request.options.model_dump() if request.options else {}
        )
        
        routes_storage[route_id]["progress"] = 90
        
        if not path:
            routes_storage[route_id]["status"] = RouteStatus.FAILED
            routes_storage[route_id]["message"] = stats.get("error", "No route found")
        else:
            routes_storage[route_id]["status"] = RouteStatus.COMPLETED
            routes_storage[route_id]["path"] = path
            routes_storage[route_id]["stats"] = stats
            routes_storage[route_id]["progress"] = 100
            
    except Exception as e:
        logger.error(f"Error processing route {route_id}: {str(e)}")
        routes_storage[route_id]["status"] = RouteStatus.FAILED
        routes_storage[route_id]["message"] = f"Processing error: {str(e)}"


@app.post("/api/routes/calculate", response_model=RouteResponse, status_code=status.HTTP_202_ACCEPTED)
async def calculate_route(request: RouteRequest, background_tasks: BackgroundTasks):
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
    
    # Start background processing
    background_tasks.add_task(process_route, route_id, request)
    
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
    
    if route["status"] != RouteStatus.COMPLETED:
        raise HTTPException(
            status_code=400, 
            detail=f"Route is not ready. Status: {route['status']}"
        )
    
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