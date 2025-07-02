from fastapi import FastAPI, HTTPException, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uuid
from datetime import datetime, timezone
from typing import Dict
import asyncio
import logging
import numpy as np

from app.models.route import (
    RouteRequest, RouteResponse, RouteStatus, 
    RouteStatusResponse, RouteResult, Coordinate
)
from app.services.trail_finder import TrailFinderService
from app.services.obstacle_config import ObstaclePresets
from app.services.path_preferences import PathPreferences, PathPreferencePresets
from app.services.dem_tile_cache import DEMTileCache

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
    allow_origins=["http://localhost:3000", "http://localhost:9002", "http://localhost:5173", "http://localhost:5174"],  # Frontend dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for now (will be replaced with Redis)
routes_storage: Dict[str, dict] = {}

# Initialize trail finder services
trail_finder = TrailFinderService()
debug_trail_finder = TrailFinderService(debug_mode=True)


def get_obstacle_config_for_profile(profile: str):
    """Get obstacle configuration for a given user profile"""
    profile_map = {
        "easy": ObstaclePresets.easy_hiker(),
        "experienced": ObstaclePresets.experienced_hiker(),
        "trail_runner": ObstaclePresets.trail_runner(),
        "accessibility": ObstaclePresets.accessibility_focused(),
    }
    return profile_map.get(profile, ObstaclePresets.easy_hiker())


def get_configs_for_profile(profile: str, options=None):
    """Get obstacle configuration and path preferences based on user profile with custom overrides"""
    # Obstacle configurations
    obstacle_map = {
        "easy": ObstaclePresets.easy_hiker(),
        "experienced": ObstaclePresets.experienced_hiker(),
        "trail_runner": ObstaclePresets.trail_runner(),
        "accessibility": ObstaclePresets.accessibility_focused(),
    }
    
    # Path preference configurations
    path_pref_map = {
        "easy": PathPreferencePresets.urban_walker(),  # Prefers sidewalks and paths
        "experienced": PathPreferencePresets.trail_seeker(),  # Prefers natural trails
        "trail_runner": PathPreferencePresets.flexible_hiker(),  # Mild path preference
        "accessibility": PathPreferencePresets.urban_walker(),  # Strongly prefers paved paths
        "default": PathPreferences(),  # Default preferences
    }
    
    obstacle_config = obstacle_map.get(profile, ObstaclePresets.easy_hiker())
    path_preferences = path_pref_map.get(profile, PathPreferences())
    
    # Enable continuous slope function by default
    obstacle_config.use_continuous_slope = True
    
    # Map user profiles to slope profiles
    slope_profile_map = {
        "easy": "city_walker",
        "experienced": "mountain_goat",
        "trail_runner": "trail_runner",
        "accessibility": "wheelchair",
        "mountain_goat": "mountain_goat",  # Direct mapping
        "city_walker": "city_walker",     # Direct mapping
        "wheelchair": "wheelchair",       # Direct mapping
    }
    obstacle_config.slope_profile = slope_profile_map.get(profile, 'default')
    
    # Apply custom configurations if provided
    if options:
        # Apply custom slope costs
        if options.customSlopeCosts:
            # Convert to the format expected by ObstacleConfig
            slope_costs = [(sc.slope_degrees, sc.cost_multiplier) for sc in options.customSlopeCosts]
            # Add a final point for vertical slopes if not already included
            if slope_costs[-1][0] < 90:
                slope_costs.append((90, np.inf))
            obstacle_config.slope_costs = slope_costs
            logger.info(f"Applied custom slope costs: {slope_costs}")
        
        # Apply max slope if specified
        if options.maxSlope is not None:
            # Add infinite cost for slopes above the max
            current_costs = obstacle_config.slope_costs
            new_costs = []
            for slope, cost in current_costs:
                if slope < options.maxSlope:
                    new_costs.append((slope, cost))
                else:
                    new_costs.append((options.maxSlope, np.inf))
                    break
            obstacle_config.slope_costs = new_costs
            logger.info(f"Applied max slope limit: {options.maxSlope}°")
        
        # Apply custom path costs
        if options.customPathCosts:
            custom = options.customPathCosts
            if custom.footway is not None:
                path_preferences.path_costs['footway'] = custom.footway
            if custom.path is not None:
                path_preferences.path_costs['path'] = custom.path
            if custom.trail is not None:
                path_preferences.path_costs['trail'] = custom.trail
            if custom.residential is not None:
                path_preferences.path_costs['residential'] = custom.residential
            if custom.off_path is not None:
                path_preferences.path_costs['off_path'] = custom.off_path
            logger.info(f"Applied custom path costs")
    
    return obstacle_config, path_preferences


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "trail-finder-api"}


async def process_route(route_id: str, request: RouteRequest):
    """Background task to process route calculation"""
    try:
        # Update progress
        routes_storage[route_id]["progress"] = 10
        
        # Get configurations based on user profile and custom options
        profile = request.options.userProfile if request.options else "default"
        obstacle_config, path_preferences = get_configs_for_profile(profile, request.options)
        
        # Create trail finder with user's configurations
        profile_trail_finder = TrailFinderService(
            obstacle_config=obstacle_config,
            path_preferences=path_preferences
        )
        
        # Validate request
        if not profile_trail_finder.validate_route_request(request.start, request.end):
            routes_storage[route_id]["status"] = RouteStatus.FAILED
            routes_storage[route_id]["message"] = "Invalid route request"
            return
        
        routes_storage[route_id]["progress"] = 30
        
        # Find the route
        path, stats = await profile_trail_finder.find_route(
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


@app.post("/api/terrain/slopes")
async def get_terrain_slopes(bounds: dict):
    """Get slope data for visualization"""
    try:
        min_lat = bounds.get("minLat")
        max_lat = bounds.get("maxLat")
        min_lon = bounds.get("minLon") 
        max_lon = bounds.get("maxLon")
        
        if not all([min_lat, max_lat, min_lon, max_lon]):
            raise HTTPException(status_code=400, detail="Invalid bounds")
        
        # Create a DEM cache instance
        dem_cache = DEMTileCache()
        
        # Download DEM data for the area
        dem_file = dem_cache.download_dem(min_lat, max_lat, min_lon, max_lon)
        if not dem_file:
            raise HTTPException(status_code=500, detail="Failed to download elevation data")
        
        # Read DEM data
        dem, out_trans, crs = dem_cache.read_dem(dem_file)
        if dem is None:
            raise HTTPException(status_code=500, detail="Failed to read elevation data")
        
        # Reproject if needed
        dem, out_trans, crs = dem_cache.reproject_dem(dem, out_trans, crs)
        
        # Calculate slopes
        cell_size_x = out_trans.a
        cell_size_y = -out_trans.e
        dzdx, dzdy = np.gradient(dem, cell_size_x, cell_size_y)
        slope_radians = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
        slope_degrees = np.degrees(slope_radians)
        
        # Store original transformation for later use
        downsample_step = 1
        
        # Downsample for visualization if too large
        max_size = 200
        if slope_degrees.shape[0] > max_size or slope_degrees.shape[1] > max_size:
            # Simple downsampling
            downsample_step = max(slope_degrees.shape[0] // max_size, slope_degrees.shape[1] // max_size)
            slope_degrees = slope_degrees[::downsample_step, ::downsample_step]
            
        # Convert to lat/lon grid
        height, width = slope_degrees.shape
        lats = []
        lons = []
        slopes = []
        
        from pyproj import Transformer
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        
        # Sample points for visualization
        sample_step = max(1, height // 50)  # Limit to ~50x50 grid for sparser heatmap
        
        for i in range(0, height, sample_step):
            for j in range(0, width, sample_step):
                # Calculate position accounting for downsampling
                actual_i = i * downsample_step
                actual_j = j * downsample_step
                x = out_trans.c + actual_j * out_trans.a + out_trans.a * downsample_step / 2
                y = out_trans.f + actual_i * out_trans.e + out_trans.e * downsample_step / 2
                lon, lat = transformer.transform(x, y)
                
                if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
                    lats.append(lat)
                    lons.append(lon)
                    slopes.append(float(slope_degrees[i, j]))
        
        return {
            "lats": lats,
            "lons": lons,
            "slopes": slopes,
            "bounds": {
                "minLat": min_lat,
                "maxLat": max_lat,
                "minLon": min_lon,
                "maxLon": max_lon
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting terrain slopes: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing terrain data: {str(e)}")


@app.post("/api/routes/debug", response_model=RouteResult)
async def calculate_debug_route(request: RouteRequest):
    """Calculate route with debug information"""
    try:
        # Get configurations based on user profile and custom options
        profile = request.options.userProfile if request.options else "default"
        obstacle_config, path_preferences = get_configs_for_profile(profile, request.options)
        
        # Create debug trail finder with user's configurations
        profile_debug_finder = TrailFinderService(
            debug_mode=True, 
            obstacle_config=obstacle_config,
            path_preferences=path_preferences
        )
        
        # Find the route with debug mode enabled
        path, stats = await profile_debug_finder.find_route(
            request.start, 
            request.end,
            request.options.model_dump() if request.options else {}
        )
        
        if not path:
            # Return failed route with debug data if available
            return RouteResult(
                routeId="debug",
                status=RouteStatus.FAILED,
                path=[],
                stats=stats,
                createdAt=datetime.now(timezone.utc).isoformat()
            )
        
        # Return successful route
        return RouteResult(
            routeId="debug",
            status=RouteStatus.COMPLETED,
            path=path,
            stats=stats,
            createdAt=datetime.now(timezone.utc).isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating debug route: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)