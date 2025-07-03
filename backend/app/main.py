from fastapi import FastAPI, HTTPException, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uuid
from datetime import datetime, timezone
from typing import Dict
import asyncio
import logging
import numpy as np
import os

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

# Create shared DEM cache for all trail finder instances
logger.info("Creating shared DEM caches...")

# Log HTTP cache location
http_cache_path = os.environ.get('HYRIVER_CACHE_NAME', os.path.abspath('cache/aiohttp_cache.sqlite'))
if not os.path.isabs(http_cache_path):
    http_cache_path = os.path.abspath(http_cache_path)

if os.path.exists(http_cache_path):
    cache_size_mb = os.path.getsize(http_cache_path) / (1024 * 1024)
    logger.info(f"HTTP cache location: {http_cache_path} ({cache_size_mb:.1f} MB)")
else:
    logger.info(f"HTTP cache will be created at: {http_cache_path}")

shared_dem_cache = DEMTileCache(debug_mode=False)
shared_debug_dem_cache = DEMTileCache(debug_mode=True)
logger.info("Shared DEM caches created")

# Initialize trail finder services with shared caches
trail_finder = TrailFinderService(dem_cache=shared_dem_cache)
debug_trail_finder = TrailFinderService(debug_mode=True, dem_cache=shared_debug_dem_cache)

# Preload popular areas on startup (optional)
PRELOAD_AREAS = [
    # Park City, UT
    {
        "name": "Park City, UT",
        "bounds": {
            "min_lat": 40.5961,
            "max_lat": 40.6961,
            "min_lon": -111.5480,
            "max_lon": -111.4480
        }
    },
    # Add more areas as needed
]

async def preload_areas():
    """Preload popular areas on startup"""
    for area in PRELOAD_AREAS:
        try:
            logger.info(f"Preloading area: {area['name']}")
            bounds = area['bounds']
            
            # Download terrain
            result = trail_finder.dem_cache.predownload_area(
                bounds['min_lat'], bounds['max_lat'], 
                bounds['min_lon'], bounds['max_lon']
            )
            
            if result['status'] == 'success':
                # Preprocess
                preprocess_result = trail_finder.dem_cache.preprocess_area(
                    bounds['min_lat'], bounds['max_lat'],
                    bounds['min_lon'], bounds['max_lon']
                )
                logger.info(f"Preloaded {area['name']}: {preprocess_result['status']}")
            else:
                logger.warning(f"Failed to preload {area['name']}: {result}")
                
        except Exception as e:
            logger.error(f"Error preloading {area['name']}: {str(e)}")

# Run preloading in background after startup
@app.on_event("startup")
async def startup_event():
    """Run tasks on startup"""
    import asyncio
    # Run preloading in background so it doesn't block startup
    asyncio.create_task(preload_areas())


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
        
        # Create trail finder with user's configurations and shared cache
        profile_trail_finder = TrailFinderService(
            obstacle_config=obstacle_config,
            path_preferences=path_preferences,
            dem_cache=shared_dem_cache
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
    from fastapi.responses import Response
    from app.services.gpx_generator import GPXGenerator
    
    if route_id not in routes_storage:
        raise HTTPException(status_code=404, detail="Route not found")
    
    route = routes_storage[route_id]
    
    if route["status"] != RouteStatus.COMPLETED:
        raise HTTPException(
            status_code=400, 
            detail=f"Route is not ready. Status: {route['status']}"
        )
    
    # Get path and stats
    path = route.get("path", [])
    stats = route.get("stats", {})
    
    # Generate route name from coordinates
    if path:
        start_coord = path[0]
        end_coord = path[-1]
        route_name = f"Trail Route {start_coord.lat:.4f},{start_coord.lon:.4f} to {end_coord.lat:.4f},{end_coord.lon:.4f}"
    else:
        route_name = "Trail Route"
    
    # Create description with statistics
    description_parts = []
    if 'distance_km' in stats:
        description_parts.append(f"Distance: {stats['distance_km']:.2f} km")
    if 'elevation_gain_m' in stats:
        description_parts.append(f"Elevation gain: {stats['elevation_gain_m']} m")
    if 'max_slope' in stats:
        description_parts.append(f"Max slope: {stats['max_slope']:.1f}°")
    if 'difficulty' in stats:
        description_parts.append(f"Difficulty: {stats['difficulty']}")
    
    route_description = " | ".join(description_parts)
    
    # Check if we have path with slopes
    if 'path_with_slopes' in stats:
        gpx_content = GPXGenerator.create_gpx(
            stats['path_with_slopes'],
            route_name=route_name,
            route_description=route_description,
            stats=stats
        )
    else:
        # Fall back to simple path
        simple_path = [(coord.lon, coord.lat) for coord in path]
        gpx_content = GPXGenerator.create_simple_gpx(simple_path, route_name)
    
    # Return as downloadable file
    filename = f"trail_route_{route_id[:8]}.gpx"
    return Response(
        content=gpx_content,
        media_type="application/gpx+xml",
        headers={
            "Content-Disposition": f"attachment; filename={filename}"
        }
    )


@app.post("/api/routes/export/gpx")
async def export_route_as_gpx(request: RouteRequest):
    """Export a route directly as GPX without storing it"""
    from fastapi.responses import Response
    from app.services.gpx_generator import GPXGenerator
    
    # Get configurations based on user profile and custom options
    profile = request.options.userProfile if request.options else "default"
    obstacle_config, path_preferences = get_configs_for_profile(profile, request.options)
    
    # Create trail finder with user's configurations and shared cache
    profile_trail_finder = TrailFinderService(
        obstacle_config=obstacle_config,
        path_preferences=path_preferences,
        dem_cache=shared_dem_cache
    )
    
    # Find the route
    path, stats = await profile_trail_finder.find_route(
        request.start, 
        request.end,
        request.options.model_dump() if request.options else {}
    )
    
    if not path:
        raise HTTPException(status_code=404, detail=stats.get("error", "No route found"))
    
    # Generate route name
    route_name = f"Trail Route {request.start.lat:.4f},{request.start.lon:.4f} to {request.end.lat:.4f},{request.end.lon:.4f}"
    
    # Create description with statistics
    description_parts = []
    if 'distance_km' in stats:
        description_parts.append(f"Distance: {stats['distance_km']:.2f} km")
    if 'elevation_gain_m' in stats:
        description_parts.append(f"Elevation gain: {stats['elevation_gain_m']} m")
    if 'max_slope' in stats:
        description_parts.append(f"Max slope: {stats['max_slope']:.1f}°")
    if 'difficulty' in stats:
        description_parts.append(f"Difficulty: {stats['difficulty']}")
    
    route_description = " | ".join(description_parts)
    
    # Generate GPX
    if 'path_with_slopes' in stats:
        gpx_content = GPXGenerator.create_gpx(
            stats['path_with_slopes'],
            route_name=route_name,
            route_description=route_description,
            stats=stats
        )
    else:
        # Fall back to simple path
        simple_path = [(coord.lon, coord.lat) for coord in path]
        gpx_content = GPXGenerator.create_simple_gpx(simple_path, route_name)
    
    # Return as downloadable file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"trail_route_{timestamp}.gpx"
    
    return Response(
        content=gpx_content,
        media_type="application/gpx+xml",
        headers={
            "Content-Disposition": f"attachment; filename={filename}"
        }
    )


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
        
        # Use shared DEM cache instance
        dem_cache = shared_dem_cache
        
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


@app.post("/api/cache/predownload")
async def predownload_area(bounds: dict):
    """Pre-download terrain data for a specific area"""
    try:
        min_lat = bounds.get("minLat")
        max_lat = bounds.get("maxLat")
        min_lon = bounds.get("minLon")
        max_lon = bounds.get("maxLon")
        
        if not all([min_lat, max_lat, min_lon, max_lon]):
            raise HTTPException(status_code=400, detail="Invalid bounds")
        
        # Use the global trail_finder's cache or create a dedicated one
        result = trail_finder.dem_cache.predownload_area(
            min_lat, max_lat, min_lon, max_lon
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error pre-downloading area: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error pre-downloading area: {str(e)}")


@app.post("/api/cache/preprocess")
async def preprocess_area(bounds: dict, force: bool = False):
    """Preprocess terrain data for faster pathfinding"""
    try:
        min_lat = bounds.get("minLat")
        max_lat = bounds.get("maxLat")
        min_lon = bounds.get("minLon")
        max_lon = bounds.get("maxLon")
        
        if not all([min_lat, max_lat, min_lon, max_lon]):
            raise HTTPException(status_code=400, detail="Invalid bounds")
        
        # Use the global trail_finder's cache
        result = trail_finder.dem_cache.preprocess_area(
            min_lat, max_lat, min_lon, max_lon, force=force
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error preprocessing area: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error preprocessing area: {str(e)}")


@app.get("/api/cache/status")
async def get_cache_status():
    """Get current cache status and statistics"""
    try:
        # Get cache status from both trail finders
        normal_status = trail_finder.dem_cache.get_cache_status()
        debug_status = debug_trail_finder.dem_cache.get_cache_status()
        
        # Combine the statistics
        return {
            "normal_cache": normal_status,
            "debug_cache": debug_status,
            "combined": {
                "total_memory_mb": normal_status["total_memory_mb"] + debug_status["total_memory_mb"],
                "total_terrain_entries": normal_status["terrain_cache"]["count"] + debug_status["terrain_cache"]["count"],
                "total_cost_surface_entries": normal_status["cost_surface_cache"]["count"] + debug_status["cost_surface_cache"]["count"],
                "total_preprocessing_entries": normal_status["preprocessing_cache"]["count"] + debug_status["preprocessing_cache"]["count"]
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting cache status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting cache status: {str(e)}")


@app.post("/api/routes/debug", response_model=RouteResult)
async def calculate_debug_route(request: RouteRequest):
    """Calculate route with debug information"""
    try:
        # Get configurations based on user profile and custom options
        profile = request.options.userProfile if request.options else "default"
        obstacle_config, path_preferences = get_configs_for_profile(profile, request.options)
        
        # Create debug trail finder with user's configurations and shared debug cache
        profile_debug_finder = TrailFinderService(
            debug_mode=True, 
            obstacle_config=obstacle_config,
            path_preferences=path_preferences,
            dem_cache=shared_debug_dem_cache
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