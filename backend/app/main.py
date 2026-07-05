import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Dict

import numpy as np
from app.engine_v2.service import TrailFinderServiceV2
from app.models.eval import EvalCase, ScoredPath, ScorePathRequest
from app.models.route import (
    RouteRequest,
    RouteResponse,
    RouteResult,
    RouteStatus,
    RouteStatusResponse,
    RouteVariantsRequest,
)
from app.services.dem_tile_cache import DEMTileCache
from app.services.eval_store import EvalStore
from app.services.obstacle_config import ObstaclePresets
from app.services.path_preferences import PathPreferencePresets, PathPreferences
from app.services.trail_finder import TrailFinderService
from fastapi import BackgroundTasks, FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Trail Finder API", description="API for finding optimal hiking trails", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:9002",
        "http://localhost:5173",
        "http://localhost:5174",
    ],  # Frontend dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for now (will be replaced with Redis)
routes_storage: Dict[str, dict] = {}

# File-backed store for saved eval cases. os.path.dirname(__file__) is
# .../backend/app, so ../../evals is the repo-root `evals/` directory.
eval_store = EvalStore(os.environ.get("EVAL_DIR") or os.path.join(os.path.dirname(__file__), "..", "..", "evals"))

# Create shared DEM cache for all trail finder instances
logger.info("Creating shared DEM caches...")

# Log HTTP cache location
http_cache_path = os.environ.get("HYRIVER_CACHE_NAME", os.path.abspath("cache/aiohttp_cache.sqlite"))
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

# Initialize v2 engine service (cheap construction: creates dirs, no downloads)
trail_finder_v2 = TrailFinderServiceV2()

# Preload popular areas on startup (optional)
PRELOAD_AREAS = [
    # Park City, UT
    {
        "name": "Park City, UT",
        "bounds": {"min_lat": 40.5961, "max_lat": 40.6961, "min_lon": -111.5480, "max_lon": -111.4480},
    },
    # Add more areas as needed
]


async def preload_areas():
    """Preload popular areas on startup"""
    for area in PRELOAD_AREAS:
        try:
            logger.info(f"Preloading area: {area['name']}")
            bounds = area["bounds"]

            # Download terrain
            result = trail_finder.dem_cache.predownload_area(
                bounds["min_lat"], bounds["max_lat"], bounds["min_lon"], bounds["max_lon"]
            )

            if result["status"] == "success":
                # Preprocess
                preprocess_result = trail_finder.dem_cache.preprocess_area(
                    bounds["min_lat"], bounds["max_lat"], bounds["min_lon"], bounds["max_lon"]
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
    logger.info("=" * 60)
    logger.info("Trail Finder API Starting...")
    logger.info("=" * 60)

    # Disable preloading for faster startup
    # import asyncio
    # asyncio.create_task(preload_areas())

    logger.info("✓ CORS middleware configured")
    logger.info("✓ Trail finder services initialized")
    logger.info("✓ DEM caches ready")
    logger.info("=" * 60)
    logger.info("🚀 API READY - Listening on port 9001")
    logger.info("=" * 60)


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
        "city_walker": "city_walker",  # Direct mapping
        "wheelchair": "wheelchair",  # Direct mapping
    }
    obstacle_config.slope_profile = slope_profile_map.get(profile, "default")

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
                path_preferences.path_costs["footway"] = custom.footway
            if custom.path is not None:
                path_preferences.path_costs["path"] = custom.path
            if custom.trail is not None:
                path_preferences.path_costs["trail"] = custom.trail
            if custom.residential is not None:
                path_preferences.path_costs["residential"] = custom.residential
            if custom.off_path is not None:
                path_preferences.path_costs["off_path"] = custom.off_path
            logger.info("Applied custom path costs")

    # Apply gradient and trail preferences if provided
    if options:
        if hasattr(options, "gradientPreference"):
            obstacle_config.gradient_preference = options.gradientPreference
            logger.info(f"Applied gradient preference: {options.gradientPreference}")
        if hasattr(options, "trailPreference"):
            path_preferences.trail_preference = options.trailPreference
            logger.info(f"Applied trail preference: {options.trailPreference}")

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
        engine = request.options.engine if request.options else "v2"
        points = request.normalized_points()
        options_dict = request.options.model_dump() if request.options else {}

        if engine == "v2":
            routes_storage[route_id]["progress"] = 30
            if len(points) > 2:
                path, stats = await trail_finder_v2.find_multi_route(points, options_dict)
            else:
                path, stats = await trail_finder_v2.find_route(points[0], points[-1], options_dict)
        else:
            # Multi-waypoint routing is a v2-only feature.
            if len(points) > 2:
                routes_storage[route_id]["status"] = RouteStatus.FAILED
                routes_storage[route_id][
                    "message"
                ] = "Multi-waypoint routing requires engine v2; v1 supports start/end only."
                return

            obstacle_config, path_preferences = get_configs_for_profile(profile, request.options)

            # Create trail finder with user's configurations and shared cache
            profile_trail_finder = TrailFinderService(
                obstacle_config=obstacle_config, path_preferences=path_preferences, dem_cache=shared_dem_cache
            )

            # Validate request
            if not profile_trail_finder.validate_route_request(points[0], points[-1]):
                routes_storage[route_id]["status"] = RouteStatus.FAILED
                routes_storage[route_id]["message"] = "Invalid route request"
                return

            routes_storage[route_id]["progress"] = 30

            # Find the route
            path, stats = await profile_trail_finder.find_route(points[0], points[-1], options_dict)

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
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    # Start background processing
    background_tasks.add_task(process_route, route_id, request)

    return RouteResponse(routeId=route_id, status=RouteStatus.PROCESSING)


@app.post("/api/routes/variants")
async def route_variants(request: RouteVariantsRequest):
    """Route the same start/end at several hiker expertise levels in one call
    (v2 engine, DEM loaded once). Returns one variant per level — a family of
    options from gentle/turny to direct/committing — with identical lines
    marked ``duplicateOf`` so clients draw each distinct route once.
    """
    options = request.options.model_dump() if request.options else {}
    try:
        variants = await trail_finder_v2.find_route_variants(request.start, request.end, options, request.levels)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error computing route variants: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="internal error computing route variants")
    return {"variants": variants, "count": len(variants)}


@app.get("/api/routes/{route_id}/status", response_model=RouteStatusResponse)
async def get_route_status(route_id: str):
    """Get route calculation status"""
    if route_id not in routes_storage:
        raise HTTPException(status_code=404, detail="Route not found")

    route = routes_storage[route_id]
    return RouteStatusResponse(status=route["status"], progress=route["progress"], message=route.get("message"))


@app.get("/api/routes/{route_id}", response_model=RouteResult)
async def get_route(route_id: str):
    """Get calculated route"""
    if route_id not in routes_storage:
        raise HTTPException(status_code=404, detail="Route not found")

    route = routes_storage[route_id]

    if route["status"] != RouteStatus.COMPLETED:
        raise HTTPException(status_code=400, detail=f"Route is not ready. Status: {route['status']}")

    return RouteResult(
        routeId=route_id,
        status=route["status"],
        path=route.get("path", []),
        stats=route.get("stats", {}),
        createdAt=route["created_at"],
    )


@app.get("/api/routes/{route_id}/gpx")
async def download_gpx(route_id: str):
    """Download route as GPX file"""
    from app.services.gpx_generator import GPXGenerator
    from fastapi.responses import Response

    if route_id not in routes_storage:
        raise HTTPException(status_code=404, detail="Route not found")

    route = routes_storage[route_id]

    if route["status"] != RouteStatus.COMPLETED:
        raise HTTPException(status_code=400, detail=f"Route is not ready. Status: {route['status']}")

    # Get path and stats
    path = route.get("path", [])
    stats = route.get("stats", {})

    # Generate route name from coordinates
    if path:
        start_coord = path[0]
        end_coord = path[-1]
        route_name = (
            f"Trail Route {start_coord.lat:.4f},{start_coord.lon:.4f} to {end_coord.lat:.4f},{end_coord.lon:.4f}"
        )
    else:
        route_name = "Trail Route"

    # Create description with statistics
    description_parts = []
    if "distance_km" in stats:
        description_parts.append(f"Distance: {stats['distance_km']:.2f} km")
    if "elevation_gain_m" in stats:
        description_parts.append(f"Elevation gain: {stats['elevation_gain_m']} m")
    if "max_slope" in stats:
        description_parts.append(f"Max slope: {stats['max_slope']:.1f}°")
    if "difficulty" in stats:
        description_parts.append(f"Difficulty: {stats['difficulty']}")

    route_description = " | ".join(description_parts)

    # Check if we have path with slopes
    if "path_with_slopes" in stats:
        gpx_content = GPXGenerator.create_gpx(
            stats["path_with_slopes"], route_name=route_name, route_description=route_description, stats=stats
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
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.post("/api/routes/export/gpx")
async def export_route_as_gpx(request: RouteRequest):
    """Export a route directly as GPX without storing it"""
    from app.services.gpx_generator import GPXGenerator
    from fastapi.responses import Response

    # Get configurations based on user profile and custom options
    profile = request.options.userProfile if request.options else "default"
    obstacle_config, path_preferences = get_configs_for_profile(profile, request.options)

    # Create trail finder with user's configurations and shared cache
    profile_trail_finder = TrailFinderService(
        obstacle_config=obstacle_config, path_preferences=path_preferences, dem_cache=shared_dem_cache
    )

    # Find the route (v2 stitches multi-point; v1/legacy uses first & last).
    points = request.normalized_points()
    options_dict = request.options.model_dump() if request.options else {}
    engine = request.options.engine if request.options else "v2"
    if engine != "v2" and len(points) > 2:
        raise HTTPException(
            status_code=400,
            detail="Multi-waypoint GPX export requires engine v2; v1 supports start/end only.",
        )
    if engine == "v2" and len(points) > 2:
        path, stats = await trail_finder_v2.find_multi_route(points, options_dict)
    else:
        path, stats = await profile_trail_finder.find_route(points[0], points[-1], options_dict)

    if not path:
        raise HTTPException(status_code=404, detail=stats.get("error", "No route found"))

    # Generate route name
    route_name = f"Trail Route {points[0].lat:.4f},{points[0].lon:.4f} to {points[-1].lat:.4f},{points[-1].lon:.4f}"

    # Create description with statistics
    description_parts = []
    if "distance_km" in stats:
        description_parts.append(f"Distance: {stats['distance_km']:.2f} km")
    if "elevation_gain_m" in stats:
        description_parts.append(f"Elevation gain: {stats['elevation_gain_m']} m")
    if "max_slope" in stats:
        description_parts.append(f"Max slope: {stats['max_slope']:.1f}°")
    if "difficulty" in stats:
        description_parts.append(f"Difficulty: {stats['difficulty']}")

    route_description = " | ".join(description_parts)

    # Generate GPX
    if "path_with_slopes" in stats:
        gpx_content = GPXGenerator.create_gpx(
            stats["path_with_slopes"], route_name=route_name, route_description=route_description, stats=stats
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
        headers={"Content-Disposition": f"attachment; filename={filename}"},
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
            "bounds": {"minLat": min_lat, "maxLat": max_lat, "minLon": min_lon, "maxLon": max_lon},
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
        result = trail_finder.dem_cache.predownload_area(min_lat, max_lat, min_lon, max_lon)

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
        result = trail_finder.dem_cache.preprocess_area(min_lat, max_lat, min_lon, max_lon, force=force)

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
                "total_terrain_entries": normal_status["terrain_cache"]["count"]
                + debug_status["terrain_cache"]["count"],
                "total_cost_surface_entries": normal_status["cost_surface_cache"]["count"]
                + debug_status["cost_surface_cache"]["count"],
                "total_preprocessing_entries": normal_status["preprocessing_cache"]["count"]
                + debug_status["preprocessing_cache"]["count"],
            },
        }

    except Exception as e:
        logger.error(f"Error getting cache status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting cache status: {str(e)}")


@app.post("/api/cache/prepopulate-box")
async def prepopulate_bounding_box(corners: dict):
    """
    Prepopulate cache for a bounding box area defined by two corner points.

    Request body:
    {
        "corner1": {"lat": 40.65, "lon": -111.57},
        "corner2": {"lat": 40.66, "lon": -111.56}
    }
    """
    try:
        # Extract corners
        corner1 = corners.get("corner1")
        corner2 = corners.get("corner2")

        if not corner1 or not corner2:
            raise HTTPException(status_code=400, detail="Both corner1 and corner2 are required")

        lat1 = corner1.get("lat")
        lon1 = corner1.get("lon")
        lat2 = corner2.get("lat")
        lon2 = corner2.get("lon")

        if None in [lat1, lon1, lat2, lon2]:
            raise HTTPException(status_code=400, detail="Invalid corner coordinates")

        # Calculate bounding box
        min_lat = min(lat1, lat2)
        max_lat = max(lat1, lat2)
        min_lon = min(lon1, lon2)
        max_lon = max(lon1, lon2)

        # Calculate area size
        lat_diff = max_lat - min_lat
        lon_diff = max_lon - min_lon
        area_km2 = lat_diff * 111 * lon_diff * 111 * 0.7

        logger.info(
            f"Prepopulating area: ({min_lat:.4f}, {min_lon:.4f}) to ({max_lat:.4f}, {max_lon:.4f}), ~{area_km2:.1f} km²"
        )

        # Get initial status
        initial_status = trail_finder.dem_cache.get_cache_status()

        # Step 1: Download terrain data
        result = trail_finder.dem_cache.predownload_area(min_lat, max_lat, min_lon, max_lon)
        if result["status"] != "success":
            raise HTTPException(status_code=500, detail=f"Failed to download terrain: {result}")

        # Step 2: Preprocess the area (compute cost surfaces)
        preprocess_result = trail_finder.dem_cache.preprocess_area(min_lat, max_lat, min_lon, max_lon, force=False)

        # Get final status
        final_status = trail_finder.dem_cache.get_cache_status()

        # Calculate what was added
        terrain_added = final_status["terrain_cache"]["count"] - initial_status["terrain_cache"]["count"]
        cost_added = final_status["cost_surface_cache"]["count"] - initial_status["cost_surface_cache"]["count"]
        memory_added = final_status["total_memory_mb"] - initial_status["total_memory_mb"]

        return {
            "status": "success",
            "area": {
                "min_lat": min_lat,
                "max_lat": max_lat,
                "min_lon": min_lon,
                "max_lon": max_lon,
                "area_km2": round(area_km2, 1),
            },
            "cache_growth": {
                "terrain_entries_added": terrain_added,
                "cost_surfaces_added": cost_added,
                "memory_added_mb": round(memory_added, 1),
            },
            "final_cache_status": final_status,
            "download_result": result,
            "preprocess_result": preprocess_result,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error prepopulating area: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error prepopulating area: {str(e)}")


@app.post("/api/eval/score-path", response_model=ScoredPath)
async def score_path_endpoint(request: ScorePathRequest):
    """Score an arbitrary polyline with the engine's cost function.

    Returns per-segment cost attribution so the Eval UI can explain why a
    drawn path costs more (or less) than the engine's optimal route. Both are
    scored the same way under the same options, so their costs are comparable.
    """
    if len(request.path) < 2:
        raise HTTPException(status_code=400, detail="path needs at least two points")
    options = request.options.model_dump()
    path = request.path
    snapped = False
    if request.snap == "trail":
        try:
            path, snapped = await trail_finder_v2.snap_to_trails(path, options)
        except Exception as e:
            # Snapping is best-effort; fall back to scoring exactly what was drawn.
            logger.warning(f"snap-to-trail failed, scoring drawn path: {e}")
    try:
        result = await trail_finder_v2.score_path(path, options)
        if snapped:
            result.snapped = True
        return result
    except HTTPException:
        raise
    except ValueError as e:
        # Rejected input (too many points, extent too large, degenerate path).
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error scoring path: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="internal error scoring path")


@app.get("/api/eval/trails")
async def eval_trails_endpoint(south: float, west: float, north: float, east: float):
    """Trail/path geometry the engine routes on (OSM highway=* ways) within a
    viewport, for a display overlay. Returns lists of [lat, lon] polylines from
    the cached OSM tiles; empty where nothing is cached."""
    try:
        lines = await trail_finder_v2.trail_lines_in_bounds(south, west, north, east)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error loading trails: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="internal error loading trails")
    return {"lines": lines, "count": len(lines)}


@app.get("/api/eval/cases", response_model=list[EvalCase])
async def list_eval_cases():
    """List all saved eval cases, sorted by id."""
    return eval_store.list()


@app.post("/api/eval/cases", response_model=EvalCase)
async def save_eval_case(case: EvalCase):
    """Create or replace a saved eval case (keyed by its id)."""
    try:
        return eval_store.save(case)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/api/eval/cases/{case_id}", status_code=204)
async def delete_eval_case(case_id: str):
    """Delete a saved eval case. Idempotent: 204 even if it did not exist."""
    eval_store.delete(case_id)
    return Response(status_code=204)


@app.post("/api/terrain/cost-point")
async def get_cost_at_point(request: dict):
    """
    Get the cost information for a single point.
    Much faster than generating entire cost surface.
    """
    try:
        lat = request.get("lat")
        lon = request.get("lon")

        if lat is None or lon is None:
            raise HTTPException(status_code=400, detail="lat and lon are required")

        # Create DEM cache with default settings
        dem_cache = DEMTileCache(
            obstacle_config=ObstaclePresets.experienced_hiker(), path_preferences=PathPreferencePresets.trail_seeker()
        )

        # Get cost data for a small area around the point
        buffer = 0.0001  # Very small buffer ~11 meters
        result = dem_cache.get_cost_at_point(lat, lon, buffer)

        if result is None:
            raise HTTPException(status_code=500, detail="Failed to get cost data")

        # Check if it's an error response
        if "error" in result and not result.get("precomputed", True):
            # Return 404 to indicate data not available
            raise HTTPException(
                status_code=404, detail=result["error"], headers={"X-Tile": result.get("tile", "unknown")}
            )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting cost at point: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/terrain/osm-data")
async def get_osm_data_at_point(request: dict):
    """
    Get raw OSM data for a specific point.
    Separate endpoint to avoid slowing down the main cost-point endpoint.
    """
    try:
        import osmnx as ox
        import pandas as pd
        from shapely.geometry import Point, box

        lat = request.get("lat")
        lon = request.get("lon")

        if lat is None or lon is None:
            raise HTTPException(status_code=400, detail="lat and lon are required")

        # Create a small bounding box around the point
        bbox_buffer = 0.0002  # About 22 meters - increased for better coverage
        bbox = box(lon - bbox_buffer, lat - bbox_buffer, lon + bbox_buffer, lat + bbox_buffer)

        # Get path preferences to know what tags to search for
        path_preferences = PathPreferencePresets.trail_seeker()

        # Fetch OSM features at this location
        from app.services.osm_settings import apply_osm_settings

        apply_osm_settings(ox)
        ox.settings.log_console = False
        logger.info(f"Fetching OSM data for ({lat}, {lon}) with buffer {bbox_buffer}")
        features = ox.features_from_polygon(bbox, path_preferences.preferred_path_tags)

        raw_osm_data = []
        logger.info(f"Found {len(features)} OSM features")

        if not features.empty:
            # Get features that contain or are very close to our point
            point = Point(lon, lat)

            for idx, feature in features.iterrows():
                # More lenient distance check - within about 10 meters
                if feature.geometry.contains(point) or feature.geometry.distance(point) < 0.0001:
                    # Extract all tags
                    tags = {}
                    for col in features.columns:
                        if col != "geometry" and pd.notna(feature[col]) and not col.startswith("osm"):
                            tags[col] = str(feature[col])

                    # Determine our interpretation
                    path_type = "off_path"
                    if "highway" in tags:
                        path_type = tags["highway"]
                    elif "leisure" in tags:
                        path_type = tags["leisure"]
                    elif "landuse" in tags:
                        path_type = tags["landuse"]
                    elif "natural" in tags:
                        natural_type = tags["natural"]
                        if natural_type in ["grassland", "heath"]:
                            path_type = "grass"
                        elif natural_type == "meadow":
                            path_type = "meadow"
                        elif natural_type in ["beach", "sand"]:
                            path_type = "beach"
                        else:
                            path_type = natural_type
                    elif "piste:type" in tags:
                        path_type = tags["piste:type"]

                    raw_osm_data.append(
                        {
                            "osm_id": str(idx),
                            "tags": tags,
                            "interpreted_type": path_type,
                            "cost_multiplier": path_preferences.get_path_cost_multiplier(path_type),
                        }
                    )

        return {"lat": lat, "lon": lon, "features_found": len(raw_osm_data), "osm_data": raw_osm_data}

    except Exception as e:
        logger.error(f"Error getting OSM data: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/terrain/cost-surface")
async def get_cost_surface(request: dict):
    """
    Get the cost surface data for visualization.
    Can be called with either:
    - start/end coordinates (for route-based view)
    - bounds only (for general exploration)
    """
    try:
        # Check if this is a bounds-only request
        if "bounds" in request and not request.get("start"):
            bounds = request["bounds"]
            min_lat = bounds["south"]
            max_lat = bounds["north"]
            min_lon = bounds["west"]
            max_lon = bounds["east"]

            # Use center point as placeholder for start/end
            center_lat = (min_lat + max_lat) / 2
            center_lon = (min_lon + max_lon) / 2
            start_lat = center_lat
            start_lon = center_lon
            end_lat = center_lat
            end_lon = center_lon
        else:
            # Traditional route-based request
            start = request["start"]
            end = request["end"]
            start_lat = start["lat"]
            start_lon = start["lon"]
            end_lat = end["lat"]
            end_lon = end["lon"]

            # Define area of interest around route
            min_lat = min(start_lat, end_lat) - 0.001
            max_lat = max(start_lat, end_lat) + 0.001
            min_lon = min(start_lon, end_lon) - 0.001
            max_lon = max(start_lon, end_lon) + 0.001

        # Create DEM cache with default settings
        dem_cache = DEMTileCache(
            obstacle_config=ObstaclePresets.experienced_hiker(), path_preferences=PathPreferencePresets.trail_seeker()
        )

        # Get the cost surface data
        cost_data = dem_cache.get_cost_surface_for_bounds(
            min_lat, max_lat, min_lon, max_lon, start_lat, start_lon, end_lat, end_lon
        )

        if cost_data is None:
            raise HTTPException(status_code=500, detail="Failed to generate cost surface")

        return cost_data

    except Exception as e:
        logger.error(f"Error generating cost surface: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


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
            dem_cache=shared_debug_dem_cache,
        )

        # Find the route with debug mode enabled
        path, stats = await profile_debug_finder.find_route(
            request.start, request.end, request.options.model_dump() if request.options else {}
        )

        if not path:
            # Return failed route with debug data if available
            return RouteResult(
                routeId="debug",
                status=RouteStatus.FAILED,
                path=[],
                stats=stats,
                createdAt=datetime.now(timezone.utc).isoformat(),
            )

        # Return successful route
        return RouteResult(
            routeId="debug",
            status=RouteStatus.COMPLETED,
            path=path,
            stats=stats,
            createdAt=datetime.now(timezone.utc).isoformat(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating debug route: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


# --- Static frontend (single-origin serving) ------------------------------
# If FRONTEND_DIST points at a built Vite `dist/` directory, serve the SPA
# from the same origin as the API so one host/tunnel can serve both. Declared
# at the END of the module ON PURPOSE: the catch-all GET "/{full_path}" would
# otherwise shadow any /api GET route registered after it. Only enabled when
# BOTH index.html and assets/ exist, so a mis-pointed FRONTEND_DIST degrades to
# API-only instead of raising at import (StaticFiles) or 500ing per request.
_frontend_dist = os.environ.get("FRONTEND_DIST")
_frontend_index = os.path.join(_frontend_dist, "index.html") if _frontend_dist else None
_frontend_assets = os.path.join(_frontend_dist, "assets") if _frontend_dist else None
if _frontend_dist and os.path.isfile(_frontend_index or "") and os.path.isdir(_frontend_assets or ""):
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    app.mount("/assets", StaticFiles(directory=_frontend_assets), name="assets")
    _dist_root = os.path.realpath(_frontend_dist)

    @app.get("/", include_in_schema=False)
    async def _serve_index():
        return FileResponse(_frontend_index)

    @app.get("/{full_path:path}", include_in_schema=False)
    async def _serve_spa(full_path: str):
        # Serve a real file if present, else SPA-fallback to index.html.
        # Contain the resolved path within the dist root to block traversal
        # (this handler is reachable over a public tunnel).
        candidate = os.path.realpath(os.path.join(_frontend_dist, full_path))
        if (candidate == _dist_root or candidate.startswith(_dist_root + os.sep)) and os.path.isfile(candidate):
            return FileResponse(candidate)
        return FileResponse(_frontend_index)

    logger.info(f"Serving frontend SPA from: {_frontend_dist}")
elif _frontend_dist:
    logger.warning("FRONTEND_DIST=%s missing index.html or assets/; API-only mode", _frontend_dist)
else:
    logger.info("FRONTEND_DIST not set; API-only mode")


if __name__ == "__main__":
    import uvicorn

    # Host/port configurable via env; default to loopback (safe default — set
    # HOST=0.0.0.0 to expose on the LAN). Frontend dev historically expects
    # port 9001: `python -m uvicorn app.main:app --reload --port 9001`.
    _host = os.environ.get("HOST", "127.0.0.1")
    _port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host=_host, port=_port)
