import math
import logging
from typing import Dict, List, Tuple, Optional
from app.models.route import Coordinate
from app.services.dem_tile_cache import DEMTileCache
from app.services.obstacle_config import ObstacleConfig, ObstaclePresets
from app.services.path_preferences import PathPreferences, PathPreferencePresets

logger = logging.getLogger(__name__)


class TrailFinderService:
    """Service for finding hiking trails between coordinates"""
    
    def __init__(self, buffer: float = 0.05, debug_mode: bool = False, 
                 obstacle_config: ObstacleConfig = None, path_preferences: PathPreferences = None,
                 dem_cache: DEMTileCache = None):
        self.buffer = buffer
        self.max_distance_km = 50  # Maximum allowed distance between points
        self.debug_mode = debug_mode
        self.obstacle_config = obstacle_config
        self.path_preferences = path_preferences
        
        # Use provided cache or create a new one
        if dem_cache is not None:
            self.dem_cache = dem_cache
            logger.info(f"TrailFinderService using shared DEM cache")
        else:
            self.dem_cache = DEMTileCache(
                buffer=buffer, 
                debug_mode=debug_mode, 
                obstacle_config=obstacle_config,
                path_preferences=path_preferences
            )
            logger.info(f"TrailFinderService created new DEM cache")
        logger.info(f"TrailFinderService initialized with debug_mode={debug_mode}")
    
    def validate_route_request(self, start: Coordinate, end: Coordinate) -> bool:
        """Validate that the route request is reasonable"""
        # Check if coordinates are the same
        if start.lat == end.lat and start.lon == end.lon:
            return False
        
        # Check distance between points
        distance = self._calculate_distance(start, end)
        if distance > self.max_distance_km:
            return False
        
        return True
    
    def calculate_bounding_box(self, start: Coordinate, end: Coordinate) -> Dict[str, float]:
        """Calculate bounding box for the area of interest"""
        min_lat = min(start.lat, end.lat) - self.buffer
        max_lat = max(start.lat, end.lat) + self.buffer
        min_lon = min(start.lon, end.lon) - self.buffer
        max_lon = max(start.lon, end.lon) + self.buffer
        
        return {
            "min_lat": min_lat,
            "max_lat": max_lat,
            "min_lon": min_lon,
            "max_lon": max_lon
        }
    
    def _calculate_distance(self, coord1: Coordinate, coord2: Coordinate) -> float:
        """Calculate distance between two coordinates in kilometers using Haversine formula"""
        R = 6371  # Earth's radius in kilometers
        
        lat1, lon1 = math.radians(coord1.lat), math.radians(coord1.lon)
        lat2, lon2 = math.radians(coord2.lat), math.radians(coord2.lon)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c
    
    async def find_route(self, start: Coordinate, end: Coordinate, options: dict) -> Tuple[List[Coordinate], dict]:
        """
        Find optimal route between two coordinates
        Returns: (path_coordinates, statistics)
        """
        try:
            # Validate route request first
            if not self.validate_route_request(start, end):
                error_stats = {"error": "Invalid route request: coordinates too far apart or identical"}
                
                # Add debug data field for consistency
                if self.debug_mode:
                    error_stats["debug_data"] = None
                
                return [], error_stats
            
            # Use the DEM tile cache to find the route
            path_coords = self.dem_cache.find_route(
                start.lat, start.lon, 
                end.lat, end.lon
            )
            
            if not path_coords:
                logger.error("No route found by DEM cache")
                error_stats = {"error": "No route found"}
                
                # Add debug data even for failed routes if in debug mode
                if self.debug_mode:
                    debug_data = self.dem_cache.get_debug_data()
                    error_stats["debug_data"] = debug_data
                
                return [], error_stats
            
            # Convert path_coords with slopes to Coordinate objects
            path = []
            for point in path_coords:
                if isinstance(point, dict):
                    # New format with slope data
                    path.append(Coordinate(lat=point['lat'], lon=point['lon']))
                else:
                    # Old format (lon, lat) tuple
                    lon, lat = point
                    path.append(Coordinate(lat=lat, lon=lon))
            
            # Calculate statistics
            total_distance = 0
            total_elevation_gain = 0
            max_slope = 0
            
            for i in range(1, len(path)):
                total_distance += self._calculate_distance(path[i-1], path[i])
                
                # Calculate elevation gain from slope data if available
                if isinstance(path_coords[i-1], dict) and 'elevation' in path_coords[i-1]:
                    if i < len(path_coords) and isinstance(path_coords[i], dict) and 'elevation' in path_coords[i]:
                        # Check that both elevations are not None
                        if path_coords[i]['elevation'] is not None and path_coords[i-1]['elevation'] is not None:
                            elevation_change = path_coords[i]['elevation'] - path_coords[i-1]['elevation']
                            if elevation_change > 0:
                                total_elevation_gain += elevation_change
                        
                        if 'slope' in path_coords[i-1]:
                            max_slope = max(max_slope, abs(path_coords[i-1]['slope']))
            
            stats = {
                "distance_km": round(total_distance, 2),
                "elevation_gain_m": round(total_elevation_gain),
                "estimated_time_min": int(total_distance * 15),  # Rough estimate: 4km/h
                "difficulty": self._estimate_difficulty_with_slope(total_distance, max_slope),
                "waypoints": len(path),
                "max_slope": round(max_slope, 1)
            }
            
            # Include the path with slope data if available
            if isinstance(path_coords[0], dict):
                stats["path_with_slopes"] = path_coords
            
            # Add debug data if in debug mode
            if self.debug_mode:
                debug_data = self.dem_cache.get_debug_data()
                logger.info(f"Debug data retrieved: {debug_data is not None}")
                if debug_data:
                    stats["debug_data"] = debug_data
                    logger.info(f"Debug data added to stats: {len(debug_data.get('explored_nodes', []))} explored nodes")
                else:
                    # Ensure debug_data field exists even if empty
                    stats["debug_data"] = None
                    logger.warning("Debug mode enabled but no debug data collected")
            
            return path, stats
            
        except Exception as e:
            logger.error(f"Error finding route: {str(e)}")
            error_stats = {"error": str(e)}
            
            # Add debug data even for failed routes if in debug mode
            if self.debug_mode:
                debug_data = self.dem_cache.get_debug_data()
                error_stats["debug_data"] = debug_data
            
            return [], error_stats
    
    def _estimate_difficulty(self, distance_km: float) -> str:
        """Estimate difficulty based on distance"""
        if distance_km < 5:
            return "easy"
        elif distance_km < 10:
            return "moderate"
        else:
            return "hard"
    
    def _estimate_difficulty_with_slope(self, distance_km: float, max_slope: float) -> str:
        """Estimate difficulty based on distance and slope"""
        # Base difficulty from distance
        if distance_km < 3:
            base_difficulty = 1
        elif distance_km < 8:
            base_difficulty = 2
        else:
            base_difficulty = 3
            
        # Slope difficulty
        if max_slope < 10:
            slope_difficulty = 1
        elif max_slope < 20:
            slope_difficulty = 2
        else:
            slope_difficulty = 3
            
        # Combined difficulty
        total_difficulty = max(base_difficulty, slope_difficulty)
        
        if total_difficulty == 1:
            return "easy"
        elif total_difficulty == 2:
            return "moderate"
        else:
            return "hard"