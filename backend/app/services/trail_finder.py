import math
import logging
from typing import Dict, List, Tuple, Optional
from app.models.route import Coordinate
from app.services.dem_tile_cache import DEMTileCache

logger = logging.getLogger(__name__)


class TrailFinderService:
    """Service for finding hiking trails between coordinates"""
    
    def __init__(self, buffer: float = 0.05):
        self.buffer = buffer
        self.max_distance_km = 50  # Maximum allowed distance between points
        self.dem_cache = DEMTileCache(buffer=buffer)
    
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
            # Use the DEM tile cache to find the route
            path_coords = self.dem_cache.find_route(
                start.lat, start.lon, 
                end.lat, end.lon
            )
            
            if not path_coords:
                logger.error("No route found by DEM cache")
                return [], {"error": "No route found"}
            
            # Convert to Coordinate objects
            path = [Coordinate(lat=lat, lon=lon) for lon, lat in path_coords]
            
            # Calculate statistics
            total_distance = 0
            for i in range(1, len(path)):
                total_distance += self._calculate_distance(path[i-1], path[i])
            
            stats = {
                "distance_km": round(total_distance, 2),
                "elevation_gain_m": 0,  # TODO: Calculate from DEM data
                "estimated_time_min": int(total_distance * 15),  # Rough estimate: 4km/h
                "difficulty": self._estimate_difficulty(total_distance),
                "waypoints": len(path)
            }
            
            return path, stats
            
        except Exception as e:
            logger.error(f"Error finding route: {str(e)}")
            return [], {"error": str(e)}
    
    def _estimate_difficulty(self, distance_km: float) -> str:
        """Estimate difficulty based on distance"""
        if distance_km < 5:
            return "easy"
        elif distance_km < 10:
            return "moderate"
        else:
            return "hard"