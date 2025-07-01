import math
from typing import Dict, List, Tuple, Optional
from app.models.route import Coordinate


class TrailFinderService:
    """Service for finding hiking trails between coordinates"""
    
    def __init__(self, buffer: float = 0.05):
        self.buffer = buffer
        self.max_distance_km = 50  # Maximum allowed distance between points
    
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
        # TODO: Integrate with t7_fixed.py logic
        # For now, return a mock route
        path = [
            start,
            Coordinate(lat=(start.lat + end.lat) / 2, lon=(start.lon + end.lon) / 2),
            end
        ]
        
        stats = {
            "distance_km": self._calculate_distance(start, end),
            "elevation_gain_m": 150,
            "estimated_time_min": 45,
            "difficulty": "moderate"
        }
        
        return path, stats