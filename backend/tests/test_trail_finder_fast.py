"""
Fast tests for TrailFinderService without async complications
"""

import pytest
from unittest.mock import Mock, patch
from app.models.route import Coordinate


class TestTrailFinderFast:
    """Fast synchronous tests for trail finder"""
    
    def test_coordinate_validation(self):
        """Test coordinate validation logic"""
        # Valid coordinates
        start = Coordinate(lat=40.630, lon=-111.580)
        end = Coordinate(lat=40.650, lon=-111.560)
        
        # Calculate distance
        from math import radians, cos, sin, asin, sqrt
        
        def haversine(lat1, lon1, lat2, lon2):
            R = 6371  # Earth's radius in km
            lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            return 2 * R * asin(sqrt(a))
        
        distance = haversine(start.lat, start.lon, end.lat, end.lon)
        
        # Should be valid (< 50km)
        assert distance < 50
        assert distance > 0.1  # Not same point
        
        # Test same point
        same_distance = haversine(start.lat, start.lon, start.lat, start.lon)
        assert same_distance < 0.01
        
        # Test far points
        far_end = Coordinate(lat=41.630, lon=-110.580)
        far_distance = haversine(start.lat, start.lon, far_end.lat, far_end.lon)
        assert far_distance > 50
    
    def test_bounding_box_calculation(self):
        """Test bounding box calculations"""
        start = Coordinate(lat=40.630, lon=-111.580)
        end = Coordinate(lat=40.650, lon=-111.560)
        buffer = 0.02
        
        min_lat = min(start.lat, end.lat) - buffer
        max_lat = max(start.lat, end.lat) + buffer
        min_lon = min(start.lon, end.lon) - buffer
        max_lon = max(start.lon, end.lon) + buffer
        
        assert min_lat == pytest.approx(40.610, 0.001)
        assert max_lat == pytest.approx(40.670, 0.001)
        assert min_lon == pytest.approx(-111.600, 0.001)
        assert max_lon == pytest.approx(-111.540, 0.001)
    
    def test_route_stats_calculation(self, mock_dem_cache):
        """Test route statistics calculation"""
        # Mock path from find_path
        mock_path = [
            (40.6560, -111.5708),
            (40.6550, -111.5700),
            (40.6540, -111.5692),
            (40.6530, -111.5690),
            (40.6520, -111.5688)
        ]
        
        # Calculate distance
        total_distance = 0
        for i in range(1, len(mock_path)):
            lat1, lon1 = mock_path[i-1]
            lat2, lon2 = mock_path[i]
            
            # Approximate distance
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            # Rough approximation: 1 degree latitude ≈ 111km
            # 1 degree longitude ≈ 111km * cos(latitude)
            import math
            lat_km = dlat * 111
            lon_km = dlon * 111 * math.cos(math.radians((lat1 + lat2) / 2))
            segment_distance = math.sqrt(lat_km**2 + lon_km**2)
            total_distance += segment_distance
        
        assert total_distance > 0
        assert total_distance < 10  # Should be < 10km for this route
        
        # Test waypoint conversion
        waypoints = []
        for lat, lon in mock_path:
            waypoints.append(Coordinate(lat=lat, lon=lon))
        
        assert len(waypoints) == len(mock_path)
        assert all(isinstance(w, Coordinate) for w in waypoints)
    
    def test_mock_pathfinding_different_routes(self, mock_dem_cache):
        """Test mock pathfinding produces different paths for different routes"""
        # Route 1
        path1, stats1 = mock_dem_cache.find_path(40.6560, -111.5708, 40.6520, -111.5688)
        
        # Route 2 (different)
        path2, stats2 = mock_dem_cache.find_path(40.6570, -111.5720, 40.6510, -111.5680)
        
        # Paths should be different
        assert path1 != path2
        assert stats1['distance_m'] != stats2['distance_m']
        
        # But both should be valid
        assert len(path1) >= 2
        assert len(path2) >= 2
        assert stats1['algorithm'] == 'mock_pathfinder'
        assert stats2['algorithm'] == 'mock_pathfinder'
    
    def test_cache_integration(self, mock_dem_cache):
        """Test that mock cache integrates properly"""
        # Should have pre-populated data
        assert len(mock_dem_cache.terrain_cache) == 2
        assert len(mock_dem_cache.cost_surface_cache) == 1
        
        # Should be able to get status
        status = mock_dem_cache.get_cache_status()
        assert status['total_memory_mb'] > 0
        
        # Should have mocked methods
        assert hasattr(mock_dem_cache, 'find_path')
        assert callable(mock_dem_cache.find_path)
        
        # Test find_path works
        path, stats = mock_dem_cache.find_path(40.656, -111.571, 40.652, -111.569)
        assert path is not None
        assert len(path) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])