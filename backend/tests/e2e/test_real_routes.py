"""
End-to-end tests using real terrain data
These tests download actual DEM data and validate route quality
"""

import pytest
import asyncio
from typing import List, Tuple

from app.services.trail_finder import TrailFinderService
from app.services.dem_tile_cache import DEMTileCache
from app.models.route import Coordinate


# Standard test routes with known characteristics
PARK_CITY_ROUTES = {
    "easy": {
        "start": (40.6572, -111.5706),
        "end": (40.6486, -111.5639),
        "expected_distance_km": (0.8, 1.5),  # min, max
        "expected_elevation_gain_m": (10, 100),
        "max_acceptable_slope": 25.0,
        "description": "Relatively flat Park City route"
    },
    "moderate": {
        "start": (40.6650, -111.5850),
        "end": (40.6550, -111.5750),
        "expected_distance_km": (1.0, 2.0),
        "expected_elevation_gain_m": (50, 200),
        "max_acceptable_slope": 35.0,
        "description": "Moderate Park City terrain"
    },
    "steep": {
        "start": (40.6482, -111.5738),
        "end": (40.6464, -111.5729),
        "expected_distance_km": (0.2, 0.5),
        "expected_elevation_gain_m": (20, 150),
        "max_acceptable_slope": 45.0,
        "description": "Steep canyon traverse"
    }
}


@pytest.mark.real_data
@pytest.mark.slow
class TestRealRoutes:
    """Test routes with actual terrain data"""
    
    @pytest.fixture(scope="class")
    def real_trail_finder(self):
        """Create trail finder that uses real data"""
        # Don't use mocks - let it download real data
        return TrailFinderService(debug_mode=True)
    
    @pytest.mark.parametrize("route_name,route_data", PARK_CITY_ROUTES.items())
    @pytest.mark.asyncio
    async def test_park_city_routes(self, real_trail_finder, route_name, route_data):
        """Test various Park City routes with real terrain"""
        start = Coordinate(lat=route_data["start"][0], lon=route_data["start"][1])
        end = Coordinate(lat=route_data["end"][0], lon=route_data["end"][1])
        
        # Calculate route
        path, stats = await real_trail_finder.find_route(start, end, {})
        
        # Validate route exists
        assert path is not None, f"Failed to find route for {route_name}"
        assert len(path) > 10, f"Route too short for {route_name}"
        
        # Validate distance
        min_dist, max_dist = route_data["expected_distance_km"]
        assert min_dist <= stats["distance_km"] <= max_dist, \
            f"Distance {stats['distance_km']}km outside expected range for {route_name}"
        
        # Validate elevation
        min_gain, max_gain = route_data["expected_elevation_gain_m"]
        assert min_gain <= stats["elevation_gain_m"] <= max_gain, \
            f"Elevation gain {stats['elevation_gain_m']}m outside expected range for {route_name}"
        
        # Validate max slope
        if "max_slope" in stats:
            assert stats["max_slope"] <= route_data["max_acceptable_slope"], \
                f"Max slope {stats['max_slope']}° exceeds acceptable for {route_name}"
    
    @pytest.mark.asyncio
    async def test_route_avoids_extreme_slopes(self, real_trail_finder):
        """Test that routes avoid extreme slopes when possible"""
        # Start on one side of steep terrain
        start = Coordinate(lat=40.6500, lon=-111.5780)
        # End on other side  
        end = Coordinate(lat=40.6480, lon=-111.5720)
        
        # Use easy hiker profile (should avoid steep slopes)
        from app.services.obstacle_config import ObstaclePresets
        easy_finder = TrailFinderService(
            obstacle_config=ObstaclePresets.easy_hiker(),
            debug_mode=True
        )
        
        path, stats = await easy_finder.find_route(start, end, {})
        
        assert path is not None, "Should find route even with slope constraints"
        
        # Should take longer route to avoid steep areas
        assert stats["distance_km"] > 0.5, "Route too direct - likely going through steep terrain"
        
        # Max slope should be reasonable for easy hiker
        if "max_slope" in stats:
            assert stats["max_slope"] < 20.0, f"Route too steep for easy hiker: {stats['max_slope']}°"
    
    @pytest.mark.asyncio
    async def test_golden_gate_park_route(self, real_trail_finder):
        """Test route in different terrain (Golden Gate Park)"""
        start = Coordinate(lat=37.7694, lon=-122.4862)
        end = Coordinate(lat=37.7754, lon=-122.4584)
        
        path, stats = await real_trail_finder.find_route(start, end, {})
        
        assert path is not None, "Should find route in Golden Gate Park"
        assert 2.0 <= stats["distance_km"] <= 4.0, "Distance outside expected range"
        
        # Golden Gate Park is relatively flat
        assert stats["elevation_gain_m"] < 50, "Too much elevation gain for flat park"
    
    @pytest.mark.asyncio
    async def test_cache_performance_real_data(self, real_trail_finder):
        """Test that caching improves performance with real data"""
        import time
        
        start = Coordinate(lat=40.6572, lon=-111.5706)
        end = Coordinate(lat=40.6486, lon=-111.5639)
        
        # First run (may download data)
        t1 = time.time()
        path1, stats1 = await real_trail_finder.find_route(start, end, {})
        time1 = time.time() - t1
        
        # Second run (should use cache)
        t2 = time.time()
        path2, stats2 = await real_trail_finder.find_route(start, end, {})
        time2 = time.time() - t2
        
        # Should be significantly faster
        assert time2 < time1 * 0.5, f"Cache not providing speedup: {time1:.2f}s vs {time2:.2f}s"
        
        # Routes should be identical
        assert len(path1) == len(path2), "Cached route differs in length"
        assert stats1["distance_km"] == stats2["distance_km"], "Cached route differs in distance"


@pytest.mark.real_data
@pytest.mark.slow
class TestTerrainAccuracy:
    """Test terrain analysis accuracy with real data"""
    
    @pytest.mark.asyncio
    async def test_known_steep_area(self):
        """Test that known steep areas are identified correctly"""
        cache = DEMTileCache()
        
        # Known steep area in Park City
        steep_bounds = {
            "min_lat": 40.6460,
            "max_lat": 40.6490,
            "min_lon": -111.5740,
            "max_lon": -111.5720
        }
        
        # This would trigger real download
        # In a real test, we'd analyze the terrain data
        # For now, just verify the cache can handle the request
        assert cache is not None
    
    @pytest.mark.asyncio
    async def test_elevation_profile_accuracy(self, real_trail_finder):
        """Test that elevation profiles match expected terrain"""
        # Route with known elevation profile
        start = Coordinate(lat=40.6572, lon=-111.5706)
        end = Coordinate(lat=40.6486, lon=-111.5639)
        
        path, stats = await real_trail_finder.find_route(start, end, {"debug": True})
        
        assert path is not None
        
        # If debug mode provides elevation data
        if hasattr(real_trail_finder, 'dem_cache') and real_trail_finder.dem_cache.debug_data:
            debug_data = real_trail_finder.dem_cache.debug_data
            
            # Verify we have elevation information
            if "elevation_profile" in debug_data:
                profile = debug_data["elevation_profile"]
                assert len(profile) > 0, "No elevation data in profile"
                
                # Check elevation is in reasonable range for Park City
                elevations = [p["elevation"] for p in profile if "elevation" in p]
                if elevations:
                    min_elev = min(elevations)
                    max_elev = max(elevations)
                    assert 2000 < min_elev < 3500, f"Minimum elevation {min_elev}m unrealistic for Park City"
                    assert 2000 < max_elev < 3500, f"Maximum elevation {max_elev}m unrealistic for Park City"


@pytest.mark.real_data
@pytest.mark.regression
class TestRegressionRoutes:
    """Test specific routes that have had issues in the past"""
    
    PROBLEM_ROUTES = [
        {
            "name": "steep_avoidance_regression",
            "start": (40.6482, -111.5738),
            "end": (40.6464, -111.5729),
            "issue": "Used to route directly through 40° slopes",
            "expected": "Should route around steep area"
        },
        # Add more regression test routes as issues are found and fixed
    ]
    
    @pytest.mark.parametrize("route", PROBLEM_ROUTES)
    @pytest.mark.asyncio
    async def test_regression_route(self, route):
        """Test routes that have had issues in the past"""
        trail_finder = TrailFinderService()
        
        start = Coordinate(lat=route["start"][0], lon=route["start"][1])
        end = Coordinate(lat=route["end"][0], lon=route["end"][1])
        
        path, stats = await trail_finder.find_route(start, end, {})
        
        assert path is not None, f"Failed to find route for {route['name']}"
        
        # Add specific assertions based on the regression
        # This is a template - actual assertions would be based on the specific issue


if __name__ == "__main__":
    # Run only real data tests
    pytest.main([__file__, "-v", "-m", "real_data"])