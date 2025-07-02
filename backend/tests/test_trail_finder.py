import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from app.services.trail_finder import TrailFinderService
from app.models.route import Coordinate


class TestTrailFinder:
    """Test the trail finding service with mocked DEM data"""
    
    @pytest.fixture
    def trail_finder(self):
        """Create a trail finder instance with mocked DEM cache"""
        with patch('app.services.trail_finder.DEMTileCache'):
            service = TrailFinderService()
            return service
    
    @pytest.fixture
    def mock_dem_cache(self):
        """Create a mock DEM cache that returns predictable paths"""
        mock = Mock()
        
        def mock_find_route(lat1, lon1, lat2, lon2):
            # Generate a simple path between two points
            # This simulates a path with several waypoints
            num_points = 10
            lats = np.linspace(lat1, lat2, num_points)
            lons = np.linspace(lon1, lon2, num_points)
            
            # Add some variation to simulate a non-straight path
            for i in range(1, num_points - 1):
                lats[i] += (np.random.random() - 0.5) * 0.001
                lons[i] += (np.random.random() - 0.5) * 0.001
            
            # Return as (lon, lat) tuples as expected by the service
            return [(lons[i], lats[i]) for i in range(num_points)]
        
        mock.find_route = mock_find_route
        return mock
    
    def test_validate_route_request_valid(self, trail_finder):
        """Test validation accepts valid route requests"""
        start = Coordinate(lat=40.6573, lon=-111.5705)
        end = Coordinate(lat=40.6382, lon=-111.5774)
        
        assert trail_finder.validate_route_request(start, end) == True
    
    def test_validate_route_request_same_point(self, trail_finder):
        """Test validation rejects routes with same start and end"""
        start = Coordinate(lat=40.6573, lon=-111.5705)
        end = Coordinate(lat=40.6573, lon=-111.5705)
        
        assert trail_finder.validate_route_request(start, end) == False
    
    def test_validate_route_request_too_far(self, trail_finder):
        """Test validation rejects routes that are too far apart"""
        start = Coordinate(lat=40.0, lon=-111.0)
        end = Coordinate(lat=41.0, lon=-112.0)  # >50km apart
        
        assert trail_finder.validate_route_request(start, end) == False
    
    def test_calculate_bounding_box(self, trail_finder):
        """Test bounding box calculation"""
        start = Coordinate(lat=40.630, lon=-111.580)
        end = Coordinate(lat=40.650, lon=-111.560)
        
        bbox = trail_finder.calculate_bounding_box(start, end)
        
        # Check buffer is applied (0.05 degrees)
        assert bbox["min_lat"] == pytest.approx(40.630 - 0.05)
        assert bbox["max_lat"] == pytest.approx(40.650 + 0.05)
        assert bbox["min_lon"] == pytest.approx(-111.580 - 0.05)
        assert bbox["max_lon"] == pytest.approx(-111.560 + 0.05)
    
    @pytest.mark.asyncio
    async def test_find_route_success(self, trail_finder, mock_dem_cache):
        """Test successful route finding"""
        trail_finder.dem_cache = mock_dem_cache
        
        start = Coordinate(lat=40.630, lon=-111.580)
        end = Coordinate(lat=40.650, lon=-111.560)
        
        path, stats = await trail_finder.find_route(start, end, {})
        
        # Check path is returned
        assert len(path) > 0
        assert path[0].lat == pytest.approx(start.lat, rel=0.01)
        assert path[0].lon == pytest.approx(start.lon, rel=0.01)
        assert path[-1].lat == pytest.approx(end.lat, rel=0.01)
        assert path[-1].lon == pytest.approx(end.lon, rel=0.01)
        
        # Check stats
        assert "distance_km" in stats
        assert "estimated_time_min" in stats
        assert "difficulty" in stats
        assert stats["distance_km"] > 0
        assert stats["estimated_time_min"] > 0
    
    @pytest.mark.asyncio
    async def test_find_route_no_path(self, trail_finder):
        """Test when no route can be found"""
        # Mock DEM cache to return None (no path found)
        mock_cache = Mock()
        mock_cache.find_route.return_value = None
        trail_finder.dem_cache = mock_cache
        
        start = Coordinate(lat=40.630, lon=-111.580)
        end = Coordinate(lat=40.650, lon=-111.560)
        
        path, stats = await trail_finder.find_route(start, end, {})
        
        assert path == []
        assert "error" in stats
        assert stats["error"] == "No route found"
    
    @pytest.mark.asyncio
    async def test_find_route_exception(self, trail_finder):
        """Test error handling when exception occurs"""
        # Mock DEM cache to raise exception
        mock_cache = Mock()
        mock_cache.find_route.side_effect = Exception("DEM error")
        trail_finder.dem_cache = mock_cache
        
        start = Coordinate(lat=40.630, lon=-111.580)
        end = Coordinate(lat=40.650, lon=-111.560)
        
        path, stats = await trail_finder.find_route(start, end, {})
        
        assert path == []
        assert "error" in stats
        assert "DEM error" in stats["error"]
    
    def test_estimate_difficulty_easy(self, trail_finder):
        """Test difficulty estimation for short routes"""
        assert trail_finder._estimate_difficulty(3.0) == "easy"
        assert trail_finder._estimate_difficulty(4.9) == "easy"
    
    def test_estimate_difficulty_moderate(self, trail_finder):
        """Test difficulty estimation for medium routes"""
        assert trail_finder._estimate_difficulty(5.0) == "moderate"
        assert trail_finder._estimate_difficulty(7.5) == "moderate"
        assert trail_finder._estimate_difficulty(9.9) == "moderate"
    
    def test_estimate_difficulty_hard(self, trail_finder):
        """Test difficulty estimation for long routes"""
        assert trail_finder._estimate_difficulty(10.0) == "hard"
        assert trail_finder._estimate_difficulty(15.0) == "hard"
    
    def test_distance_calculation(self, trail_finder):
        """Test Haversine distance calculation"""
        # Test known distance: approximately 2.85 km
        coord1 = Coordinate(lat=40.630, lon=-111.580)
        coord2 = Coordinate(lat=40.650, lon=-111.560)
        
        distance = trail_finder._calculate_distance(coord1, coord2)
        
        # Should be approximately 2.85 km
        assert distance == pytest.approx(2.85, rel=0.1)
    
    @pytest.mark.asyncio
    async def test_route_statistics_calculation(self, trail_finder, mock_dem_cache):
        """Test that route statistics are calculated correctly"""
        trail_finder.dem_cache = mock_dem_cache
        
        start = Coordinate(lat=40.630, lon=-111.580)
        end = Coordinate(lat=40.635, lon=-111.575)  # Short route
        
        path, stats = await trail_finder.find_route(start, end, {})
        
        # Check all expected stats are present
        assert "distance_km" in stats
        assert "elevation_gain_m" in stats
        assert "estimated_time_min" in stats
        assert "difficulty" in stats
        assert "waypoints" in stats
        
        # Check values are reasonable
        assert 0 < stats["distance_km"] < 2.0  # Short route
        assert stats["elevation_gain_m"] == 0  # Not implemented yet
        assert stats["estimated_time_min"] > 0
        assert stats["difficulty"] == "easy"  # Short route
        assert stats["waypoints"] == len(path)
    
    @pytest.mark.asyncio
    async def test_real_dem_route_finding(self):
        """Test route finding with real DEM data and py3dep"""
        # This test uses the actual DEM cache without mocking
        # Testing coordinates from Utah area where we know there's elevation data
        from app.services.trail_finder import TrailFinderService
        
        service = TrailFinderService()
        
        # Close coordinates in Utah mountains - should work with real DEM data
        start = Coordinate(lat=40.6573, lon=-111.5705)  # Near Salt Lake City
        end = Coordinate(lat=40.6582, lon=-111.5774)    # Close by, small area
        
        # This might take a while as it downloads DEM data
        path, stats = await service.find_route(start, end, {})
        
        # Should either find a path or fail gracefully
        if path:
            # If successful, check the path is reasonable
            assert len(path) >= 2  # At least start and end
            assert path[0].lat == pytest.approx(start.lat, rel=0.01)
            assert path[0].lon == pytest.approx(start.lon, rel=0.01)
            assert path[-1].lat == pytest.approx(end.lat, rel=0.01)
            assert path[-1].lon == pytest.approx(end.lon, rel=0.01)
            
            # Check stats
            assert "distance_km" in stats
            assert stats["distance_km"] > 0
            assert "estimated_time_min" in stats
            assert stats["estimated_time_min"] > 0
        else:
            # If no path found, should have error in stats
            assert "error" in stats
            assert len(stats["error"]) > 0
            
    @pytest.mark.asyncio 
    async def test_real_dem_multiple_coordinates(self):
        """Test multiple coordinate pairs that should work with py3dep"""
        from app.services.trail_finder import TrailFinderService
        
        service = TrailFinderService()
        
        # Test pairs from Utah area
        test_pairs = [
            # Very close coordinates
            (Coordinate(lat=40.6573, lon=-111.5705), Coordinate(lat=40.6575, lon=-111.5710)),
            # Slightly further apart
            (Coordinate(lat=40.6580, lon=-111.5720), Coordinate(lat=40.6590, lon=-111.5730)),
        ]
        
        for i, (start, end) in enumerate(test_pairs):
            print(f"Testing pair {i+1}: {start.lat},{start.lon} -> {end.lat},{end.lon}")
            
            path, stats = await service.find_route(start, end, {})
            
            # At minimum, should not crash and should return some response
            assert isinstance(path, list)
            assert isinstance(stats, dict)
            
            if path:
                print(f"  Found path with {len(path)} waypoints")
                assert len(path) >= 2
            else:
                print(f"  No path found: {stats.get('error', 'Unknown error')}")
                assert "error" in stats
    
    @pytest.mark.asyncio
    async def test_debug_mode_functionality(self):
        """Test that debug mode collects pathfinding information"""
        from app.services.trail_finder import TrailFinderService
        
        # Create debug-enabled service
        debug_service = TrailFinderService(debug_mode=True)
        
        # Use very close coordinates to ensure fast processing
        start = Coordinate(lat=40.6573, lon=-111.5705)
        end = Coordinate(lat=40.6575, lon=-111.5710)
        
        path, stats = await debug_service.find_route(start, end, {})
        
        # Should have debug data in stats
        assert "debug_data" in stats
        debug_data = stats["debug_data"]
        
        # Check that debug data has expected structure
        assert "explored_nodes" in debug_data
        assert "decision_points" in debug_data
        assert "grid_exploration" in debug_data
        assert "terrain_costs" in debug_data
        assert "bounds" in debug_data
        
        # Check that we explored some nodes
        assert len(debug_data["explored_nodes"]) > 0
        assert len(debug_data["decision_points"]) > 0
        
        # Check grid exploration data
        grid = debug_data["grid_exploration"]
        assert "shape" in grid
        assert "g_scores" in grid
        assert "f_scores" in grid
        assert "explored" in grid
        assert "in_path" in grid
        
        print(f"Debug mode explored {len(debug_data['explored_nodes'])} nodes")
        print(f"Made {len(debug_data['decision_points'])} decision points")
        print(f"Grid shape: {grid['shape']}")
        
        # Verify that some nodes were marked as explored
        explored_count = sum(sum(row) for row in grid["explored"])
        assert explored_count > 0
        print(f"Marked {explored_count} grid cells as explored")
    
    @pytest.mark.asyncio
    async def test_debug_mode_with_problematic_coordinates(self):
        """Test debug mode with coordinates that might cause issues"""
        from app.services.trail_finder import TrailFinderService
        
        # Create debug-enabled service
        debug_service = TrailFinderService(debug_mode=True)
        
        # Use the coordinates that caused the frontend error
        start = Coordinate(lat=40.6465, lon=-111.5754)
        end = Coordinate(lat=40.6554, lon=-111.5715)
        
        path, stats = await debug_service.find_route(start, end, {})
        
        # Should always return stats, even if no path found
        assert isinstance(stats, dict)
        
        if path:
            # If path found, should have debug data
            assert "debug_data" in stats
            assert stats["debug_data"] is not None
            print(f"Successful route with {len(path)} points and debug data")
        else:
            # If no path found, should have error message but might not have debug data
            assert "error" in stats
            print(f"No route found: {stats['error']}")
            
        # Check if debug_data exists and is properly structured
        if "debug_data" in stats and stats["debug_data"]:
            debug_data = stats["debug_data"]
            assert "explored_nodes" in debug_data
            assert "decision_points" in debug_data
            print(f"Debug data present with {len(debug_data['explored_nodes'])} explored nodes")
        else:
            print("No debug data returned (this might happen with failed routes)")
    
    @pytest.mark.asyncio
    async def test_debug_endpoint_error_handling(self):
        """Test the debug endpoint handles errors gracefully"""
        import json
        from app.main import debug_trail_finder
        from app.models.route import RouteResult, RouteStatus
        from datetime import datetime, timezone
        
        # Test with problematic coordinates
        start = Coordinate(lat=40.6465, lon=-111.5754)
        end = Coordinate(lat=40.6554, lon=-111.5715)
        
        # Simulate what the endpoint does
        try:
            path, stats = await debug_trail_finder.find_route(start, end, {})
            
            # Should be able to create RouteResult even if no path
            if not path:
                # Ensure error is properly handled
                assert "error" in stats
                route_result = RouteResult(
                    routeId="debug",
                    status=RouteStatus.FAILED,
                    path=[],
                    stats=stats,
                    createdAt=datetime.now(timezone.utc).isoformat()
                )
            else:
                route_result = RouteResult(
                    routeId="debug", 
                    status=RouteStatus.COMPLETED,
                    path=path,
                    stats=stats,
                    createdAt=datetime.now(timezone.utc).isoformat()
                )
            
            # Should be JSON serializable
            json_str = json.dumps(route_result.model_dump())
            assert len(json_str) > 0
            print("Route result successfully serialized")
            
        except Exception as e:
            print(f"Unexpected error: {e}")
            assert False, f"Debug endpoint should handle errors gracefully: {e}"
    
    @pytest.mark.asyncio
    async def test_debug_with_impossible_route(self):
        """Test debug mode with coordinates that should fail to find a route"""
        from app.services.trail_finder import TrailFinderService
        
        # Create debug-enabled service
        debug_service = TrailFinderService(debug_mode=True)
        
        # Use coordinates that are very far apart (should exceed max distance)
        start = Coordinate(lat=40.0, lon=-111.0)
        end = Coordinate(lat=41.0, lon=-112.0)  # >50km apart
        
        path, stats = await debug_service.find_route(start, end, {})
        
        # Should not find a path due to distance validation
        assert len(path) == 0
        assert "error" in stats or len(path) == 0
        
        # Should still have debug_data field (even if None)
        assert "debug_data" in stats
        print(f"Failed route stats keys: {list(stats.keys())}")
        
        if stats["debug_data"]:
            print("Debug data collected even for failed route")
        else:
            print("No debug data for failed route (expected for validation failures)")
    
    @pytest.mark.asyncio
    async def test_regular_vs_debug_parity(self):
        """Test that regular and debug modes find the same route"""
        from app.services.trail_finder import TrailFinderService
        
        # Create both regular and debug services
        regular_service = TrailFinderService(debug_mode=False)
        debug_service = TrailFinderService(debug_mode=True)
        
        # Use the coordinates that the user reported as problematic
        start = Coordinate(lat=40.6465, lon=-111.5754)
        end = Coordinate(lat=40.6554, lon=-111.5715)
        
        # Find routes with both services
        regular_path, regular_stats = await regular_service.find_route(start, end, {})
        debug_path, debug_stats = await debug_service.find_route(start, end, {})
        
        print(f"Regular route: {len(regular_path)} points, distance: {regular_stats.get('distance_km', 'N/A')}km")
        print(f"Debug route: {len(debug_path)} points, distance: {debug_stats.get('distance_km', 'N/A')}km")
        
        # Both should find routes (or both should fail)
        if regular_path:
            assert debug_path, f"Regular found route ({len(regular_path)} points) but debug failed: {debug_stats.get('error', 'unknown error')}"
            
            # Paths should be identical or very similar
            assert abs(len(regular_path) - len(debug_path)) <= 1, f"Path lengths differ significantly: regular={len(regular_path)}, debug={len(debug_path)}"
            
            # Distances should be identical or very close
            regular_dist = regular_stats.get('distance_km', 0)
            debug_dist = debug_stats.get('distance_km', 0)
            assert abs(regular_dist - debug_dist) < 0.01, f"Distances differ: regular={regular_dist}, debug={debug_dist}"
            
        else:
            assert not debug_path, f"Regular failed ({regular_stats.get('error', 'unknown')}) but debug succeeded ({len(debug_path)} points)"
        
        # Debug should have additional debug_data
        if debug_path:
            assert "debug_data" in debug_stats, "Debug mode should include debug_data"
            assert debug_stats["debug_data"] is not None, "Debug data should not be None for successful routes"
            
        print("✓ Regular and debug routes are consistent")