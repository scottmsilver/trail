#!/usr/bin/env python3
"""
Comprehensive route calculation tests
Consolidates various route testing scenarios
"""

import pytest
import asyncio
import time
import numpy as np
from typing import List, Tuple

from app.services.trail_finder import TrailFinderService
from app.services.dem_tile_cache import DEMTileCache
from app.services.obstacle_config import ObstacleConfig, ObstaclePresets
from app.services.path_preferences import PathPreferences, PathPreferencePresets
from app.models.route import Coordinate


class TestRouteCalculations:
    """Test various route calculation scenarios"""
    
    @pytest.fixture
    def trail_finder(self):
        """Create trail finder instance"""
        return TrailFinderService(debug_mode=False)
    
    @pytest.fixture
    def debug_trail_finder(self):
        """Create debug trail finder instance"""
        return TrailFinderService(debug_mode=True)
    
    # Test various route scenarios
    @pytest.mark.asyncio
    @pytest.mark.parametrize("start,end,name", [
        # Park City routes
        ((40.6572, -111.5706), (40.6486, -111.5639), "Park City standard"),
        ((40.6571, -111.5705), (40.6477, -111.5631), "Park City cache test"),
        ((40.6560, -111.5708), (40.6520, -111.5688), "Park City short"),
        # Golden Gate Park
        ((37.7694, -122.4862), (37.7754, -122.4584), "Golden Gate Park"),
        # Steep terrain
        ((40.6482, -111.5738), (40.6464, -111.5729), "Steep terrain"),
    ])
    async def test_route_calculation(self, trail_finder, start, end, name):
        """Test basic route calculation"""
        start_coord = Coordinate(lat=start[0], lon=start[1])
        end_coord = Coordinate(lat=end[0], lon=end[1])
        
        path, stats = await trail_finder.find_route(start_coord, end_coord, {})
        
        assert path is not None, f"Failed to find route for {name}"
        assert len(path) > 0, f"Empty path for {name}"
        assert stats is not None, f"No stats for {name}"
        
        # Verify path endpoints
        assert abs(path[0].lat - start[0]) < 0.001, "Path doesn't start at start point"
        assert abs(path[-1].lat - end[0]) < 0.001, "Path doesn't end at end point"
        
        # Verify stats
        assert stats.get('distance_km', 0) > 0, "Invalid distance"
        assert stats.get('waypoints', 0) == len(path), "Waypoint count mismatch"
    
    @pytest.mark.asyncio
    async def test_route_with_custom_slope_costs(self, trail_finder):
        """Test route with custom slope configuration"""
        start = Coordinate(lat=40.6482, lon=-111.5738)
        end = Coordinate(lat=40.6464, lon=-111.5729)
        
        # Very strict slope limits
        strict_config = ObstacleConfig()
        strict_config.slope_costs = [
            (0, 1.0),
            (3, 2.0),
            (5, 10.0),
            (10, float('inf'))
        ]
        strict_config.use_continuous_slope = True
        
        strict_finder = TrailFinderService(obstacle_config=strict_config)
        
        path, stats = await strict_finder.find_route(start, end, {})
        
        assert path is not None, "Should find route even with strict slopes"
        assert stats['max_slope'] <= 10.0, f"Max slope {stats['max_slope']} exceeds limit"
    
    @pytest.mark.asyncio
    async def test_route_with_path_preferences(self):
        """Test route with different path preferences"""
        start = Coordinate(lat=37.7694, lon=-122.4862)
        end = Coordinate(lat=37.7704, lon=-122.4852)
        
        # Test urban walker (prefers paved paths)
        urban_prefs = PathPreferencePresets.urban_walker()
        urban_finder = TrailFinderService(path_preferences=urban_prefs)
        
        urban_path, urban_stats = await urban_finder.find_route(start, end, {})
        
        # Test trail seeker (prefers natural trails)
        trail_prefs = PathPreferencePresets.trail_seeker()
        trail_finder = TrailFinderService(path_preferences=trail_prefs)
        
        trail_path, trail_stats = await trail_finder.find_route(start, end, {})
        
        assert urban_path is not None
        assert trail_path is not None
        
        # Paths may differ based on preferences
        # Both should be valid routes
    
    @pytest.mark.asyncio
    async def test_bidirectional_pathfinding(self, trail_finder):
        """Test that bidirectional search works correctly"""
        start = Coordinate(lat=40.6572, lon=-111.5706)
        end = Coordinate(lat=40.6486, lon=-111.5639)
        
        # This should trigger bidirectional search (medium distance)
        path, stats = await trail_finder.find_route(start, end, {})
        
        assert path is not None
        assert len(path) > 50, "Medium route should have many points"
        
        # Check if bidirectional was used (if debug info available)
        if 'algorithm' in stats:
            assert 'bidirectional' in stats['algorithm'].lower()
    
    @pytest.mark.asyncio
    async def test_route_consistency(self, trail_finder):
        """Test that same route calculated multiple times is consistent"""
        start = Coordinate(lat=40.6560, lon=-111.5708)
        end = Coordinate(lat=40.6520, lon=-111.5688)
        
        # Calculate same route 3 times
        results = []
        for i in range(3):
            path, stats = await trail_finder.find_route(start, end, {})
            assert path is not None, f"Run {i+1} failed"
            results.append((path, stats))
        
        # All results should be identical
        for i in range(1, len(results)):
            path1, stats1 = results[0]
            path2, stats2 = results[i]
            
            assert len(path1) == len(path2), f"Path lengths differ: {len(path1)} vs {len(path2)}"
            assert stats1['distance_km'] == stats2['distance_km'], "Distances differ"
            assert stats1['elevation_gain_m'] == stats2['elevation_gain_m'], "Elevations differ"
    
    @pytest.mark.asyncio
    async def test_impossible_route(self, trail_finder):
        """Test handling of impossible routes"""
        # Try to route across ocean or impassable terrain
        start = Coordinate(lat=37.7749, lon=-122.4194)  # San Francisco
        end = Coordinate(lat=37.3382, lon=-121.8863)    # San Jose (if blocked)
        
        # This test depends on terrain data and obstacles
        # It should either find a route or fail gracefully
        path, stats = await trail_finder.find_route(start, end, {})
        
        # Either result is acceptable as long as no exception
        if path is None:
            assert 'error' in stats or 'message' in stats
    
    @pytest.mark.asyncio
    async def test_very_short_route(self, trail_finder):
        """Test very short routes (< 100m)"""
        start = Coordinate(lat=40.6560, lon=-111.5708)
        end = Coordinate(lat=40.6558, lon=-111.5706)  # Very close
        
        path, stats = await trail_finder.find_route(start, end, {})
        
        assert path is not None, "Should handle very short routes"
        assert stats['distance_km'] < 0.1, "Distance should be < 100m"
        assert len(path) >= 2, "Even short routes need start and end points"
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("profile", ["easy", "experienced", "trail_runner", "accessibility"])
    async def test_user_profiles(self, profile):
        """Test different user profiles"""
        start = Coordinate(lat=40.6482, lon=-111.5738)
        end = Coordinate(lat=40.6464, lon=-111.5729)
        
        # Get configurations for profile
        obstacle_config = {
            "easy": ObstaclePresets.easy_hiker(),
            "experienced": ObstaclePresets.experienced_hiker(),
            "trail_runner": ObstaclePresets.trail_runner(),
            "accessibility": ObstaclePresets.accessibility_focused(),
        }[profile]
        
        path_prefs = {
            "easy": PathPreferencePresets.urban_walker(),
            "experienced": PathPreferencePresets.trail_seeker(),
            "trail_runner": PathPreferencePresets.flexible_hiker(),
            "accessibility": PathPreferencePresets.urban_walker(),
        }[profile]
        
        finder = TrailFinderService(
            obstacle_config=obstacle_config,
            path_preferences=path_prefs
        )
        
        path, stats = await finder.find_route(start, end, {})
        
        assert path is not None, f"Failed to find route for {profile} profile"
        
        # Accessibility profile should avoid steep slopes
        if profile == "accessibility":
            assert stats['max_slope'] < 10.0, "Accessibility route too steep"