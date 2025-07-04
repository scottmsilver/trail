#!/usr/bin/env python3
"""
Fast route calculation tests using pre-populated cache fixtures
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.trail_finder import TrailFinderService
from app.services.obstacle_config import ObstacleConfig, ObstaclePresets
from app.services.path_preferences import PathPreferences, PathPreferencePresets
from app.models.route import Coordinate
from tests.fixtures.cache_fixtures import (
    mock_dem_cache,
    sample_route_request,
    expected_route_stats,
    sample_dem_data,
    sample_cost_surface
)


class TestFastRouteCalculations:
    """Fast route calculation tests with mocked data"""
    
    @pytest.fixture
    def trail_finder_with_cache(self, mock_dem_cache):
        """Create trail finder with mocked cache"""
        trail_finder = TrailFinderService(debug_mode=False)
        # Replace the DEM cache with our mock
        trail_finder.dem_cache = mock_dem_cache
        return trail_finder
    
    @pytest.mark.asyncio
    async def test_basic_route_calculation(self, trail_finder_with_cache, sample_route_request):
        """Test basic route calculation with mock cache"""
        start = Coordinate(**sample_route_request['start'])
        end = Coordinate(**sample_route_request['end'])
        
        path, stats = await trail_finder_with_cache.find_route(
            start, end,
            sample_route_request['options']
        )
        
        assert path is not None, "Should find route"
        assert len(path) >= 2, "Path should have at least start and end"
        assert stats is not None, "Should return statistics"
        
        # Verify endpoints
        assert abs(path[0].lat - start.lat) < 0.01, "Path should start at start point"
        assert abs(path[-1].lat - end.lat) < 0.01, "Path should end at end point"
    
    @pytest.mark.asyncio
    async def test_route_statistics(self, trail_finder_with_cache, sample_route_request, expected_route_stats):
        """Test that route statistics are reasonable"""
        start = Coordinate(**sample_route_request['start'])
        end = Coordinate(**sample_route_request['end'])
        
        path, stats = await trail_finder_with_cache.find_route(
            start, end,
            sample_route_request['options']
        )
        
        # Check statistics are within expected ranges
        assert stats['distance_km'] == expected_route_stats['distance_km']
        assert stats.get('elevation_gain_m', 0) == expected_route_stats['elevation_gain_m']
        assert stats['waypoints'] == expected_route_stats['waypoints']
        if 'max_slope' in stats:
            assert stats['max_slope'] == expected_route_stats['max_slope']
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("profile,max_expected_slope", [
        ("easy", 15.0),
        ("experienced", 35.0),
        ("accessibility", 10.0),
    ])
    async def test_user_profiles_respect_limits(self, mock_dem_cache, sample_route_request, profile, max_expected_slope):
        """Test that different user profiles respect slope limits"""
        # Get profile configurations
        obstacle_config = {
            "easy": ObstaclePresets.easy_hiker(),
            "experienced": ObstaclePresets.experienced_hiker(),
            "accessibility": ObstaclePresets.accessibility_focused(),
        }[profile]
        
        trail_finder = TrailFinderService(
            obstacle_config=obstacle_config,
            debug_mode=False
        )
        trail_finder.dem_cache = mock_dem_cache
        
        start = Coordinate(**sample_route_request['start'])
        end = Coordinate(**sample_route_request['end'])
        
        path, stats = await trail_finder.find_route(
            start, end,
            sample_route_request['options']
        )
        
        assert path is not None, f"Should find route for {profile} profile"
        
        # Check slope limits are respected
        if 'max_slope' in stats:
            assert stats['max_slope'] <= max_expected_slope, \
                f"{profile} profile exceeded slope limit: {stats['max_slope']} > {max_expected_slope}"
    
    @pytest.mark.asyncio
    async def test_very_short_route(self, trail_finder_with_cache):
        """Test handling of very short routes"""
        # Points very close together (< 50m apart)
        start = Coordinate(lat=40.6560, lon=-111.5708)
        end = Coordinate(lat=40.6561, lon=-111.5707)
        
        path, stats = await trail_finder_with_cache.find_route(start, end, {})
        
        assert path is not None, "Should handle very short routes"
        assert len(path) >= 2, "Even short routes need at least 2 points"
        assert stats['distance_km'] < 0.1, "Distance should be < 100m"
    
    @pytest.mark.asyncio
    async def test_path_preferences(self, mock_dem_cache, sample_route_request):
        """Test that path preferences affect route choice"""
        start = Coordinate(**sample_route_request['start'])
        end = Coordinate(**sample_route_request['end'])
        
        # Test with different preferences
        preferences = [
            PathPreferencePresets.urban_walker(),
            PathPreferencePresets.trail_seeker(),
            PathPreferencePresets.flexible_hiker()
        ]
        
        results = []
        for pref in preferences:
            trail_finder = TrailFinderService(
                path_preferences=pref,
                debug_mode=False
            )
            trail_finder.dem_cache = mock_dem_cache
            
            path, stats = await trail_finder.find_route(start, end, sample_route_request['options'])
            assert path is not None, f"Should find route with {pref} preferences"
            results.append(len(path))
        
        # Different preferences may result in different path lengths
        # but all should be valid routes
        assert all(r > 0 for r in results), "All preferences should produce valid routes"
    
    @pytest.mark.asyncio
    async def test_cache_hit_performance(self, trail_finder_with_cache, sample_route_request):
        """Test that repeated calculations are fast due to cache"""
        start = Coordinate(**sample_route_request['start'])
        end = Coordinate(**sample_route_request['end'])
        
        # First calculation
        import time
        t1 = time.time()
        path1, stats1 = await trail_finder_with_cache.find_route(
            start, end,
            sample_route_request['options']
        )
        time1 = time.time() - t1
        
        # Second calculation (should use cache)
        t2 = time.time()
        path2, stats2 = await trail_finder_with_cache.find_route(
            start, end,
            sample_route_request['options']
        )
        time2 = time.time() - t2
        
        # Should produce identical results
        assert len(path1) == len(path2), "Paths should be identical"
        assert stats1['distance_km'] == stats2['distance_km'], "Stats should be identical"
        
        # Second run should be at least as fast (usually faster)
        assert time2 <= time1 * 1.1, "Cached run should not be slower"
    
    @pytest.mark.asyncio
    async def test_resolution_options(self, trail_finder_with_cache, sample_route_request):
        """Test different resolution settings"""
        start = Coordinate(**sample_route_request['start'])
        end = Coordinate(**sample_route_request['end'])
        
        path_lengths = {}
        for resolution in ["low", "medium", "high"]:
            options = sample_route_request['options'].copy()
            options['resolution'] = resolution
            
            path, stats = await trail_finder_with_cache.find_route(start, end, options)
            assert path is not None, f"Should find route with {resolution} resolution"
            path_lengths[resolution] = len(path)
        
        # Higher resolution should generally produce more points
        # (though not always strictly true for very simple routes)
        assert path_lengths['high'] >= path_lengths['low'] * 0.8, \
            "High resolution should produce reasonably detailed path"
    
    @pytest.mark.asyncio
    async def test_impossible_route_handling(self, trail_finder_with_cache):
        """Test handling of routes that can't be calculated"""
        # Create a scenario where route is impossible
        # Mock the pathfinding to return None
        with patch.object(trail_finder_with_cache.dem_cache, 'find_path', return_value=(None, {})):
            start = Coordinate(lat=40.6560, lon=-111.5708)
            end = Coordinate(lat=40.6520, lon=-111.5688)
            
            path, stats = await trail_finder_with_cache.find_route(start, end, {})
            
            # Should handle gracefully
            assert path is None or len(path) == 0, "Should return None or empty path"
            assert stats is not None, "Should still return stats/error info"
    
    @pytest.mark.asyncio
    async def test_consistency_across_multiple_runs(self, trail_finder_with_cache, sample_route_request):
        """Test that route calculation is deterministic"""
        start = Coordinate(**sample_route_request['start'])
        end = Coordinate(**sample_route_request['end'])
        
        # Calculate route 5 times
        results = []
        for _ in range(5):
            path, stats = await trail_finder_with_cache.find_route(
                start, end,
                sample_route_request['options']
            )
            results.append({
                'length': len(path),
                'distance': stats['distance_km'],
                'waypoints': stats['waypoints'],
                'first_point': (path[0].lat, path[0].lon) if path else None,
                'last_point': (path[-1].lat, path[-1].lon) if path else None
            })
        
        # All runs should produce identical results
        first_result = results[0]
        for i, result in enumerate(results[1:], 1):
            assert result == first_result, f"Run {i+1} differs from run 1"


if __name__ == "__main__":
    # Run with: pytest tests/test_route_calculations_fast.py -v
    pytest.main([__file__, "-v"])