#!/usr/bin/env python3
"""
Fast cache consistency tests using pre-populated fixtures
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.trail_finder import TrailFinderService
from app.models.route import Coordinate
from tests.fixtures.cache_fixtures import (
    mock_dem_cache, 
    sample_route_request, 
    expected_route_stats,
    populated_tile_cache,
    test_data_dir
)


class TestFastCacheConsistency:
    """Test cache consistency with pre-populated fixtures"""
    
    @pytest.mark.asyncio
    async def test_route_with_mock_cache(self, mock_dem_cache, sample_route_request):
        """Test route calculation using mocked cache"""
        # Create trail finder with mocked cache
        trail_finder = TrailFinderService(debug_mode=False)
        
        # Mock the DEM cache
        with patch.object(trail_finder, 'dem_cache', mock_dem_cache):
            start = Coordinate(**sample_route_request['start'])
            end = Coordinate(**sample_route_request['end'])
            
            # Calculate route
            path, stats = await trail_finder.find_route(
                start, end, 
                sample_route_request['options']
            )
            
            assert path is not None, "Should find route with mock cache"
            assert len(path) > 0, "Path should have points"
            assert stats is not None, "Should return stats"
    
    @pytest.mark.asyncio
    async def test_consistency_between_runs(self, mock_dem_cache, sample_route_request):
        """Test that multiple runs produce identical results"""
        trail_finder = TrailFinderService(debug_mode=False)
        
        with patch.object(trail_finder, 'dem_cache', mock_dem_cache):
            start = Coordinate(**sample_route_request['start'])
            end = Coordinate(**sample_route_request['end'])
            
            # Run route calculation 3 times
            results = []
            for _ in range(3):
                path, stats = await trail_finder.find_route(
                    start, end,
                    sample_route_request['options']
                )
                results.append((path, stats))
            
            # All results should be identical
            for i in range(1, len(results)):
                path1, stats1 = results[0]
                path2, stats2 = results[i]
                
                assert len(path1) == len(path2), f"Path lengths differ"
                assert stats1['distance_km'] == stats2['distance_km'], "Distances differ"
                assert stats1['waypoints'] == stats2['waypoints'], "Waypoint counts differ"
    
    def test_mock_cache_has_expected_data(self, mock_dem_cache):
        """Test that mock cache contains expected data"""
        # Check terrain cache
        assert len(mock_dem_cache.terrain_cache) > 0, "Terrain cache should have data"
        
        # Check cost surface cache
        assert len(mock_dem_cache.cost_surface_cache) > 0, "Cost cache should have data"
        
        # Check cache status
        status = mock_dem_cache.get_cache_status()
        assert status['terrain_cache']['count'] > 0, "Should report terrain cache entries"
        assert status['cost_surface_cache']['count'] > 0, "Should report cost cache entries"
        assert status['total_memory_mb'] > 0, "Should report memory usage"
    
    @pytest.mark.asyncio
    async def test_tile_cache_usage(self, populated_tile_cache, test_data_dir, sample_route_request):
        """Test that tile cache is used correctly"""
        # Create trail finder with specific cache directory
        trail_finder = TrailFinderService(debug_mode=False)
        
        # Mock cache directory
        with patch('app.services.dem_tile_cache.TILE_CACHE_DIR', str(test_data_dir / 'tile_cache')):
            trail_finder.dem_cache.tiled_cache = populated_tile_cache
            
            start = Coordinate(**sample_route_request['start'])
            end = Coordinate(**sample_route_request['end'])
            
            # Calculate route
            path, stats = await trail_finder.find_route(
                start, end,
                sample_route_request['options']
            )
            
            assert path is not None, "Should find route with tile cache"
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("resolution", ["high", "medium", "low"])
    async def test_different_resolutions(self, mock_dem_cache, sample_route_request, resolution):
        """Test cache consistency across different resolutions"""
        trail_finder = TrailFinderService(debug_mode=False)
        
        with patch.object(trail_finder, 'dem_cache', mock_dem_cache):
            start = Coordinate(**sample_route_request['start'])
            end = Coordinate(**sample_route_request['end'])
            
            # Update resolution
            options = sample_route_request['options'].copy()
            options['resolution'] = resolution
            
            # Calculate route
            path, stats = await trail_finder.find_route(start, end, options)
            
            assert path is not None, f"Should find route with {resolution} resolution"
            assert len(path) > 0, "Path should have points"
    
    def test_cache_memory_usage(self, mock_dem_cache):
        """Test that cache reports reasonable memory usage"""
        status = mock_dem_cache.get_cache_status()
        
        assert 'total_memory_mb' in status, "Should report memory usage"
        assert status['total_memory_mb'] > 0, "Should use some memory"
        assert status['total_memory_mb'] < 1000, "Should not use excessive memory"
        
        # Check detailed cache info
        assert status['terrain_cache']['count'] == 2, "Should have 2 terrain entries"
        assert status['cost_surface_cache']['count'] == 1, "Should have 1 cost surface entry"
    
    @pytest.mark.asyncio
    async def test_obstacle_avoidance_with_cache(self, mock_dem_cache, sample_route_request):
        """Test that obstacle avoidance works with cached data"""
        from app.services.obstacle_config import ObstaclePresets
        
        # Create trail finder with strict obstacle config
        obstacle_config = ObstaclePresets.accessibility_focused()
        trail_finder = TrailFinderService(
            obstacle_config=obstacle_config,
            debug_mode=False
        )
        
        with patch.object(trail_finder, 'dem_cache', mock_dem_cache):
            start = Coordinate(**sample_route_request['start'])
            end = Coordinate(**sample_route_request['end'])
            
            # Calculate route
            path, stats = await trail_finder.find_route(
                start, end,
                sample_route_request['options']
            )
            
            # With accessibility config, max slope should be limited
            if 'max_slope' in stats:
                assert stats['max_slope'] < 20.0, "Accessibility route should avoid steep slopes"


if __name__ == "__main__":
    # Run with: pytest tests/test_cache_consistency_fast.py -v
    pytest.main([__file__, "-v"])