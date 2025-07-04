"""
Fast tests for DEM cache using pre-populated fixtures
"""

import pytest
import numpy as np
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from app.services.dem_tile_cache import DEMTileCache
from app.services.obstacle_config import ObstacleConfig


class TestDEMCacheFast:
    """Fast DEM cache tests using fixtures"""
    
    def test_cache_initialization(self, mock_dem_cache):
        """Test cache is properly initialized with mock data"""
        assert mock_dem_cache is not None
        assert len(mock_dem_cache.terrain_cache) > 0
        assert len(mock_dem_cache.cost_surface_cache) > 0
    
    def test_cache_status(self, mock_dem_cache):
        """Test cache status reporting"""
        status = mock_dem_cache.get_cache_status()
        
        assert status['terrain_cache']['count'] == 2
        assert status['cost_surface_cache']['count'] == 1
        assert status['total_memory_mb'] > 0
        
        # Check entries
        terrain_entries = status['terrain_cache']['entries']
        assert len(terrain_entries) == 2
        assert all('shape' in entry for entry in terrain_entries)
        assert all('size_mb' in entry for entry in terrain_entries)
    
    def test_mock_find_path(self, mock_dem_cache):
        """Test mocked find_path method"""
        # Test coordinates within our mock bounds
        lat1, lon1 = 40.6560, -111.5708
        lat2, lon2 = 40.6520, -111.5688
        
        path, stats = mock_dem_cache.find_path(lat1, lon1, lat2, lon2)
        
        assert path is not None
        assert len(path) > 2  # At least start, middle, end
        assert stats is not None
        assert 'distance_m' in stats
        assert 'elevation_gain_m' in stats
        assert 'algorithm' in stats
        assert stats['algorithm'] == 'mock_pathfinder'
        
        # Check path starts and ends at correct locations
        assert abs(path[0][0] - lat1) < 0.01
        assert abs(path[0][1] - lon1) < 0.01
        assert abs(path[-1][0] - lat2) < 0.01
        assert abs(path[-1][1] - lon2) < 0.01
    
    def test_terrain_cache_format(self, mock_dem_cache):
        """Test terrain cache has correct format"""
        for key, value in mock_dem_cache.terrain_cache.items():
            assert isinstance(value, tuple)
            assert len(value) == 3
            dem, transform, crs = value
            assert isinstance(dem, np.ndarray)
            assert dem.dtype == np.float32
            assert crs == 'EPSG:3857'
    
    def test_cost_cache_format(self, mock_dem_cache):
        """Test cost surface cache has correct format"""
        for key, value in mock_dem_cache.cost_surface_cache.items():
            assert isinstance(value, tuple)
            assert len(value) == 3
            cost_surface, slope_degrees, indices = value
            assert isinstance(cost_surface, np.ndarray)
            assert isinstance(slope_degrees, np.ndarray)
            assert isinstance(indices, np.ndarray)
            assert indices.dtype == np.int32
    
    def test_tile_cache_exists(self, mock_dem_cache):
        """Test tile cache is properly set"""
        assert mock_dem_cache.tiled_cache is not None
        assert mock_dem_cache.tiled_cache.tile_size == 0.01
        assert os.path.exists(mock_dem_cache.tiled_cache.cache_dir)
    
    def test_different_routes(self, mock_dem_cache):
        """Test finding paths between different points"""
        test_routes = [
            ((40.6560, -111.5708), (40.6520, -111.5688)),
            ((40.6550, -111.5750), (40.6530, -111.5730)),
            ((40.6570, -111.5790), (40.6510, -111.5710)),
        ]
        
        for (start, end) in test_routes:
            path, stats = mock_dem_cache.find_path(
                start[0], start[1], end[0], end[1]
            )
            
            assert path is not None, f"Failed to find path from {start} to {end}"
            assert len(path) >= 2, f"Path too short from {start} to {end}"
            assert stats['distance_m'] > 0, f"Invalid distance from {start} to {end}"
    
    def test_cache_memory_usage(self, mock_dem_cache):
        """Test memory usage tracking"""
        status = mock_dem_cache.get_cache_status()
        
        # Calculate expected memory
        expected_mb = 0
        for entry in status['terrain_cache']['entries']:
            expected_mb += entry['size_mb']
        for entry in status['cost_surface_cache']['entries']:
            expected_mb += entry['size_mb']
        
        assert abs(status['total_memory_mb'] - expected_mb) < 0.1
    
    def test_preprocessing_cache_empty(self, mock_dem_cache):
        """Test preprocessing cache starts empty"""
        assert len(mock_dem_cache.preprocessing_cache) == 0
        
        status = mock_dem_cache.get_cache_status()
        assert status['preprocessing_cache']['count'] == 0
    
    def test_buffer_setting(self, mock_dem_cache):
        """Test buffer configuration"""
        assert mock_dem_cache.buffer == 0.05  # Default buffer
        
        # Create new cache with custom buffer
        custom_cache = DEMTileCache(buffer=0.1)
        assert custom_cache.buffer == 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])