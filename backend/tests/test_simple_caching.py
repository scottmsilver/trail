#!/usr/bin/env python3
"""
Simple caching test that doesn't require clean state
"""

import pytest
import os
from pathlib import Path

from app.services.dem_tile_cache import DEMTileCache
from app.services.obstacle_config import ObstacleConfig


class TestSimpleCaching:
    """Basic caching tests that work with existing cache"""
    
    def test_cache_initialization(self):
        """Test that cache can be initialized"""
        cache = DEMTileCache(obstacle_config=ObstacleConfig())
        assert cache is not None
        
    def test_cache_status(self):
        """Test cache status reporting"""
        cache = DEMTileCache(obstacle_config=ObstacleConfig())
        status = cache.get_cache_status()
        
        assert isinstance(status, dict)
        assert 'terrain_cache' in status
        assert 'cost_surface_cache' in status
        assert 'total_memory_mb' in status
        
    def test_cache_directories_exist(self):
        """Test that cache directories are created"""
        cache = DEMTileCache(obstacle_config=ObstacleConfig())
        
        # At least one of these should exist after initialization
        cache_dirs = ['cache', 'dem_data', 'tile_cache']
        exists = any(os.path.exists(d) for d in cache_dirs)
        assert exists, "No cache directories found"
        
    def test_tile_size_config(self):
        """Test tile size configuration"""
        cache = DEMTileCache(obstacle_config=ObstacleConfig())
        
        # Check if tiled cache is configured
        assert hasattr(cache, 'tiled_cache')
        if cache.tiled_cache:
            assert cache.tiled_cache.tile_size > 0
            assert cache.tiled_cache.tile_size <= 1.0  # Should be in degrees