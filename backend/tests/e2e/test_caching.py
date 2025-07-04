#!/usr/bin/env python3
"""
Test terrain caching functionality
"""

import pytest
import time
import os
import shutil
from pathlib import Path

from app.services.dem_tile_cache import DEMTileCache
from app.services.obstacle_config import ObstacleConfig
from app.services.path_preferences import PathPreferences


@pytest.fixture
def clean_cache_dir():
    """Ensure clean cache directory for tests"""
    cache_dirs = ['cache', 'dem_data', 'tile_cache']
    
    # Backup existing caches
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            backup = f"{cache_dir}.pytest_backup"
            if os.path.exists(backup):
                shutil.rmtree(backup)
            shutil.move(cache_dir, backup)
    
    # Create clean directories
    os.makedirs('cache', exist_ok=True)
    
    yield
    
    # Cleanup and restore
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
        backup = f"{cache_dir}.pytest_backup"
        if os.path.exists(backup):
            shutil.move(backup, cache_dir)


@pytest.mark.real_data
@pytest.mark.slow
class TestTerrainCaching:
    """Test terrain caching functionality"""
    
    # Test coordinates (small area for faster tests)
    START = (37.7694, -122.4762)  # Stow Lake
    END = (37.7704, -122.4752)     # Very close
    
    def test_first_run_downloads_terrain(self, clean_cache_dir):
        """Test that first run downloads terrain data"""
        cache = DEMTileCache(
            obstacle_config=ObstacleConfig(),
            path_preferences=PathPreferences()
        )
        
        # Check no DEM files exist
        dem_files = list(Path('dem_data').glob('*.tif')) if os.path.exists('dem_data') else []
        assert len(dem_files) == 0, "DEM files exist before first run"
        
        # First run
        start_time = time.time()
        path = cache.find_route(*self.START, *self.END)
        elapsed = time.time() - start_time
        
        assert path is not None, "First run should find a path"
        assert len(path) > 0, "Path should have points"
        
        # Check DEM files were created
        dem_files = list(Path('dem_data').glob('*.tif')) if os.path.exists('dem_data') else []
        assert len(dem_files) > 0, "No DEM files created after first run"
    
    def test_second_run_uses_cache(self, clean_cache_dir):
        """Test that second run is faster due to caching"""
        cache = DEMTileCache(
            obstacle_config=ObstacleConfig(),
            path_preferences=PathPreferences()
        )
        
        # First run
        start_time = time.time()
        path1 = cache.find_route(*self.START, *self.END)
        time1 = time.time() - start_time
        
        assert path1 is not None, "First run failed"
        
        # Second run (should use cache)
        start_time = time.time()
        path2 = cache.find_route(*self.START, *self.END)
        time2 = time.time() - start_time
        
        assert path2 is not None, "Second run failed"
        
        # Second run should be faster
        assert time2 < time1, f"Second run ({time2:.1f}s) not faster than first ({time1:.1f}s)"
        
        # Paths should be identical
        assert len(path1) == len(path2), "Path lengths differ between runs"
    
    def test_cache_consistency(self, clean_cache_dir):
        """Test that cached results are consistent"""
        cache = DEMTileCache(
            obstacle_config=ObstacleConfig(),
            path_preferences=PathPreferences()
        )
        
        # Run same route 3 times
        paths = []
        for i in range(3):
            path = cache.find_route(*self.START, *self.END)
            assert path is not None, f"Run {i+1} failed"
            paths.append(path)
        
        # All paths should be identical
        for i in range(1, len(paths)):
            assert len(paths[0]) == len(paths[i]), f"Path {i+1} has different length"
            # Check first and last points match
            assert paths[0][0] == paths[i][0], f"Path {i+1} has different start"
            assert paths[0][-1] == paths[i][-1], f"Path {i+1} has different end"
    
    def test_cache_status_reporting(self):
        """Test cache status information"""
        cache = DEMTileCache(
            obstacle_config=ObstacleConfig(),
            path_preferences=PathPreferences()
        )
        
        status = cache.get_cache_status()
        
        assert isinstance(status, dict), "Cache status should be a dictionary"
        assert 'terrain_cache' in status, "Should report terrain cache status"
        assert 'cost_surface_cache' in status, "Should report cost surface cache status"
        
        # Check cache entry counts
        assert status['terrain_cache']['count'] >= 0, "Invalid terrain cache count"
        assert status['cost_surface_cache']['count'] >= 0, "Invalid cost surface cache count"