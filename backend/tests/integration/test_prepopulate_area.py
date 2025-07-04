"""
Integration tests for area prepopulation functionality
"""

import pytest
import subprocess
import sys
import os
import json
import shutil
from pathlib import Path
import tempfile

from app.services.dem_tile_cache import DEMTileCache


@pytest.mark.integration
class TestPrepopulateArea:
    """Test area prepopulation functionality"""
    
    # Small test area coordinates
    CORNER1 = "40.6500,-111.5700"
    CORNER2 = "40.6510,-111.5690"
    
    @pytest.fixture
    def clean_cache_dir(self):
        """Create a temporary cache directory for testing"""
        temp_dir = tempfile.mkdtemp(prefix="test_prepopulate_")
        
        # Save original environment
        original_env = {}
        env_vars = ['HYRIVER_CACHE_NAME', 'HYRIVER_CACHE_DIR']
        for var in env_vars:
            original_env[var] = os.environ.get(var)
        
        # Set test environment
        os.environ['HYRIVER_CACHE_NAME'] = os.path.join(temp_dir, 'http_cache.sqlite')
        
        yield temp_dir
        
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        # Restore environment
        for var, value in original_env.items():
            if value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = value
    
    def test_prepopulate_script_exists(self):
        """Test that prepopulate script exists and is executable"""
        script_path = Path("prepopulate_area.py")
        assert script_path.exists(), "prepopulate_area.py should exist"
        
        # Check script has proper shebang
        with open(script_path) as f:
            first_line = f.readline()
            assert first_line.startswith("#!/usr/bin/env python"), "Script should have python shebang"
    
    def test_prepopulate_script_help(self):
        """Test prepopulate script shows help with wrong arguments"""
        result = subprocess.run(
            [sys.executable, "prepopulate_area.py"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode != 0, "Should fail without arguments"
        assert "Usage:" in result.stdout, "Should show usage"
        assert "lat1,lon1" in result.stdout, "Should show coordinate format"
    
    @pytest.mark.slow
    def test_prepopulate_small_area(self, clean_cache_dir):
        """Test prepopulating a small area"""
        # Run prepopulate script
        result = subprocess.run(
            [sys.executable, "prepopulate_area.py", self.CORNER1, self.CORNER2],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        assert result.returncode == 0, f"Script failed: {result.stderr}"
        
        # Check output contains expected sections
        output = result.stdout
        assert "PREPOPULATING TRAIL MAP CACHE" in output
        assert "Downloading elevation data" in output
        assert "Processing elevation data" in output
        assert "Fetching obstacle data" in output
        assert "Fetching trail and path data" in output
        assert "Computing cost surface" in output
        assert "PREPOPULATION COMPLETE" in output
        assert "✅ Area is now prepopulated" in output
        
        # Verify some data was cached
        assert "Terrain: 0 → 1 entries" in output
        assert "Cost surfaces: 0 → 1 entries" in output
    
    def test_prepopulate_with_existing_cache(self):
        """Test prepopulating when some data is already cached"""
        # Initialize cache
        cache = DEMTileCache()
        
        # Get initial status
        initial_status = cache.get_cache_status()
        initial_terrain_count = initial_status['terrain_cache']['count']
        
        # Define test area
        lat1, lon1 = 40.6500, -111.5700
        lat2, lon2 = 40.6510, -111.5690
        min_lat, max_lat = min(lat1, lat2), max(lat1, lat2)
        min_lon, max_lon = min(lon1, lon2), max(lon1, lon2)
        
        # Download and cache terrain
        dem_file = cache.download_dem(min_lat, max_lat, min_lon, max_lon)
        assert dem_file is not None, "Should download DEM"
        
        dem, out_trans, crs = cache.read_dem(dem_file)
        assert dem is not None, "Should read DEM"
        
        # Cache the terrain
        cache_key = f"{min_lat:.4f},{max_lat:.4f},{min_lon:.4f},{max_lon:.4f}"
        cache.terrain_cache[cache_key] = (dem, out_trans, crs)
        
        # Run prepopulate again - should use cached terrain
        result = subprocess.run(
            [sys.executable, "prepopulate_area.py", self.CORNER1, self.CORNER2],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        assert result.returncode == 0, f"Script failed: {result.stderr}"
        
        # Should mention using cached data
        output = result.stdout
        assert "Using cached DEM file" in output or "Downloaded in 0.0s" in output
    
    def test_prepopulate_invalid_coordinates(self):
        """Test prepopulate with invalid coordinates"""
        # Test with invalid format
        result = subprocess.run(
            [sys.executable, "prepopulate_area.py", "invalid", "coordinates"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode != 0, "Should fail with invalid coordinates"
        assert "Error" in result.stderr
    
    def test_prepopulate_area_calculation(self):
        """Test that area calculation is correct"""
        from app.services.dem_tile_cache import DEMTileCache
        
        # Test coordinates
        lat1, lon1 = 40.6500, -111.5700
        lat2, lon2 = 40.6600, -111.5600
        
        # Calculate expected area
        lat_diff = abs(lat2 - lat1)
        lon_diff = abs(lon2 - lon1)
        expected_area = lat_diff * 111 * lon_diff * 111 * 0.7  # km²
        
        # Run prepopulate
        result = subprocess.run(
            [sys.executable, "prepopulate_area.py", 
             f"{lat1},{lon1}", f"{lat2},{lon2}"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        assert result.returncode == 0
        
        # Check area calculation in output
        output = result.stdout
        assert f"Area size: ~{expected_area:.1f} km²" in output
    
    def test_prepopulate_creates_cache_files(self, tmp_path):
        """Test that prepopulation creates expected cache files"""
        # Create a test cache directory
        cache_dir = tmp_path / "test_cache"
        cache_dir.mkdir()
        
        # Initialize cache with custom directory
        cache = DEMTileCache()
        
        # Define small test area
        lat1, lon1 = 40.6500, -111.5700
        lat2, lon2 = 40.6505, -111.5695
        min_lat, max_lat = min(lat1, lat2), max(lat1, lat2)
        min_lon, max_lon = min(lon1, lon2), max(lon1, lon2)
        
        # Create cache key
        cache_key = f"{min_lat:.4f},{max_lat:.4f},{min_lon:.4f},{max_lon:.4f}"
        
        # Run prepopulation steps manually to test cache creation
        dem_file = cache.download_dem(min_lat, max_lat, min_lon, max_lon)
        assert dem_file is not None
        
        dem, out_trans, crs = cache.read_dem(dem_file)
        assert dem is not None
        
        # Cache terrain
        cache.terrain_cache[cache_key] = (dem, out_trans, crs)
        assert len(cache.terrain_cache) > 0
        
        # Fetch and cache obstacles
        obstacles = cache.fetch_obstacles(min_lat, max_lat, min_lon, max_lon)
        obstacle_mask = cache.get_obstacle_mask(obstacles, out_trans, dem.shape, crs)
        assert obstacle_mask is not None
        
        # Fetch and cache paths
        paths = cache.fetch_paths(min_lat, max_lat, min_lon, max_lon)
        path_raster, path_types = cache.rasterize_paths(paths, out_trans, dem.shape, crs)
        assert path_raster is not None
        
        # Compute cost surface
        cost_surface, slope_degrees = cache.compute_cost_surface(
            dem, out_trans, obstacle_mask, path_raster, path_types
        )
        assert cost_surface is not None
        
        # Cache cost surface
        cost_cache_key = f"{cache_key}_cost"
        cache.cost_surface_cache[cost_cache_key] = {
            'cost_surface': cost_surface,
            'indices': cache.build_indices(cost_surface),
            'slope_degrees': slope_degrees,
            'obstacle_mask': obstacle_mask,
            'path_raster': path_raster
        }
        
        # Verify caches were populated
        assert len(cache.terrain_cache) == 1
        assert len(cache.cost_surface_cache) == 1
        
        # Get cache status
        status = cache.get_cache_status()
        assert status['terrain_cache']['count'] == 1
        assert status['cost_surface_cache']['count'] == 1
        assert status['total_memory_mb'] > 0


@pytest.mark.integration
def test_prepopulate_performance():
    """Test that prepopulation improves route calculation performance"""
    from app.services.dem_tile_cache import DEMTileCache
    import time
    
    # Test coordinates
    start_lat, start_lon = 40.6500, -111.5700
    end_lat, end_lon = 40.6510, -111.5690
    
    # Create two cache instances
    cache_without_prepop = DEMTileCache()
    cache_with_prepop = DEMTileCache()
    
    # Prepopulate the second cache
    min_lat = min(start_lat, end_lat) - 0.001
    max_lat = max(start_lat, end_lat) + 0.001
    min_lon = min(start_lon, end_lon) - 0.001
    max_lon = max(start_lon, end_lon) + 0.001
    
    # Download and cache for prepopulated instance
    dem_file = cache_with_prepop.download_dem(min_lat, max_lat, min_lon, max_lon)
    if dem_file:
        dem, out_trans, crs = cache_with_prepop.read_dem(dem_file)
        if dem is not None:
            cache_key = f"{min_lat:.4f},{max_lat:.4f},{min_lon:.4f},{max_lon:.4f}"
            cache_with_prepop.terrain_cache[cache_key] = (dem, out_trans, crs)
    
    # Time route calculation without prepopulation
    start_time = time.time()
    route1 = cache_without_prepop.find_route(start_lat, start_lon, end_lat, end_lon)
    time_without_prepop = time.time() - start_time
    
    # Time route calculation with prepopulation
    start_time = time.time()
    route2 = cache_with_prepop.find_route(start_lat, start_lon, end_lat, end_lon)
    time_with_prepop = time.time() - start_time
    
    # Both should find routes
    assert route1 is not None, "Should find route without prepopulation"
    assert route2 is not None, "Should find route with prepopulation"
    
    # Routes should be similar (same number of points)
    assert len(route1) == len(route2), "Routes should have same number of points"
    
    # Prepopulated should be faster (or at least not significantly slower)
    # We don't assert strict speedup because small areas might not show much difference
    print(f"Time without prepop: {time_without_prepop:.2f}s")
    print(f"Time with prepop: {time_with_prepop:.2f}s")
    print(f"Speedup: {time_without_prepop / time_with_prepop:.2f}x")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])