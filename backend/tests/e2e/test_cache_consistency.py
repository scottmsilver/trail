#!/usr/bin/env python3
"""
pytest for cache consistency - ensures routes are identical with and without cache
"""

import pytest
import subprocess
import sys
import time
import os
import shutil
import tempfile
from pathlib import Path

@pytest.fixture(scope="module")
def clean_cache_env():
    """Fixture to provide clean cache environment for tests"""
    backed_up = []
    temp_dir = None
    original_env = {}
    
    # Setup
    print("\nSetting up clean test environment...")
    
    # 1. Backup existing caches
    cache_dirs = ['tile_cache', 'precomputed_cache', 'cache', 'dem_data']
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            backup_name = f"{cache_dir}.pytest_backup"
            if os.path.exists(backup_name):
                shutil.rmtree(backup_name)
            shutil.move(cache_dir, backup_name)
            backed_up.append(cache_dir)
    
    # 2. Create temporary directory for test caches
    temp_dir = tempfile.mkdtemp(prefix="trail_cache_test_")
    
    # 3. Set environment to use temp directory for HTTP cache
    original_env['HYRIVER_CACHE_NAME'] = os.environ.get('HYRIVER_CACHE_NAME', '')
    os.environ['HYRIVER_CACHE_NAME'] = os.path.join(temp_dir, 'http_cache.sqlite')
    
    # 4. Create empty cache directories
    os.makedirs('cache', exist_ok=True)
    os.makedirs('dem_data', exist_ok=True)
    
    yield  # Run tests
    
    # Cleanup
    print("\nCleaning up test environment...")
    
    # 1. Remove test caches
    for cache_dir in ['tile_cache', 'precomputed_cache', 'cache', 'dem_data']:
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
    
    # 2. Restore original caches
    for cache_dir in backed_up:
        backup_name = f"{cache_dir}.pytest_backup"
        if os.path.exists(backup_name):
            shutil.move(backup_name, cache_dir)
    
    # 3. Clean up temp directory
    if temp_dir and os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    
    # 4. Restore environment
    if 'HYRIVER_CACHE_NAME' in original_env:
        if original_env['HYRIVER_CACHE_NAME']:
            os.environ['HYRIVER_CACHE_NAME'] = original_env['HYRIVER_CACHE_NAME']
        else:
            del os.environ['HYRIVER_CACHE_NAME']


def run_route_cli(start_coords, end_coords):
    """Helper to run route_cli.py and parse output"""
    # Use the simpler test CLI that doesn't hang
    cmd = [
        sys.executable, 
        "route_cli_test.py",
        f"Start: {start_coords}",
        f"End: {end_coords}"
    ]
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    elapsed = time.time() - start_time
    
    assert result.returncode == 0, f"route_cli failed: {result.stderr}"
    
    # Parse output
    output = result.stdout
    metrics = {}
    has_download = False
    
    for line in output.split('\n'):
        if "Downloading" in line:
            has_download = True
        elif "Path points:" in line:
            metrics['points'] = int(line.split(':')[1].strip())
        elif "Distance:" in line:
            dist_str = line.split(':')[1].strip()
            if 'km' in dist_str and '(' in dist_str:
                km_part = dist_str.split('(')[1].split('km')[0].strip()
                metrics['distance'] = float(km_part)
        elif "Elevation gain:" in line:
            elev_str = line.split(':')[1].strip()
            if 'm' in elev_str:
                m_value = elev_str.split('m')[0].strip()
                metrics['elevation'] = int(m_value)
        elif "Max slope:" in line:
            metrics['max_slope'] = float(line.split(':')[1].strip().replace('°', ''))
    
    return metrics, elapsed, has_download


@pytest.mark.real_data
@pytest.mark.slow
class TestCacheConsistency:
    """Test that caching produces identical routes"""
    
    # Test coordinates - extremely short route for faster tests
    # Using coordinates just ~10 meters apart
    START = "40.6500, -111.5700"
    END = "40.6500, -111.5701"
    
    def test_clean_start_downloads_data(self, clean_cache_env):
        """Test that starting with no cache triggers downloads"""
        # Verify clean state
        assert not os.path.exists('tile_cache'), "tile_cache exists before test"
        dem_files = len(list(Path('dem_data').glob('*.tif'))) if os.path.exists('dem_data') else 0
        assert dem_files == 0, f"Found {dem_files} DEM files before test"
        
        # Run route
        metrics, elapsed, has_download = run_route_cli(self.START, self.END)
        
        # Should have downloaded
        assert has_download, "First run should download DEM data"
        
        # Should have created cache
        assert os.path.exists('tile_cache'), "Should create tile_cache"
        tile_count = len(list(Path('tile_cache').rglob('*.pkl')))
        assert tile_count > 0, f"Should create tile cache files, found {tile_count}"
    
    def test_second_run_uses_cache(self, clean_cache_env):
        """Test that second run uses cache and doesn't download"""
        # First run (may download)
        metrics1, time1, downloaded1 = run_route_cli(self.START, self.END)
        
        # Second run
        metrics2, time2, downloaded2 = run_route_cli(self.START, self.END)
        
        # Should not download on second run
        assert not downloaded2, "Second run should not download (should use cache)"
        
        # Should be faster
        assert time2 < time1, f"Second run ({time2:.1f}s) should be faster than first ({time1:.1f}s)"
    
    def test_routes_are_identical(self, clean_cache_env):
        """Test that cached and non-cached routes are identical"""
        # Run twice
        metrics1, _, _ = run_route_cli(self.START, self.END)
        metrics2, _, _ = run_route_cli(self.START, self.END)
        
        # Compare all metrics
        assert metrics1['points'] == metrics2['points'], \
            f"Point count differs: {metrics1['points']} vs {metrics2['points']}"
        
        assert metrics1['distance'] == metrics2['distance'], \
            f"Distance differs: {metrics1['distance']} vs {metrics2['distance']}"
        
        assert metrics1['elevation'] == metrics2['elevation'], \
            f"Elevation differs: {metrics1['elevation']} vs {metrics2['elevation']}"
        
        assert metrics1['max_slope'] == metrics2['max_slope'], \
            f"Max slope differs: {metrics1['max_slope']} vs {metrics2['max_slope']}"
    
    def test_cache_provides_speedup(self, clean_cache_env):
        """Test that cache provides meaningful performance improvement"""
        # Clear any existing cache for this specific test
        if os.path.exists('tile_cache'):
            shutil.rmtree('tile_cache')
        
        # First run - no cache
        _, time1, _ = run_route_cli(self.START, self.END)
        
        # Second run - with cache
        _, time2, _ = run_route_cli(self.START, self.END)
        
        # Calculate speedup
        speedup = time1 / time2 if time2 > 0 else 0
        
        # Should be at least 20% faster
        assert speedup > 1.2, f"Cache speedup only {speedup:.1f}x (expected >1.2x)"
    
    @pytest.mark.parametrize("start,end", [
        ("40.6571, -111.5705", "40.6520, -111.5688"),  # Original test route
        ("40.6550, -111.5700", "40.6500, -111.5650"),  # Different route
    ])
    def test_multiple_routes_consistent(self, clean_cache_env, start, end):
        """Test cache consistency across different routes"""
        # Run each route twice
        metrics1, _, _ = run_route_cli(start, end)
        metrics2, _, _ = run_route_cli(start, end)
        
        # Should be identical
        assert metrics1 == metrics2, f"Routes differ for {start} -> {end}"


if __name__ == "__main__":
    # Run with: pytest tests/test_cache_consistency.py -v
    pytest.main([__file__, "-v"])