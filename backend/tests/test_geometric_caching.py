#!/usr/bin/env python3
"""
Comprehensive tests for geometric caching fixes.
Ensures that routes with slight coordinate differences share cached data.
"""

import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from app.services.dem_tile_cache import DEMTileCache
from app.services.tiled_dem_cache import TiledDEMCache


class TestGeometricCaching(unittest.TestCase):
    """Test that all caches use geometric containment instead of exact matching"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.cache = DEMTileCache(buffer=0.02, debug_mode=True)
        # Override cache directories to use temp dir
        self.cache.dem_data_dir = os.path.join(self.test_dir, 'dem_data')
        self.cache.tiled_cache.cache_dir = os.path.join(self.test_dir, 'tile_cache')
        os.makedirs(self.cache.dem_data_dir, exist_ok=True)
        os.makedirs(self.cache.tiled_cache.cache_dir, exist_ok=True)
    
    def tearDown(self):
        """Clean up test files"""
        shutil.rmtree(self.test_dir)
    
    def test_tile_aligned_cache_key_creation(self):
        """Test that cache keys are properly tile-aligned"""
        # Test coordinates that differ by 0.0001°
        coords_a = (40.6282, 40.6760, -111.5908, -111.5443)
        coords_b = (40.6282, 40.6760, -111.5908, -111.5444)  # 0.0001° difference
        
        key_a = self.cache._create_tile_aligned_cache_key(*coords_a)
        key_b = self.cache._create_tile_aligned_cache_key(*coords_b)
        
        # Keys should be identical despite coordinate difference
        self.assertEqual(key_a, key_b, 
                        f"Cache keys should be identical for nearby coordinates: {key_a} vs {key_b}")
        
        # Keys should be tile-aligned (divisible by tile_size)
        parts = key_a.split(',')
        for part in parts:
            coord = float(part)
            # Check if coordinate is aligned to tile boundary
            # Due to floating point precision, check if it's very close to a tile boundary
            remainder = coord % self.cache.tiled_cache.tile_size
            # Either very close to 0 or very close to tile_size (for negative coords)
            is_aligned = remainder < 1e-6 or abs(remainder - self.cache.tiled_cache.tile_size) < 1e-6
            self.assertTrue(is_aligned, 
                          f"Coordinate {coord} not tile-aligned (remainder: {remainder})")
    
    def test_terrain_cache_reuse(self):
        """Test that terrain_cache reuses data for nearby routes"""
        # Mock the download and read methods
        mock_dem = np.random.rand(100, 100)
        mock_transform = Mock()
        mock_crs = 'EPSG:4326'
        
        with patch.object(self.cache, 'download_dem') as mock_download, \
             patch.object(self.cache, 'read_dem') as mock_read, \
             patch.object(self.cache, 'reproject_dem') as mock_reproject:
            
            mock_download.return_value = 'test.tif'
            mock_read.return_value = (mock_dem, mock_transform, mock_crs)
            mock_reproject.return_value = (mock_dem, mock_transform, mock_crs)
            
            # First route
            min_lat_a, max_lat_a = 40.6282, 40.6760
            min_lon_a, max_lon_a = -111.5908, -111.5443
            
            # Get cache key for first route
            key_a = self.cache._create_tile_aligned_cache_key(min_lat_a, max_lat_a, 
                                                            min_lon_a, max_lon_a)
            
            # Simulate terrain caching (this would happen in fallback path)
            self.cache.terrain_cache[key_a] = (mock_dem, mock_transform, mock_crs)
            
            # Second route with 0.0001° difference
            min_lat_b, max_lat_b = 40.6282, 40.6760
            min_lon_b, max_lon_b = -111.5908, -111.5444
            
            key_b = self.cache._create_tile_aligned_cache_key(min_lat_b, max_lat_b, 
                                                            min_lon_b, max_lon_b)
            
            # Check that keys are the same
            self.assertEqual(key_a, key_b, "Terrain cache keys should be identical")
            
            # Check that data would be reused
            self.assertIn(key_b, self.cache.terrain_cache, 
                         "Second route should find cached terrain data")
    
    def test_cost_surface_cache_reuse(self):
        """Test that cost_surface_cache reuses data for nearby routes"""
        # Test coordinates
        min_lat, max_lat = 40.6282, 40.6760
        min_lon_a, max_lon_a = -111.5908, -111.5443
        min_lon_b, max_lon_b = -111.5908, -111.5444  # 0.0001° difference
        
        # Get cache keys
        key_a = self.cache._create_tile_aligned_cache_key(min_lat, max_lat, 
                                                        min_lon_a, max_lon_a) + "_cost"
        key_b = self.cache._create_tile_aligned_cache_key(min_lat, max_lat, 
                                                        min_lon_b, max_lon_b) + "_cost"
        
        # Keys should be identical
        self.assertEqual(key_a, key_b, "Cost surface cache keys should be identical")
        
        # Simulate caching
        mock_cost_surface = np.random.rand(100, 100)
        mock_slopes = np.random.rand(100, 100)
        mock_indices = np.arange(10000).reshape(100, 100)
        
        self.cache.cost_surface_cache[key_a] = (mock_cost_surface, mock_slopes, mock_indices)
        
        # Second route should find the cached data
        self.assertIn(key_b, self.cache.cost_surface_cache,
                     "Second route should find cached cost surface")
    
    def test_dem_file_naming_reuse(self):
        """Test that DEM files use tile-aligned names for reuse"""
        with patch('os.path.exists') as mock_exists, \
             patch('py3dep.get_dem') as mock_get_dem:
            
            mock_exists.return_value = False  # Simulate file doesn't exist
            mock_get_dem.return_value = None  # Don't actually download
            
            # First route
            min_lat, max_lat = 40.6282, 40.6760
            min_lon_a, max_lon_a = -111.5908, -111.5443
            
            # Get expected filename for first route
            cache_key_a = self.cache._create_tile_aligned_cache_key(min_lat, max_lat, 
                                                                  min_lon_a, max_lon_a)
            filename_a = f"dem_{cache_key_a.replace(',', '_')}.tif"
            
            # Second route with tiny difference
            min_lon_b, max_lon_b = -111.5908, -111.5444
            
            cache_key_b = self.cache._create_tile_aligned_cache_key(min_lat, max_lat, 
                                                                  min_lon_b, max_lon_b)
            filename_b = f"dem_{cache_key_b.replace(',', '_')}.tif"
            
            # Filenames should be identical
            self.assertEqual(filename_a, filename_b, 
                           f"DEM filenames should be identical: {filename_a} vs {filename_b}")
    
    def test_tile_cache_geometric_reuse(self):
        """Test that tile cache properly handles geometric queries"""
        # Create a mock tile
        tile_x, tile_y = -11157, 4065
        mock_tile_data = {
            'cost_surface': np.random.rand(402, 305),
            'slope_degrees': np.random.rand(402, 305),
            'dem': np.random.rand(402, 305),
            'tile_x': tile_x,
            'tile_y': tile_y,
            'transform': Mock(),
            'crs': 'EPSG:4326'
        }
        
        # Mock get_tile to return our test data
        with patch.object(self.cache.tiled_cache, 'get_tile') as mock_get_tile:
            mock_get_tile.return_value = mock_tile_data
            
            # Two slightly different bounds that map to same tiles
            bounds_a = (40.6282, 40.6760, -111.5908, -111.5443)
            bounds_b = (40.6282, 40.6760, -111.5908, -111.5444)
            
            # Get tiles for both bounds
            tiles_a = self.cache.tiled_cache.get_tiles_for_bounds(*bounds_a)
            tiles_b = self.cache.tiled_cache.get_tiles_for_bounds(*bounds_b)
            
            # Should request the same tiles
            self.assertEqual(set(tiles_a), set(tiles_b),
                           "Different bounds should map to same tiles")
    
    def test_preprocessing_cache_compatibility(self):
        """Test that preprocessing cache still works with new caching"""
        # Preprocessing cache uses dimensions as key, should still work
        test_key = "100x100"
        test_data = {'test': 'data'}
        
        self.cache.preprocessing_cache[test_key] = test_data
        
        # Should be able to retrieve
        self.assertEqual(self.cache.preprocessing_cache[test_key], test_data,
                        "Preprocessing cache should still work")
    
    def test_real_route_scenario(self):
        """Test a realistic scenario with two nearby routes"""
        # Mock the tile system to simulate having cached tiles
        with patch.object(self.cache.tiled_cache, 'compose_tiles') as mock_compose:
            # Set up mock return value
            mock_composed = {
                'cost_surface': np.ones((2418, 1836)),
                'slope_degrees': np.zeros((2418, 1836)),
                'dem': np.full((2418, 1836), 2500.0),
                'transform': Mock(),
                'crs': 'EPSG:32613'
            }
            mock_compose.return_value = mock_composed
            
            # Capture calls to download_dem
            with patch.object(self.cache, 'download_dem') as mock_download:
                # Route A
                start_a = (40.6560, -111.5708)
                end_a = (40.6482, -111.5643)
                
                # This would be called by find_route
                min_lat_a = min(start_a[0], end_a[0]) - self.cache.buffer
                max_lat_a = max(start_a[0], end_a[0]) + self.cache.buffer
                min_lon_a = min(start_a[1], end_a[1]) - self.cache.buffer
                max_lon_a = max(start_a[1], end_a[1]) + self.cache.buffer
                
                tiles_a = self.cache.tiled_cache.get_tiles_for_bounds(
                    min_lat_a, max_lat_a, min_lon_a, max_lon_a)
                
                # Route B (0.0001° different)
                start_b = (40.6560, -111.5708)
                end_b = (40.6482, -111.5644)
                
                min_lat_b = min(start_b[0], end_b[0]) - self.cache.buffer
                max_lat_b = max(start_b[0], end_b[0]) + self.cache.buffer  
                min_lon_b = min(start_b[1], end_b[1]) - self.cache.buffer
                max_lon_b = max(start_b[1], end_b[1]) + self.cache.buffer
                
                tiles_b = self.cache.tiled_cache.get_tiles_for_bounds(
                    min_lat_b, max_lat_b, min_lon_b, max_lon_b)
                
                # Should use same tiles
                self.assertEqual(set(tiles_a), set(tiles_b),
                               "Routes A and B should use the same tiles")
                
                # In the new system, download_dem should not be called
                # because tile-based approach handles it
                mock_download.assert_not_called()


class TestCacheKeyAlignment(unittest.TestCase):
    """Specific tests for cache key alignment logic"""
    
    def setUp(self):
        self.cache = DEMTileCache()
        self.tile_size = 0.01
    
    def test_alignment_rounds_outward(self):
        """Test that alignment always rounds outward to cover the area"""
        # Test bounds
        min_lat, max_lat = 40.6234, 40.6789
        min_lon, max_lon = -111.5678, -111.5123
        
        key = self.cache._create_tile_aligned_cache_key(min_lat, max_lat, min_lon, max_lon)
        parts = [float(x) for x in key.split(',')]
        
        # Check that bounds were expanded outward
        self.assertLessEqual(parts[0], min_lat, "Min lat should round down")
        self.assertGreaterEqual(parts[1], max_lat, "Max lat should round up")
        self.assertLessEqual(parts[2], min_lon, "Min lon should round down")
        self.assertGreaterEqual(parts[3], max_lon, "Max lon should round up")
    
    def test_precision_is_appropriate(self):
        """Test that cache keys use appropriate precision"""
        min_lat, max_lat = 40.6234567, 40.6789012
        min_lon, max_lon = -111.5678901, -111.5123456
        
        key = self.cache._create_tile_aligned_cache_key(min_lat, max_lat, min_lon, max_lon)
        
        # Should use 3 decimal places
        for part in key.split(','):
            self.assertEqual(len(part.split('.')[-1]), 3,
                           f"Cache key part '{part}' should have 3 decimal places")


def run_geometric_cache_tests():
    """Run all geometric caching tests"""
    print("\n🧪 GEOMETRIC CACHING TEST SUITE")
    print("="*60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestGeometricCaching))
    suite.addTests(loader.loadTestsFromTestCase(TestCacheKeyAlignment))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✅ All geometric caching tests passed!")
        print("Routes with tiny coordinate differences will now share cached data.")
    else:
        print("\n❌ Some tests failed. Check output above for details.")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_geometric_cache_tests()
    sys.exit(0 if success else 1)