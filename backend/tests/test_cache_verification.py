#!/usr/bin/env python3
"""
Test suite to verify caching behavior is working correctly.
Ensures that cached data is being used and provides expected performance improvements.
"""

import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import shutil
import tempfile
from unittest.mock import patch, MagicMock
from app.services.dem_tile_cache import DEMTileCache
from app.services.obstacle_config import ObstacleConfig
from app.services.path_preferences import PathPreferences


class TestCachingBehavior(unittest.TestCase):
    """Test suite for verifying caching functionality"""
    
    def setUp(self):
        """Set up test environment"""
        # Create temporary cache directory
        self.temp_dir = tempfile.mkdtemp()
        self.original_cache_dir = os.environ.get('TILE_CACHE_DIR', 'tile_cache')
        os.environ['TILE_CACHE_DIR'] = os.path.join(self.temp_dir, 'test_cache')
        
        # Initialize cache with test configuration
        self.cache = DEMTileCache(
            obstacle_config=ObstacleConfig(),
            path_preferences=PathPreferences()
        )
        
        # Test coordinates (Park City area)
        self.test_routes = [
            {
                'name': 'Short route',
                'start': (40.6572, -111.5709),
                'end': (40.6472, -111.5671)
            },
            {
                'name': 'Overlapping route',
                'start': (40.6550, -111.5700),
                'end': (40.6480, -111.5675)
            }
        ]
    
    def tearDown(self):
        """Clean up test environment"""
        # Restore original cache directory
        if self.original_cache_dir:
            os.environ['TILE_CACHE_DIR'] = self.original_cache_dir
        else:
            del os.environ['TILE_CACHE_DIR']
        
        # Clean up temporary directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_terrain_cache_hit(self):
        """Test that terrain data is cached and reused"""
        route = self.test_routes[0]
        
        # First run - should cache terrain
        start_time1 = time.time()
        path1 = self.cache.find_route(
            route['start'][0], route['start'][1],
            route['end'][0], route['end'][1]
        )
        time1 = time.time() - start_time1
        
        # Check that terrain was cached
        self.assertGreater(len(self.cache.terrain_cache), 0, 
                          "Terrain cache should contain entries after first run")
        
        # Second run - should use cached terrain
        start_time2 = time.time()
        path2 = self.cache.find_route(
            route['start'][0], route['start'][1],
            route['end'][0], route['end'][1]
        )
        time2 = time.time() - start_time2
        
        # Verify paths are identical
        self.assertIsNotNone(path1, "First path should be found")
        self.assertIsNotNone(path2, "Second path should be found")
        self.assertEqual(len(path1), len(path2), "Paths should have same length")
        
        # Second run should be faster
        self.assertLess(time2, time1 * 0.8, 
                       f"Cached run ({time2:.2f}s) should be at least 20% faster than first run ({time1:.2f}s)")
        
        print(f"\n✓ Terrain cache verification:")
        print(f"  First run: {time1:.2f}s")
        print(f"  Cached run: {time2:.2f}s")
        print(f"  Speedup: {time1/time2:.1f}x")
    
    def test_cost_surface_cache_hit(self):
        """Test that cost surfaces are cached and reused"""
        route = self.test_routes[0]
        
        # Clear terrain cache to isolate cost surface caching
        self.cache.terrain_cache.clear()
        
        initial_cost_cache_size = len(self.cache.cost_surface_cache)
        
        # First run
        path1 = self.cache.find_route(
            route['start'][0], route['start'][1],
            route['end'][0], route['end'][1]
        )
        
        # Check that cost surface was cached
        self.assertGreater(len(self.cache.cost_surface_cache), initial_cost_cache_size,
                          "Cost surface cache should grow after first run")
        
        # Track download calls
        with patch.object(self.cache, 'fetch_obstacles') as mock_fetch:
            # Second run - should not fetch obstacles again
            path2 = self.cache.find_route(
                route['start'][0], route['start'][1],
                route['end'][0], route['end'][1]
            )
            
            # Should not have called fetch_obstacles
            mock_fetch.assert_not_called()
        
        print(f"\n✓ Cost surface cache verification:")
        print(f"  Cost surfaces cached: {len(self.cache.cost_surface_cache)}")
    
    def test_tile_cache_persistence(self):
        """Test that tile cache persists between instances"""
        route = self.test_routes[0]
        
        # Create first cache instance and run route
        cache1 = DEMTileCache(
            obstacle_config=ObstacleConfig(),
            path_preferences=PathPreferences()
        )
        cache1.tiled_cache.cache_dir = os.path.join(self.temp_dir, 'tile_test')
        
        path1 = cache1.find_route(
            route['start'][0], route['start'][1],
            route['end'][0], route['end'][1]
        )
        
        # Count cached tiles
        tile_count = 0
        if os.path.exists(cache1.tiled_cache.cache_dir):
            for subdir in ['terrain', 'cost', 'obstacles']:
                path = os.path.join(cache1.tiled_cache.cache_dir, subdir)
                if os.path.exists(path):
                    tile_count += len([f for f in os.listdir(path) if f.endswith('.pkl')])
        
        self.assertGreater(tile_count, 0, "Should have cached tiles on disk")
        
        # Create new cache instance
        cache2 = DEMTileCache(
            obstacle_config=ObstacleConfig(),
            path_preferences=PathPreferences()
        )
        cache2.tiled_cache.cache_dir = os.path.join(self.temp_dir, 'tile_test')
        
        # Run same route - should load from disk cache
        start_time = time.time()
        path2 = cache2.find_route(
            route['start'][0], route['start'][1],
            route['end'][0], route['end'][1]
        )
        cached_time = time.time() - start_time
        
        self.assertIsNotNone(path2, "Should find path using cached tiles")
        
        print(f"\n✓ Tile cache persistence verification:")
        print(f"  Tiles cached to disk: {tile_count}")
        print(f"  New instance load time: {cached_time:.2f}s")
    
    def test_overlapping_routes_cache_efficiency(self):
        """Test that overlapping routes efficiently reuse cached data"""
        # Run first route
        route1 = self.test_routes[0]
        start_time1 = time.time()
        path1 = self.cache.find_route(
            route1['start'][0], route1['start'][1],
            route1['end'][0], route1['end'][1]
        )
        time1 = time.time() - start_time1
        
        initial_terrain_size = len(self.cache.terrain_cache)
        initial_cost_size = len(self.cache.cost_surface_cache)
        
        # Run overlapping route
        route2 = self.test_routes[1]
        start_time2 = time.time()
        path2 = self.cache.find_route(
            route2['start'][0], route2['start'][1],
            route2['end'][0], route2['end'][1]
        )
        time2 = time.time() - start_time2
        
        # Check cache growth
        final_terrain_size = len(self.cache.terrain_cache)
        final_cost_size = len(self.cache.cost_surface_cache)
        
        # For overlapping routes, cache shouldn't grow much
        terrain_growth = final_terrain_size - initial_terrain_size
        cost_growth = final_cost_size - initial_cost_size
        
        print(f"\n✓ Overlapping routes cache efficiency:")
        print(f"  Route 1 time: {time1:.2f}s")
        print(f"  Route 2 time: {time2:.2f}s (overlapping)")
        print(f"  Terrain cache growth: {terrain_growth} entries")
        print(f"  Cost cache growth: {cost_growth} entries")
        
        # Second route should be faster due to cache reuse
        self.assertLess(time2, time1 * 0.9, 
                       "Overlapping route should benefit from cached data")
    
    def test_cache_memory_limits(self):
        """Test that cache respects memory limits"""
        # Get initial memory usage
        initial_status = self.cache.get_cache_status()
        initial_memory = initial_status['total_memory_mb']
        
        # Run multiple routes to fill cache
        for i in range(3):
            # Offset each route slightly to create new cache entries
            offset = i * 0.01
            self.cache.find_route(
                40.65 + offset, -111.57,
                40.64 + offset, -111.56
            )
        
        # Check final memory usage
        final_status = self.cache.get_cache_status()
        final_memory = final_status['total_memory_mb']
        
        print(f"\n✓ Cache memory management:")
        print(f"  Initial memory: {initial_memory:.1f} MB")
        print(f"  Final memory: {final_memory:.1f} MB")
        print(f"  Terrain entries: {final_status['terrain_cache']['count']}")
        print(f"  Cost entries: {final_status['cost_surface_cache']['count']}")
        
        # Memory should have increased but be reasonable
        self.assertGreater(final_memory, initial_memory, 
                          "Cache memory should increase with use")
        self.assertLess(final_memory, 1000, 
                       "Cache memory should stay within reasonable limits")


class TestCachePerformance(unittest.TestCase):
    """Performance tests for caching"""
    
    def setUp(self):
        """Set up performance test environment"""
        self.cache = DEMTileCache(
            obstacle_config=ObstacleConfig(),
            path_preferences=PathPreferences()
        )
    
    def test_cache_warmup_performance(self):
        """Test performance improvement from cold to warm cache"""
        # Test route
        start_lat, start_lon = 40.6572, -111.5709
        end_lat, end_lon = 40.6472, -111.5671
        
        # Clear all caches
        self.cache.terrain_cache.clear()
        self.cache.cost_surface_cache.clear()
        self.cache.preprocessing_cache.clear()
        self.cache.tiled_cache.memory_cache.clear()
        
        # Cold cache run
        cold_start = time.time()
        cold_path = self.cache.find_route(start_lat, start_lon, end_lat, end_lon)
        cold_time = time.time() - cold_start
        
        # Warm cache runs
        warm_times = []
        for i in range(3):
            warm_start = time.time()
            warm_path = self.cache.find_route(start_lat, start_lon, end_lat, end_lon)
            warm_times.append(time.time() - warm_start)
        
        avg_warm_time = sum(warm_times) / len(warm_times)
        speedup = cold_time / avg_warm_time
        
        print(f"\n✓ Cache warmup performance:")
        print(f"  Cold cache: {cold_time:.2f}s")
        print(f"  Warm cache avg: {avg_warm_time:.2f}s")
        print(f"  Speedup: {speedup:.1f}x")
        
        self.assertGreater(speedup, 1.5, 
                          f"Warm cache should be at least 1.5x faster (got {speedup:.1f}x)")


def run_cache_verification():
    """Run all cache verification tests"""
    print("\n🔍 CACHE VERIFICATION TEST SUITE")
    print("="*60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestCachingBehavior))
    suite.addTests(loader.loadTestsFromTestCase(TestCachePerformance))
    
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
        print("\n✅ All cache verification tests passed!")
    else:
        print("\n❌ Some tests failed. Check output above for details.")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    
    success = run_cache_verification()
    sys.exit(0 if success else 1)