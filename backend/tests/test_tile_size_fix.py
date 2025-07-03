#!/usr/bin/env python3
"""Test that tile size variations are handled correctly after the fix"""

import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from unittest.mock import Mock, patch
from app.services.tiled_dem_cache import TiledDEMCache
import logging


class TestTileSizeFix(unittest.TestCase):
    """Test the tile size mismatch fix"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.cache = TiledDEMCache(tile_size_degrees=0.01)
        # Capture log messages from the specific module logger
        self.log_messages = []
        self.handler = logging.StreamHandler()
        self.handler.emit = lambda record: self.log_messages.append(record.getMessage())
        # Get the logger used by tiled_dem_cache module
        self.logger = logging.getLogger('app.services.tiled_dem_cache')
        self.logger.addHandler(self.handler)
        self.logger.setLevel(logging.INFO)
    
    def tearDown(self):
        """Clean up after tests"""
        self.logger.removeHandler(self.handler)
    
    def create_test_tile(self, tile_x, tile_y, height, width, base_elevation=2500):
        """Create a test tile with specified dimensions"""
        from rasterio.transform import from_origin
        
        dem = np.full((height, width), base_elevation, dtype=np.float32)
        dem += np.random.uniform(-50, 50, (height, width))
        
        slope_degrees = np.random.uniform(0, 30, (height, width))
        cost_surface = 1 + 0.1 * slope_degrees
        
        west = tile_x * self.cache.tile_size
        north = (tile_y + 1) * self.cache.tile_size
        pixel_size = self.cache.tile_size / width
        transform = from_origin(west, north, pixel_size, pixel_size)
        
        return {
            'tile_x': tile_x,
            'tile_y': tile_y,
            'cost_surface': cost_surface.astype(np.float32),
            'slope_degrees': slope_degrees.astype(np.float32),
            'dem': dem.astype(np.float32),
            'transform': transform,
            'crs': 'EPSG:4326',
            'bounds': (tile_y * self.cache.tile_size, 
                      (tile_y + 1) * self.cache.tile_size,
                      tile_x * self.cache.tile_size, 
                      (tile_x + 1) * self.cache.tile_size)
        }
    
    def test_no_warnings_for_common_variations(self):
        """Test that common tile size variations don't produce warnings"""
        # Create tiles with common size variations
        tiles = []
        tile_data_list = []
        
        # Most tiles are 402x305
        for i in range(3):
            for j in range(3):
                tile_x = -111 + i
                tile_y = 40 + j
                tiles.append((tile_x, tile_y))
                # Most tiles have one size
                if (i, j) != (1, 1):
                    tile_data = self.create_test_tile(tile_x, tile_y, 402, 305)
                else:
                    # One tile has slightly different size
                    tile_data = self.create_test_tile(tile_x, tile_y, 403, 306)
                tile_data_list.append(tile_data)
        
        # Mock get_tile
        def mock_get_tile(tx, ty, data_type, compute_func=None, **kwargs):
            for td in tile_data_list:
                if td['tile_x'] == tx and td['tile_y'] == ty:
                    return td
            return None
        
        self.cache.get_tile = mock_get_tile
        
        # Clear log messages
        self.log_messages.clear()
        
        # Compose tiles
        target_bounds = (40, 43, -111, -108)
        result = self.cache.compose_tiles(tiles, 'cost', target_bounds)
        
        self.assertIsNotNone(result)
        # Check that no warning messages were logged
        warning_messages = [msg for msg in self.log_messages if 'mismatch' in msg.lower()]
        self.assertEqual(len(warning_messages), 0, f"Found warning messages: {warning_messages}")
        
        # Check that informative messages were logged instead
        info_messages = [msg for msg in self.log_messages if 'size variations detected' in msg]
        self.assertGreater(len(info_messages), 0, "Should log size variations informatively")
    
    def test_outlier_tile_handling(self):
        """Test that outlier tiles (very small) are handled gracefully"""
        tiles = []
        tile_data_list = []
        
        # Create normal tiles and one outlier
        tile_data_list.append(self.create_test_tile(-111, 40, 402, 305))
        tile_data_list.append(self.create_test_tile(-110, 40, 402, 305))
        tile_data_list.append(self.create_test_tile(-111, 41, 122, 91))  # Outlier
        tile_data_list.append(self.create_test_tile(-110, 41, 402, 305))
        
        tiles = [(-111, 40), (-110, 40), (-111, 41), (-110, 41)]
        
        # Mock get_tile
        def mock_get_tile(tx, ty, data_type, compute_func=None, **kwargs):
            for td in tile_data_list:
                if td['tile_x'] == tx and td['tile_y'] == ty:
                    return td
            return None
        
        self.cache.get_tile = mock_get_tile
        
        # Clear log messages
        self.log_messages.clear()
        
        # Compose tiles
        target_bounds = (40, 42, -111, -109)
        result = self.cache.compose_tiles(tiles, 'cost', target_bounds)
        
        self.assertIsNotNone(result)
        
        # Check that outlier was logged
        outlier_messages = [msg for msg in self.log_messages if 'Outlier tile' in msg]
        self.assertGreater(len(outlier_messages), 0, "Should log outlier tiles")
        
        # Verify the outlier tile info is in the message
        self.assertTrue(any('122x91' in msg for msg in outlier_messages))
    
    def test_all_tiles_same_size(self):
        """Test that no size variation messages are logged when all tiles are the same size"""
        tiles = []
        tile_data_list = []
        
        # All tiles have the same size
        for i in range(2):
            for j in range(2):
                tile_x = -111 + i
                tile_y = 40 + j
                tiles.append((tile_x, tile_y))
                tile_data = self.create_test_tile(tile_x, tile_y, 402, 305)
                tile_data_list.append(tile_data)
        
        # Mock get_tile
        def mock_get_tile(tx, ty, data_type, compute_func=None, **kwargs):
            for td in tile_data_list:
                if td['tile_x'] == tx and td['tile_y'] == ty:
                    return td
            return None
        
        self.cache.get_tile = mock_get_tile
        
        # Clear log messages
        self.log_messages.clear()
        
        # Compose tiles
        target_bounds = (40, 42, -111, -109)
        result = self.cache.compose_tiles(tiles, 'cost', target_bounds)
        
        self.assertIsNotNone(result)
        
        # Check that no size variation messages were logged
        variation_messages = [msg for msg in self.log_messages if 'size variations detected' in msg]
        self.assertEqual(len(variation_messages), 0, "Should not log size variations when all tiles are the same size")


def run_tile_size_fix_tests():
    """Run tile size fix tests"""
    print("\n🧪 TILE SIZE FIX TEST SUITE")
    print("="*60)
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTileSizeFix)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if result.wasSuccessful():
        print("\n✅ All tile size fix tests passed!")
    else:
        print("\n❌ Some tests failed.")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    
    success = run_tile_size_fix_tests()
    sys.exit(0 if success else 1)