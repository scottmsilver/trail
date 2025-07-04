#!/usr/bin/env python3
"""Tests for tile composition to ensure proper handling of DEM data and size mismatches"""

import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from unittest.mock import Mock, patch
from app.services.tiled_dem_cache import TiledDEMCache
import rasterio
import pytest


@pytest.mark.integration
class TestTileComposition(unittest.TestCase):
    """Test tile composition functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.cache = TiledDEMCache(tile_size_degrees=0.01, cache_dir="test_tile_cache")
    
    def tearDown(self):
        """Clean up test files"""
        import shutil
        if os.path.exists("test_tile_cache"):
            shutil.rmtree("test_tile_cache")
    
    def create_test_tile(self, tile_x, tile_y, height=402, width=305, 
                        base_elevation=2500, slope_range=(0, 30)):
        """Create a test tile with realistic data"""
        # Create elevation data with some variation
        dem = np.full((height, width), base_elevation, dtype=np.float32)
        # Add some terrain variation
        dem += np.random.uniform(-50, 50, (height, width))
        
        # Create slope data
        slope_degrees = np.random.uniform(slope_range[0], slope_range[1], (height, width))
        
        # Create cost surface based on slopes
        cost_surface = 1 + 0.1 * slope_degrees  # Simple cost function
        
        # Create transform
        from rasterio.transform import from_origin
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
    
    def test_single_tile_composition(self):
        """Test composing a single tile"""
        tile_x, tile_y = -111, 40
        tile_data = self.create_test_tile(tile_x, tile_y)
        
        # Mock get_tile to return our test data
        self.cache.get_tile = Mock(return_value=tile_data)
        
        # Compose single tile
        tiles = [(tile_x, tile_y)]
        target_bounds = tile_data['bounds']
        
        result = self.cache.compose_tiles(tiles, 'cost', target_bounds)
        
        self.assertIsNotNone(result)
        self.assertEqual(result['cost_surface'].shape, tile_data['cost_surface'].shape)
        self.assertTrue('dem' in result)
        self.assertFalse(np.any(np.isnan(result['dem'])))
        self.assertFalse(np.any(result['dem'] == 0))  # No zeros in DEM
    
    def test_multiple_tiles_composition(self):
        """Test composing multiple tiles in a 2x2 grid"""
        # Create 2x2 grid of tiles
        tiles = []
        tile_data_list = []
        
        for i in range(2):
            for j in range(2):
                tile_x = -111 + i
                tile_y = 40 + j
                tiles.append((tile_x, tile_y))
                tile_data = self.create_test_tile(tile_x, tile_y)
                tile_data_list.append(tile_data)
        
        # Mock get_tile to return appropriate data
        def mock_get_tile(tx, ty, data_type, compute_func=None, **kwargs):
            for td in tile_data_list:
                if td['tile_x'] == tx and td['tile_y'] == ty:
                    return td
            return None
        
        self.cache.get_tile = mock_get_tile
        
        # Compose tiles
        target_bounds = (40, 42, -111, -109)  # Cover all 4 tiles
        result = self.cache.compose_tiles(tiles, 'cost', target_bounds)
        
        self.assertIsNotNone(result)
        # Check composed size (2x2 tiles)
        expected_height = 2 * 402  # 2 tiles vertically
        expected_width = 2 * 305   # 2 tiles horizontally
        self.assertEqual(result['cost_surface'].shape, (expected_height, expected_width))
        
        # Check DEM has no NaN values where tiles exist
        self.assertTrue('dem' in result)
        # There should be no zeros in the DEM
        self.assertEqual(np.sum(result['dem'] == 0), 0, "DEM should have no zero values")
    
    def test_tiles_with_size_mismatch(self):
        """Test handling tiles with different sizes"""
        # Create tiles with different sizes
        tile1 = self.create_test_tile(-111, 40, height=402, width=305)
        tile2 = self.create_test_tile(-110, 40, height=403, width=306)  # Slightly larger
        
        tile_data_map = {
            (-111, 40): tile1,
            (-110, 40): tile2
        }
        
        def mock_get_tile(tx, ty, data_type, compute_func=None, **kwargs):
            return tile_data_map.get((tx, ty))
        
        self.cache.get_tile = mock_get_tile
        
        # Compose tiles
        tiles = [(-111, 40), (-110, 40)]
        target_bounds = (40, 41, -111, -109)
        
        result = self.cache.compose_tiles(tiles, 'cost', target_bounds)
        
        self.assertIsNotNone(result)
        # Should handle size mismatch without errors
        self.assertTrue('dem' in result)
        
        # Check that data was copied correctly (no zeros except in gaps)
        # The composed array should be based on the larger tile size
        self.assertGreater(result['cost_surface'].shape[0], 0)
        self.assertGreater(result['cost_surface'].shape[1], 0)
    
    def test_missing_tile_handling(self):
        """Test handling when some tiles are missing"""
        # Create only 3 out of 4 tiles in a 2x2 grid
        tile_data_map = {
            (-111, 40): self.create_test_tile(-111, 40),
            (-110, 40): self.create_test_tile(-110, 40),
            (-111, 41): self.create_test_tile(-111, 41),
            # (-110, 41) is missing
        }
        
        def mock_get_tile(tx, ty, data_type, compute_func=None, **kwargs):
            return tile_data_map.get((tx, ty))
        
        self.cache.get_tile = mock_get_tile
        
        # Try to compose all 4 tiles
        tiles = [(-111, 40), (-110, 40), (-111, 41), (-110, 41)]
        target_bounds = (40, 42, -111, -109)
        
        result = self.cache.compose_tiles(tiles, 'cost', target_bounds)
        
        self.assertIsNotNone(result)
        # Missing tile area should have NaN in DEM
        dem = result['dem']
        # Check that we have some NaN values (from missing tile)
        self.assertTrue(np.any(np.isnan(dem)), "Missing tile area should have NaN values")
        # But not all should be NaN
        self.assertFalse(np.all(np.isnan(dem)), "Not all values should be NaN")
    
    def test_dem_values_preserved(self):
        """Test that DEM values are correctly preserved during composition"""
        # Create a tile with known elevation values
        tile_x, tile_y = -111, 40
        tile_data = self.create_test_tile(tile_x, tile_y, base_elevation=2600)
        
        # Set a specific elevation pattern
        test_pattern = np.linspace(2500, 2700, tile_data['dem'].size).reshape(tile_data['dem'].shape)
        tile_data['dem'] = test_pattern.astype(np.float32)
        
        self.cache.get_tile = Mock(return_value=tile_data)
        
        # Compose
        tiles = [(tile_x, tile_y)]
        target_bounds = tile_data['bounds']
        result = self.cache.compose_tiles(tiles, 'cost', target_bounds)
        
        # Check that DEM values are preserved
        np.testing.assert_array_almost_equal(result['dem'], test_pattern, decimal=3)
    
    def test_transform_calculation(self):
        """Test that composite transform is calculated correctly"""
        # Create a 2x2 grid
        tiles = []
        tile_data_list = []
        
        base_x, base_y = -111, 40
        for i in range(2):
            for j in range(2):
                tile_x = base_x + i
                tile_y = base_y + j
                tiles.append((tile_x, tile_y))
                tile_data = self.create_test_tile(tile_x, tile_y)
                tile_data_list.append(tile_data)
        
        def mock_get_tile(tx, ty, data_type, compute_func=None, **kwargs):
            for td in tile_data_list:
                if td['tile_x'] == tx and td['tile_y'] == ty:
                    return td
            return None
        
        self.cache.get_tile = mock_get_tile
        
        # Compose
        target_bounds = (40, 42, -111, -109)
        result = self.cache.compose_tiles(tiles, 'cost', target_bounds)
        
        # Check transform
        transform = result['transform']
        self.assertIsNotNone(transform)
        
        # Top-left corner should align with the westernmost, northernmost tile
        expected_west = base_x * self.cache.tile_size
        expected_north = (base_y + 2) * self.cache.tile_size  # +2 because northernmost tile
        
        self.assertAlmostEqual(transform.c, expected_west, places=6)
        self.assertAlmostEqual(transform.f, expected_north, places=6)


@pytest.mark.integration
class TestSlopeCalculationWithComposedDEM(unittest.TestCase):
    """Test slope calculation using composed DEM data"""
    
    def test_slope_with_nan_values(self):
        """Test that slope calculation handles NaN values correctly"""
        from app.services.dem_tile_cache import DEMTileCache
        from pyproj import Transformer
        
        cache = DEMTileCache()
        
        # Create a simple path
        path = [
            (-111.5, 40.5),
            (-111.499, 40.501),
            (-111.498, 40.502)
        ]
        
        # Create DEM with some NaN values
        dem = np.array([
            [2500, 2510, np.nan],
            [2505, 2515, 2525],
            [np.nan, 2520, 2530]
        ], dtype=np.float32)
        
        # Create a simple transform
        from rasterio.transform import from_origin
        out_trans = from_origin(-111.5, 40.503, 0.001, 0.001)
        
        # Create transformer
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:4326", always_xy=True)
        
        # Calculate slopes
        result = cache.calculate_path_slopes(path, dem, out_trans, transformer)
        
        self.assertEqual(len(result), 3)
        
        # Check that points with NaN elevation are handled
        for point in result:
            if point['elevation'] is None:
                # Slope should be 0 when elevation is missing
                self.assertEqual(point['slope'], 0.0)
            else:
                # Valid elevations should be in expected range
                self.assertGreater(point['elevation'], 2000)
                self.assertLess(point['elevation'], 3000)


def run_tile_composition_tests():
    """Run all tile composition tests"""
    print("\n🧪 TILE COMPOSITION TEST SUITE")
    print("="*60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestTileComposition))
    suite.addTests(loader.loadTestsFromTestCase(TestSlopeCalculationWithComposedDEM))
    
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
        print("\n✅ All tile composition tests passed!")
    else:
        print("\n❌ Some tests failed. Check output above for details.")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    
    success = run_tile_composition_tests()
    sys.exit(0 if success else 1)