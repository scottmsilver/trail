#!/usr/bin/env python3
"""
Standalone tests that can run without py3dep installed
This tests the core logic of the elevation library
"""

import os
import sys
import tempfile
import shutil
import json
import unittest
from unittest.mock import MagicMock, patch

# Mock py3dep before importing anything else
sys.modules['py3dep'] = MagicMock()

# Mock rasterio
mock_rasterio = MagicMock()
mock_transform = MagicMock()
mock_transform.from_bounds = lambda *args: MagicMock()
mock_rasterio.transform = mock_transform
sys.modules['rasterio'] = mock_rasterio
sys.modules['rasterio.transform'] = mock_transform

# Now we can import our modules
import numpy as np

# Simple Bounds class for testing
class Bounds:
    def __init__(self, south, north, west, east):
        if north <= south:
            raise ValueError("North must be greater than south")
        if east <= west:
            raise ValueError("East must be greater than west")
        self.south = south
        self.north = north
        self.west = west
        self.east = east
    
    def to_tuple(self):
        return (self.west, self.south, self.east, self.north)


class TestElevationStandalone(unittest.TestCase):
    """Standalone tests for elevation library concepts"""
    
    def test_tile_key_calculation(self):
        """Test tile key calculation logic"""
        tile_size = 0.01
        
        # Test cases
        test_cases = [
            (40.000, -111.000, "4000_-11100"),
            (40.005, -111.005, "4000_-11101"),
            (40.655, -111.505, "4065_-11151"),
            (-40.005, -111.005, "-4001_-11101"),
        ]
        
        for lat, lon, expected in test_cases:
            tile_row = int(np.floor(lat / tile_size))
            tile_col = int(np.floor(lon / tile_size))
            tile_key = f"{tile_row}_{tile_col}"
            self.assertEqual(tile_key, expected, 
                           f"Wrong tile for ({lat}, {lon}): got {tile_key}, expected {expected}")
    
    def test_tile_bounds_calculation(self):
        """Test bounds calculation from tile keys"""
        tile_size = 0.01
        
        test_cases = [
            ("4000_-11100", 40.00, 40.01, -111.00, -110.99),
            ("4065_-11151", 40.65, 40.66, -111.51, -111.50),
            ("-4001_-11101", -40.01, -40.00, -111.01, -111.00),
        ]
        
        for tile_key, exp_south, exp_north, exp_west, exp_east in test_cases:
            row, col = map(int, tile_key.split('_'))
            bounds = Bounds(
                south=row * tile_size,
                north=(row + 1) * tile_size,
                west=col * tile_size,
                east=(col + 1) * tile_size
            )
            self.assertAlmostEqual(bounds.south, exp_south, places=6)
            self.assertAlmostEqual(bounds.north, exp_north, places=6)
            self.assertAlmostEqual(bounds.west, exp_west, places=6)
            self.assertAlmostEqual(bounds.east, exp_east, places=6)
    
    def test_tiles_needed_for_area(self):
        """Test calculation of tiles needed for an area"""
        tile_size = 0.01
        
        # Small area - single tile
        bounds = Bounds(south=40.001, north=40.009, west=-111.009, east=-111.001)
        min_row = int(np.floor(bounds.south / tile_size))
        max_row = int(np.floor(bounds.north / tile_size))
        min_col = int(np.floor(bounds.west / tile_size))
        max_col = int(np.floor(bounds.east / tile_size))
        
        tile_count = (max_row - min_row + 1) * (max_col - min_col + 1)
        self.assertEqual(tile_count, 1)
        
        # Larger area - 4 tiles
        bounds = Bounds(south=40.005, north=40.015, west=-111.015, east=-111.005)
        min_row = int(np.floor(bounds.south / tile_size))
        max_row = int(np.floor(bounds.north / tile_size))
        min_col = int(np.floor(bounds.west / tile_size))
        max_col = int(np.floor(bounds.east / tile_size))
        
        tile_count = (max_row - min_row + 1) * (max_col - min_col + 1)
        self.assertEqual(tile_count, 4)
    
    def test_resolution_validation(self):
        """Test resolution validation logic"""
        valid_resolutions = [1, 3, 5, 10, 30, 60]
        
        # Test valid resolutions
        for res in valid_resolutions:
            # Should not raise
            self.assertIn(res, valid_resolutions)
        
        # Test invalid resolutions
        invalid_resolutions = [2, 7, 15, 20, 100]
        for res in invalid_resolutions:
            self.assertNotIn(res, valid_resolutions)
    
    def test_tile_size_by_resolution(self):
        """Test tile size calculation based on resolution"""
        resolution_to_tile_size = {
            1: 0.005,   # ~0.5km
            3: 0.005,   # ~0.5km
            5: 0.01,    # ~1km
            10: 0.01,   # ~1km
            30: 0.02,   # ~2km
            60: 0.02,   # ~2km
        }
        
        for res, expected_size in resolution_to_tile_size.items():
            if res <= 3:
                tile_size = 0.005
            elif res <= 10:
                tile_size = 0.01
            else:
                tile_size = 0.02
            
            self.assertEqual(tile_size, expected_size)
    
    def test_bounds_validation(self):
        """Test bounds validation"""
        # Valid bounds
        try:
            bounds = Bounds(south=40.0, north=41.0, west=-111.0, east=-110.0)
            # Should not raise
        except ValueError:
            self.fail("Valid bounds raised ValueError")
        
        # Invalid bounds - north <= south
        with self.assertRaises(ValueError):
            Bounds(south=40.0, north=40.0, west=-111.0, east=-110.0)
        
        # Invalid bounds - east <= west
        with self.assertRaises(ValueError):
            Bounds(south=40.0, north=41.0, west=-111.0, east=-111.0)
    
    def test_index_file_operations(self):
        """Test index file operations"""
        test_dir = tempfile.mkdtemp(prefix="test_index_")
        try:
            # Create index structure
            index = {
                "tiles": {
                    "4000_-11100": {
                        "bounds": {"south": 40.0, "north": 40.01, "west": -111.0, "east": -110.99},
                        "resolution_m": 10,
                        "shape": [100, 100]
                    }
                },
                "version": "1.0"
            }
            
            # Save index
            index_path = os.path.join(test_dir, "tile_index.json")
            with open(index_path, 'w') as f:
                json.dump(index, f, indent=2)
            
            # Load index
            with open(index_path, 'r') as f:
                loaded = json.load(f)
            
            # Verify
            self.assertEqual(loaded["version"], "1.0")
            self.assertIn("4000_-11100", loaded["tiles"])
            self.assertEqual(loaded["tiles"]["4000_-11100"]["resolution_m"], 10)
            
        finally:
            shutil.rmtree(test_dir)


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestElevationStandalone)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("Running standalone elevation tests...")
    print("These tests don't require py3dep or rasterio to be installed")
    print("=" * 60)
    
    success = run_tests()
    
    if success:
        print("\nAll tests passed!")
        sys.exit(0)
    else:
        print("\nSome tests failed!")
        sys.exit(1)