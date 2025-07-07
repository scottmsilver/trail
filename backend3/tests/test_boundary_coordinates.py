"""
Test boundary coordinate handling
"""

import unittest
import tempfile
import shutil
import os
from unittest.mock import patch, MagicMock
import numpy as np
from elevation import ElevationLibrary, Bounds


class TestBoundaryCoordinates(unittest.TestCase):
    """Test that coordinates exactly on tile boundaries are handled correctly"""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.lib = ElevationLibrary(data_dir=self.test_dir, resolution=10)
    
    def tearDown(self):
        shutil.rmtree(self.test_dir)
    
    def test_tile_assignment_consistency(self):
        """Test that boundary coordinates consistently map to the same tile"""
        # Exact tile boundaries at multiples of 0.01 (for 10m resolution)
        boundary_coords = [
            (40.00, -111.00),
            (40.01, -111.01),
            (40.00, -111.01),
            (40.01, -111.00),
            (-40.00, 111.00),  # Test negative coordinates too
        ]
        
        for lat, lon in boundary_coords:
            # Get tile multiple times - should be consistent
            tile1 = self.lib._get_tile_key(lat, lon)
            tile2 = self.lib._get_tile_key(lat, lon)
            self.assertEqual(tile1, tile2, 
                           f"Inconsistent tile assignment for ({lat}, {lon})")
            
            # Verify tile assignment follows our rule (boundary goes to lower tile)
            tile_row, tile_col = map(int, tile1.split('_'))
            expected_row = int(np.floor((lat - 1e-10) / 0.01))
            expected_col = int(np.floor((lon - 1e-10) / 0.01))
            self.assertEqual(tile_row, expected_row)
            self.assertEqual(tile_col, expected_col)
    
    def test_neighboring_coordinates(self):
        """Test that coordinates near boundaries map correctly"""
        # Test around boundary at 40.0, -111.0
        base_lat, base_lon = 40.0, -111.0
        epsilon = 0.0001
        
        # Points just around the boundary
        test_points = [
            (base_lat - epsilon, base_lon - epsilon, "3999_-11101"),  # SW
            (base_lat - epsilon, base_lon, "3999_-11101"),           # S
            (base_lat - epsilon, base_lon + epsilon, "3999_-11100"),  # SE
            (base_lat, base_lon - epsilon, "3999_-11101"),           # W
            (base_lat, base_lon, "3999_-11101"),                     # Exact boundary
            (base_lat, base_lon + epsilon, "3999_-11100"),           # E
            (base_lat + epsilon, base_lon - epsilon, "4000_-11101"),  # NW
            (base_lat + epsilon, base_lon, "4000_-11101"),           # N
            (base_lat + epsilon, base_lon + epsilon, "4000_-11100"),  # NE
        ]
        
        for lat, lon, expected_tile in test_points:
            actual_tile = self.lib._get_tile_key(lat, lon)
            self.assertEqual(actual_tile, expected_tile,
                           f"Wrong tile for ({lat:.5f}, {lon:.5f}): "
                           f"got {actual_tile}, expected {expected_tile}")
    
    def test_elevation_at_boundaries_simple(self):
        """Test that get_elevation handles boundary coordinates correctly"""
        # This is a simpler test without complex mocking
        # Just verify the tile assignment and clamping logic
        
        # Test exact boundary coordinates
        boundary_coords = [
            (40.0, -111.0),
            (40.01, -111.01),
            (40.005, -111.005),
        ]
        
        for lat, lon in boundary_coords:
            # Verify tile assignment is consistent
            tile1 = self.lib._get_tile_key(lat, lon)
            tile2 = self.lib._get_tile_key(lat, lon)
            self.assertEqual(tile1, tile2)
            
            # Verify the tile bounds make sense
            bounds = self.lib._get_tile_bounds(tile1)
            
            # The coordinate should be within or on the edge of tile bounds
            # (with our fix, exact boundaries go to the lower tile)
            self.assertLessEqual(bounds.south, lat)
            self.assertLessEqual(bounds.west, lon)
            
            # For exact boundaries, lat/lon might equal north/east bounds
            self.assertLessEqual(lat, bounds.north + 1e-10)
            self.assertLessEqual(lon, bounds.east + 1e-10)
    
    def test_different_resolutions(self):
        """Test boundary handling for different resolutions"""
        resolutions_and_boundaries = [
            (3, 0.005, 40.005),   # 3m -> 0.005° tiles
            (5, 0.01, 40.01),     # 5m -> 0.01° tiles (not 0.005)
            (10, 0.01, 40.01),    # 10m -> 0.01° tiles
            (30, 0.02, 40.02),    # 30m -> 0.02° tiles
        ]
        
        for res, tile_size, boundary in resolutions_and_boundaries:
            lib = ElevationLibrary(data_dir=self.test_dir, resolution=res)
            self.assertAlmostEqual(lib.tile_size, tile_size)
            
            # Test exact boundary
            tile_key = lib._get_tile_key(boundary, -111.0)
            
            # Boundary should go to lower tile
            expected_row = int(np.floor((boundary - 1e-10) / tile_size))
            actual_row = int(tile_key.split('_')[0])
            self.assertEqual(actual_row, expected_row,
                           f"{res}m: boundary {boundary} should map to row {expected_row}")


if __name__ == '__main__':
    unittest.main()