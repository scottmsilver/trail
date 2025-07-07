#!/usr/bin/env python3
"""
Simple tests for elevation library focused on loading, removing, and finding data
"""

import os
import shutil
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import json


class TestElevationLoadRemoveFind(unittest.TestCase):
    """Test loading, removing, and finding data"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp(prefix="elev_simple_")
        # Mock py3dep at module level
        self.py3dep_patch = patch.dict('sys.modules', {'py3dep': MagicMock()})
        self.py3dep_patch.start()
        
        # Import after mocking
        from elevation import ElevationLibrary, Bounds
        self.ElevationLibrary = ElevationLibrary
        self.Bounds = Bounds
        self.lib = self.ElevationLibrary(data_dir=self.test_dir, resolution=10)
    
    def tearDown(self):
        """Clean up"""
        self.py3dep_patch.stop()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def add_fake_tile(self, tile_key, lat, lon):
        """Add a fake tile to the index"""
        self.lib.tile_index["tiles"][tile_key] = {
            "bounds": {
                "south": lat,
                "north": lat + 0.01,
                "west": lon,
                "east": lon + 0.01
            },
            "resolution_m": 10,
            "shape": [100, 100],
            "file": f"tile_{tile_key}.tif",
            "size_bytes": 50000
        }
        # Create fake file
        tile_path = os.path.join(self.test_dir, "10m", f"tile_{tile_key}.tif")
        with open(tile_path, 'wb') as f:
            f.write(b'FAKE_DATA')
        self.lib._save_index()
    
    def test_data_not_found_before_loading(self):
        """Test that data is not found before loading"""
        # Single point
        with self.assertRaises(ValueError) as ctx:
            self.lib.get_elevation(40.5, -111.5)
        self.assertIn("not available", str(ctx.exception))
        
        # Array
        bounds = self.Bounds(south=40.5, north=40.51, west=-111.51, east=-111.5)
        with self.assertRaises(ValueError) as ctx:
            self.lib.get_elevation_array(bounds)
        self.assertIn("not available", str(ctx.exception))
    
    def test_data_found_after_loading(self):
        """Test that data is found after simulated loading"""
        # Add fake tiles
        self.add_fake_tile("4050_-11150", 40.50, -111.50)
        self.add_fake_tile("4050_-11151", 40.50, -111.51)
        
        # Check tile info is available
        info = self.lib.get_tile_info(40.505, -111.505)
        self.assertIsNotNone(info)
        self.assertEqual(info["resolution_m"], 10)
        
        # List should show tiles
        listing = self.lib.list_loaded_areas()
        self.assertEqual(listing["total_tiles"], 2)
        self.assertGreater(listing["total_size_mb"], 0)
    
    def test_data_not_found_after_removal(self):
        """Test that data is not found after removal"""
        # Add fake tile - need correct tile key for 40.505, -111.505
        # 40.505 / 0.01 = 4050.5, floor = 4050
        # -111.505 / 0.01 = -11150.5, floor = -11151
        self.add_fake_tile("4050_-11151", 40.50, -111.51)
        
        # Verify it exists
        info = self.lib.get_tile_info(40.505, -111.505)
        self.assertIsNotNone(info)
        
        # Remove it - need bounds that include the tile
        bounds = self.Bounds(south=40.50, north=40.51, west=-111.51, east=-111.50)
        result = self.lib.remove_area(bounds)
        self.assertEqual(result["tiles_removed"], 1)
        
        # Verify it's gone
        info = self.lib.get_tile_info(40.505, -111.505)
        self.assertIsNone(info)
        
        # Should error when trying to get elevation
        with self.assertRaises(ValueError) as ctx:
            self.lib.get_elevation(40.505, -111.505)
        self.assertIn("not available", str(ctx.exception))
    
    def test_partial_data_fails(self):
        """Test that partial data coverage fails appropriately"""
        # Add only one tile out of four needed
        self.add_fake_tile("4050_-11151", 40.50, -111.51)
        
        # Request area that needs 4 tiles
        bounds = self.Bounds(south=40.505, north=40.515, west=-111.515, east=-111.505)
        
        with self.assertRaises(ValueError) as ctx:
            self.lib.get_elevation_array(bounds)
        
        error_msg = str(ctx.exception)
        self.assertIn("3 tiles", error_msg)  # 3 missing (we added one)
        self.assertIn("Missing tiles:", error_msg)
    
    def test_remove_all_functionality(self):
        """Test remove_all clears everything"""
        # Add multiple tiles
        for i in range(3):
            self.add_fake_tile(f"4050_-1115{i}", 40.50, -111.50 - i*0.01)
        
        # Verify they exist
        listing = self.lib.list_loaded_areas()
        self.assertEqual(listing["total_tiles"], 3)
        
        # Remove all
        result = self.lib.remove_all()
        self.assertEqual(result["tiles_removed"], 3)
        
        # Verify all gone
        listing = self.lib.list_loaded_areas()
        self.assertEqual(listing["total_tiles"], 0)
    
    def test_finding_correct_tile_for_coordinate(self):
        """Test that correct tile is found for given coordinates"""
        # Test various coordinates and their expected tiles
        # Note: coordinates exactly on boundaries go to the lower-numbered tile
        test_cases = [
            (40.000, -111.000, "3999_-11101"),  # Exact boundary -> lower tile
            (40.005, -111.005, "4000_-11101"),
            (40.655, -111.505, "4065_-11151"),
            (-40.005, -111.005, "-4001_-11101"),
            (40.001, -111.001, "4000_-11101"),  # Just inside tile
            (39.999, -110.999, "3999_-11100"),  # Just inside lower tile
        ]
        
        for lat, lon, expected_key in test_cases:
            tile_key = self.lib._get_tile_key(lat, lon)
            self.assertEqual(tile_key, expected_key, 
                           f"Wrong tile for ({lat}, {lon}): got {tile_key}, expected {expected_key}")
    
    def test_bounds_calculation_for_tiles(self):
        """Test that tile bounds are calculated correctly"""
        test_tiles = [
            ("4000_-11100", 40.00, 40.01, -111.00, -110.99),
            ("4065_-11151", 40.65, 40.66, -111.51, -111.50),
            ("-4001_-11101", -40.01, -40.00, -111.01, -111.00),
        ]
        
        for tile_key, exp_south, exp_north, exp_west, exp_east in test_tiles:
            bounds = self.lib._get_tile_bounds(tile_key)
            self.assertAlmostEqual(bounds.south, exp_south, places=6)
            self.assertAlmostEqual(bounds.north, exp_north, places=6)
            self.assertAlmostEqual(bounds.west, exp_west, places=6)
            self.assertAlmostEqual(bounds.east, exp_east, places=6)


class TestDataDirectory(unittest.TestCase):
    """Test data directory requirements"""
    
    def test_data_dir_required(self):
        """Test that data directory is required"""
        # Mock py3dep
        with patch.dict('sys.modules', {'py3dep': MagicMock()}):
            from elevation import ElevationLibrary
            
            # Should fail with no data_dir
            with self.assertRaises(ValueError) as ctx:
                ElevationLibrary(data_dir=None)
            self.assertIn("data_dir is required", str(ctx.exception))
            
            # Should fail with empty string
            with self.assertRaises(ValueError) as ctx:
                ElevationLibrary(data_dir="")
            self.assertIn("data_dir is required", str(ctx.exception))


class TestResolutionSupport(unittest.TestCase):
    """Test resolution support"""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="elev_res_")
        self.py3dep_patch = patch.dict('sys.modules', {'py3dep': MagicMock()})
        self.py3dep_patch.start()
    
    def tearDown(self):
        self.py3dep_patch.stop()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_valid_resolutions(self):
        """Test that valid resolutions are accepted"""
        from elevation import ElevationLibrary
        
        valid_resolutions = [1, 3, 5, 10, 30, 60]
        for res in valid_resolutions:
            lib = ElevationLibrary(data_dir=self.test_dir, resolution=res)
            self.assertEqual(lib.resolution, res)
            
            # Check directory created
            res_dir = os.path.join(self.test_dir, f"{res}m")
            self.assertTrue(os.path.exists(res_dir))
    
    def test_invalid_resolution(self):
        """Test that invalid resolutions are rejected"""
        from elevation import ElevationLibrary
        
        invalid_resolutions = [2, 7, 15, 20, 100]
        for res in invalid_resolutions:
            with self.assertRaises(ValueError) as ctx:
                ElevationLibrary(data_dir=self.test_dir, resolution=res)
            self.assertIn(f"Invalid resolution {res}m", str(ctx.exception))
    
    def test_tile_sizes_by_resolution(self):
        """Test that tile sizes vary by resolution"""
        from elevation import ElevationLibrary
        
        expected_tile_sizes = {
            1: 0.005,   # ~0.5km
            3: 0.005,   # ~0.5km
            5: 0.01,    # ~1km
            10: 0.01,   # ~1km
            30: 0.02,   # ~2km
            60: 0.02,   # ~2km
        }
        
        for res, expected_size in expected_tile_sizes.items():
            lib = ElevationLibrary(data_dir=self.test_dir, resolution=res)
            self.assertEqual(lib.tile_size, expected_size)


if __name__ == "__main__":
    unittest.main(verbosity=2)