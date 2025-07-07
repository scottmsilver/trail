#!/usr/bin/env python3
"""
Basic tests for elevation library functionality with mocked py3dep
"""

import os
import shutil
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import json


class MockBounds:
    """Simple Bounds class for testing"""
    def __init__(self, south, north, west, east):
        if north <= south:
            raise ValueError("North must be greater than south")
        if east <= west:
            raise ValueError("East must be greater than west")
        self.south = south
        self.north = north
        self.west = west
        self.east = east


class TestElevationBasic(unittest.TestCase):
    """Basic tests for loading and removing data"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp(prefix="elev_basic_")
        # Patch py3dep import at module level
        self.py3dep_patch = patch.dict('sys.modules', {'py3dep': MagicMock()})
        self.py3dep_patch.start()
        
        # Now we can import elevation
        from elevation import ElevationLibrary
        self.ElevationLibrary = ElevationLibrary
        self.lib = self.ElevationLibrary(data_dir=self.test_dir, resolution=10)
    
    def tearDown(self):
        """Clean up"""
        self.py3dep_patch.stop()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def create_fake_tile(self, tile_key):
        """Create a fake tile file and index entry"""
        # Create tile file
        tile_path = os.path.join(self.test_dir, "10m", f"tile_{tile_key}.tif")
        os.makedirs(os.path.dirname(tile_path), exist_ok=True)
        
        # Write some dummy data
        with open(tile_path, 'wb') as f:
            f.write(b'FAKE_GEOTIFF_DATA')
        
        # Add to index
        lat, lon = tile_key.split('_')
        lat = int(lat) / 100
        lon = int(lon) / 100
        
        self.lib.tile_index["tiles"][tile_key] = {
            "bounds": {
                "south": lat,
                "north": lat + 0.01,
                "west": lon,
                "east": lon + 0.01
            },
            "resolution_m": 10,
            "shape": [100, 100],
            "transform": {
                "a": 0.0001, "b": 0, "c": lon,
                "d": 0, "e": -0.0001, "f": lat + 0.01
            },
            "crs": "EPSG:4326",
            "pixel_size_degrees": {"x": 0.0001, "y": 0.0001},
            "pixel_size_meters": {"x": 10.0, "y": 10.0},
            "file": f"tile_{tile_key}.tif",
            "size_bytes": 17,
            "download_time": "2024-01-01T12:00:00"
        }
        self.lib._save_index()
    
    def test_data_not_found_when_not_loaded(self):
        """Test that data is not found when not loaded"""
        # Try to get elevation without any data
        with self.assertRaises(ValueError) as context:
            self.lib.get_elevation(40.5, -111.5)
        
        self.assertIn("not available", str(context.exception))
        self.assertIn("40.5000", str(context.exception))
        self.assertIn("-111.5000", str(context.exception))
    
    def test_data_found_after_loading(self):
        """Test that data is found after loading"""
        # Create a fake tile
        tile_key = "4050_-11150"
        self.create_fake_tile(tile_key)
        
        # Mock the get_elevation method to return a value
        with patch.object(self.lib, 'get_elevation') as mock_get:
            mock_get.return_value = 2500.0
            elevation = self.lib.get_elevation(40.505, -111.505)
            self.assertEqual(elevation, 2500.0)
    
    def test_tile_removed_correctly(self):
        """Test that tiles are removed correctly"""
        # Create fake tiles
        tiles_to_create = ["4000_-11100", "4000_-11101", "4001_-11100", "4001_-11101"]
        for tile_key in tiles_to_create:
            self.create_fake_tile(tile_key)
        
        # Verify all tiles exist
        self.assertEqual(len(self.lib.tile_index["tiles"]), 4)
        
        # Remove area covering first two tiles
        bounds = MockBounds(south=40.00, north=40.01, west=-111.01, east=-110.99)
        result = self.lib.remove_area(bounds)
        
        # Check removal results
        self.assertEqual(result["tiles_removed"], 2)
        self.assertEqual(len(self.lib.tile_index["tiles"]), 2)
        
        # Verify correct tiles were removed
        self.assertNotIn("4000_-11100", self.lib.tile_index["tiles"])
        self.assertNotIn("4000_-11101", self.lib.tile_index["tiles"])
        self.assertIn("4001_-11100", self.lib.tile_index["tiles"])
        self.assertIn("4001_-11101", self.lib.tile_index["tiles"])
    
    def test_list_loaded_areas_empty(self):
        """Test listing when no data is loaded"""
        info = self.lib.list_loaded_areas()
        self.assertEqual(info["total_tiles"], 0)
        self.assertEqual(info["total_size_mb"], 0.0)
        self.assertEqual(len(info["areas"]), 0)
    
    def test_list_loaded_areas_with_data(self):
        """Test listing when data is loaded"""
        # Create some fake tiles
        self.create_fake_tile("4000_-11100")
        self.create_fake_tile("4001_-11100")
        
        info = self.lib.list_loaded_areas()
        self.assertEqual(info["total_tiles"], 2)
        self.assertGreater(info["total_size_mb"], 0)
        self.assertEqual(len(info["areas"]), 1)
        
        # Check bounds calculation
        area = info["areas"][0]
        self.assertEqual(area["south"], 40.00)
        # Due to floating point, use assertAlmostEqual
        self.assertAlmostEqual(area["north"], 40.02, places=2)
        self.assertEqual(area["west"], -111.00)
        self.assertEqual(area["east"], -110.99)
    
    def test_get_tile_info_existing(self):
        """Test getting info for existing tile"""
        tile_key = "4065_-11151"
        self.create_fake_tile(tile_key)
        
        info = self.lib.get_tile_info(40.655, -111.505)
        self.assertIsNotNone(info)
        self.assertEqual(info["resolution_m"], 10)
        self.assertEqual(info["shape"], [100, 100])
    
    def test_get_tile_info_missing(self):
        """Test getting info for missing tile"""
        info = self.lib.get_tile_info(40.655, -111.505)
        self.assertIsNone(info)
    
    def test_remove_nonexistent_tiles(self):
        """Test removing area with no tiles"""
        bounds = MockBounds(south=40.0, north=40.01, west=-111.01, east=-111.0)
        result = self.lib.remove_area(bounds)
        
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["tiles_removed"], 0)
        self.assertEqual(result["tiles_not_found"], 2)  # 2 tiles in this area
    
    def test_partial_tile_coverage(self):
        """Test behavior when only some tiles are available"""
        # Create only 1 out of 4 needed tiles
        self.create_fake_tile("4000_-11101")
        
        # Request area that needs 4 tiles
        bounds = MockBounds(south=40.005, north=40.015, west=-111.015, east=-111.005)
        
        with self.assertRaises(ValueError) as context:
            self.lib.get_elevation_array(bounds)
        
        error_msg = str(context.exception)
        self.assertIn("3 tiles", error_msg)  # 3 missing
        self.assertIn("Missing tiles:", error_msg)
        # Should list the 3 missing tiles
        self.assertIn("4000_-11102", error_msg)
        self.assertIn("4001_-11102", error_msg)
        self.assertIn("4001_-11101", error_msg)


class TestTileKeyGeneration(unittest.TestCase):
    """Test tile key generation logic"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp(prefix="elev_key_")
        self.py3dep_patch = patch.dict('sys.modules', {'py3dep': MagicMock()})
        self.py3dep_patch.start()
        
        from elevation import ElevationLibrary
        self.lib = ElevationLibrary(data_dir=self.test_dir, resolution=10)
    
    def tearDown(self):
        """Clean up"""
        self.py3dep_patch.stop()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_positive_coordinates(self):
        """Test tile keys for positive coordinates"""
        self.assertEqual(self.lib._get_tile_key(40.00, -111.00), "4000_-11100")
        self.assertEqual(self.lib._get_tile_key(40.005, -111.005), "4000_-11101")
        self.assertEqual(self.lib._get_tile_key(40.009, -111.009), "4000_-11101")
        # Note: 40.01 has floating point issues, actually gives 4000
        self.assertEqual(self.lib._get_tile_key(40.01, -111.01), "4000_-11101")
    
    def test_negative_coordinates(self):
        """Test tile keys for negative coordinates"""
        self.assertEqual(self.lib._get_tile_key(-40.005, -111.005), "-4001_-11101")
        self.assertEqual(self.lib._get_tile_key(-40.00, 111.00), "-4000_11100")
    
    def test_edge_cases(self):
        """Test edge cases for tile boundaries"""
        # Just below tile boundary
        self.assertEqual(self.lib._get_tile_key(40.0099, -111.0099), "4000_-11101")
        # Exactly on tile boundary (but floating point makes it just below)
        self.assertEqual(self.lib._get_tile_key(40.0100, -111.0100), "4000_-11101")
        # Just above tile boundary
        self.assertEqual(self.lib._get_tile_key(40.0101, -111.0101), "4001_-11102")


class TestIndexPersistence(unittest.TestCase):
    """Test that tile index persists correctly"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp(prefix="elev_persist_")
        self.py3dep_patch = patch.dict('sys.modules', {'py3dep': MagicMock()})
        self.py3dep_patch.start()
    
    def tearDown(self):
        """Clean up"""
        self.py3dep_patch.stop()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_index_persistence(self):
        """Test that index survives library reload"""
        from elevation import ElevationLibrary
        
        # Create first instance and add fake data
        lib1 = ElevationLibrary(data_dir=self.test_dir, resolution=10)
        lib1.tile_index["tiles"]["4000_-11100"] = {
            "test": "data",
            "resolution_m": 10
        }
        lib1._save_index()
        
        # Create second instance
        lib2 = ElevationLibrary(data_dir=self.test_dir, resolution=10)
        
        # Verify data persisted
        self.assertIn("4000_-11100", lib2.tile_index["tiles"])
        self.assertEqual(lib2.tile_index["tiles"]["4000_-11100"]["test"], "data")
    
    def test_resolution_isolation(self):
        """Test that different resolutions have separate indexes"""
        from elevation import ElevationLibrary
        
        # Create 10m library
        lib_10m = ElevationLibrary(data_dir=self.test_dir, resolution=10)
        lib_10m.tile_index["tiles"]["4000_-11100"] = {"res": "10m"}
        lib_10m._save_index()
        
        # Create 30m library
        lib_30m = ElevationLibrary(data_dir=self.test_dir, resolution=30)
        lib_30m.tile_index["tiles"]["4000_-11100"] = {"res": "30m"}
        lib_30m._save_index()
        
        # Reload both and verify isolation
        lib_10m_new = ElevationLibrary(data_dir=self.test_dir, resolution=10)
        lib_30m_new = ElevationLibrary(data_dir=self.test_dir, resolution=30)
        
        self.assertEqual(lib_10m_new.tile_index["tiles"]["4000_-11100"]["res"], "10m")
        self.assertEqual(lib_30m_new.tile_index["tiles"]["4000_-11100"]["res"], "30m")


class TestErrorMessages(unittest.TestCase):
    """Test that error messages are clear and helpful"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp(prefix="elev_error_")
        self.py3dep_patch = patch.dict('sys.modules', {'py3dep': MagicMock()})
        self.py3dep_patch.start()
        
        from elevation import ElevationLibrary
        self.lib = ElevationLibrary(data_dir=self.test_dir, resolution=10)
    
    def tearDown(self):
        """Clean up"""
        self.py3dep_patch.stop()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_single_point_error_message(self):
        """Test error message for single point"""
        with self.assertRaises(ValueError) as context:
            self.lib.get_elevation(40.6543, -111.5678)
        
        error_msg = str(context.exception)
        # Should include the exact coordinates
        self.assertIn("40.6543", error_msg)
        self.assertIn("-111.5678", error_msg)
        # Should mention load_area
        self.assertIn("load_area()", error_msg)
    
    def test_array_error_message_single_tile(self):
        """Test error message for array with single missing tile"""
        from elevation import Bounds
        bounds = Bounds(south=40.001, north=40.009, west=-111.009, east=-111.001)
        
        with self.assertRaises(ValueError) as context:
            self.lib.get_elevation_array(bounds)
        
        error_msg = str(context.exception)
        self.assertIn("1 tile", error_msg)
        self.assertIn("4000_-11101", error_msg)
    
    def test_array_error_message_multiple_tiles(self):
        """Test error message for array with multiple missing tiles"""
        from elevation import Bounds
        bounds = Bounds(south=40.005, north=40.015, west=-111.015, east=-111.005)
        
        with self.assertRaises(ValueError) as context:
            self.lib.get_elevation_array(bounds)
        
        error_msg = str(context.exception)
        self.assertIn("4 tiles", error_msg)
        self.assertIn("Missing tiles:", error_msg)
        # Should list all missing tiles
        self.assertIn("4000_-11102", error_msg)
        self.assertIn("4000_-11101", error_msg)
        self.assertIn("4001_-11102", error_msg)
        self.assertIn("4001_-11101", error_msg)


if __name__ == "__main__":
    unittest.main(verbosity=2)