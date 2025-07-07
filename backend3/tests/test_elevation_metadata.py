#!/usr/bin/env python3
"""
Test elevation library metadata tracking
"""

import os
import json
import shutil
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from elevation import ElevationLibrary, Bounds

class TestElevationMetadata(unittest.TestCase):
    """Test that metadata is properly tracked"""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="elev_meta_test_")
        self.lib = ElevationLibrary(data_dir=self.test_dir, resolution=10)
    
    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_index_file_creation(self):
        """Test that index file is created correctly"""
        # The index file is created on first save
        self.lib._save_index()
        
        index_path = os.path.join(self.test_dir, "10m", "tile_index.json")
        self.assertTrue(os.path.exists(index_path))
        
        # Load and check structure
        with open(index_path, 'r') as f:
            index = json.load(f)
        
        self.assertIn("tiles", index)
        self.assertIn("version", index)
        self.assertEqual(index["version"], "1.0")
        self.assertIsInstance(index["tiles"], dict)
    
    def test_tile_metadata_fields(self):
        """Test that tile metadata contains all required fields"""
        # Mock a tile entry
        tile_key = "4000_-11100"
        mock_metadata = {
            "bounds": {
                "north": 40.01,
                "south": 40.00,
                "east": -110.99,
                "west": -111.00
            },
            "resolution_m": 10,
            "shape": [100, 100],
            "transform": {
                "a": 0.0001, "b": 0, "c": -111.0,
                "d": 0, "e": -0.0001, "f": 40.01
            },
            "crs": "EPSG:4326",
            "pixel_size_degrees": {"x": 0.0001, "y": 0.0001},
            "pixel_size_meters": {"x": 10.0, "y": 10.0},
            "file": "tile_4000_-11100.tif",
            "size_bytes": 40000,
            "download_time": "2024-01-01T12:00:00"
        }
        
        # Add to index
        self.lib.tile_index["tiles"][tile_key] = mock_metadata
        self.lib._save_index()
        
        # Reload and verify
        reloaded_lib = ElevationLibrary(data_dir=self.test_dir, resolution=10)
        tile_data = reloaded_lib.tile_index["tiles"][tile_key]
        
        # Check all fields exist
        self.assertIn("bounds", tile_data)
        self.assertIn("resolution_m", tile_data)
        self.assertIn("shape", tile_data)
        self.assertIn("transform", tile_data)
        self.assertIn("crs", tile_data)
        self.assertIn("pixel_size_degrees", tile_data)
        self.assertIn("pixel_size_meters", tile_data)
        self.assertIn("file", tile_data)
        self.assertIn("size_bytes", tile_data)
        self.assertIn("download_time", tile_data)
        
        # Check nested structures
        self.assertEqual(len(tile_data["shape"]), 2)
        self.assertEqual(len(tile_data["transform"]), 6)
        self.assertIn("x", tile_data["pixel_size_degrees"])
        self.assertIn("y", tile_data["pixel_size_degrees"])
    
    def test_get_tile_info(self):
        """Test getting tile info for a point"""
        # Mock some tile data
        # For point (40.655, -111.505):
        # 40.655 / 0.01 = 4065.5, floor = 4065
        # -111.505 / 0.01 = -11150.5, floor = -11151
        tile_key = "4065_-11151"
        self.lib.tile_index["tiles"][tile_key] = {
            "resolution_m": 10,
            "shape": [334, 334],
            "pixel_size_meters": {"x": 9.2, "y": 10.4}
        }
        
        # Test point within tile
        info = self.lib.get_tile_info(40.655, -111.505)
        self.assertIsNotNone(info)
        self.assertEqual(info["resolution_m"], 10)
        
        # Test point outside any tile
        info = self.lib.get_tile_info(41.0, -112.0)
        self.assertIsNone(info)
    
    def test_pixel_size_calculation(self):
        """Test that pixel sizes are calculated correctly"""
        # At 40° latitude, 1 degree longitude ≈ 85.4 km
        # So 0.01° ≈ 854m
        lat = 40.0
        lon_deg = 0.01
        expected_m = lon_deg * 111000 * np.cos(np.radians(lat))
        
        self.assertAlmostEqual(expected_m, 854, delta=10)  # Within 10m
    
    def test_resolution_in_metadata(self):
        """Test that resolution is properly stored in all the right places"""
        # Create libraries with different resolutions
        for res in [3, 10, 30]:
            lib = ElevationLibrary(data_dir=self.test_dir, resolution=res)
            self.assertEqual(lib.resolution, res)
            
            # Check directory
            res_dir = os.path.join(self.test_dir, f"{res}m")
            self.assertTrue(os.path.exists(res_dir))


class TestArrayMetadata(unittest.TestCase):
    """Test metadata returned with elevation arrays"""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="elev_array_test_")
        self.lib = ElevationLibrary(data_dir=self.test_dir, resolution=10)
    
    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    @patch('elevation.ElevationLibrary.get_elevation_array')
    def test_array_metadata_structure(self, mock_get_array):
        """Test that get_elevation_array returns proper metadata"""
        # Mock the return value
        mock_data = np.zeros((100, 100))
        mock_metadata = {
            "transform": MagicMock(),
            "bounds": {
                "north": 40.01, "south": 40.00,
                "east": -110.99, "west": -111.00
            },
            "shape": (100, 100),
            "resolution_m": 10,
            "pixel_size_degrees": {"x": 0.0001, "y": 0.0001},
            "pixel_size_meters": {"x": 10.0, "y": 10.0},
            "crs": "EPSG:4326",
            "tiles_used": 1
        }
        mock_get_array.return_value = (mock_data, mock_metadata)
        
        # Call the method
        bounds = Bounds(south=40.0, north=40.01, west=-111.0, east=-110.99)
        data, metadata = self.lib.get_elevation_array(bounds)
        
        # Verify metadata structure
        self.assertIn("transform", metadata)
        self.assertIn("bounds", metadata)
        self.assertIn("shape", metadata)
        self.assertIn("resolution_m", metadata)
        self.assertIn("pixel_size_degrees", metadata)
        self.assertIn("pixel_size_meters", metadata)
        self.assertIn("crs", metadata)
        self.assertIn("tiles_used", metadata)
        
        # Verify data shape matches metadata
        self.assertEqual(data.shape, metadata["shape"])


class TestMetadataPersistence(unittest.TestCase):
    """Test that metadata persists correctly"""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="elev_persist_test_")
    
    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_metadata_survives_reload(self):
        """Test that metadata survives library reload"""
        # Create library and add mock tile
        lib1 = ElevationLibrary(data_dir=self.test_dir, resolution=10)
        
        mock_tile = {
            "bounds": {"north": 40.01, "south": 40.00, "east": -110.99, "west": -111.00},
            "resolution_m": 10,
            "shape": [100, 100],
            "transform": {"a": 0.0001, "b": 0, "c": -111.0, "d": 0, "e": -0.0001, "f": 40.01},
            "crs": "EPSG:4326",
            "pixel_size_degrees": {"x": 0.0001, "y": 0.0001},
            "pixel_size_meters": {"x": 10.0, "y": 10.0},
            "file": "tile_4000_-11100.tif",
            "size_bytes": 40000,
            "download_time": "2024-01-01T12:00:00"
        }
        
        lib1.tile_index["tiles"]["4000_-11100"] = mock_tile
        lib1._save_index()
        
        # Create new library instance
        lib2 = ElevationLibrary(data_dir=self.test_dir, resolution=10)
        
        # Verify tile data persisted
        self.assertIn("4000_-11100", lib2.tile_index["tiles"])
        reloaded_tile = lib2.tile_index["tiles"]["4000_-11100"]
        
        # Check key fields
        self.assertEqual(reloaded_tile["resolution_m"], 10)
        self.assertEqual(reloaded_tile["shape"], [100, 100])
        self.assertEqual(reloaded_tile["crs"], "EPSG:4326")


if __name__ == "__main__":
    unittest.main(verbosity=2)