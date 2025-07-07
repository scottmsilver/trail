#!/usr/bin/env python3
"""
Integration tests for elevation library
Tests the full workflow without actually downloading data
"""

import os
import shutil
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from elevation import ElevationLibrary, Bounds

class TestElevationIntegration(unittest.TestCase):
    """Integration tests for full workflows"""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="elev_integration_")
        self.lib = ElevationLibrary(data_dir=self.test_dir, resolution=10)
    
    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def create_mock_tile(self, tile_key, bounds):
        """Create a mock elevation tile file"""
        height, width = 100, 100
        data = np.random.uniform(2000, 2500, (height, width)).astype(np.float32)
        
        transform = from_bounds(
            bounds.west, bounds.south,
            bounds.east, bounds.north,
            width, height
        )
        
        filename = self.lib._get_tile_filename(tile_key)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with rasterio.open(
            filename, 'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=data.dtype,
            crs='EPSG:4326',
            transform=transform,
            compress='lzw'
        ) as dst:
            dst.write(data, 1)
        
        # Update index
        self.lib.tile_index["tiles"][tile_key] = {
            "bounds": {
                "north": bounds.north,
                "south": bounds.south,
                "east": bounds.east,
                "west": bounds.west
            },
            "resolution_m": self.lib.resolution,
            "shape": [height, width],
            "transform": {
                "a": transform.a,
                "b": transform.b,
                "c": transform.c,
                "d": transform.d,
                "e": transform.e,
                "f": transform.f
            },
            "crs": "EPSG:4326",
            "pixel_size_degrees": {
                "x": transform.a,
                "y": abs(transform.e)
            },
            "pixel_size_meters": {
                "x": transform.a * 111000 * np.cos(np.radians((bounds.north + bounds.south) / 2)),
                "y": abs(transform.e) * 111000
            },
            "file": os.path.basename(filename),
            "size_bytes": os.path.getsize(filename),
            "download_time": "2024-01-01T12:00:00"
        }
        self.lib._save_index()
        
        return data
    
    def test_single_point_retrieval(self):
        """Test getting elevation at a single point"""
        # Create mock tile
        tile_bounds = Bounds(south=40.00, north=40.01, west=-111.01, east=-111.00)
        tile_key = "4000_-11101"
        mock_data = self.create_mock_tile(tile_key, tile_bounds)
        
        # Test point in middle of tile
        elevation = self.lib.get_elevation(40.005, -111.005)
        
        # Should be within range of mock data
        self.assertGreaterEqual(elevation, 2000)
        self.assertLessEqual(elevation, 2500)
    
    def test_array_retrieval_single_tile(self):
        """Test getting array that fits in single tile"""
        # Create mock tile
        tile_bounds = Bounds(south=40.00, north=40.01, west=-111.01, east=-111.00)
        tile_key = "4000_-11101"
        mock_data = self.create_mock_tile(tile_key, tile_bounds)
        
        # Request subset of tile
        request_bounds = Bounds(south=40.002, north=40.008, west=-111.008, east=-111.002)
        data, metadata = self.lib.get_elevation_array(request_bounds)
        
        # Check data
        self.assertIsInstance(data, np.ndarray)
        self.assertEqual(len(data.shape), 2)
        
        # Check metadata
        self.assertEqual(metadata["resolution_m"], 10)
        self.assertEqual(metadata["crs"], "EPSG:4326")
        self.assertEqual(metadata["tiles_used"], 1)
        
        # Data should be within expected range
        self.assertGreaterEqual(data.min(), 2000)
        self.assertLessEqual(data.max(), 2500)
    
    def test_array_retrieval_multiple_tiles(self):
        """Test getting array that spans multiple tiles"""
        # Create 2x2 tile grid
        tiles = [
            ("4000_-11102", Bounds(north=40.01, south=40.00, east=-111.01, west=-111.02)),
            ("4000_-11101", Bounds(north=40.01, south=40.00, east=-111.00, west=-111.01)),
            ("4001_-11102", Bounds(north=40.02, south=40.01, east=-111.01, west=-111.02)),
            ("4001_-11101", Bounds(north=40.02, south=40.01, east=-111.00, west=-111.01)),
        ]
        
        for tile_key, bounds in tiles:
            self.create_mock_tile(tile_key, bounds)
        
        # Request area spanning all tiles
        request_bounds = Bounds(south=40.005, north=40.015, west=-111.015, east=-111.005)
        data, metadata = self.lib.get_elevation_array(request_bounds)
        
        # Check that we used 4 tiles
        self.assertEqual(metadata["tiles_used"], 4)
        
        # Array should be larger than single tile
        self.assertGreater(data.shape[0], 100)
        self.assertGreater(data.shape[1], 100)
    
    def test_remove_area(self):
        """Test removing tiles"""
        # Create mock tiles
        tile_bounds = Bounds(south=40.00, north=40.01, west=-111.01, east=-111.00)
        tile_key = "4000_-11101"
        self.create_mock_tile(tile_key, tile_bounds)
        
        # Verify tile exists
        self.assertIn(tile_key, self.lib.tile_index["tiles"])
        tile_file = self.lib._get_tile_filename(tile_key)
        self.assertTrue(os.path.exists(tile_file))
        
        # Remove area
        result = self.lib.remove_area(tile_bounds)
        self.assertEqual(result["tiles_removed"], 1)
        
        # Verify tile is gone
        self.assertNotIn(tile_key, self.lib.tile_index["tiles"])
        self.assertFalse(os.path.exists(tile_file))
    
    def test_list_loaded_areas(self):
        """Test listing loaded areas"""
        # Start with empty
        info = self.lib.list_loaded_areas()
        self.assertEqual(info["total_tiles"], 0)
        
        # Add some tiles
        tiles = [
            ("4000_-11101", Bounds(north=40.01, south=40.00, east=-111.00, west=-111.01)),
            ("4001_-11101", Bounds(north=40.02, south=40.01, east=-111.00, west=-111.01)),
        ]
        
        for tile_key, bounds in tiles:
            self.create_mock_tile(tile_key, bounds)
        
        # List again
        info = self.lib.list_loaded_areas()
        self.assertEqual(info["total_tiles"], 2)
        self.assertGreater(info["total_size_mb"], 0)
        
        # Check overall bounds
        self.assertEqual(len(info["areas"]), 1)
        area = info["areas"][0]
        self.assertEqual(area["south"], 40.00)
        self.assertEqual(area["north"], 40.02)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases"""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="elev_error_")
        self.lib = ElevationLibrary(data_dir=self.test_dir, resolution=10)
    
    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_invalid_bounds(self):
        """Test handling of invalid bounds"""
        # North <= South
        with self.assertRaises(ValueError):
            Bounds(south=40.0, north=40.0, west=-111.0, east=-110.0)
        
        # East <= West
        with self.assertRaises(ValueError):
            Bounds(south=40.0, north=41.0, west=-111.0, east=-111.0)
    
    def test_missing_data_clear_errors(self):
        """Test clear error messages for missing data"""
        # Single point
        with self.assertRaises(ValueError) as context:
            self.lib.get_elevation(40.5, -111.5)
        
        self.assertIn("40.5000", str(context.exception))
        self.assertIn("-111.5000", str(context.exception))
        self.assertIn("load_area()", str(context.exception))
        
        # Array with multiple missing tiles
        bounds = Bounds(south=40.0, north=40.02, west=-111.02, east=-111.0)
        with self.assertRaises(ValueError) as context:
            self.lib.get_elevation_array(bounds)
        
        # Check that it mentions multiple tiles (exact count may vary based on tile size)
        self.assertIn("tiles", str(context.exception))
        self.assertIn("Missing tiles:", str(context.exception))
    
    def test_remove_nonexistent_area(self):
        """Test removing area with no tiles"""
        bounds = Bounds(south=40.0, north=40.01, west=-111.01, east=-111.0)
        result = self.lib.remove_area(bounds)
        
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["tiles_removed"], 0)
        self.assertEqual(result["tiles_not_found"], 2)  # This area covers 2 tiles


class TestResolutionHandling(unittest.TestCase):
    """Test resolution-specific behavior"""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="elev_res_")
    
    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_invalid_resolution(self):
        """Test that invalid resolutions are rejected"""
        with self.assertRaises(ValueError) as context:
            ElevationLibrary(data_dir=self.test_dir, resolution=15)
        
        self.assertIn("Invalid resolution 15m", str(context.exception))
        self.assertIn("[1, 3, 5, 10, 30, 60]", str(context.exception))
    
    def test_resolution_isolation(self):
        """Test that different resolutions are isolated"""
        # Create 10m library and mock tile
        lib_10m = ElevationLibrary(data_dir=self.test_dir, resolution=10)
        lib_10m.tile_index["tiles"]["4000_-11100"] = {"test": "10m"}
        lib_10m._save_index()
        
        # Create 30m library
        lib_30m = ElevationLibrary(data_dir=self.test_dir, resolution=30)
        
        # 30m should not see 10m tiles
        self.assertEqual(len(lib_30m.tile_index["tiles"]), 0)
        
        # Add tile to 30m
        lib_30m.tile_index["tiles"]["4000_-11100"] = {"test": "30m"}
        lib_30m._save_index()
        
        # Reload 10m - should still have its own data
        lib_10m_new = ElevationLibrary(data_dir=self.test_dir, resolution=10)
        self.assertEqual(lib_10m_new.tile_index["tiles"]["4000_-11100"]["test"], "10m")


if __name__ == "__main__":
    unittest.main(verbosity=2)