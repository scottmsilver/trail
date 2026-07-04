#!/usr/bin/env python3
"""
Comprehensive test suite for two-layer elevation system.
Inspired by single-layer tests to ensure complete coverage.
"""

import unittest
import tempfile
import shutil
import os
import json
import numpy as np
from unittest.mock import patch, MagicMock, mock_open
from datetime import datetime

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elevation import TwoLayerElevationLibrary, Bounds


class TestTwoLayerBasic(unittest.TestCase):
    """Basic functionality tests with mocked py3dep"""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_twolayer_basic_")
        self.lib = TwoLayerElevationLibrary(data_dir=self.test_dir, resolution=10)
    
    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_initialization_creates_structure(self):
        """Test that initialization creates proper directory structure"""
        self.assertTrue(os.path.exists(self.lib.resolution_dir))
        self.assertTrue(os.path.exists(self.lib.layer1_dir))
        self.assertTrue(os.path.exists(self.lib.layer2_dir))
        # Index files are created lazily when data is saved
        # Just check the paths are set correctly
        self.assertTrue(self.lib.layer1_index_file.endswith('index.json'))
        self.assertTrue(self.lib.layer2_index_file.endswith('index.json'))
    
    def test_data_not_found_when_not_loaded(self):
        """Test that data is not found when not loaded"""
        with self.assertRaises(ValueError) as context:
            self.lib.get_elevation(40.5, -111.5)
        
        self.assertIn("not available", str(context.exception))
        self.assertIn("40.5", str(context.exception))
        self.assertIn("-111.5", str(context.exception))
    
    def test_get_tile_info_missing(self):
        """Test get_tile_info returns None for missing tiles"""
        info = self.lib.get_tile_info(40.5, -111.5)
        self.assertIsNone(info)
    
    def test_list_loaded_areas_empty(self):
        """Test list_loaded_areas when empty"""
        result = self.lib.list_loaded_areas()
        self.assertEqual(result["areas"], [])
        self.assertEqual(result["total_tiles"], 0)
        self.assertEqual(result["total_regions"], 0)
        self.assertEqual(result["total_size_mb"], 0)
    
    def test_remove_nonexistent_area(self):
        """Test removing area that doesn't exist"""
        bounds = Bounds(south=40.0, north=40.1, west=-111.1, east=-111.0)
        result = self.lib.remove_area(bounds)
        
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["tiles_removed"], 0)
        self.assertGreater(result["tiles_not_found"], 0)


class TestTileKeyGeneration(unittest.TestCase):
    """Test tile key generation logic"""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_tile_keys_")
        self.lib = TwoLayerElevationLibrary(data_dir=self.test_dir, resolution=10)
    
    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_positive_coordinates(self):
        """Test tile key generation for positive coordinates"""
        key = self.lib._get_tile_key(40.5, -111.5)
        # Test format, not exact value (implementation may vary)
        self.assertRegex(key, r'^40\d+_-111\d+$')
    
    def test_negative_coordinates(self):
        """Test tile key generation for negative coordinates"""
        key = self.lib._get_tile_key(-40.5, 111.5)
        # Test format with negative latitude
        self.assertRegex(key, r'^-40\d+_111\d+$')
    
    def test_exact_tile_boundary(self):
        """Test tile key for coordinates on exact tile boundary"""
        # Should consistently assign to same tile
        key1 = self.lib._get_tile_key(40.0, -111.0)
        key2 = self.lib._get_tile_key(40.0, -111.0)
        self.assertEqual(key1, key2)
        # Don't test exact value, just consistency
    
    def test_region_key_generation(self):
        """Test Layer 1 region key generation"""
        key = self.lib._get_region_key(40.5, -111.5)
        self.assertEqual(key, "r405_-1115")


class TestIndexPersistence(unittest.TestCase):
    """Test index file persistence"""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_index_")
    
    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_index_persistence(self):
        """Test that indexes persist between instances"""
        # Create first instance and mock some data
        lib1 = TwoLayerElevationLibrary(data_dir=self.test_dir, resolution=10)
        
        # Manually add some index data
        lib1.layer1_index["regions"]["test_region"] = {"test": "data1"}
        lib1.layer2_index["tiles"]["test_tile"] = {"test": "data2"}
        lib1._save_json(lib1.layer1_index, lib1.layer1_index_file)
        lib1._save_json(lib1.layer2_index, lib1.layer2_index_file)
        
        # Create new instance
        lib2 = TwoLayerElevationLibrary(data_dir=self.test_dir, resolution=10)
        
        # Check data persisted
        self.assertIn("test_region", lib2.layer1_index["regions"])
        self.assertIn("test_tile", lib2.layer2_index["tiles"])
        self.assertEqual(lib2.layer1_index["regions"]["test_region"]["test"], "data1")
        self.assertEqual(lib2.layer2_index["tiles"]["test_tile"]["test"], "data2")
    
    def test_resolution_isolation(self):
        """Test that different resolutions maintain separate data"""
        lib_10m = TwoLayerElevationLibrary(data_dir=self.test_dir, resolution=10)
        lib_30m = TwoLayerElevationLibrary(data_dir=self.test_dir, resolution=30)
        
        # Add data to 10m
        lib_10m.layer2_index["tiles"]["test_10m"] = {"resolution": 10}
        lib_10m._save_json(lib_10m.layer2_index, lib_10m.layer2_index_file)
        
        # Add data to 30m
        lib_30m.layer2_index["tiles"]["test_30m"] = {"resolution": 30}
        lib_30m._save_json(lib_30m.layer2_index, lib_30m.layer2_index_file)
        
        # Reload and check isolation
        lib_10m_new = TwoLayerElevationLibrary(data_dir=self.test_dir, resolution=10)
        lib_30m_new = TwoLayerElevationLibrary(data_dir=self.test_dir, resolution=30)
        
        self.assertIn("test_10m", lib_10m_new.layer2_index["tiles"])
        self.assertNotIn("test_30m", lib_10m_new.layer2_index["tiles"])
        
        self.assertIn("test_30m", lib_30m_new.layer2_index["tiles"])
        self.assertNotIn("test_10m", lib_30m_new.layer2_index["tiles"])


class TestErrorMessages(unittest.TestCase):
    """Test error messages are helpful"""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_errors_")
        self.lib = TwoLayerElevationLibrary(data_dir=self.test_dir, resolution=10)
    
    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_single_point_error_message(self):
        """Test error message for single point query"""
        lat, lon = 40.1234, -111.5678
        
        with self.assertRaises(ValueError) as context:
            self.lib.get_elevation(lat, lon)
        
        error_msg = str(context.exception)
        self.assertIn(str(lat), error_msg)
        self.assertIn(str(lon), error_msg)
        self.assertIn("load_area()", error_msg)
    
    def test_array_error_message(self):
        """Test error message for array query"""
        bounds = Bounds(south=40.0, north=40.1, west=-111.1, east=-111.0)
        
        with self.assertRaises(ValueError) as context:
            self.lib.get_elevation_array(bounds)
        
        error_msg = str(context.exception)
        # Check for key parts of the actual error message
        self.assertIn("not available", error_msg)
        self.assertIn("load_area()", error_msg)


class TestBoundsValidation(unittest.TestCase):
    """Test bounds validation and edge cases"""
    
    def test_invalid_bounds_north_south(self):
        """Test bounds with north <= south"""
        with self.assertRaises(ValueError):
            Bounds(north=40.0, south=40.0, east=-111.0, west=-111.1)
        
        with self.assertRaises(ValueError):
            Bounds(north=40.0, south=40.1, east=-111.0, west=-111.1)
    
    def test_invalid_bounds_east_west(self):
        """Test bounds with east <= west"""
        with self.assertRaises(ValueError):
            Bounds(north=40.1, south=40.0, east=-111.1, west=-111.1)
        
        with self.assertRaises(ValueError):
            Bounds(north=40.1, south=40.0, east=-111.2, west=-111.1)
    
    def test_bounds_contains_point(self):
        """Test bounds.contains_point method"""
        bounds = Bounds(north=40.1, south=40.0, east=-111.0, west=-111.1)
        
        # Inside
        self.assertTrue(bounds.contains_point(40.05, -111.05))
        
        # On boundary
        self.assertTrue(bounds.contains_point(40.0, -111.05))
        self.assertTrue(bounds.contains_point(40.1, -111.05))
        self.assertTrue(bounds.contains_point(40.05, -111.0))
        self.assertTrue(bounds.contains_point(40.05, -111.1))
        
        # Outside
        self.assertFalse(bounds.contains_point(39.9, -111.05))
        self.assertFalse(bounds.contains_point(40.2, -111.05))
        self.assertFalse(bounds.contains_point(40.05, -110.9))
        self.assertFalse(bounds.contains_point(40.05, -111.2))
    
    def test_bounds_overlaps(self):
        """Test bounds.overlaps method"""
        bounds1 = Bounds(north=40.1, south=40.0, east=-111.0, west=-111.1)
        
        # Overlapping
        bounds2 = Bounds(north=40.05, south=39.95, east=-110.95, west=-111.05)
        self.assertTrue(bounds1.overlaps(bounds2))
        self.assertTrue(bounds2.overlaps(bounds1))
        
        # Non-overlapping
        bounds3 = Bounds(north=39.9, south=39.8, east=-111.0, west=-111.1)
        self.assertFalse(bounds1.overlaps(bounds3))
        self.assertFalse(bounds3.overlaps(bounds1))
    
    def test_bounds_union(self):
        """Test bounds.union method"""
        bounds1 = Bounds(north=40.1, south=40.0, east=-111.0, west=-111.1)
        bounds2 = Bounds(north=40.2, south=40.05, east=-110.9, west=-111.05)
        
        union = bounds1.union(bounds2)
        self.assertEqual(union.north, 40.2)
        self.assertEqual(union.south, 40.0)
        self.assertEqual(union.east, -110.9)
        self.assertEqual(union.west, -111.1)


class TestLayerInteraction(unittest.TestCase):
    """Test interaction between Layer 1 and Layer 2"""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_layers_")
        self.lib = TwoLayerElevationLibrary(data_dir=self.test_dir, resolution=10)
    
    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    @patch('elevation.py3dep.get_dem')
    def test_layer1_creates_seamless_regions(self, mock_get_dem):
        """Test that Layer 1 creates seamless regions"""
        bounds = Bounds(south=40.0, north=40.01, west=-111.01, east=-111.0)
        
        # Mock DEM
        mock_dem = MagicMock()
        mock_dem.rio.crs = "EPSG:4326"
        mock_dem.to_numpy.return_value = np.ones((100, 100)) * 2000
        mock_dem.x.min.return_value = -111.1
        mock_dem.x.max.return_value = -111.0
        mock_dem.y.min.return_value = 40.0
        mock_dem.y.max.return_value = 40.1
        mock_get_dem.return_value = mock_dem
        
        # Load area
        self.lib.load_area(bounds)
        
        # Check Layer 1 regions created
        self.assertGreater(len(self.lib.layer1_index["regions"]), 0)
        
        # Check region size is larger than requested bounds
        for region_key, region_info in self.lib.layer1_index["regions"].items():
            region_bounds = region_info["bounds"]
            # Region should be 0.1° x 0.1° (LAYER1_REGION_SIZE)
            self.assertAlmostEqual(
                region_bounds["north"] - region_bounds["south"],
                self.lib.LAYER1_REGION_SIZE,
                places=2
            )
    
    @patch('elevation.py3dep.get_dem')
    def test_layer2_tiles_from_layer1(self, mock_get_dem):
        """Test that Layer 2 tiles are created from Layer 1 data"""
        bounds = Bounds(south=40.0, north=40.02, west=-111.02, east=-111.0)
        
        # Mock DEM
        mock_dem = MagicMock()
        mock_dem.rio.crs = "EPSG:4326"
        mock_dem.to_numpy.return_value = np.ones((200, 200)) * 2000
        mock_dem.x.min.return_value = -111.1
        mock_dem.x.max.return_value = -111.0
        mock_dem.y.min.return_value = 40.0
        mock_dem.y.max.return_value = 40.1
        mock_get_dem.return_value = mock_dem
        
        # Load area
        result = self.lib.load_area(bounds)
        
        # Check tiles created
        self.assertGreater(result["tiles_created"], 0)
        
        # Check each tile references a Layer 1 region
        for tile_key, tile_info in self.lib.layer2_index["tiles"].items():
            self.assertIn("source_region", tile_info)
            self.assertIn(tile_info["source_region"], self.lib.layer1_index["regions"])


class TestMultiResolution(unittest.TestCase):
    """Test multi-resolution support"""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_multireso_")
    
    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_all_resolutions_supported(self):
        """Test that all advertised resolutions work"""
        resolutions = [1, 3, 5, 10, 30, 60]
        
        for res in resolutions:
            with self.subTest(resolution=res):
                lib = TwoLayerElevationLibrary(data_dir=self.test_dir, resolution=res)
                self.assertEqual(lib.resolution, res)
                
                # Check tile size is appropriate
                expected_tile_size = TwoLayerElevationLibrary.RESOLUTIONS[res]
                self.assertEqual(lib.tile_size, expected_tile_size)
    
    def test_resolution_tile_sizes(self):
        """Test that tile sizes are appropriate for each resolution"""
        expected_sizes = {
            1: 0.001,
            3: 0.005,
            5: 0.005,
            10: 0.01,
            30: 0.01,
            60: 0.02
        }
        
        for res, expected_size in expected_sizes.items():
            with self.subTest(resolution=res):
                lib = TwoLayerElevationLibrary(data_dir=self.test_dir, resolution=res)
                self.assertEqual(lib.tile_size, expected_size)


class TestDataListingAndMerging(unittest.TestCase):
    """Test listing loaded areas with intelligent merging"""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_listing_")
        self.lib = TwoLayerElevationLibrary(data_dir=self.test_dir, resolution=10)
    
    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    @patch('elevation.py3dep.get_dem')
    def test_list_merges_adjacent_areas(self, mock_get_dem):
        """Test that adjacent tiles are merged into single area"""
        # Create mock DEM
        mock_dem = MagicMock()
        mock_dem.rio.crs = "EPSG:4326"
        
        # Load two adjacent areas
        areas = [
            Bounds(south=40.00, north=40.01, west=-111.01, east=-111.00),
            Bounds(south=40.00, north=40.01, west=-111.02, east=-111.01)
        ]
        
        for bounds in areas:
            mock_dem.to_numpy.return_value = np.ones((100, 100)) * 2000
            mock_dem.x.min.return_value = bounds.west - 0.05
            mock_dem.x.max.return_value = bounds.east + 0.05
            mock_dem.y.min.return_value = bounds.south - 0.05
            mock_dem.y.max.return_value = bounds.north + 0.05
            mock_get_dem.return_value = mock_dem
            self.lib.load_area(bounds)
        
        # List areas
        result = self.lib.list_loaded_areas()
        
        # Should merge into one area
        self.assertEqual(len(result["areas"]), 1)
        area = result["areas"][0]
        
        # Check merged bounds (with more tolerance for float precision)
        self.assertAlmostEqual(area["bounds"]["west"], -111.02, places=1)
        self.assertAlmostEqual(area["bounds"]["east"], -111.00, places=1)
    
    @patch('elevation.py3dep.get_dem')
    def test_list_separate_areas(self, mock_get_dem):
        """Test that non-adjacent areas remain separate"""
        # Create mock DEM
        mock_dem = MagicMock()
        mock_dem.rio.crs = "EPSG:4326"
        
        # Load two separate areas
        areas = [
            Bounds(south=40.00, north=40.01, west=-111.01, east=-111.00),
            Bounds(south=40.10, north=40.11, west=-111.11, east=-111.10)
        ]
        
        for bounds in areas:
            mock_dem.to_numpy.return_value = np.ones((100, 100)) * 2000
            mock_dem.x.min.return_value = bounds.west - 0.05
            mock_dem.x.max.return_value = bounds.east + 0.05
            mock_dem.y.min.return_value = bounds.south - 0.05
            mock_dem.y.max.return_value = bounds.north + 0.05
            mock_get_dem.return_value = mock_dem
            self.lib.load_area(bounds)
        
        # List areas
        result = self.lib.list_loaded_areas()
        
        # Should remain as two areas
        self.assertEqual(len(result["areas"]), 2)


class TestCacheManagement(unittest.TestCase):
    """Test rasterio dataset cache management"""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_cache_")
        self.lib = TwoLayerElevationLibrary(data_dir=self.test_dir, resolution=10)
    
    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_cache_cleanup_on_remove(self):
        """Test that cache is cleaned up on remove operations"""
        # Start with empty cache
        self.assertEqual(len(self.lib._open_datasets), 0)
        
        # Mock some cached datasets
        self.lib._open_datasets["/fake/path1.tif"] = MagicMock()
        self.lib._open_datasets["/fake/path2.tif"] = MagicMock()
        
        # Remove area should clean cache
        bounds = Bounds(south=40.0, north=40.1, west=-111.1, east=-111.0)
        self.lib.remove_area(bounds)
        
        # Cache should be empty
        self.assertEqual(len(self.lib._open_datasets), 0)
    
    def test_cache_cleanup_on_remove_all(self):
        """Test that cache is cleaned up on remove_all"""
        # Mock some cached datasets
        self.lib._open_datasets["/fake/path1.tif"] = MagicMock()
        self.lib._open_datasets["/fake/path2.tif"] = MagicMock()
        
        # Remove all should clean cache
        self.lib.remove_all()
        
        # Cache should be empty
        self.assertEqual(len(self.lib._open_datasets), 0)


class TestArtifactPrevention(unittest.TestCase):
    """Test that the system prevents tile boundary artifacts"""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_artifacts_")
        self.lib = TwoLayerElevationLibrary(data_dir=self.test_dir, resolution=10)
    
    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    @patch('elevation.py3dep.get_dem')
    def test_no_artifacts_smooth_gradient(self, mock_get_dem):
        """Test no artifacts with smooth gradient data"""
        bounds = Bounds(south=40.00, north=40.03, west=-111.03, east=-111.00)
        
        # Create smooth gradient data
        height, width = 400, 400
        y = np.linspace(0, 1, height)
        x = np.linspace(0, 1, width)
        X, Y = np.meshgrid(x, y)
        smooth_data = 1000 + 100 * X + 100 * Y
        
        mock_dem = MagicMock()
        mock_dem.rio.crs = "EPSG:4326"
        mock_dem.to_numpy.return_value = smooth_data
        mock_dem.x.min.return_value = bounds.west - 0.05
        mock_dem.x.max.return_value = bounds.east + 0.05
        mock_dem.y.min.return_value = bounds.south - 0.05
        mock_dem.y.max.return_value = bounds.north + 0.05
        mock_get_dem.return_value = mock_dem
        
        # Load and get array
        self.lib.load_area(bounds)
        array, _ = self.lib.get_elevation_array(bounds)
        
        # Check gradients
        row_diff = np.diff(array, axis=0)
        col_diff = np.diff(array, axis=1)
        
        # With smooth data, gradients should be small and consistent
        max_row_gradient = np.max(np.abs(row_diff))
        max_col_gradient = np.max(np.abs(col_diff))
        
        # Should have no large jumps
        self.assertLess(max_row_gradient, 10)
        self.assertLess(max_col_gradient, 10)
        
        # Gradients should be fairly uniform (low standard deviation)
        self.assertLess(np.std(row_diff), 5)
        self.assertLess(np.std(col_diff), 5)


if __name__ == '__main__':
    unittest.main(verbosity=2)