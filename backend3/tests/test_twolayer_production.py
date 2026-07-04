#!/usr/bin/env python3
"""
Comprehensive tests for production-ready two-layer elevation system.
Tests all APIs to ensure feature parity with single-layer system.
"""

import unittest
import tempfile
import shutil
import os
import numpy as np
import json
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elevation import TwoLayerElevationLibrary, Bounds


class TestTwoLayerProduction(unittest.TestCase):
    """Test the production-ready two-layer system"""
    
    def setUp(self):
        """Create temporary test directory"""
        self.test_dir = tempfile.mkdtemp(prefix="test_twolayer_prod_")
        
    def tearDown(self):
        """Clean up test directory"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_initialization(self):
        """Test library initialization"""
        lib = TwoLayerElevationLibrary(self.test_dir, resolution=10)
        
        # Check directories created
        self.assertTrue(os.path.exists(lib.resolution_dir))
        self.assertTrue(os.path.exists(lib.layer1_dir))
        self.assertTrue(os.path.exists(lib.layer2_dir))
        
        # Check resolution
        self.assertEqual(lib.resolution, 10)
        self.assertEqual(lib.tile_size, 0.01)
    
    def test_invalid_resolution(self):
        """Test initialization with invalid resolution"""
        with self.assertRaises(ValueError):
            TwoLayerElevationLibrary(self.test_dir, resolution=15)
    
    def test_invalid_directory(self):
        """Test initialization with non-existent directory"""
        with self.assertRaises(ValueError):
            TwoLayerElevationLibrary("/nonexistent/path", resolution=10)
    
    def test_load_area(self):
        """Test loading an area"""
        lib = TwoLayerElevationLibrary(self.test_dir, resolution=10)
        
        bounds = Bounds(
            south=40.00,
            north=40.01,
            west=-111.01,
            east=-111.00
        )
        
        # Mock py3dep to avoid real downloads
        with patch('elevation.py3dep.get_dem') as mock_get_dem:
            # Create mock DEM data
            mock_dem = MagicMock()
            mock_dem.rio.crs = "EPSG:4326"
            mock_dem.to_numpy.return_value = np.random.rand(100, 100) * 1000 + 1000
            mock_dem.x.min.return_value = bounds.west
            mock_dem.x.max.return_value = bounds.east
            mock_dem.y.min.return_value = bounds.south
            mock_dem.y.max.return_value = bounds.north
            mock_get_dem.return_value = mock_dem
            
            result = lib.load_area(bounds)
        
        self.assertEqual(result["status"], "success")
        self.assertGreater(result["total_tiles"], 0)
        self.assertEqual(result["resolution_m"], 10)
    
    def test_remove_area(self):
        """Test removing an area"""
        lib = TwoLayerElevationLibrary(self.test_dir, resolution=10)
        
        bounds = Bounds(
            south=40.00,
            north=40.01,
            west=-111.01,
            east=-111.00
        )
        
        # First load some data
        with patch('elevation.py3dep.get_dem') as mock_get_dem:
            mock_dem = MagicMock()
            mock_dem.rio.crs = "EPSG:4326"
            mock_dem.to_numpy.return_value = np.random.rand(100, 100) * 1000 + 1000
            mock_dem.x.min.return_value = bounds.west
            mock_dem.x.max.return_value = bounds.east
            mock_dem.y.min.return_value = bounds.south
            mock_dem.y.max.return_value = bounds.north
            mock_get_dem.return_value = mock_dem
            
            lib.load_area(bounds)
        
        # Now remove it
        result = lib.remove_area(bounds)
        
        self.assertEqual(result["status"], "success")
        self.assertGreater(result["tiles_removed"], 0)
        
        # Verify tiles are gone
        self.assertEqual(len(lib.layer2_index["tiles"]), 0)
    
    def test_remove_all(self):
        """Test removing all data"""
        lib = TwoLayerElevationLibrary(self.test_dir, resolution=10)
        
        # Load multiple areas
        areas = [
            Bounds(south=40.00, north=40.01, west=-111.01, east=-111.00),
            Bounds(south=40.10, north=40.11, west=-111.11, east=-111.10)
        ]
        
        with patch('elevation.py3dep.get_dem') as mock_get_dem:
            mock_dem = MagicMock()
            mock_dem.rio.crs = "EPSG:4326"
            
            for bounds in areas:
                mock_dem.to_numpy.return_value = np.random.rand(100, 100) * 1000 + 1000
                mock_dem.x.min.return_value = bounds.west
                mock_dem.x.max.return_value = bounds.east
                mock_dem.y.min.return_value = bounds.south
                mock_dem.y.max.return_value = bounds.north
                mock_get_dem.return_value = mock_dem
                
                lib.load_area(bounds)
        
        # Remove all
        result = lib.remove_all()
        
        self.assertEqual(result["status"], "success")
        self.assertGreater(result["tiles_removed"], 0)
        self.assertGreater(result["regions_removed"], 0)
        
        # Verify everything is gone
        self.assertEqual(len(lib.layer1_index["regions"]), 0)
        self.assertEqual(len(lib.layer2_index["tiles"]), 0)
    
    def test_get_elevation(self):
        """Test getting elevation at a single point"""
        lib = TwoLayerElevationLibrary(self.test_dir, resolution=10)
        
        lat, lon = 40.005, -111.005
        expected_elevation = 2000.0
        
        # First verify it fails when not loaded
        with self.assertRaises(ValueError) as context:
            lib.get_elevation(lat, lon)
        self.assertIn("not available", str(context.exception))
        
        # Now load the area
        bounds = Bounds(
            south=40.00,
            north=40.01,
            west=-111.01,
            east=-111.00
        )
        
        with patch('elevation.py3dep.get_dem') as mock_get_dem:
            mock_dem = MagicMock()
            mock_dem.rio.crs = "EPSG:4326"
            mock_dem.to_numpy.return_value = np.full((100, 100), expected_elevation)
            mock_dem.x.min.return_value = bounds.west
            mock_dem.x.max.return_value = bounds.east
            mock_dem.y.min.return_value = bounds.south
            mock_dem.y.max.return_value = bounds.north
            mock_get_dem.return_value = mock_dem
            
            # Explicitly load the area
            result = lib.load_area(bounds)
            self.assertEqual(result["status"], "success")
            
            # Now get_elevation should work
            elevation = lib.get_elevation(lat, lon)
        
        self.assertAlmostEqual(elevation, expected_elevation, delta=1.0)
    
    def test_get_elevation_array(self):
        """Test getting elevation array"""
        lib = TwoLayerElevationLibrary(self.test_dir, resolution=10)
        
        bounds = Bounds(
            south=40.00,
            north=40.01,
            west=-111.01,
            east=-111.00
        )
        
        # First verify it fails when not loaded
        with self.assertRaises(ValueError) as context:
            lib.get_elevation_array(bounds)
        self.assertIn("not available", str(context.exception))
        
        with patch('elevation.py3dep.get_dem') as mock_get_dem:
            # Create a function that returns appropriate mock data for any request
            def get_dem_side_effect(bounds_tuple, resolution, crs):
                west, south, east, north = bounds_tuple
                mock_dem = MagicMock()
                mock_dem.rio.crs = "EPSG:4326"
                mock_dem.to_numpy.return_value = np.random.rand(1000, 1000) * 1000 + 1000
                mock_dem.x.min.return_value = west
                mock_dem.x.max.return_value = east
                mock_dem.y.min.return_value = south
                mock_dem.y.max.return_value = north
                return mock_dem
            
            mock_get_dem.side_effect = get_dem_side_effect
            
            # Explicitly load the area
            result = lib.load_area(bounds)
            self.assertEqual(result["status"], "success")
            
            # Now get_elevation_array should work
            array, metadata = lib.get_elevation_array(bounds)
        
        self.assertIsInstance(array, np.ndarray)
        self.assertEqual(len(array.shape), 2)
        self.assertIn("bounds", metadata)
        self.assertIn("shape", metadata)
        self.assertIn("resolution_m", metadata)
        self.assertEqual(metadata["resolution_m"], 10)
    
    def test_list_loaded_areas(self):
        """Test listing loaded areas"""
        lib = TwoLayerElevationLibrary(self.test_dir, resolution=10)
        
        # Initially empty
        result = lib.list_loaded_areas()
        self.assertEqual(len(result["areas"]), 0)
        self.assertEqual(result["total_tiles"], 0)
        
        # Load an area
        bounds = Bounds(
            south=40.00,
            north=40.02,
            west=-111.02,
            east=-111.00
        )
        
        with patch('elevation.py3dep.get_dem') as mock_get_dem:
            mock_dem = MagicMock()
            mock_dem.rio.crs = "EPSG:4326"
            mock_dem.to_numpy.return_value = np.random.rand(200, 200) * 1000 + 1000
            mock_dem.x.min.return_value = bounds.west
            mock_dem.x.max.return_value = bounds.east
            mock_dem.y.min.return_value = bounds.south
            mock_dem.y.max.return_value = bounds.north
            mock_get_dem.return_value = mock_dem
            
            lib.load_area(bounds)
        
        # Check listing
        result = lib.list_loaded_areas()
        self.assertEqual(len(result["areas"]), 1)
        self.assertGreater(result["total_tiles"], 0)
        self.assertEqual(result["resolution_m"], 10)
        
        # Check area bounds
        area = result["areas"][0]
        self.assertIn("bounds", area)
        self.assertIn("tiles", area)
        self.assertIn("area_sq_km", area)
    
    def test_get_tile_info(self):
        """Test getting tile information"""
        lib = TwoLayerElevationLibrary(self.test_dir, resolution=10)
        
        lat, lon = 40.005, -111.005
        
        # Initially no info
        info = lib.get_tile_info(lat, lon)
        self.assertIsNone(info)
        
        # Load area
        with patch('elevation.py3dep.get_dem') as mock_get_dem:
            mock_dem = MagicMock()
            mock_dem.rio.crs = "EPSG:4326"
            mock_dem.to_numpy.return_value = np.random.rand(100, 100) * 1000 + 1000
            mock_dem.x.min.return_value = -111.01
            mock_dem.x.max.return_value = -111.00
            mock_dem.y.min.return_value = 40.00
            mock_dem.y.max.return_value = 40.01
            mock_get_dem.return_value = mock_dem
            
            bounds = Bounds(south=40.00, north=40.01, west=-111.01, east=-111.00)
            lib.load_area(bounds)
        
        # Now should have info
        info = lib.get_tile_info(lat, lon)
        self.assertIsNotNone(info)
        self.assertIn("tile_key", info)
        self.assertIn("bounds", info)
        self.assertIn("resolution_m", info)
        self.assertEqual(info["resolution_m"], 10)
    
    def test_multi_resolution_support(self):
        """Test that different resolutions work"""
        resolutions = [1, 3, 5, 10, 30, 60]
        
        for res in resolutions:
            res_dir = os.path.join(self.test_dir, f"res_{res}")
            os.makedirs(res_dir)
            
            lib = TwoLayerElevationLibrary(res_dir, resolution=res)
            self.assertEqual(lib.resolution, res)
            
            # Check tile size is appropriate
            expected_tile_size = TwoLayerElevationLibrary.RESOLUTIONS[res]
            self.assertEqual(lib.tile_size, expected_tile_size)
    
    def test_bounds_validation(self):
        """Test bounds validation"""
        # Invalid bounds (north <= south)
        with self.assertRaises(ValueError):
            Bounds(north=40.0, south=40.0, east=-111.0, west=-111.1)
        
        # Invalid bounds (east <= west)
        with self.assertRaises(ValueError):
            Bounds(north=40.1, south=40.0, east=-111.1, west=-111.1)
    
    def test_no_artifacts(self):
        """Test that the system produces no tile boundary artifacts"""
        lib = TwoLayerElevationLibrary(self.test_dir, resolution=10)
        
        # Area that spans multiple tiles
        bounds = Bounds(
            south=40.00,
            north=40.02,
            west=-111.02,
            east=-111.00
        )
        
        # Create smooth gradient data for testing
        height, width = 1000, 1000
        y = np.linspace(0, 1, height)
        x = np.linspace(0, 1, width)
        X, Y = np.meshgrid(x, y)
        smooth_data = 1000 + 100 * X + 100 * Y  # Smooth gradient
        
        with patch('elevation.py3dep.get_dem') as mock_get_dem:
            # Create a function that returns appropriate mock data for any request
            def get_dem_side_effect(bounds_tuple, resolution, crs):
                west, south, east, north = bounds_tuple
                mock_dem = MagicMock()
                mock_dem.rio.crs = "EPSG:4326"
                mock_dem.to_numpy.return_value = smooth_data
                mock_dem.x.min.return_value = west
                mock_dem.x.max.return_value = east
                mock_dem.y.min.return_value = south
                mock_dem.y.max.return_value = north
                return mock_dem
            
            mock_get_dem.side_effect = get_dem_side_effect
            
            lib.load_area(bounds)
            array, _ = lib.get_elevation_array(bounds)
        
        # Check for artifacts (large discontinuities)
        threshold = 50  # meters
        row_diff = np.diff(array, axis=0)
        col_diff = np.diff(array, axis=1)
        
        # With smooth gradient data, there should be no large jumps
        large_jumps = np.sum(np.abs(row_diff) > threshold) + np.sum(np.abs(col_diff) > threshold)
        self.assertEqual(large_jumps, 0, f"Found {large_jumps} discontinuities > {threshold}m")


class TestRealData(unittest.TestCase):
    """Test with real elevation data (integration tests)"""
    
    @classmethod
    def setUpClass(cls):
        """Create a temporary directory for all real data tests"""
        cls.test_data_dir = tempfile.mkdtemp(prefix="elev_twolayer_real_")
        print(f"\nUsing test data directory: {cls.test_data_dir}")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up all downloaded data"""
        if os.path.exists(cls.test_data_dir):
            print(f"Cleaning up test data directory: {cls.test_data_dir}")
            shutil.rmtree(cls.test_data_dir)
    
    def setUp(self):
        """Create library instance for each test"""
        self.lib = TwoLayerElevationLibrary(data_dir=self.test_data_dir, resolution=10)
    
    def test_park_city_area(self):
        """Test loading Park City area with real data"""
        bounds = Bounds(
            south=40.6448,
            north=40.6588,
            west=-111.5780,
            east=-111.5595
        )
        
        # Load area
        result = self.lib.load_area(bounds)
        self.assertEqual(result["status"], "success")
        
        # Get elevation array
        array, metadata = self.lib.get_elevation_array(bounds)
        
        # Check array properties
        self.assertIsInstance(array, np.ndarray)
        self.assertEqual(len(array.shape), 2)
        self.assertGreater(array.shape[0], 0)
        self.assertGreater(array.shape[1], 0)
        
        # Check for reasonable elevation values (Park City is ~2000-3000m)
        # Filter out zeros and NaNs
        valid_values = array[(array > 0) & ~np.isnan(array)]
        if len(valid_values) > 0:
            self.assertGreater(np.min(valid_values), 1500)  # Above 1500m
            self.assertLess(np.max(valid_values), 4000)     # Below 4000m
        
        # Check no artifacts - but ignore edges where there might be zeros
        # Create a mask for valid data (non-zero, non-NaN)
        valid_mask = (array > 0) & ~np.isnan(array)
        
        # Only check gradients between valid pixels
        interior_gradients = []
        for i in range(1, array.shape[0] - 1):
            for j in range(1, array.shape[1] - 1):
                if valid_mask[i, j]:
                    # Check gradients to neighbors that are also valid
                    if valid_mask[i-1, j]:
                        interior_gradients.append(abs(array[i, j] - array[i-1, j]))
                    if valid_mask[i+1, j]:
                        interior_gradients.append(abs(array[i, j] - array[i+1, j]))
                    if valid_mask[i, j-1]:
                        interior_gradients.append(abs(array[i, j] - array[i, j-1]))
                    if valid_mask[i, j+1]:
                        interior_gradients.append(abs(array[i, j] - array[i, j+1]))
        
        if interior_gradients:
            max_gradient = max(interior_gradients)
            # Use a more reasonable threshold for real terrain
            self.assertLess(max_gradient, 500, f"Max gradient {max_gradient}m is too high for adjacent pixels")
        
        # Test single point elevation
        center_lat = (bounds.north + bounds.south) / 2
        center_lon = (bounds.east + bounds.west) / 2
        elevation = self.lib.get_elevation(center_lat, center_lon)
        self.assertGreater(elevation, 1500)
        self.assertLess(elevation, 4000)
        
        # Clean up
        self.lib.remove_area(bounds)


if __name__ == '__main__':
    unittest.main(verbosity=2)