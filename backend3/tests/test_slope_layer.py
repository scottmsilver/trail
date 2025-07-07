#!/usr/bin/env python3
"""
Tests for the slope analysis layer.
"""

import unittest
import tempfile
import shutil
import os
import numpy as np
from unittest.mock import MagicMock, patch

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elevation import TwoLayerElevationLibrary, Bounds
from slope_layer import SlopeLayer, SlopeData


class TestSlopeLayer(unittest.TestCase):
    """Test the slope analysis layer"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp(prefix="test_slope_")
        self.elev_dir = os.path.join(self.test_dir, "elevation")
        self.slope_dir = os.path.join(self.test_dir, "slope")
        
        # Create directories
        os.makedirs(self.elev_dir, exist_ok=True)
        os.makedirs(self.slope_dir, exist_ok=True)
        
        # Create elevation library
        self.elev_lib = TwoLayerElevationLibrary(self.elev_dir, resolution=10)
        
        # Create slope layer
        self.slope_layer = SlopeLayer(self.elev_lib, data_dir=self.slope_dir)
    
    def tearDown(self):
        """Clean up test environment"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_initialization(self):
        """Test slope layer initialization"""
        # Check directories created
        self.assertTrue(os.path.exists(self.slope_layer.data_dir))
        self.assertTrue(os.path.exists(self.slope_layer.tiles_dir))
        
        # Check resolution matches elevation
        self.assertEqual(self.slope_layer.resolution, 10)
        self.assertEqual(self.slope_layer.tile_size, self.elev_lib.tile_size)
    
    def test_compute_area_without_elevation(self):
        """Test computing slopes when elevation data not loaded"""
        bounds = Bounds(south=40.0, north=40.01, west=-111.01, east=-111.0)
        
        result = self.slope_layer.compute_area(bounds)
        
        self.assertEqual(result["status"], "error")
        self.assertIn("Elevation data not available", result["message"])
    
    @patch('elevation.py3dep.get_dem')
    def test_compute_area_with_elevation(self, mock_get_dem):
        """Test computing slopes from elevation data"""
        bounds = Bounds(south=40.0, north=40.01, west=-111.01, east=-111.0)
        
        # Create elevation data with a simple slope
        height, width = 100, 100
        y = np.linspace(0, 1, height)
        x = np.linspace(0, 1, width)
        X, Y = np.meshgrid(x, y)
        # Create a plane tilted in both directions
        elevation = 2000 + 100 * X + 50 * Y
        
        # Mock elevation data
        mock_dem = MagicMock()
        mock_dem.rio.crs = "EPSG:4326"
        mock_dem.to_numpy.return_value = elevation
        mock_dem.x.min.return_value = bounds.west - 0.05
        mock_dem.x.max.return_value = bounds.east + 0.05
        mock_dem.y.min.return_value = bounds.south - 0.05
        mock_dem.y.max.return_value = bounds.north + 0.05
        mock_get_dem.return_value = mock_dem
        
        # First load elevation data
        self.elev_lib.load_area(bounds)
        
        # Now compute slopes
        result = self.slope_layer.compute_area(bounds)
        
        self.assertEqual(result["status"], "success")
        self.assertGreater(result["tiles_created"], 0)
        self.assertEqual(result["resolution_m"], 10)
    
    def test_slope_calculation(self):
        """Test slope calculation accuracy"""
        # Create a simple inclined plane
        height, width = 50, 50
        elevation = np.zeros((height, width))
        
        # Create a 45-degree slope in X direction
        # Rise of 1m per 1m horizontal distance
        for i in range(width):
            elevation[:, i] = i
        
        bounds = Bounds(south=40.0, north=40.005, west=-111.005, east=-111.0)
        pixel_size_m = 10.0  # 10m pixels
        
        # Use private method for testing
        slope_data = self.slope_layer._compute_slopes(elevation, bounds, pixel_size_m)
        
        # Check slope is approximately 45 degrees (accounting for pixel size)
        # Gradient is 1 elevation unit per pixel, pixel is 10m
        # So gradient is 1/10 = 0.1, arctan(0.1) ≈ 5.7 degrees
        expected_slope = np.degrees(np.arctan(1.0 / pixel_size_m))
        mean_slope = np.mean(slope_data.slope[1:-1, 1:-1])  # Exclude edges
        
        self.assertAlmostEqual(mean_slope, expected_slope, delta=0.5)
        
        # Check aspect is approximately East (90 degrees)
        mean_aspect = np.mean(slope_data.aspect[1:-1, 1:-1])
        self.assertAlmostEqual(mean_aspect, 90, delta=5)
    
    def test_get_slope_not_computed(self):
        """Test getting slope when not computed"""
        with self.assertRaises(ValueError) as context:
            self.slope_layer.get_slope(40.5, -111.5)
        
        self.assertIn("Slope data not available", str(context.exception))
        self.assertIn("compute_area()", str(context.exception))
    
    @patch('elevation.py3dep.get_dem')
    def test_get_slope_after_compute(self, mock_get_dem):
        """Test getting slope after computing"""
        bounds = Bounds(south=40.0, north=40.01, west=-111.01, east=-111.0)
        
        # Create simple elevation data
        elevation = np.ones((100, 100)) * 2000
        
        mock_dem = MagicMock()
        mock_dem.rio.crs = "EPSG:4326"
        mock_dem.to_numpy.return_value = elevation
        mock_dem.x.min.return_value = bounds.west - 0.05
        mock_dem.x.max.return_value = bounds.east + 0.05
        mock_dem.y.min.return_value = bounds.south - 0.05
        mock_dem.y.max.return_value = bounds.north + 0.05
        mock_get_dem.return_value = mock_dem
        
        # Load elevation and compute slopes
        self.elev_lib.load_area(bounds)
        self.slope_layer.compute_area(bounds)
        
        # Now should be able to get slope
        slope = self.slope_layer.get_slope(40.005, -111.005)
        
        # Flat terrain should have ~0 slope
        self.assertLess(slope, 1.0)
    
    def test_get_slope_change(self):
        """Test getting slope change (curvature)"""
        # Would test with elevation data that has curvature
        # For now, just ensure the method exists
        self.assertTrue(hasattr(self.slope_layer, 'get_slope_change'))
    
    def test_get_aspect(self):
        """Test getting aspect"""
        # Would test with elevation data that has clear aspect
        # For now, just ensure the method exists
        self.assertTrue(hasattr(self.slope_layer, 'get_aspect'))
    
    def test_remove_area(self):
        """Test removing computed slope data"""
        # First compute some data (mocked)
        bounds = Bounds(south=40.0, north=40.01, west=-111.01, east=-111.0)
        
        # Manually add a tile to the index
        tile_key = "0400_-0111"
        tile_path = os.path.join(self.slope_layer.tiles_dir, f"{tile_key}.tif")
        
        # Create dummy file
        with open(tile_path, 'w') as f:
            f.write("dummy")
        
        self.slope_layer.index["tiles"][tile_key] = {
            "bounds": {
                "north": 40.01,
                "south": 40.0,
                "east": -111.0,
                "west": -111.01
            },
            "path": tile_path,
            "resolution_m": 10
        }
        self.slope_layer._save_index()
        
        # Remove area
        result = self.slope_layer.remove_area(bounds)
        
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["tiles_removed"], 1)
        self.assertFalse(os.path.exists(tile_path))
        self.assertEqual(len(self.slope_layer.index["tiles"]), 0)
    
    def test_list_computed_areas_empty(self):
        """Test listing when no areas computed"""
        result = self.slope_layer.list_computed_areas()
        
        self.assertEqual(result["areas"], [])
        self.assertEqual(result["total_tiles"], 0)
        self.assertEqual(result["total_size_mb"], 0)
    
    def test_consistent_tiling(self):
        """Test that slope tiles align with elevation tiles"""
        # The tile keys should use the same scheme
        lat, lon = 40.6543, -111.5432
        
        elev_tile_key = self.elev_lib._get_tile_key(lat, lon)
        slope_tile_key = self.slope_layer._get_tile_key(lat, lon)
        
        # Should produce the same tile key
        self.assertEqual(elev_tile_key, slope_tile_key)


class TestSlopeCalculations(unittest.TestCase):
    """Test mathematical correctness of slope calculations"""
    
    def test_flat_terrain(self):
        """Test slope of flat terrain"""
        slope_layer = SlopeLayer(MagicMock(), data_dir="/tmp/test")
        
        # Flat elevation
        elevation = np.ones((50, 50)) * 1000
        bounds = Bounds(south=40.0, north=40.01, west=-111.01, east=-111.0)
        
        slope_data = slope_layer._compute_slopes(elevation, bounds, pixel_size_m=10)
        
        # Should have near-zero slope
        self.assertLess(np.max(slope_data.slope), 0.1)
        self.assertLess(np.max(slope_data.slope_change), 0.01)
    
    def test_constant_slope(self):
        """Test terrain with constant slope"""
        slope_layer = SlopeLayer(MagicMock(), data_dir="/tmp/test")
        
        # Create 30-degree slope to the east
        height, width = 50, 50
        elevation = np.zeros((height, width))
        pixel_size_m = 10.0
        
        # tan(30°) ≈ 0.577
        rise_per_pixel = pixel_size_m * np.tan(np.radians(30))
        
        for i in range(width):
            elevation[:, i] = i * rise_per_pixel
        
        bounds = Bounds(south=40.0, north=40.01, west=-111.01, east=-111.0)
        slope_data = slope_layer._compute_slopes(elevation, bounds, pixel_size_m)
        
        # Check slope is approximately 30 degrees
        mean_slope = np.mean(slope_data.slope[5:-5, 5:-5])  # Exclude edges
        self.assertAlmostEqual(mean_slope, 30, delta=1)
        
        # Constant slope should have near-zero curvature
        mean_curvature = np.mean(slope_data.slope_change[5:-5, 5:-5])
        self.assertLess(mean_curvature, 0.1)
        
        # Aspect should be East (90 degrees)
        mean_aspect = np.mean(slope_data.aspect[5:-5, 5:-5])
        self.assertAlmostEqual(mean_aspect, 90, delta=5)
    
    def test_ridge_terrain(self):
        """Test terrain with a ridge (positive curvature)"""
        slope_layer = SlopeLayer(MagicMock(), data_dir="/tmp/test")
        
        # Create a ridge running north-south
        height, width = 50, 50
        elevation = np.zeros((height, width))
        
        center = width // 2
        for i in range(width):
            # Parabolic cross-section
            distance_from_center = abs(i - center)
            elevation[:, i] = 1000 - distance_from_center ** 2
        
        bounds = Bounds(south=40.0, north=40.01, west=-111.01, east=-111.0)
        slope_data = slope_layer._compute_slopes(elevation, bounds, pixel_size_m=10)
        
        # Ridge should have positive curvature at the top
        ridge_curvature = slope_data.slope_change[:, center]
        self.assertGreater(np.mean(ridge_curvature), 0)
    
    def test_aspect_directions(self):
        """Test aspect calculations for all directions"""
        slope_layer = SlopeLayer(MagicMock(), data_dir="/tmp/test")
        
        # Create a cone-shaped hill
        height, width = 51, 51  # Odd size for clear center
        elevation = np.zeros((height, width))
        
        center_y, center_x = height // 2, width // 2
        
        for i in range(height):
            for j in range(width):
                distance = np.sqrt((i - center_y)**2 + (j - center_x)**2)
                elevation[i, j] = 1000 - distance * 10
        
        bounds = Bounds(south=40.0, north=40.01, west=-111.01, east=-111.0)
        slope_data = slope_layer._compute_slopes(elevation, bounds, pixel_size_m=10)
        
        # Check aspects at cardinal directions from center
        # North of center should face North (0/360)
        north_aspect = slope_data.aspect[center_y - 10, center_x]
        self.assertTrue(north_aspect < 45 or north_aspect > 315)
        
        # East of center should face East (90)
        east_aspect = slope_data.aspect[center_y, center_x + 10]
        self.assertAlmostEqual(east_aspect, 90, delta=45)
        
        # South of center should face South (180)
        south_aspect = slope_data.aspect[center_y + 10, center_x]
        self.assertAlmostEqual(south_aspect, 180, delta=45)
        
        # West of center should face West (270)
        west_aspect = slope_data.aspect[center_y, center_x - 10]
        self.assertAlmostEqual(west_aspect, 270, delta=45)


if __name__ == '__main__':
    unittest.main(verbosity=2)