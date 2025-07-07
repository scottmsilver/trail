#!/usr/bin/env python3
"""
Integration tests with real elevation data.
Tests known mountain peaks to verify accuracy.
"""

import os
import shutil
import tempfile
import unittest
from elevation import ElevationLibrary, Bounds


class TestRealElevationData(unittest.TestCase):
    """Test with real USGS elevation data for known peaks
    
    Note: We use generous tolerances (100-200m) because:
    1. DEM data has inherent resolution limits
    2. Peak coordinates may not align exactly with highest DEM cell
    3. Different data sources may have slight variations
    """
    
    @classmethod
    def setUpClass(cls):
        """Create a temporary directory for all tests"""
        cls.test_data_dir = tempfile.mkdtemp(prefix="elev_real_test_")
        print(f"\nUsing test data directory: {cls.test_data_dir}")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up all downloaded data"""
        if os.path.exists(cls.test_data_dir):
            print(f"Cleaning up test data directory: {cls.test_data_dir}")
            shutil.rmtree(cls.test_data_dir)
    
    def setUp(self):
        """Create library instance for each test"""
        self.lib = ElevationLibrary(data_dir=self.test_data_dir, resolution=10)
    
    def test_mount_whitney_california(self):
        """Test Mount Whitney - highest peak in continental US"""
        # Mount Whitney coordinates and expected elevation
        lat, lon = 36.5785, -118.2923
        expected_elevation = 4421  # meters
        tolerance = 200  # meters - increased for DEM resolution variations
        
        # Define a small area around the peak
        bounds = Bounds(
            south=lat - 0.01,
            north=lat + 0.01,
            west=lon - 0.01,
            east=lon + 0.01
        )
        
        # Load the area
        result = self.lib.load_area(bounds)
        self.assertEqual(result["status"], "success")
        
        # Get elevation at peak
        elevation = self.lib.get_elevation(lat, lon)
        
        # Check it's within tolerance
        self.assertAlmostEqual(elevation, expected_elevation, delta=tolerance,
                             msg=f"Mount Whitney elevation {elevation}m not within {tolerance}m of expected {expected_elevation}m")
        
        # Clean up this specific area
        remove_result = self.lib.remove_area(bounds)
        self.assertGreater(remove_result["tiles_removed"], 0)
    
    def test_pikes_peak_colorado(self):
        """Test Pikes Peak - famous Colorado fourteener"""
        # Pikes Peak coordinates and expected elevation
        lat, lon = 38.8409, -105.0422
        expected_elevation = 4302  # meters
        tolerance = 50  # meters
        
        # Define a small area around the peak
        bounds = Bounds(
            south=lat - 0.01,
            north=lat + 0.01,
            west=lon - 0.01,
            east=lon + 0.01
        )
        
        # Load the area
        result = self.lib.load_area(bounds)
        self.assertEqual(result["status"], "success")
        
        # Get elevation at peak
        elevation = self.lib.get_elevation(lat, lon)
        
        # Check it's within tolerance
        self.assertAlmostEqual(elevation, expected_elevation, delta=tolerance,
                             msg=f"Pikes Peak elevation {elevation}m not within {tolerance}m of expected {expected_elevation}m")
        
        # Clean up
        remove_result = self.lib.remove_area(bounds)
        self.assertGreater(remove_result["tiles_removed"], 0)
    
    def test_mount_rainier_washington(self):
        """Test Mount Rainier - prominent Cascade volcano"""
        # Mount Rainier coordinates and expected elevation
        lat, lon = 46.8523, -121.7603
        expected_elevation = 4392  # meters
        tolerance = 100  # meters - increased for DEM resolution variations
        
        # Define a small area around the peak
        bounds = Bounds(
            south=lat - 0.01,
            north=lat + 0.01,
            west=lon - 0.01,
            east=lon + 0.01
        )
        
        # Load the area
        result = self.lib.load_area(bounds)
        self.assertEqual(result["status"], "success")
        
        # Get elevation at peak
        elevation = self.lib.get_elevation(lat, lon)
        
        # Check it's within tolerance
        self.assertAlmostEqual(elevation, expected_elevation, delta=tolerance,
                             msg=f"Mount Rainier elevation {elevation}m not within {tolerance}m of expected {expected_elevation}m")
        
        # Clean up
        remove_result = self.lib.remove_area(bounds)
        self.assertGreater(remove_result["tiles_removed"], 0)
    
    def test_death_valley_california(self):
        """Test Death Valley - lowest point in North America"""
        # Badwater Basin coordinates and expected elevation
        lat, lon = 36.2300, -116.7670
        expected_elevation = -86  # meters below sea level
        tolerance = 10  # meters
        
        # Define a small area
        bounds = Bounds(
            south=lat - 0.01,
            north=lat + 0.01,
            west=lon - 0.01,
            east=lon + 0.01
        )
        
        # Load the area
        result = self.lib.load_area(bounds)
        self.assertEqual(result["status"], "success")
        
        # Get elevation
        elevation = self.lib.get_elevation(lat, lon)
        
        # Check it's below sea level and within tolerance
        self.assertLess(elevation, 0, "Death Valley should be below sea level")
        self.assertAlmostEqual(elevation, expected_elevation, delta=tolerance,
                             msg=f"Death Valley elevation {elevation}m not within {tolerance}m of expected {expected_elevation}m")
        
        # Clean up
        remove_result = self.lib.remove_area(bounds)
        self.assertGreater(remove_result["tiles_removed"], 0)
    
    def test_cleanup_verification(self):
        """Verify that remove_all actually cleans up files"""
        # Load a small test area
        bounds = Bounds(south=40.0, north=40.01, west=-111.01, east=-111.0)
        self.lib.load_area(bounds)
        
        # Verify data exists
        info = self.lib.list_loaded_areas()
        self.assertGreater(info["total_tiles"], 0)
        self.assertGreater(info["total_size_mb"], 0)
        
        # Remove all
        result = self.lib.remove_all()
        self.assertGreater(result["tiles_removed"], 0)
        
        # Verify data is gone
        info_after = self.lib.list_loaded_areas()
        self.assertEqual(info_after["total_tiles"], 0)
        self.assertEqual(info_after["total_size_mb"], 0)
        
        # Verify files are actually deleted
        data_dir = os.path.join(self.test_data_dir, "10m")
        if os.path.exists(data_dir):
            tif_files = [f for f in os.listdir(data_dir) if f.endswith('.tif')]
            self.assertEqual(len(tif_files), 0, "TIF files should be deleted")


class TestMultiResolution(unittest.TestCase):
    """Test different resolutions for the same location"""
    
    def setUp(self):
        self.test_data_dir = tempfile.mkdtemp(prefix="elev_multi_res_")
        # Use a location we know has multiple resolutions available
        # Avoid exact tile boundaries (like 40.0) which can cause edge cases
        self.test_lat = 39.9999
        self.test_lon = -110.9999
        self.bounds = Bounds(
            south=self.test_lat - 0.005,
            north=self.test_lat + 0.005,
            west=self.test_lon - 0.005,
            east=self.test_lon + 0.005
        )
    
    def tearDown(self):
        """Clean up test data"""
        if os.path.exists(self.test_data_dir):
            shutil.rmtree(self.test_data_dir)
    
    def test_resolution_comparison(self):
        """Compare data from different resolutions"""
        resolutions = [10, 30]  # Test 10m and 30m
        elevations = {}
        
        for res in resolutions:
            lib = ElevationLibrary(data_dir=self.test_data_dir, resolution=res)
            
            # Load area
            result = lib.load_area(self.bounds)
            if result["status"] != "success":
                self.skipTest(f"{res}m resolution not available for test area")
            
            # Get elevation
            elevations[res] = lib.get_elevation(self.test_lat, self.test_lon)
            
            # Verify resolution-specific storage
            res_dir = os.path.join(self.test_data_dir, f"{res}m")
            self.assertTrue(os.path.exists(res_dir))
            
            # Check tile size differences
            self.assertEqual(lib.tile_size, 0.01 if res == 10 else 0.02)
        
        # Elevations should be similar but not identical
        if len(elevations) == 2:
            diff = abs(elevations[10] - elevations[30])
            self.assertLess(diff, 50, 
                f"Large elevation difference between resolutions: {diff}m")


class TestMultiTileAreas(unittest.TestCase):
    """Test areas that span multiple tiles to verify stitching"""
    
    def setUp(self):
        self.test_data_dir = tempfile.mkdtemp(prefix="elev_multitile_")
        self.lib = ElevationLibrary(data_dir=self.test_data_dir, resolution=10)
    
    def tearDown(self):
        """Clean up test data"""
        if os.path.exists(self.test_data_dir):
            shutil.rmtree(self.test_data_dir)
    
    def test_death_valley_transect(self):
        """Test a transect across Death Valley spanning multiple tiles"""
        # Create a line from Telescope Peak to Badwater Basin
        # This should span multiple tiles and show elevation change
        
        # Telescope Peak (high point near Death Valley)
        peak_lat, peak_lon = 36.1699, -117.0892
        
        # Badwater Basin (lowest point)
        basin_lat, basin_lon = 36.2300, -116.7670
        
        # Load area covering both points (spans ~0.32 degrees longitude)
        bounds = Bounds(
            south=min(peak_lat, basin_lat) - 0.01,
            north=max(peak_lat, basin_lat) + 0.01,
            west=min(peak_lon, basin_lon) - 0.01,
            east=max(peak_lon, basin_lon) + 0.01
        )
        
        print(f"\nLoading Death Valley transect: {bounds}")
        result = self.lib.load_area(bounds)
        self.assertEqual(result["status"], "success")
        
        # Should have loaded multiple tiles
        self.assertGreaterEqual(result["tiles_downloaded"] + result["tiles_existing"], 
                               4, "Should span at least 4 tiles")
        
        # Get elevations at both ends
        peak_elevation = self.lib.get_elevation(peak_lat, peak_lon)
        basin_elevation = self.lib.get_elevation(basin_lat, basin_lon)
        
        print(f"Telescope Peak elevation: {peak_elevation:.1f}m")
        print(f"Badwater Basin elevation: {basin_elevation:.1f}m")
        
        # Verify expected elevation difference
        self.assertGreater(peak_elevation, 3000, "Telescope Peak should be > 3000m")
        self.assertLess(basin_elevation, 0, "Badwater Basin should be below sea level")
        
        # Sample points along the transect
        num_samples = 5
        for i in range(num_samples):
            t = i / (num_samples - 1)
            lat = peak_lat + t * (basin_lat - peak_lat)
            lon = peak_lon + t * (basin_lon - peak_lon)
            elev = self.lib.get_elevation(lat, lon)
            print(f"  Point {i}: ({lat:.4f}, {lon:.4f}) = {elev:.1f}m")
    
    def test_sierra_nevada_ridge(self):
        """Test along Sierra Nevada ridge spanning multiple tiles"""
        # Test area along the Sierra Nevada crest
        # From near Mount Whitney northward
        
        south_lat, south_lon = 36.5785, -118.2923  # Near Mount Whitney
        north_lat, north_lon = 36.6499, -118.3199  # North along ridge (avoid exact boundary)
        
        # Load area covering the ridge
        bounds = Bounds(
            south=south_lat - 0.01,
            north=north_lat + 0.01,
            west=min(south_lon, north_lon) - 0.01,
            east=max(south_lon, north_lon) + 0.01
        )
        
        print(f"\nLoading Sierra Nevada ridge: {bounds}")
        result = self.lib.load_area(bounds)
        self.assertEqual(result["status"], "success")
        
        # Should span multiple tiles
        total_tiles = result["tiles_downloaded"] + result["tiles_existing"]
        self.assertGreaterEqual(total_tiles, 4, "Should span at least 4 tiles")
        
        # Sample elevations along the ridge
        elevations = []
        num_samples = 5
        for i in range(num_samples):
            t = i / (num_samples - 1)
            lat = south_lat + t * (north_lat - south_lat)
            lon = south_lon + t * (north_lon - south_lon)
            elev = self.lib.get_elevation(lat, lon)
            elevations.append(elev)
            print(f"  Ridge point {i}: ({lat:.4f}, {lon:.4f}) = {elev:.1f}m")
        
        # All points should be at high elevation (ridge line)
        for elev in elevations:
            self.assertGreater(elev, 3500, f"Ridge elevation {elev} should be > 3500m")
    
    def test_cross_tile_boundary_precision(self):
        """Test that elevations are consistent across tile boundaries"""
        # Pick a point that will be near a tile boundary
        # Tiles are 0.01 degrees, so pick near a boundary
        base_lat = 40.005  # Near middle of a tile
        base_lon = -111.005
        
        # Load a 2x2 tile area
        bounds = Bounds(
            south=base_lat - 0.015,
            north=base_lat + 0.015,
            west=base_lon - 0.015,
            east=base_lon + 0.015
        )
        
        print(f"\nLoading tile boundary test area: {bounds}")
        result = self.lib.load_area(bounds)
        self.assertEqual(result["status"], "success")
        self.assertGreaterEqual(result["tiles_downloaded"] + result["tiles_existing"], 4)
        
        # Test points right at tile boundaries
        # Tile boundary should be at lat=40.01
        boundary_lat = 40.01
        
        # Get elevations just before and after the boundary
        before_elev = self.lib.get_elevation(boundary_lat - 0.0001, base_lon)
        after_elev = self.lib.get_elevation(boundary_lat + 0.0001, base_lon)
        
        print(f"Elevation just before tile boundary: {before_elev:.2f}m")
        print(f"Elevation just after tile boundary: {after_elev:.2f}m")
        
        # Elevations should be similar (smooth transition)
        # Allow for some natural terrain variation but not huge jumps
        self.assertLess(abs(after_elev - before_elev), 100,
                       "Elevation change across tile boundary should be < 100m")


if __name__ == "__main__":
    # Run with verbose output
    unittest.main(verbosity=2)