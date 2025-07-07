#!/usr/bin/env python3
"""Debug script to test boundary coordinate issue"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile
import shutil
from elevation import ElevationLibrary, Bounds

# Create temporary directory
test_dir = tempfile.mkdtemp(prefix="elev_boundary_test_")
print(f"Test directory: {test_dir}")

try:
    # Create library
    lib = ElevationLibrary(data_dir=test_dir, resolution=30)
    
    # Test coordinates exactly at 40.0, -111.0
    test_lat = 40.0
    test_lon = -111.0
    
    # Load a small area around the point
    bounds = Bounds(
        south=test_lat - 0.005,
        north=test_lat + 0.005,
        west=test_lon - 0.005,
        east=test_lon + 0.005
    )
    
    print(f"\nLoading area: {bounds}")
    result = lib.load_area(bounds)
    print(f"Load result: {result}")
    
    # Check which tile the coordinate maps to
    tile_key = lib._get_tile_key(test_lat, test_lon)
    print(f"\nCoordinate ({test_lat}, {test_lon}) maps to tile: {tile_key}")
    
    # Try to get elevation
    try:
        elevation = lib.get_elevation(test_lat, test_lon)
        print(f"Elevation: {elevation}m")
    except Exception as e:
        print(f"Error getting elevation: {e}")
        
        # Let's try a slightly offset coordinate
        offset_lat = test_lat - 0.0001
        offset_lon = test_lon + 0.0001
        print(f"\nTrying offset coordinates ({offset_lat}, {offset_lon})")
        
        try:
            elevation = lib.get_elevation(offset_lat, offset_lon)
            print(f"Elevation at offset: {elevation}m")
        except Exception as e2:
            print(f"Still failed: {e2}")

finally:
    # Clean up
    if test_dir and test_dir.startswith("/tmp"):
        shutil.rmtree(test_dir)
        print(f"\nCleaned up {test_dir}")