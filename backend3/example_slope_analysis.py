#!/usr/bin/env python3
"""
Example: Using the slope analysis layer to get terrain characteristics.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from elevation import TwoLayerElevationLibrary, Bounds
from slope_layer import SlopeLayer


def main():
    """Demonstrate slope analysis capabilities"""
    # Define area of interest
    bounds = Bounds(
        south=40.650,
        north=40.655,
        west=-111.575,
        east=-111.570
    )
    
    # Initialize libraries
    elev_lib = TwoLayerElevationLibrary("./elevation_data", resolution=10)
    slope_layer = SlopeLayer(elev_lib, "./slope_data")
    
    print("Slope Analysis Example")
    print("=" * 50)
    
    # 1. Load elevation data if needed
    try:
        elev_array, _ = elev_lib.get_elevation_array(bounds)
        print("✓ Elevation data available")
    except ValueError:
        print("Loading elevation data...")
        result = elev_lib.load_area(bounds)
        print(f"✓ Loaded {result['total_tiles']} elevation tiles")
    
    # 2. Compute slopes
    print("\nComputing slopes...")
    result = slope_layer.compute_area(bounds)
    if result["status"] == "success":
        print(f"✓ Created {result['tiles_created']} slope tiles")
    else:
        print(f"✗ Error: {result}")
        return
    
    # 3. Query specific points
    print("\nQuerying specific locations:")
    test_points = [
        (40.652, -111.573, "Point A"),
        (40.653, -111.572, "Point B"),
        (40.654, -111.571, "Point C")
    ]
    
    for lat, lon, name in test_points:
        try:
            slope = slope_layer.get_slope(lat, lon)
            slope_change = slope_layer.get_slope_change(lat, lon)
            aspect = slope_layer.get_aspect(lat, lon)
            elevation = elev_lib.get_elevation(lat, lon)
            
            # Convert aspect to compass direction
            compass_dirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
            dir_index = int((aspect + 22.5) % 360 / 45)
            direction = compass_dirs[dir_index]
            
            print(f"\n{name} ({lat:.4f}, {lon:.4f}):")
            print(f"  Elevation: {elevation:.1f}m")
            print(f"  Slope: {slope:.1f}°")
            print(f"  Slope change: {slope_change:.2f}°/m")
            print(f"  Aspect: {aspect:.1f}° ({direction})")
            
        except ValueError as e:
            print(f"\n{name}: Error - {e}")
    
    # 4. Get statistics for the area
    print("\nArea statistics:")
    slope_data, metadata = slope_layer.get_slope_array(bounds)
    
    print(f"  Min slope: {slope_data.slope.min():.1f}°")
    print(f"  Max slope: {slope_data.slope.max():.1f}°")
    print(f"  Mean slope: {slope_data.slope.mean():.1f}°")
    
    # Find steepest areas
    steep_threshold = 30  # degrees
    steep_pixels = (slope_data.slope > steep_threshold).sum()
    total_pixels = slope_data.slope.size
    steep_percent = 100 * steep_pixels / total_pixels
    
    print(f"\n  Steep terrain (>{steep_threshold}°): {steep_percent:.1f}% of area")
    
    # 5. Save the computed data
    areas = slope_layer.list_computed_areas()
    print(f"\nTotal slope data stored: {areas['total_size_mb']:.2f} MB")


if __name__ == "__main__":
    main()