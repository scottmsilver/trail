#!/usr/bin/env python3
"""
Debug why slope arrays are smaller than elevation arrays.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from elevation import TwoLayerElevationLibrary, Bounds
from slope_layer import SlopeLayer


def debug_size_issue():
    """Debug the size mismatch between elevation and slope arrays"""
    
    bounds = Bounds(
        south=40.650,
        north=40.655,
        west=-111.575,
        east=-111.570
    )
    
    # Initialize
    elev_lib = TwoLayerElevationLibrary("./elevation_data", resolution=10)
    slope_layer = SlopeLayer(elev_lib, "./slope_data")
    
    print("Debugging Size Mismatch")
    print("=" * 50)
    
    # Get elevation array
    elev_array, elev_meta = elev_lib.get_elevation_array(bounds)
    print(f"Requested bounds: {bounds}")
    print(f"Elevation array shape: {elev_array.shape}")
    print(f"Elevation metadata: {elev_meta}")
    
    # Check what happens during slope computation
    print("\nDuring slope computation:")
    
    # Look at the pixel size calculation
    transform = elev_meta.get("transform", {})
    if transform:
        pixel_width_deg = abs(transform.get("a", 0))
        pixel_height_deg = abs(transform.get("e", 0))
        print(f"Pixel width (degrees): {pixel_width_deg}")
        print(f"Pixel height (degrees): {pixel_height_deg}")
        
        # This is what slope_layer does
        lat_center = (bounds.north + bounds.south) / 2
        meters_per_degree_lat = 111320.0
        meters_per_degree_lon = 111320.0 * np.cos(np.radians(lat_center))
        pixel_width_m = pixel_width_deg * meters_per_degree_lon
        pixel_height_m = pixel_height_deg * meters_per_degree_lat
        pixel_size_m = (pixel_width_m + pixel_height_m) / 2
        
        print(f"Calculated pixel size: {pixel_size_m:.2f}m")
    
    # Now check slope array
    slope_data, slope_meta = slope_layer.get_slope_array(bounds)
    print(f"\nSlope array shape: {slope_data.slope.shape}")
    print(f"Slope metadata: {slope_meta}")
    
    # Check the tile creation process
    print("\nChecking tile dimensions:")
    
    # List slope tiles
    slope_tiles = slope_layer.list_computed_areas()
    print(f"Number of slope tiles: {slope_tiles['total_tiles']}")
    
    # Look at individual tile info
    for tile_key, tile_info in list(slope_layer.index["tiles"].items())[:2]:
        print(f"\nTile {tile_key}:")
        print(f"  Bounds: {tile_info['bounds']}")
        
        # Try to open the tile and check its size
        import rasterio
        with rasterio.open(tile_info['path']) as src:
            print(f"  Raster shape: {src.height} x {src.width}")
            print(f"  Transform: {src.transform}")
    
    return elev_array, slope_data


if __name__ == "__main__":
    import numpy as np
    debug_size_issue()