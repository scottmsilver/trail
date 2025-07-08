#!/usr/bin/env python3
"""
Fix the slope layer tiling issue.
The problem: slope tiles are being extracted incorrectly from the computed data.
"""

import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from elevation import TwoLayerElevationLibrary, Bounds
from slope_layer import SlopeLayer


def analyze_tiling_issue():
    """Analyze why slope tiles have wrong dimensions"""
    
    # Small test area
    bounds = Bounds(
        south=40.650,
        north=40.652,  # Smaller area
        west=-111.572,
        east=-111.570
    )
    
    elev_lib = TwoLayerElevationLibrary("./elevation_data", resolution=10)
    slope_layer = SlopeLayer(elev_lib, "./slope_data")
    
    print("Analyzing Tiling Issue")
    print("=" * 50)
    print(f"Test bounds: {bounds}")
    print(f"Tile size: {slope_layer.tile_size}")
    
    # Get elevation data
    elev_array, elev_meta = elev_lib.get_elevation_array(bounds)
    print(f"\nElevation array shape: {elev_array.shape}")
    
    # Manually compute slopes
    print("\nManual slope computation:")
    pixel_size_m = 10
    grad_y, grad_x = np.gradient(elev_array)
    grad_x = grad_x / pixel_size_m
    grad_y = grad_y / pixel_size_m
    slope_rad = np.arctan(np.sqrt(grad_x**2 + grad_y**2))
    slope_deg = np.degrees(slope_rad)
    print(f"Computed slope array shape: {slope_deg.shape}")
    
    # Now let's see what tiles would be created
    print("\nTile calculation:")
    south_tiles = int(np.floor(bounds.south / slope_layer.tile_size))
    north_tiles = int(np.floor(bounds.north / slope_layer.tile_size))
    west_tiles = int(np.floor(bounds.west / slope_layer.tile_size))
    east_tiles = int(np.floor(bounds.east / slope_layer.tile_size))
    
    print(f"Tile indices: lat {south_tiles} to {north_tiles}, lon {west_tiles} to {east_tiles}")
    
    # The issue is likely in how pixels are mapped to tiles
    # Let's trace through the tile creation logic
    height, width = slope_deg.shape
    lat_range = bounds.north - bounds.south
    lon_range = bounds.east - bounds.west
    
    print(f"\nData dimensions:")
    print(f"Height: {height} pixels, Lat range: {lat_range}°")
    print(f"Width: {width} pixels, Lon range: {lon_range}°")
    print(f"Pixels per degree: {height/lat_range:.1f} (lat), {width/lon_range:.1f} (lon)")
    
    # Check what happens for one tile
    tile_south = south_tiles * slope_layer.tile_size
    tile_north = (south_tiles + 1) * slope_layer.tile_size
    tile_west = west_tiles * slope_layer.tile_size
    tile_east = (west_tiles + 1) * slope_layer.tile_size
    
    print(f"\nFirst tile bounds:")
    print(f"Geographic: {tile_south:.3f} to {tile_north:.3f}, {tile_west:.3f} to {tile_east:.3f}")
    
    # Calculate pixel extraction
    px_south = int((tile_south - bounds.south) / lat_range * height)
    px_north = int((tile_north - bounds.south) / lat_range * height)
    px_west = int((tile_west - bounds.west) / lon_range * width)
    px_east = int((tile_east - bounds.west) / lon_range * width)
    
    print(f"Pixel indices: [{px_south}:{px_north}, {px_west}:{px_east}]")
    
    if px_south >= 0 and px_north <= height and px_west >= 0 and px_east <= width:
        extracted_shape = (px_north - px_south, px_east - px_west)
        print(f"Extracted tile shape would be: {extracted_shape}")
    
    # The fix: we need to ensure tiles are extracted with proper alignment
    print("\nSuggested fix:")
    print("1. Use the same pixel-to-geographic mapping as the elevation layer")
    print("2. Ensure transform matrices are correctly calculated")
    print("3. Consider computing slopes tile-by-tile rather than all at once")


if __name__ == "__main__":
    analyze_tiling_issue()