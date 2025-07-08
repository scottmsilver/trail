#!/usr/bin/env python3
"""
Test the fixed slope layer implementation.
"""

import os
import sys
import shutil
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from elevation import TwoLayerElevationLibrary, Bounds
from slope_layer import SlopeLayer


def test_fixed_slopes():
    """Test that the fixed slope layer eliminates banding"""
    
    # Test area
    bounds = Bounds(
        south=40.650,
        north=40.655,
        west=-111.575,
        east=-111.570
    )
    
    # Remove old slope data
    slope_dir = "./slope_data"
    if os.path.exists(slope_dir):
        print("Removing old slope data...")
        shutil.rmtree(slope_dir)
    
    # Initialize libraries
    elev_lib = TwoLayerElevationLibrary("./elevation_data", resolution=10)
    slope_layer = SlopeLayer(elev_lib, slope_dir)
    
    print("Testing Fixed Slope Layer")
    print("=" * 50)
    
    # Ensure elevation data is loaded
    try:
        elev_array, elev_meta = elev_lib.get_elevation_array(bounds)
        print(f"Elevation array shape: {elev_array.shape}")
    except ValueError:
        print("Loading elevation data...")
        elev_lib.load_area(bounds)
        elev_array, elev_meta = elev_lib.get_elevation_array(bounds)
    
    # Compute slopes with new method
    print("\nComputing slopes with fixed method...")
    result = slope_layer.compute_area(bounds)
    print(f"Result: {result}")
    
    # Get slope array
    slope_data, slope_meta = slope_layer.get_slope_array(bounds)
    print(f"\nSlope array shape: {slope_data.slope.shape}")
    print(f"Slope metadata: {slope_meta}")
    
    # Compare shapes
    print(f"\nShape comparison:")
    print(f"  Elevation: {elev_array.shape}")
    print(f"  Slope: {slope_data.slope.shape}")
    
    # Create comparison visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Elevation
    ax1 = axes[0, 0]
    im1 = ax1.imshow(elev_array, cmap='terrain', aspect='equal')
    ax1.set_title(f'Elevation ({elev_array.shape})')
    plt.colorbar(im1, ax=ax1, label='Elevation (m)')
    
    # 2. New slope calculation
    ax2 = axes[0, 1]
    im2 = ax2.imshow(slope_data.slope, cmap='YlOrRd', aspect='equal', vmin=0, vmax=40)
    ax2.set_title(f'Fixed Slope Calculation ({slope_data.slope.shape})')
    plt.colorbar(im2, ax=ax2, label='Slope (degrees)')
    
    # 3. Slope histogram
    ax3 = axes[1, 0]
    ax3.hist(slope_data.slope.flatten(), bins=100, edgecolor='black', alpha=0.7)
    ax3.set_xlabel('Slope (degrees)')
    ax3.set_ylabel('Count')
    ax3.set_title('Slope Distribution (Fixed)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Cross-section
    ax4 = axes[1, 1]
    mid_row = min(elev_array.shape[0], slope_data.slope.shape[0]) // 2
    max_cols = min(elev_array.shape[1], slope_data.slope.shape[1])
    
    x_coords = np.arange(max_cols)
    
    # Normalize elevation for display
    elev_profile = elev_array[mid_row, :max_cols]
    elev_norm = (elev_profile - elev_profile.min()) / (elev_profile.max() - elev_profile.min())
    
    ax4.plot(x_coords, elev_norm, 'b-', linewidth=2, label='Elevation (normalized)')
    
    if slope_data.slope.shape[0] > mid_row:
        slope_profile = slope_data.slope[mid_row, :max_cols]
        slope_norm = slope_profile / 40  # Normalize to 0-1 assuming max 40 degrees
        ax4.plot(x_coords, slope_norm, 'r-', linewidth=2, label='Slope/40°')
    
    ax4.set_xlabel('Pixel')
    ax4.set_ylabel('Normalized Value')
    ax4.set_title('Cross-section Profiles')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Fixed Slope Layer Test', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('fixed_slopes_test.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to: fixed_slopes_test.png")
    
    # Check tile info
    print("\nChecking tile information:")
    sample_lat, sample_lon = 40.652, -111.572
    tile_info = slope_layer.get_tile_info(sample_lat, sample_lon)
    if tile_info:
        print(f"Tile at ({sample_lat}, {sample_lon}): {tile_info}")
    
    # Check for size consistency
    print("\nAnalyzing tiles:")
    for i, (tile_key, tile_data) in enumerate(list(slope_layer.index["tiles"].items())[:3]):
        print(f"  Tile {tile_key}: bounds={tile_data['bounds']}")
        
        # Check the actual file
        import rasterio
        with rasterio.open(tile_data['path']) as src:
            print(f"    Shape: {src.height}x{src.width}, Transform: {src.transform}")
    
    return elev_array, slope_data


if __name__ == "__main__":
    test_fixed_slopes()