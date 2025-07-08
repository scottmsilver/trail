#!/usr/bin/env python3
"""
Investigate banding artifacts in slope calculations more deeply.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from elevation import TwoLayerElevationLibrary, Bounds
from slope_layer import SlopeLayer


def investigate_banding():
    """Deep dive into banding artifacts"""
    
    bounds = Bounds(
        south=40.650,
        north=40.655,
        west=-111.575,
        east=-111.570
    )
    
    # Initialize
    elev_lib = TwoLayerElevationLibrary("./elevation_data", resolution=10)
    slope_layer = SlopeLayer(elev_lib, "./slope_data")
    
    # Get data
    elev_array, elev_meta = elev_lib.get_elevation_array(bounds)
    slope_data, _ = slope_layer.get_slope_array(bounds)
    
    print("Investigating Slope Banding")
    print("=" * 50)
    print(f"Elevation array shape: {elev_array.shape}")
    print(f"Slope array shape: {slope_data.slope.shape}")
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Original slope map with different colormaps
    ax1 = axes[0, 0]
    im1 = ax1.imshow(slope_data.slope, cmap='viridis', aspect='equal')
    ax1.set_title('Slopes - Continuous Colormap (viridis)')
    plt.colorbar(im1, ax=ax1, label='Slope (degrees)')
    
    # 2. Slope map with discrete colormap
    ax2 = axes[0, 1]
    # Create discrete colormap with specific bins
    n_bins = 20
    cmap = plt.get_cmap('RdYlGn_r')
    discrete_cmap = LinearSegmentedColormap.from_list(
        'discrete', cmap(np.linspace(0, 1, n_bins)), N=n_bins
    )
    im2 = ax2.imshow(slope_data.slope, cmap=discrete_cmap, aspect='equal',
                     vmin=0, vmax=40)
    ax2.set_title(f'Slopes - Discrete Colormap ({n_bins} bins)')
    plt.colorbar(im2, ax=ax2, label='Slope (degrees)')
    
    # 3. Gradient magnitude from elevation
    ax3 = axes[0, 2]
    grad_y, grad_x = np.gradient(elev_array)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    im3 = ax3.imshow(grad_mag, cmap='plasma', aspect='equal')
    ax3.set_title('Raw Gradient Magnitude')
    plt.colorbar(im3, ax=ax3, label='Gradient (m/pixel)')
    
    # 4. Slope calculation verification
    ax4 = axes[1, 0]
    # Recalculate slopes using simple method
    pixel_size = 10  # meters
    simple_slope = np.degrees(np.arctan(grad_mag / pixel_size))
    im4 = ax4.imshow(simple_slope, cmap='viridis', aspect='equal')
    ax4.set_title('Simple Slope Calculation')
    plt.colorbar(im4, ax=ax4, label='Slope (degrees)')
    
    # 5. Difference between methods
    ax5 = axes[1, 1]
    # Resize arrays to match
    min_height = min(slope_data.slope.shape[0], simple_slope.shape[0])
    min_width = min(slope_data.slope.shape[1], simple_slope.shape[1])
    slope_trimmed = slope_data.slope[:min_height, :min_width]
    simple_trimmed = simple_slope[:min_height, :min_width]
    
    slope_diff = slope_trimmed - simple_trimmed
    im5 = ax5.imshow(slope_diff, cmap='RdBu_r', aspect='equal',
                     vmin=-5, vmax=5)
    ax5.set_title('Difference (Layer - Simple)')
    plt.colorbar(im5, ax=ax5, label='Difference (degrees)')
    
    # 6. Histogram of slope values
    ax6 = axes[1, 2]
    # Plot both histograms
    bins = np.linspace(0, 40, 100)
    ax6.hist(slope_data.slope.flatten(), bins=bins, alpha=0.5, 
             label='Slope Layer', density=True)
    ax6.hist(simple_slope.flatten(), bins=bins, alpha=0.5,
             label='Simple Calc', density=True)
    ax6.set_xlabel('Slope (degrees)')
    ax6.set_ylabel('Density')
    ax6.set_title('Slope Distribution Comparison')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('Slope Banding Investigation', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('banding_investigation.png', dpi=150, bbox_inches='tight')
    print("Investigation saved to: banding_investigation.png")
    
    # Additional analysis
    print(f"\nSlope statistics:")
    print(f"  Slope layer - min: {slope_data.slope.min():.2f}°, max: {slope_data.slope.max():.2f}°")
    print(f"  Simple calc - min: {simple_slope.min():.2f}°, max: {simple_slope.max():.2f}°")
    print(f"  Mean difference: {slope_diff.mean():.3f}°")
    print(f"  Std difference: {slope_diff.std():.3f}°")
    
    # Check for quantization
    unique_slopes = np.unique(slope_data.slope)
    slope_steps = np.diff(unique_slopes)
    if len(slope_steps) > 0:
        common_step = np.median(slope_steps[slope_steps > 0.001])
        print(f"\nMedian step between unique slope values: {common_step:.4f}°")
    
    # Analysis of flat areas
    flat_mask = slope_data.slope < 5  # Areas less than 5 degrees
    flat_percent = 100 * flat_mask.sum() / slope_data.slope.size
    print(f"\nFlat areas (<5°): {flat_percent:.1f}% of terrain")
    
    # The issue might be that on relatively flat terrain,
    # small elevation changes create discrete slope bands
    print("\nPossible causes of banding:")
    print("1. On gentle terrain, small elevation steps create visible slope bands")
    print("2. The gradient calculation amplifies small differences")
    print("3. Visualization colormaps can exaggerate subtle differences")
    print("4. Original DEM data may have subtle terracing from processing")
    
    return elev_array, slope_data, simple_slope


if __name__ == "__main__":
    investigate_banding()