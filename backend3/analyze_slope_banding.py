#!/usr/bin/env python3
"""
Analyze potential banding in slope calculations.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from elevation import TwoLayerElevationLibrary, Bounds
from slope_layer import SlopeLayer


def analyze_banding():
    """Analyze what might be causing banding in slope maps"""
    
    bounds = Bounds(
        south=40.650,
        north=40.655,
        west=-111.575,
        east=-111.570
    )
    
    # Initialize
    elev_lib = TwoLayerElevationLibrary("./elevation_data", resolution=10)
    slope_layer = SlopeLayer(elev_lib, "./slope_data")
    
    # Get elevation data
    elev_array, elev_meta = elev_lib.get_elevation_array(bounds)
    
    print("Elevation Data Analysis")
    print("=" * 50)
    print(f"Data type: {elev_array.dtype}")
    print(f"Min elevation: {elev_array.min():.6f}")
    print(f"Max elevation: {elev_array.max():.6f}")
    
    # Check for discrete steps in elevation
    unique_values = np.unique(elev_array)
    print(f"\nUnique elevation values: {len(unique_values)}")
    
    # Check elevation precision
    if len(unique_values) < 100:
        print("WARNING: Very few unique elevation values - data may be quantized!")
        print("First 10 unique values:", unique_values[:10])
    
    # Look at elevation differences
    elev_diffs = np.diff(np.sort(unique_values))
    if len(elev_diffs) > 0:
        min_diff = elev_diffs[elev_diffs > 0].min() if any(elev_diffs > 0) else 0
        print(f"Minimum elevation step: {min_diff:.6f}m")
    
    # Check if values appear to be integers
    rounded = np.round(elev_array)
    if np.allclose(elev_array, rounded):
        print("\nWARNING: Elevation data appears to be integer values!")
    
    # Check data source resolution
    print(f"\nData shape: {elev_array.shape}")
    print(f"Approximate area covered: {elev_array.shape[0]*10}m x {elev_array.shape[1]*10}m")
    
    # Look for periodic patterns in differences
    row_diffs = np.diff(elev_array[20, :])  # Sample row
    col_diffs = np.diff(elev_array[:, 20])  # Sample column
    print(f"Std dev of elevation differences (row): {np.std(row_diffs):.4f}m")
    print(f"Std dev of elevation differences (col): {np.std(col_diffs):.4f}m")
    
    # Get slope data
    slope_data, _ = slope_layer.get_slope_array(bounds)
    
    print("\n\nSlope Data Analysis")
    print("=" * 50)
    
    # Check for discrete slope values
    unique_slopes = np.unique(slope_data.slope)
    print(f"Unique slope values: {len(unique_slopes)}")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Elevation histogram
    ax1 = axes[0, 0]
    ax1.hist(elev_array.flatten(), bins=100, edgecolor='black', alpha=0.7)
    ax1.set_title('Elevation Distribution')
    ax1.set_xlabel('Elevation (m)')
    ax1.set_ylabel('Count')
    
    # 2. Slope histogram
    ax2 = axes[0, 1]
    ax2.hist(slope_data.slope.flatten(), bins=100, edgecolor='black', alpha=0.7)
    ax2.set_title('Slope Distribution')
    ax2.set_xlabel('Slope (degrees)')
    ax2.set_ylabel('Count')
    
    # 3. Elevation gradients
    ax3 = axes[0, 2]
    grad_y, grad_x = np.gradient(elev_array)
    gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
    ax3.hist(gradient_mag.flatten(), bins=100, edgecolor='black', alpha=0.7)
    ax3.set_title('Raw Gradient Magnitude')
    ax3.set_xlabel('Gradient (m/pixel)')
    ax3.set_ylabel('Count')
    
    # 4. Small section of elevation
    ax4 = axes[1, 0]
    small_section = elev_array[20:40, 20:40]
    im4 = ax4.imshow(small_section, cmap='terrain', interpolation='nearest')
    ax4.set_title('Elevation (20x20 pixel section)')
    plt.colorbar(im4, ax=ax4, label='Elevation (m)')
    
    # 5. Same section slopes
    ax5 = axes[1, 1]
    small_slopes = slope_data.slope[20:40, 20:40]
    im5 = ax5.imshow(small_slopes, cmap='hot', interpolation='nearest')
    ax5.set_title('Slopes (same section)')
    plt.colorbar(im5, ax=ax5, label='Slope (degrees)')
    
    # 6. Cross-section plot
    ax6 = axes[1, 2]
    # Use minimum of both array dimensions
    mid_row = min(elev_array.shape[0], slope_data.slope.shape[0]) // 2
    max_cols = min(elev_array.shape[1], slope_data.slope.shape[1])
    
    elevation_profile = elev_array[mid_row, :max_cols]
    slope_profile = slope_data.slope[mid_row, :max_cols]
    x_coords = np.arange(len(elevation_profile))
    
    ax6.plot(x_coords, elevation_profile, 'b-', linewidth=2, label='Elevation')
    ax6.set_xlabel('Pixel')
    ax6.set_ylabel('Elevation (m)', color='b')
    ax6.tick_params(axis='y', labelcolor='b')
    
    # Add slope on secondary axis
    ax6_twin = ax6.twinx()
    ax6_twin.plot(x_coords, slope_profile, 'r-', linewidth=2, label='Slope', alpha=0.7)
    ax6_twin.set_ylabel('Slope (degrees)', color='r')
    ax6_twin.tick_params(axis='y', labelcolor='r')
    ax6.set_title('Cross-section Profile')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('banding_analysis.png', dpi=150, bbox_inches='tight')
    print("\nAnalysis saved to: banding_analysis.png")
    
    # Additional analysis: Check for patterns
    print("\n\nPattern Analysis")
    print("=" * 50)
    
    # Check if slopes cluster around certain values
    slope_hist, bins = np.histogram(slope_data.slope.flatten(), bins=360)
    peak_indices = np.where(slope_hist > np.mean(slope_hist) + 2*np.std(slope_hist))[0]
    if len(peak_indices) > 0:
        print(f"Slope values cluster around: {[f'{bins[i]:.1f}' for i in peak_indices]} degrees")
    
    # Check gradient consistency
    # Calculate theoretical slope from elevation gradient
    pixel_size_m = 10  # approximate
    theoretical_slope = np.degrees(np.arctan(gradient_mag / pixel_size_m))
    slope_diff = np.abs(slope_data.slope - theoretical_slope)
    print(f"Mean difference between calculated and theoretical slope: {slope_diff.mean():.2f}°")
    
    return elev_array, slope_data


if __name__ == "__main__":
    analyze_banding()