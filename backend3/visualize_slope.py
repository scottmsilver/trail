#!/usr/bin/env python3
"""
Visualize slope data from the slope analysis layer.
Creates maps showing slope, slope change, and aspect.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from elevation import TwoLayerElevationLibrary, Bounds
from slope_layer import SlopeLayer


def create_slope_maps(elev_lib: TwoLayerElevationLibrary, slope_layer: SlopeLayer,
                     bounds: Bounds, output_prefix: str = "slope_map"):
    """
    Create visualizations of slope data.
    
    Args:
        elev_lib: Elevation library instance
        slope_layer: Slope layer instance
        bounds: Geographic bounds to visualize
        output_prefix: Prefix for output files
    """
    # Get slope data
    try:
        slope_data, metadata = slope_layer.get_slope_array(bounds)
    except ValueError as e:
        print(f"Error: {e}")
        print("Computing slopes for this area...")
        result = slope_layer.compute_area(bounds)
        if result["status"] != "success":
            print(f"Failed to compute slopes: {result}")
            return
        slope_data, metadata = slope_layer.get_slope_array(bounds)
    
    print(f"Slope array shape: {slope_data.slope.shape}")
    print(f"Slope range: {np.min(slope_data.slope):.1f}° to {np.max(slope_data.slope):.1f}°")
    print(f"Max slope change: {np.max(slope_data.slope_change):.2f}°/m")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Calculate extent for plots
    extent = [bounds.west, bounds.east, bounds.south, bounds.north]
    
    # 1. Slope map (top left)
    ax1 = axes[0, 0]
    
    # Create custom colormap for slopes
    # Green (flat) -> Yellow (moderate) -> Red (steep)
    slope_colors = ['#2d7f2d', '#7fbf7f', '#ffff00', '#ff7f00', '#ff0000', '#8b0000']
    n_bins = 100
    slope_cmap = LinearSegmentedColormap.from_list('slope', slope_colors, N=n_bins)
    
    im1 = ax1.imshow(slope_data.slope, extent=extent, origin='lower',
                     cmap=slope_cmap, vmin=0, vmax=45, aspect='equal')
    ax1.set_title('Slope (degrees)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    
    # Add contour lines
    contour_levels = [5, 10, 15, 20, 25, 30, 35, 40]
    contours = ax1.contour(slope_data.slope, levels=contour_levels,
                          extent=extent, colors='black', alpha=0.3, linewidths=0.5)
    ax1.clabel(contours, inline=True, fontsize=8, fmt='%d°')
    
    # Colorbar
    cbar1 = plt.colorbar(im1, ax=ax1, label='Slope (degrees)')
    
    # 2. Slope change/curvature map (top right)
    ax2 = axes[0, 1]
    
    # Use diverging colormap for curvature (blue=concave, red=convex)
    max_curve = np.percentile(np.abs(slope_data.slope_change), 95)
    im2 = ax2.imshow(slope_data.slope_change, extent=extent, origin='lower',
                     cmap='RdBu_r', vmin=-max_curve, vmax=max_curve, aspect='equal')
    ax2.set_title('Slope Change (Curvature)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    
    # Colorbar
    cbar2 = plt.colorbar(im2, ax=ax2, label='Slope change (°/m)')
    
    # 3. Aspect map (bottom left)
    ax3 = axes[1, 0]
    
    # Create circular colormap for aspect
    # North=Red, East=Yellow, South=Cyan, West=Blue, back to North=Red
    aspect_cmap = cm.get_cmap('hsv')
    
    im3 = ax3.imshow(slope_data.aspect, extent=extent, origin='lower',
                     cmap=aspect_cmap, vmin=0, vmax=360, aspect='equal')
    ax3.set_title('Aspect (slope direction)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Longitude')
    ax3.set_ylabel('Latitude')
    
    # Colorbar with compass directions
    cbar3 = plt.colorbar(im3, ax=ax3, label='Direction (degrees)')
    cbar3.set_ticks([0, 90, 180, 270, 360])
    cbar3.set_ticklabels(['N', 'E', 'S', 'W', 'N'])
    
    # 4. Combined visualization (bottom right)
    ax4 = axes[1, 1]
    
    # Get elevation data for hillshade
    try:
        elev_array, _ = elev_lib.get_elevation_array(bounds)
    except:
        elev_array = None
    
    if elev_array is not None:
        # Create hillshade
        from matplotlib.colors import LightSource
        ls = LightSource(azdeg=315, altdeg=45)
        
        # Combine slope with hillshade
        rgb = ls.shade(elev_array, cmap=slope_cmap, 
                      blend_mode='overlay', vmin=0, vmax=45,
                      vert_exag=2)
        
        # Overlay slope data
        slope_overlay = ls.shade(slope_data.slope, cmap=slope_cmap,
                               blend_mode='overlay', vmin=0, vmax=45)
        
        ax4.imshow(slope_overlay, extent=extent, origin='lower', aspect='equal')
        ax4.set_title('Slope with Terrain Shading', fontsize=14, fontweight='bold')
    else:
        # Just show slope again if no elevation
        im4 = ax4.imshow(slope_data.slope, extent=extent, origin='lower',
                        cmap=slope_cmap, vmin=0, vmax=45, aspect='equal')
        ax4.set_title('Slope (degrees)', fontsize=14, fontweight='bold')
    
    ax4.set_xlabel('Longitude')
    ax4.set_ylabel('Latitude')
    
    # Overall title
    fig.suptitle(f'Slope Analysis - {slope_data.resolution_m}m Resolution', 
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    output_file = f"{output_prefix}_{slope_data.resolution_m}m.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSlope map saved to: {output_file}")
    
    plt.close()
    
    # Also create individual maps
    create_single_slope_map(slope_data, bounds, f"{output_prefix}_slope_only")
    create_aspect_wheel(slope_data, bounds, f"{output_prefix}_aspect_wheel")


def create_single_slope_map(slope_data, bounds, output_prefix):
    """Create a single slope map with better styling"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    extent = [bounds.west, bounds.east, bounds.south, bounds.north]
    
    # Create custom colormap
    slope_colors = ['#2d7f2d', '#7fbf7f', '#ffff00', '#ff7f00', '#ff0000', '#8b0000']
    slope_cmap = LinearSegmentedColormap.from_list('slope', slope_colors, N=100)
    
    im = ax.imshow(slope_data.slope, extent=extent, origin='lower',
                   cmap=slope_cmap, vmin=0, vmax=45, aspect='equal')
    
    # Add contours
    contour_levels = [5, 10, 15, 20, 25, 30, 35, 40]
    contours = ax.contour(slope_data.slope, levels=contour_levels,
                         extent=extent, colors='black', alpha=0.4, linewidths=1)
    ax.clabel(contours, inline=True, fontsize=10, fmt='%d°')
    
    ax.set_title(f'Terrain Slope Map - {slope_data.resolution_m}m Resolution',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', 
                       pad=0.05, aspect=30)
    cbar.set_label('Slope (degrees)', fontsize=12)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_file = f"{output_prefix}_{slope_data.resolution_m}m.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Single slope map saved to: {output_file}")
    plt.close()


def create_aspect_wheel(slope_data, bounds, output_prefix):
    """Create an aspect map with a compass wheel legend"""
    fig, (ax_map, ax_wheel) = plt.subplots(1, 2, figsize=(14, 8),
                                           gridspec_kw={'width_ratios': [3, 1]})
    
    extent = [bounds.west, bounds.east, bounds.south, bounds.north]
    
    # Aspect map
    aspect_cmap = cm.get_cmap('hsv')
    im = ax_map.imshow(slope_data.aspect, extent=extent, origin='lower',
                      cmap=aspect_cmap, vmin=0, vmax=360, aspect='equal')
    
    ax_map.set_title(f'Aspect (Slope Direction) - {slope_data.resolution_m}m Resolution',
                    fontsize=16, fontweight='bold', pad=20)
    ax_map.set_xlabel('Longitude', fontsize=12)
    ax_map.set_ylabel('Latitude', fontsize=12)
    ax_map.grid(True, alpha=0.3, linestyle='--')
    
    # Compass wheel
    theta = np.linspace(0, 2*np.pi, 360)
    r = np.ones_like(theta)
    colors = np.linspace(0, 360, 360)
    
    # Create wheel using polar projection
    ax_wheel.remove()
    ax_wheel = fig.add_subplot(1, 2, 2, projection='polar')
    
    # Plot colored wheel
    for i in range(len(theta)-1):
        ax_wheel.fill_between([theta[i], theta[i+1]], 0, 1,
                            color=aspect_cmap(colors[i]/360))
    
    # Add compass labels
    ax_wheel.set_theta_zero_location('N')
    ax_wheel.set_theta_direction(-1)
    ax_wheel.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
    ax_wheel.set_xticklabels(['N', 'E', 'S', 'W'], fontsize=14, fontweight='bold')
    ax_wheel.set_ylim(0, 1)
    ax_wheel.set_yticks([])
    ax_wheel.set_title('Aspect Direction', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    output_file = f"{output_prefix}_{slope_data.resolution_m}m.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Aspect wheel map saved to: {output_file}")
    plt.close()


def main():
    """Main function to create slope visualizations"""
    # Default area (Park City)
    bounds = Bounds(
        south=40.6448,
        north=40.6588,
        west=-111.5780,
        east=-111.5595
    )
    
    # Data directories
    data_dir = "./elevation_data"
    slope_dir = "./slope_data"
    
    # Create directories if needed
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(slope_dir, exist_ok=True)
    
    print("Initializing elevation and slope libraries...")
    
    # Initialize libraries
    elev_lib = TwoLayerElevationLibrary(data_dir=data_dir, resolution=10)
    slope_layer = SlopeLayer(elev_lib, data_dir=slope_dir)
    
    # Check if elevation data is loaded
    try:
        elev_array, _ = elev_lib.get_elevation_array(bounds)
        print("Elevation data found")
    except ValueError:
        print("Elevation data not found. Loading area...")
        result = elev_lib.load_area(bounds)
        print(f"Loaded {result['total_tiles']} elevation tiles")
    
    # Create slope visualizations
    print("\nCreating slope visualizations...")
    create_slope_maps(elev_lib, slope_layer, bounds, output_prefix="slope_map")
    
    print("\nDone!")


if __name__ == "__main__":
    main()