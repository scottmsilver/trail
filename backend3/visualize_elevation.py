#!/usr/bin/env python3
"""
Visualize elevation data as a depth map for a given area.
Creates a heatmap showing elevation with optional path overlay.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from elevation import TwoLayerElevationLibrary, Bounds


def create_depth_map(data_dir: str, start_lat: float, start_lon: float, 
                     end_lat: float, end_lon: float, resolution: int = 10,
                     output_file: str = "depth_map.png"):
    """
    Create a depth map visualization of elevation data.
    
    Args:
        data_dir: Directory containing elevation data
        start_lat, start_lon: Start coordinates
        end_lat, end_lon: End coordinates
        resolution: DEM resolution in meters
        output_file: Output filename for the visualization
    """
    # Initialize library
    lib = TwoLayerElevationLibrary(data_dir=data_dir, resolution=resolution)
    
    # Create bounds that include both points with some padding
    padding = 0.005  # ~500m padding
    bounds = Bounds(
        south=min(start_lat, end_lat) - padding,
        north=max(start_lat, end_lat) + padding,
        west=min(start_lon, end_lon) - padding,
        east=max(start_lon, end_lon) + padding
    )
    
    print(f"Loading elevation data for bounds:")
    print(f"  South: {bounds.south:.4f}, North: {bounds.north:.4f}")
    print(f"  West: {bounds.west:.4f}, East: {bounds.east:.4f}")
    
    # Load the area
    result = lib.load_area(bounds)
    if result["status"] != "success":
        print(f"Failed to load area: {result}")
        return
    
    print(f"Loaded {result['total_tiles']} tiles at {resolution}m resolution")
    
    # Get elevation array
    elev_array, metadata = lib.get_elevation_array(bounds)
    
    print(f"Elevation array shape: {elev_array.shape}")
    
    # Filter out zeros and NaNs for statistics
    valid_mask = (elev_array > 0) & ~np.isnan(elev_array)
    if np.any(valid_mask):
        valid_elev = elev_array[valid_mask]
        print(f"Elevation range (valid data): {np.min(valid_elev):.1f}m to {np.max(valid_elev):.1f}m")
        print(f"Percentage of valid data: {np.sum(valid_mask) / elev_array.size * 100:.1f}%")
    else:
        print("Warning: No valid elevation data found!")
    
    # Replace zeros with NaN for better visualization
    elev_array[elev_array <= 0] = np.nan
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create custom colormap (terrain-like)
    colors = ['#2E8B57', '#3CB371', '#90EE90', '#FFFF99', '#FFD700', 
              '#FFA500', '#FF6347', '#8B4513', '#A0522D', '#D2691E']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('terrain', colors, N=n_bins)
    
    # Plot elevation as heatmap
    extent = [bounds.west, bounds.east, bounds.south, bounds.north]
    im = ax.imshow(elev_array, extent=extent, origin='lower', 
                   cmap=cmap, alpha=0.9, aspect='equal')
    
    # Add contour lines
    # Calculate positions for contours
    rows, cols = elev_array.shape
    lon_vals = np.linspace(bounds.west, bounds.east, cols)
    lat_vals = np.linspace(bounds.south, bounds.north, rows)
    lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals)
    
    # Add contours every 50m
    contour_interval = 50
    valid_elev = elev_array[~np.isnan(elev_array)]
    if len(valid_elev) > 0:
        min_elev = np.floor(np.min(valid_elev) / contour_interval) * contour_interval
        max_elev = np.ceil(np.max(valid_elev) / contour_interval) * contour_interval
        contour_levels = np.arange(min_elev, max_elev + contour_interval, contour_interval)
    else:
        contour_levels = []
    
    contours = ax.contour(lon_grid, lat_grid, elev_array, 
                         levels=contour_levels, colors='black', 
                         alpha=0.3, linewidths=0.5)
    ax.clabel(contours, inline=True, fontsize=8, fmt='%d')
    
    # Plot start and end points
    ax.plot(start_lon, start_lat, 'go', markersize=15, 
            markeredgecolor='darkgreen', markeredgewidth=2, label='Start')
    ax.plot(end_lon, end_lat, 'ro', markersize=15, 
            markeredgecolor='darkred', markeredgewidth=2, label='End')
    
    # Draw line between start and end
    ax.plot([start_lon, end_lon], [start_lat, end_lat], 'w--', 
            linewidth=2, alpha=0.7, label='Direct path')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('Elevation (meters)', fontsize=12)
    
    # Labels and title
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(f'Elevation Map\n{resolution}m Resolution', fontsize=14)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add legend
    ax.legend(loc='upper right', fontsize=10)
    
    # Set axis limits to exact bounds
    ax.set_xlim(bounds.west, bounds.east)
    ax.set_ylim(bounds.south, bounds.north)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nDepth map saved to: {output_file}")
    
    # Also create a pure grayscale depth map
    grayscale_file = output_file.replace('.png', '_grayscale.png')
    fig2, ax2 = plt.subplots(figsize=(12, 10))
    im2 = ax2.imshow(elev_array, extent=extent, origin='lower', 
                     cmap='gray', alpha=1.0, aspect='equal')
    ax2.plot(start_lon, start_lat, 'wo', markersize=15, 
             markeredgecolor='white', markeredgewidth=2)
    ax2.plot(end_lon, end_lat, 'wo', markersize=15, 
             markeredgecolor='white', markeredgewidth=2)
    ax2.plot([start_lon, end_lon], [start_lat, end_lat], 'w--', 
             linewidth=2, alpha=0.7)
    ax2.set_xlabel('Longitude', fontsize=12)
    ax2.set_ylabel('Latitude', fontsize=12) 
    ax2.set_title(f'Elevation Depth Map (Grayscale)\n{resolution}m Resolution', fontsize=14)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(bounds.west, bounds.east)
    ax2.set_ylim(bounds.south, bounds.north)
    plt.tight_layout()
    plt.savefig(grayscale_file, dpi=300, bbox_inches='tight')
    print(f"Grayscale depth map saved to: {grayscale_file}")
    
    plt.close('all')
    
    # Print elevation at start and end points
    start_elev = lib.get_elevation(start_lat, start_lon)
    end_elev = lib.get_elevation(end_lat, end_lon)
    print(f"\nElevation at start ({start_lat:.4f}, {start_lon:.4f}): {start_elev:.1f}m")
    print(f"Elevation at end ({end_lat:.4f}, {end_lon:.4f}): {end_elev:.1f}m")
    print(f"Elevation change: {end_elev - start_elev:.1f}m")
    
    return lib


def main():
    """Main function to create depth map for specified coordinates"""
    # Coordinates from the user
    start_lat, start_lon = 40.6588, -111.5780
    end_lat, end_lon = 40.6448, -111.5595
    
    # Data directory - adjust as needed
    data_dir = "/home/ssilver/development/trail/backend3/elevation_data"
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Creating data directory: {data_dir}")
        os.makedirs(data_dir)
    
    # Create depth maps at different resolutions
    resolutions = [10]  # Start with 10m, can add [3, 5, 10, 30] for comparison
    
    for res in resolutions:
        print(f"\n{'='*50}")
        print(f"Creating depth map at {res}m resolution")
        print(f"{'='*50}")
        
        output_file = f"depth_map_{res}m.png"
        try:
            lib = create_depth_map(
                data_dir=data_dir,
                start_lat=start_lat,
                start_lon=start_lon,
                end_lat=end_lat,
                end_lon=end_lon,
                resolution=res,
                output_file=output_file
            )
        except Exception as e:
            print(f"Error creating depth map: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()