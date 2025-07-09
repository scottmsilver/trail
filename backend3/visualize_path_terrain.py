#!/usr/bin/env python3
"""
Visualize path with hillshade and slope overlay to better understand terrain.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from elevation import TwoLayerElevationLibrary, Bounds
from elevation_pathfinder import ElevationPathfinder

def create_hillshade(elevation, azimuth=315, altitude=45):
    """Create hillshade from elevation data."""
    # Calculate gradients
    dy, dx = np.gradient(elevation)
    
    # Convert angles to radians
    azimuth = 360.0 - azimuth + 90
    azimuth_rad = np.radians(azimuth)
    altitude_rad = np.radians(altitude)
    
    # Calculate slope and aspect
    slope = np.arctan(np.sqrt(dx*dx + dy*dy))
    aspect = np.arctan2(-dx, dy)
    
    # Calculate hillshade
    shaded = np.sin(altitude_rad) * np.sin(slope) \
           + np.cos(altitude_rad) * np.cos(slope) \
           * np.cos(azimuth_rad - aspect)
    
    return (shaded + 1) / 2  # Normalize to 0-1

def visualize_path_with_terrain(start_lat, start_lon, goal_lat, goal_lon,
                               resolution=10, output_file="path_terrain.png"):
    """Create a detailed terrain visualization with path."""
    
    # Setup bounds
    margin = 0.002
    min_lat = min(start_lat, goal_lat) - margin
    max_lat = max(start_lat, goal_lat) + margin
    min_lon = min(start_lon, goal_lon) - margin
    max_lon = max(start_lon, goal_lon) + margin
    bounds = Bounds(north=max_lat, south=min_lat, east=max_lon, west=min_lon)
    
    # Load elevation data
    print("Loading elevation data...")
    elev_lib = TwoLayerElevationLibrary('./elevation_data', resolution)
    result = elev_lib.load_area(bounds)
    
    if result['status'] != 'success':
        print(f"Failed to load elevation data")
        return
    
    # Get elevation array
    elev_array, metadata = elev_lib.get_elevation_array(bounds)
    
    # Calculate pixel size
    lat_center = (bounds.north + bounds.south) / 2
    lon_deg_to_m = 111000 * np.cos(np.radians(lat_center))
    pixel_size = abs(metadata['transform']['a']) * lon_deg_to_m
    
    # Convert coordinates to pixels
    def latlon_to_pixel(lat, lon):
        col = int((lon - bounds.west) / (bounds.east - bounds.west) * elev_array.shape[1])
        row = int((bounds.north - lat) / (bounds.north - bounds.south) * elev_array.shape[0])
        return min(max(row, 0), elev_array.shape[0]-1), min(max(col, 0), elev_array.shape[1]-1)
    
    start_row, start_col = latlon_to_pixel(start_lat, start_lon)
    goal_row, goal_col = latlon_to_pixel(goal_lat, goal_lon)
    
    # Calculate slope
    dy, dx = np.gradient(elev_array, pixel_size)
    slope_degrees = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
    
    # Create hillshade
    hillshade = create_hillshade(elev_array)
    
    # Find multiple paths with different parameters
    paths = []
    params = [
        {"elevation_weight": 0.1, "elevation_exponent": 1.5, "label": "Direct (low avoidance)", "color": "yellow"},
        {"elevation_weight": 0.5, "elevation_exponent": 2.0, "label": "Balanced", "color": "red"},
        {"elevation_weight": 1.5, "elevation_exponent": 2.5, "label": "Avoid climbs", "color": "cyan"},
    ]
    
    for param in params:
        pathfinder = ElevationPathfinder(
            elev_array, pixel_size,
            elevation_weight=param["elevation_weight"],
            elevation_exponent=param["elevation_exponent"],
            distance_weight=0.05,
            max_slope_degrees=45
        )
        path = pathfinder.find_path(start_row, start_col, goal_row, goal_col)
        if path:
            stats = pathfinder.calculate_path_stats(path)
            paths.append({
                "path": path,
                "stats": stats,
                "label": param["label"],
                "color": param["color"]
            })
    
    # Create figure
    fig = plt.figure(figsize=(20, 12))
    
    # Main map with hillshade and paths
    ax1 = plt.subplot(2, 3, (1, 4))
    
    # Display hillshade
    ax1.imshow(hillshade, cmap='gray', origin='upper',
               extent=[bounds.west, bounds.east, bounds.south, bounds.north],
               alpha=1.0)
    
    # Overlay slope with transparency
    slope_colors = [(0.0, 0.5, 0.0), (0.5, 0.9, 0.0), (1.0, 1.0, 0.0),
                   (1.0, 0.7, 0.0), (1.0, 0.3, 0.0), (0.8, 0.0, 0.0)]
    slope_cmap = LinearSegmentedColormap.from_list('slope', slope_colors)
    
    im = ax1.imshow(slope_degrees, cmap=slope_cmap, origin='upper',
                    extent=[bounds.west, bounds.east, bounds.south, bounds.north],
                    alpha=0.4, vmin=0, vmax=45)
    
    # Plot paths
    for path_data in paths:
        path = path_data["path"]
        path_array = np.array(path)
        rows = path_array[:, 0]
        cols = path_array[:, 1]
        lons = bounds.west + (cols / elev_array.shape[1]) * (bounds.east - bounds.west)
        lats = bounds.north - (rows / elev_array.shape[0]) * (bounds.north - bounds.south)
        
        ax1.plot(lons, lats, color=path_data["color"], linewidth=3,
                label=f'{path_data["label"]} (dist: {path_data["stats"]["distance_ratio"]:.2f}x)',
                alpha=0.8)
    
    # Mark start and goal
    ax1.plot(start_lon, start_lat, 'go', markersize=12, label='Start')
    ax1.plot(goal_lon, goal_lat, 'ro', markersize=12, label='Goal')
    
    ax1.set_title('Terrain Analysis with Multiple Path Options', fontsize=16)
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Slope map
    ax2 = plt.subplot(2, 3, 2)
    im2 = ax2.imshow(slope_degrees, cmap=slope_cmap, origin='upper',
                     extent=[bounds.west, bounds.east, bounds.south, bounds.north],
                     vmin=0, vmax=45)
    ax2.set_title('Slope Map (degrees)', fontsize=14)
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    plt.colorbar(im2, ax=ax2, label='Slope (°)')
    
    # Elevation map
    ax3 = plt.subplot(2, 3, 3)
    im3 = ax3.imshow(elev_array, cmap='terrain', origin='upper',
                     extent=[bounds.west, bounds.east, bounds.south, bounds.north])
    ax3.set_title('Elevation Map', fontsize=14)
    ax3.set_xlabel('Longitude')
    ax3.set_ylabel('Latitude')
    plt.colorbar(im3, ax=ax3, label='Elevation (m)')
    
    # Elevation profiles
    ax4 = plt.subplot(2, 3, (5, 6))
    
    for path_data in paths:
        path = path_data["path"]
        elevations = []
        distances = [0]
        
        for i, (row, col) in enumerate(path):
            elevations.append(elev_array[row, col])
            if i > 0:
                dr = row - path[i-1][0]
                dc = col - path[i-1][1]
                dist = np.sqrt(dr*dr + dc*dc) * pixel_size
                distances.append(distances[-1] + dist)
        
        ax4.plot(distances, elevations, color=path_data["color"], 
                linewidth=2, label=path_data["label"], alpha=0.8)
    
    ax4.set_xlabel('Distance along path (m)')
    ax4.set_ylabel('Elevation (m)')
    ax4.set_title('Elevation Profiles Comparison', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add path statistics text
    stats_text = "Path Statistics:\n"
    for path_data in paths:
        stats = path_data["stats"]
        stats_text += f"\n{path_data['label']}:\n"
        stats_text += f"  Distance: {stats['total_distance']:.0f}m ({stats['distance_ratio']:.2f}x direct)\n"
        stats_text += f"  Ascent: {stats['total_ascent']:.0f}m, Descent: {stats['total_descent']:.0f}m\n"
    
    ax4.text(0.02, 0.98, stats_text, transform=ax4.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round',
             facecolor='white', alpha=0.8), fontsize=10)
    
    plt.suptitle(f'Path Analysis: ({start_lat:.4f}, {start_lon:.4f}) to ({goal_lat:.4f}, {goal_lon:.4f})',
                 fontsize=18)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved terrain visualization to: {output_file}")
    plt.close()

if __name__ == "__main__":
    # Use the requested coordinates
    visualize_path_with_terrain(40.6599, -111.5662, 40.6483, -111.5648)