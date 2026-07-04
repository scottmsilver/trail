#!/usr/bin/env python3
"""
Find alternative routes by exploring a wider area and allowing much longer paths.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from elevation import TwoLayerElevationLibrary, Bounds
from elevation_pathfinder import ElevationPathfinder, visualize_path

def find_alternative_routes(start_lat, start_lon, goal_lat, goal_lon,
                          resolution=10, margin=0.02):  # Much larger margin
    """Find multiple alternative routes with different strategies."""
    
    # Setup bounds with large margin
    min_lat = min(start_lat, goal_lat) - margin
    max_lat = max(start_lat, goal_lat) + margin
    min_lon = min(start_lon, goal_lon) - margin
    max_lon = max(start_lon, goal_lon) + margin
    bounds = Bounds(north=max_lat, south=min_lat, east=max_lon, west=min_lon)
    
    print(f"Search area: {bounds.south:.4f} to {bounds.north:.4f} lat, "
          f"{bounds.west:.4f} to {bounds.east:.4f} lon")
    print(f"Area size: ~{(bounds.north-bounds.south)*111:.1f} x "
          f"{(bounds.east-bounds.west)*111*np.cos(np.radians((bounds.north+bounds.south)/2)):.1f} km")
    
    # Load elevation data
    print("\nLoading elevation data for expanded search area...")
    elev_lib = TwoLayerElevationLibrary('./elevation_data', resolution)
    result = elev_lib.load_area(bounds)
    
    if result['status'] != 'success':
        print(f"Failed to load elevation data")
        return
    
    # Get elevation array
    elev_array, metadata = elev_lib.get_elevation_array(bounds)
    print(f"Loaded elevation array: {elev_array.shape}")
    
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
    
    print(f"\nStart pixel: ({start_row}, {start_col}), elevation: {elev_array[start_row, start_col]:.1f}m")
    print(f"Goal pixel: ({goal_row}, {goal_col}), elevation: {elev_array[goal_row, goal_col]:.1f}m")
    
    # Define different routing strategies
    strategies = [
        {
            "name": "Direct Route",
            "elevation_weight": 0.1,
            "elevation_exponent": 1.5,
            "distance_weight": 0.5,  # High penalty for deviation
            "max_slope": 45,
            "color": "yellow",
            "style": "-"
        },
        {
            "name": "Gentle Climb",
            "elevation_weight": 2.0,
            "elevation_exponent": 3.0,
            "distance_weight": 0.01,  # Very low penalty - allow long detours
            "max_slope": 20,  # Gentler max slope
            "color": "cyan",
            "style": "-"
        },
        {
            "name": "Valley Route",
            "elevation_weight": 1.5,
            "elevation_exponent": 2.0,
            "distance_weight": 0.02,
            "max_slope": 30,
            "color": "blue",
            "style": "--"
        },
        {
            "name": "Ridge Route",
            "elevation_weight": 0.5,
            "elevation_exponent": 2.0,
            "distance_weight": 0.03,
            "max_slope": 35,
            "color": "red",
            "style": "-."
        }
    ]
    
    # Find paths with different strategies
    paths = []
    for strategy in strategies:
        print(f"\nFinding {strategy['name']}...")
        pathfinder = ElevationPathfinder(
            elev_array, pixel_size,
            elevation_weight=strategy["elevation_weight"],
            elevation_exponent=strategy["elevation_exponent"],
            distance_weight=strategy["distance_weight"],
            max_slope_degrees=strategy["max_slope"]
        )
        
        path = pathfinder.find_path(start_row, start_col, goal_row, goal_col)
        if path:
            stats = pathfinder.calculate_path_stats(path)
            stats['pixel_size'] = pixel_size
            
            print(f"  Found path: {stats['total_distance']:.0f}m "
                  f"({stats['distance_ratio']:.2f}x direct), "
                  f"ascent: {stats['total_ascent']:.0f}m")
            
            paths.append({
                "name": strategy["name"],
                "path": path,
                "stats": stats,
                "color": strategy["color"],
                "style": strategy["style"]
            })
        else:
            print(f"  No path found!")
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 16))
    
    # Main map with all routes
    ax1 = plt.subplot(2, 2, 1)
    
    # Calculate and show slope
    dy, dx = np.gradient(elev_array, pixel_size)
    slope_degrees = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
    
    # Create hillshade
    azimuth = 315
    altitude = 45
    azimuth_rad = np.radians(360 - azimuth + 90)
    altitude_rad = np.radians(altitude)
    
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect_rad = np.arctan2(-dx, dy)
    
    hillshade = np.sin(altitude_rad) * np.sin(slope_rad) + \
                np.cos(altitude_rad) * np.cos(slope_rad) * \
                np.cos(azimuth_rad - aspect_rad)
    hillshade = (hillshade + 1) / 2
    
    # Display terrain
    ax1.imshow(hillshade, cmap='gray', origin='upper',
               extent=[bounds.west, bounds.east, bounds.south, bounds.north],
               alpha=1.0)
    
    # Overlay elevation with transparency
    im1 = ax1.imshow(elev_array, cmap='terrain', origin='upper',
                     extent=[bounds.west, bounds.east, bounds.south, bounds.north],
                     alpha=0.5)
    
    # Plot all paths
    for path_data in paths:
        path = path_data["path"]
        path_array = np.array(path)
        rows = path_array[:, 0]
        cols = path_array[:, 1]
        lons = bounds.west + (cols / elev_array.shape[1]) * (bounds.east - bounds.west)
        lats = bounds.north - (rows / elev_array.shape[0]) * (bounds.north - bounds.south)
        
        ax1.plot(lons, lats, color=path_data["color"], 
                linewidth=3, linestyle=path_data["style"],
                label=f'{path_data["name"]} ({path_data["stats"]["distance_ratio"]:.1f}x)',
                alpha=0.8)
    
    # Mark start and goal
    ax1.plot(start_lon, start_lat, 'go', markersize=15, 
             markeredgecolor='black', markeredgewidth=2, label='Start')
    ax1.plot(goal_lon, goal_lat, 'ro', markersize=15,
             markeredgecolor='black', markeredgewidth=2, label='Goal')
    
    # Direct line
    ax1.plot([start_lon, goal_lon], [start_lat, goal_lat], 
             'k:', linewidth=2, alpha=0.5, label='Direct line')
    
    ax1.set_title('Alternative Routes Analysis', fontsize=16)
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Slope map
    ax2 = plt.subplot(2, 2, 2)
    slope_colors = [(0.0, 0.5, 0.0), (0.5, 0.9, 0.0), (1.0, 1.0, 0.0),
                   (1.0, 0.7, 0.0), (1.0, 0.3, 0.0), (0.8, 0.0, 0.0)]
    slope_cmap = LinearSegmentedColormap.from_list('slope', slope_colors)
    
    im2 = ax2.imshow(slope_degrees, cmap=slope_cmap, origin='upper',
                     extent=[bounds.west, bounds.east, bounds.south, bounds.north],
                     vmin=0, vmax=45)
    
    # Plot paths on slope map
    for path_data in paths:
        path = path_data["path"]
        path_array = np.array(path)
        rows = path_array[:, 0]
        cols = path_array[:, 1]
        lons = bounds.west + (cols / elev_array.shape[1]) * (bounds.east - bounds.west)
        lats = bounds.north - (rows / elev_array.shape[0]) * (bounds.north - bounds.south)
        
        ax2.plot(lons, lats, color=path_data["color"], 
                linewidth=2, linestyle=path_data["style"], alpha=0.8)
    
    ax2.plot(start_lon, start_lat, 'go', markersize=10)
    ax2.plot(goal_lon, goal_lat, 'ro', markersize=10)
    
    ax2.set_title('Routes on Slope Map', fontsize=16)
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Slope (degrees)')
    
    # Elevation profiles
    ax3 = plt.subplot(2, 2, 3)
    
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
        
        ax3.plot(distances, elevations, color=path_data["color"],
                linewidth=2, linestyle=path_data["style"],
                label=path_data["name"], alpha=0.8)
    
    ax3.set_xlabel('Distance along path (m)')
    ax3.set_ylabel('Elevation (m)')
    ax3.set_title('Elevation Profiles', fontsize=16)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Statistics table
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    
    # Create statistics table
    headers = ['Route', 'Distance (m)', 'Ratio', 'Ascent (m)', 'Descent (m)', 'Max Slope']
    data = []
    
    for path_data in paths:
        stats = path_data["stats"]
        path = path_data["path"]
        
        # Calculate max slope along path
        max_slope = 0
        for i in range(1, len(path)):
            r1, c1 = path[i-1]
            r2, c2 = path[i]
            dr = r2 - r1
            dc = c2 - c1
            dist = np.sqrt(dr*dr + dc*dc) * pixel_size
            elev_change = abs(elev_array[r2, c2] - elev_array[r1, c1])
            if dist > 0:
                slope = np.degrees(np.arctan(elev_change / dist))
                max_slope = max(max_slope, slope)
        
        data.append([
            path_data["name"],
            f"{stats['total_distance']:.0f}",
            f"{stats['distance_ratio']:.2f}x",
            f"{stats['total_ascent']:.0f}",
            f"{stats['total_descent']:.0f}",
            f"{max_slope:.1f}°"
        ])
    
    # Add direct line stats
    direct_distance = stats['straight_distance']
    direct_elev_change = elev_array[goal_row, goal_col] - elev_array[start_row, start_col]
    data.append([
        "Direct Line",
        f"{direct_distance:.0f}",
        "1.00x",
        f"{max(0, direct_elev_change):.0f}",
        f"{max(0, -direct_elev_change):.0f}",
        "N/A"
    ])
    
    table = ax4.table(cellText=data, colLabels=headers,
                     cellLoc='center', loc='center',
                     colWidths=[0.25, 0.15, 0.1, 0.15, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax4.set_title('Route Statistics Comparison', fontsize=16, pad=20)
    
    plt.suptitle(f'Alternative Routes Analysis\n'
                 f'Start: ({start_lat:.4f}, {start_lon:.4f}) → '
                 f'Goal: ({goal_lat:.4f}, {goal_lon:.4f})',
                 fontsize=18)
    
    plt.tight_layout()
    plt.savefig('alternative_routes.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: alternative_routes.png")
    plt.close()
    
    # Also create individual detailed views for interesting routes
    if len(paths) > 2:
        # Create detailed comparison of most different routes
        longest_path = max(paths, key=lambda p: p['stats']['distance_ratio'])
        shortest_path = min(paths, key=lambda p: p['stats']['distance_ratio'])
        
        if longest_path != shortest_path:
            print(f"\nCreating detailed comparison of {shortest_path['name']} vs {longest_path['name']}...")
            
            fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            for ax, path_data, title in [(ax1, shortest_path, "Most Direct Route"),
                                         (ax2, longest_path, "Gentlest Route")]:
                ax.imshow(hillshade, cmap='gray', origin='upper',
                         extent=[bounds.west, bounds.east, bounds.south, bounds.north])
                ax.imshow(elev_array, cmap='terrain', origin='upper',
                         extent=[bounds.west, bounds.east, bounds.south, bounds.north],
                         alpha=0.5)
                
                path = path_data["path"]
                path_array = np.array(path)
                rows = path_array[:, 0]
                cols = path_array[:, 1]
                lons = bounds.west + (cols / elev_array.shape[1]) * (bounds.east - bounds.west)
                lats = bounds.north - (rows / elev_array.shape[0]) * (bounds.north - bounds.south)
                
                ax.plot(lons, lats, 'r-', linewidth=4, alpha=0.8)
                ax.plot(start_lon, start_lat, 'go', markersize=12)
                ax.plot(goal_lon, goal_lat, 'ro', markersize=12)
                
                ax.set_title(f"{title}: {path_data['name']}\n"
                           f"Distance: {path_data['stats']['total_distance']:.0f}m "
                           f"({path_data['stats']['distance_ratio']:.1f}x)")
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('route_comparison.png', dpi=150, bbox_inches='tight')
            print("Detailed comparison saved to: route_comparison.png")


if __name__ == "__main__":
    # Use the requested coordinates with large search area
    find_alternative_routes(40.6599, -111.5662, 40.6483, -111.5648, margin=0.02)