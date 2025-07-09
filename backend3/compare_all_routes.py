#!/usr/bin/env python3
"""
Compare all different routes found with various parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from elevation import TwoLayerElevationLibrary, Bounds

def load_path_coords(filename):
    """Load path coordinates from file."""
    coords = []
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    coords.append((float(parts[0]), float(parts[1])))
    return coords

def create_route_comparison():
    """Create comparison of all routes."""
    
    # Define routes to compare
    routes = [
        {
            "file": "heavy_slope_penalty_coords.txt",
            "label": "Moderate Detour (3.2x)",
            "color": "yellow",
            "style": "-",
            "width": 3
        },
        {
            "file": "extreme_gentle_coords.txt", 
            "label": "Gentle with Breaks (2.6x)",
            "color": "cyan",
            "style": "-",
            "width": 3
        },
        {
            "file": "very_wide_search_coords.txt",
            "label": "Wide Search (3.7x)", 
            "color": "orange",
            "style": "--",
            "width": 3
        },
        {
            "file": "extreme_wide_search_coords.txt",
            "label": "Extreme Detour (5.3x)",
            "color": "red",
            "style": "-",
            "width": 4
        }
    ]
    
    # Load elevation data for background
    start_lat, start_lon = 40.6599, -111.5662
    goal_lat, goal_lon = 40.6483, -111.5648
    margin = 0.05
    
    bounds = Bounds(
        north=max(start_lat, goal_lat) + margin,
        south=min(start_lat, goal_lat) - margin,
        east=max(start_lon, goal_lon) + margin,
        west=min(start_lon, goal_lon) - margin
    )
    
    print("Loading elevation data...")
    elev_lib = TwoLayerElevationLibrary('./elevation_data', 10)
    result = elev_lib.load_area(bounds)
    
    if result['status'] != 'success':
        print("Failed to load elevation data")
        return
    
    elev_array, metadata = elev_lib.get_elevation_array(bounds)
    
    # Create figure
    fig = plt.figure(figsize=(20, 16))
    
    # Main map
    ax = plt.subplot(111)
    
    # Show elevation
    im = ax.imshow(elev_array, cmap='terrain', origin='upper',
                   extent=[bounds.west, bounds.east, bounds.south, bounds.north],
                   alpha=0.8)
    
    # Plot each route
    valid_routes = []
    for route in routes:
        coords = load_path_coords(route["file"])
        if coords:
            lats, lons = zip(*coords)
            ax.plot(lons, lats, color=route["color"], 
                   linewidth=route["width"], linestyle=route["style"],
                   label=route["label"], alpha=0.8)
            valid_routes.append(route["label"])
            print(f"Loaded {route['label']}: {len(coords)} points")
    
    # Direct path
    ax.plot([start_lon, goal_lon], [start_lat, goal_lat], 
           'k:', linewidth=3, alpha=0.5, label='Direct Path (1.0x)')
    
    # Mark start and goal
    ax.plot(start_lon, start_lat, 'go', markersize=15, 
           markeredgecolor='black', markeredgewidth=2, label='Start (2492m)')
    ax.plot(goal_lon, goal_lat, 'ro', markersize=15,
           markeredgecolor='black', markeredgewidth=2, label='Goal (2826m)')
    
    # Add distance circles
    from matplotlib.patches import Circle
    lat_center = (start_lat + goal_lat) / 2
    lon_deg_per_km = 1 / (111 * np.cos(np.radians(lat_center)))
    
    for radius_km in [1, 2, 3, 4, 5]:
        circle = Circle((start_lon, start_lat), 
                       radius_km * lon_deg_per_km,
                       fill=False, edgecolor='gray', 
                       linestyle='--', alpha=0.3)
        ax.add_patch(circle)
        # Label the 1km circle
        if radius_km == 1:
            ax.text(start_lon + radius_km * lon_deg_per_km, start_lat,
                   f'{radius_km}km', fontsize=8, alpha=0.5)
    
    ax.set_title('Route Comparison: Trading Distance for Gentler Slopes\n'
                 'Wider search areas allow discovery of natural valleys and ridges',
                 fontsize=18)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Elevation (m)', fontsize=12)
    
    # Add text box with insights
    textstr = "Route Characteristics:\n"
    textstr += "• Direct: ~1km, steep climbs\n"
    textstr += "• Moderate: 3.2km, avg 6.5° slope\n"
    textstr += "• Wide Search: 3.7km, avg 5.6° slope\n"
    textstr += "• Extreme: 5.1km, avg 4.5° slope\n\n"
    textstr += "The extreme route discovers a\n"
    textstr += "natural valley system to the east"
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig('all_routes_comparison.png', dpi=150, bbox_inches='tight')
    print("\nSaved comparison to: all_routes_comparison.png")

if __name__ == "__main__":
    create_route_comparison()