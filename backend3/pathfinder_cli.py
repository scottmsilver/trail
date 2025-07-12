#!/usr/bin/env python3
"""
Command-line interface for elevation pathfinding with customizable parameters.
"""

import argparse
import sys
import os
from datetime import datetime, timezone
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from elevation import TwoLayerElevationLibrary, Bounds
from elevation_pathfinder_sustained import SustainedSlopePathfinder, visualize_path_with_fatigue
from path_layer import PathLayer, PathType
from elevation_pathfinder_terrain import TerrainAwarePathfinder
from elevation_fd_safe import FDManagedElevationLibrary


def create_gpx(lats, lons, elevations, route_name, description, filename):
    """Create a GPX file from path coordinates."""
    gpx_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="Elevation Pathfinder"
     xmlns="http://www.topografix.com/GPX/1/1"
     xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
     xsi:schemaLocation="http://www.topografix.com/GPX/1/1 http://www.topografix.com/GPX/1/1/gpx.xsd">
  <metadata>
    <name>{route_name}</name>
    <desc>{description}</desc>
    <time>{datetime.now(timezone.utc).isoformat()}Z</time>
  </metadata>
  <trk>
    <name>{route_name}</name>
    <desc>{description}</desc>
    <trkseg>
"""
    
    for lat, lon, ele in zip(lats, lons, elevations):
        gpx_content += f'      <trkpt lat="{lat:.6f}" lon="{lon:.6f}">\n'
        gpx_content += f'        <ele>{ele:.1f}</ele>\n'
        gpx_content += f'      </trkpt>\n'
    
    gpx_content += """    </trkseg>
  </trk>
</gpx>"""
    
    with open(filename, 'w') as f:
        f.write(gpx_content)
    
    return filename


def main():
    parser = argparse.ArgumentParser(
        description='Find paths with customizable elevation and distance penalties',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Heavily penalize slopes, ignore distance (great for finding easiest path):
  python pathfinder_cli.py --start 40.6599,-111.5662 --goal 40.6483,-111.5648 \\
    --elevation-weight 3.0 --elevation-exponent 3.0 --distance-weight 0.001

  # Standard balanced path:
  python pathfinder_cli.py --start 40.6599,-111.5662 --goal 40.6483,-111.5648 \\
    --elevation-weight 0.5 --elevation-exponent 2.0 --distance-weight 0.1

  # With sustained slope fatigue:
  python pathfinder_cli.py --start 40.6599,-111.5662 --goal 40.6483,-111.5648 \\
    --elevation-weight 2.0 --distance-weight 0.01 --sustained-weight 2.0 \\
    --fatigue-distance 50 --steep-threshold 12

  # Terrain-aware pathfinding (prefers trails, uses OSM data):
  python pathfinder_cli.py --start 40.6599,-111.5662 --goal 40.6483,-111.5648 \\
    --use-terrain --prefer-trails 0.1 --avoid-roads
        """
    )
    
    # Required arguments
    parser.add_argument('--start', type=str, required=True,
                       help='Start coordinates as lat,lon')
    parser.add_argument('--goal', type=str, required=True,
                       help='Goal coordinates as lat,lon')
    
    # Penalty weights
    parser.add_argument('--climb-penalty', '--elevation-weight', 
                       dest='elevation_weight', type=float, default=1.0,
                       help='Penalty for elevation gain (default: 1.0, higher=avoid climbing more, try 2-5 for strong avoidance)')
    parser.add_argument('--elevation-exponent', type=float, default=2.0,
                       help='Exponent for elevation penalty (default: 2.0, try 2.5-3.0 for stronger penalty on big changes)')
    parser.add_argument('--distance-penalty', '--distance-weight',
                       dest='distance_weight', type=float, default=0.1,
                       help='Penalty for longer paths (default: 0.1, lower=allow scenic detours, try 0.001-0.01 for long detours)')
    
    # Sustained slope parameters
    parser.add_argument('--fatigue-penalty-multiplier', '--sustained-weight',
                       dest='sustained_weight', type=float, default=0.0,
                       help='Penalty multiplier for sustained steep climbing (default: 0.0=disabled, try 1-3 to break up long climbs)')
    parser.add_argument('--steep-threshold', type=float, default=15.0,
                       help='Slope threshold in degrees for fatigue (default: 15.0)')
    parser.add_argument('--rest-threshold', type=float, default=8.0,
                       help='Slope threshold in degrees for rest (default: 8.0)')
    parser.add_argument('--fatigue-distance', type=float, default=100.0,
                       help='Distance before fatigue kicks in (default: 100.0m)')
    parser.add_argument('--fatigue-exponent', type=float, default=2.0,
                       help='How quickly fatigue accumulates (default: 2.0)')
    
    # Terrain awareness parameters
    parser.add_argument('--use-terrain', action='store_true',
                       help='Enable terrain-aware pathfinding (uses OSM path data)')
    parser.add_argument('--trail-cost-factor', '--prefer-trails',
                       dest='prefer_trails', type=float, default=0.3,
                       help='Cost multiplier for trails (default: 0.3=trails cost 30% of normal, 0.1=strong preference, 2.0=avoid trails)')
    parser.add_argument('--avoid-roads', action='store_true',
                       help='Increase cost for roads/streets when terrain-aware')
    parser.add_argument('--terrain-cost-scale', '--terrain-weight',
                       dest='terrain_weight', type=float, default=1.0,
                       help='Scales all terrain type penalties (default: 1.0, lower=terrain matters less)')
    
    # Other parameters
    parser.add_argument('--max-slope', type=float, default=45.0,
                       help='Maximum traversable slope in degrees (default: 45.0)')
    parser.add_argument('--resolution', type=int, default=10,
                       help='DEM resolution in meters (default: 10)')
    parser.add_argument('--margin', type=float, default=0.01,
                       help='Search area margin in degrees (default: 0.01)')
    parser.add_argument('--data-dir', type=str, default='./elevation_data',
                       help='Directory for elevation data')
    parser.add_argument('--output', type=str, default='pathfinder_result.png',
                       help='Output filename for visualization')
    parser.add_argument('--gpx', type=str, default=None,
                       help='Output GPX filename (if not specified, auto-generated from parameters)')
    parser.add_argument('--skip-viz', action='store_true',
                       help='Skip visualization (only generate GPX and coordinate files)')
    parser.add_argument('--skip-gpx', action='store_true',
                       help='Skip GPX file generation')
    parser.add_argument('--viz-only', action='store_true',
                       help='Only generate visualization (skip GPX and coordinate files)')
    
    args = parser.parse_args()
    
    # Validate conflicting flags
    if args.skip_viz and args.viz_only:
        parser.error("Cannot use --skip-viz and --viz-only together")
    
    # Check if old parameter names were used and suggest new ones
    old_to_new_params = {
        '--prefer-trails': '--trail-cost-factor',
        '--elevation-weight': '--climb-penalty',
        '--distance-weight': '--distance-penalty',
        '--sustained-weight': '--fatigue-penalty-multiplier',
        '--terrain-weight': '--terrain-cost-scale'
    }
    
    # Build suggested command with new parameter names
    import sys
    original_args = sys.argv[1:]
    suggested_args = []
    old_params_used = []
    
    i = 0
    while i < len(original_args):
        arg = original_args[i]
        if arg in old_to_new_params:
            old_params_used.append(arg)
            suggested_args.append(old_to_new_params[arg])
        else:
            suggested_args.append(arg)
        i += 1
    
    # Print suggestion if old parameters were used
    if old_params_used:
        print("\nNOTE: You're using deprecated parameter names. Consider using:")
        print(f"python {sys.argv[0]} {' '.join(suggested_args)}\n")
    
    # Validate parameter values
    if args.prefer_trails > 2.0:
        print(f"WARNING: trail-cost-factor={args.prefer_trails} is very high. "
              f"This will strongly avoid trails. Use 0.1-0.5 to prefer trails.")
    
    if args.distance_weight < 0:
        parser.error("distance-penalty cannot be negative")
    
    if args.elevation_weight < 0:
        parser.error("climb-penalty cannot be negative")
    
    if args.sustained_weight < 0:
        parser.error("fatigue-penalty-multiplier cannot be negative")
    
    if args.terrain_weight < 0:
        parser.error("terrain-cost-scale cannot be negative")
    
    # Parse coordinates
    try:
        start_lat, start_lon = map(float, args.start.split(','))
        goal_lat, goal_lon = map(float, args.goal.split(','))
    except:
        parser.error("Coordinates must be in format: lat,lon")
    
    # Setup bounds
    min_lat = min(start_lat, goal_lat) - args.margin
    max_lat = max(start_lat, goal_lat) + args.margin
    min_lon = min(start_lon, goal_lon) - args.margin
    max_lon = max(start_lon, goal_lon) + args.margin
    bounds = Bounds(north=max_lat, south=min_lat, east=max_lon, west=min_lon)
    
    # Load elevation data
    print(f"Loading elevation data...")
    print(f"Search area: {bounds.south:.4f} to {bounds.north:.4f} lat, "
          f"{bounds.west:.4f} to {bounds.east:.4f} lon")
    
    # Use FD-safe wrapper for large margins to avoid file descriptor limits
    base_elev_lib = TwoLayerElevationLibrary(args.data_dir, args.resolution)
    
    # Check if we need FD management (large search areas)
    area_size = (bounds.north - bounds.south) * (bounds.east - bounds.west)
    use_fd_safe = area_size > 0.01  # More than 0.01 square degrees
    
    if use_fd_safe:
        print(f"Using FD-safe elevation library for large area ({area_size:.4f} sq degrees)")
        elev_lib = FDManagedElevationLibrary(base_elev_lib, max_open_files=20)  # Reduced from 30
    else:
        elev_lib = base_elev_lib
    
    result = elev_lib.load_area(bounds) if hasattr(elev_lib, 'load_area') else base_elev_lib.load_area(bounds)
    
    if result['status'] != 'success':
        print(f"Failed to load elevation data: {result.get('message', 'Unknown error')}")
        return 1
    
    # Get elevation array
    if use_fd_safe:
        elev_array, metadata = elev_lib.get_elevation_array_safe(bounds)
    else:
        elev_array, metadata = elev_lib.get_elevation_array(bounds)
    
    # Calculate pixel size
    lat_center = (bounds.north + bounds.south) / 2
    lon_deg_to_m = 111000 * np.cos(np.radians(lat_center))
    pixel_size = abs(metadata['transform']['a']) * lon_deg_to_m
    
    # Convert coordinates
    def latlon_to_pixel(lat, lon):
        col = int((lon - bounds.west) / (bounds.east - bounds.west) * elev_array.shape[1])
        row = int((bounds.north - lat) / (bounds.north - bounds.south) * elev_array.shape[0])
        return min(max(row, 0), elev_array.shape[0]-1), min(max(col, 0), elev_array.shape[1]-1)
    
    start_row, start_col = latlon_to_pixel(start_lat, start_lon)
    goal_row, goal_col = latlon_to_pixel(goal_lat, goal_lon)
    
    print(f"\nStart: ({start_lat:.4f}, {start_lon:.4f}) -> elevation {elev_array[start_row, start_col]:.1f}m")
    print(f"Goal: ({goal_lat:.4f}, {goal_lon:.4f}) -> elevation {elev_array[goal_row, goal_col]:.1f}m")
    print(f"Elevation difference: {elev_array[goal_row, goal_col] - elev_array[start_row, start_col]:.1f}m")
    
    # Print parameters
    print(f"\nPathfinding parameters:")
    print(f"  Elevation weight: {args.elevation_weight}")
    print(f"  Elevation exponent: {args.elevation_exponent}")
    print(f"  Distance weight: {args.distance_weight}")
    if args.sustained_weight > 0:
        print(f"  Sustained slope weight: {args.sustained_weight}")
        print(f"  Steep threshold: {args.steep_threshold}°")
        print(f"  Fatigue distance: {args.fatigue_distance}m")
    if args.use_terrain:
        print(f"  Terrain-aware: YES")
        print(f"  Trail preference: {args.prefer_trails}")
        print(f"  Avoid roads: {args.avoid_roads}")
        print(f"  Terrain weight: {args.terrain_weight}")
    print(f"  Max slope: {args.max_slope}°")
    
    # Create pathfinder based on terrain awareness
    if args.use_terrain:
        # Initialize path layer
        path_layer = PathLayer()
        
        # Check if we need to create path layer tile
        tile_x = int(bounds.west * 100)
        tile_y = int(bounds.south * 100)
        if path_layer.load_tile(tile_x, tile_y) is None:
            print("\nCreating path layer tile...")
            path_layer.create_tile(tile_x, tile_y, (bounds.west, bounds.south, bounds.east, bounds.north))
        
        # Create terrain-aware pathfinder
        # Pass base library to terrain pathfinder as it needs the original interface
        pathfinder = TerrainAwarePathfinder(base_elev_lib, path_layer, bounds, resolution=args.resolution)
        
        # Set parameters
        pathfinder.elevation_weight = args.elevation_weight
        pathfinder.elevation_exponent = args.elevation_exponent
        pathfinder.terrain_weight = args.terrain_weight
        pathfinder.terrain_costs[PathType.HIKING_PATH] = args.prefer_trails
        pathfinder.max_slope_degrees = args.max_slope
        
        if args.avoid_roads:
            pathfinder.terrain_costs[PathType.STREET] = 2.0
        
        # Set sustained slope parameters
        pathfinder.sustained_slope_weight = args.sustained_weight
        pathfinder.steep_threshold = args.steep_threshold
        pathfinder.fatigue_distance = args.fatigue_distance
        pathfinder.fatigue_exponent = args.fatigue_exponent
        
        # Find path using lat/lon
        print(f"\nSearching for terrain-aware path...")
        path_coords = pathfinder.find_path(start_lat, start_lon, goal_lat, goal_lon)
        
        # Convert to pixel path for consistency with visualization
        if path_coords:
            path = []
            for lat, lon, elev in path_coords:
                row, col = latlon_to_pixel(lat, lon)
                path.append((row, col))
        else:
            path = None
            
        # Clean up FD-managed elevation library if used
        if use_fd_safe:
            elev_lib.close_all()
            
    else:
        # Use original pathfinder
        pathfinder = SustainedSlopePathfinder(
            elev_array, pixel_size,
            elevation_weight=args.elevation_weight,
            elevation_exponent=args.elevation_exponent,
            distance_weight=args.distance_weight,
            sustained_slope_weight=args.sustained_weight,
            steep_threshold_degrees=args.steep_threshold,
            rest_threshold_degrees=args.rest_threshold,
            fatigue_distance=args.fatigue_distance,
            fatigue_exponent=args.fatigue_exponent,
            max_slope_degrees=args.max_slope
        )
        
        # Find path
        print(f"\nSearching for path...")
        path = pathfinder.find_path(start_row, start_col, goal_row, goal_col)
    
    # Clean up FD-managed elevation library if used
    if use_fd_safe:
        elev_lib.close_all()
    
    if path:
        # Calculate stats based on pathfinder type
        if args.use_terrain:
            # For terrain-aware pathfinder, calculate stats manually
            stats = {
                'total_distance': 0,
                'straight_distance': pixel_size * np.sqrt((goal_row - start_row)**2 + (goal_col - start_col)**2),
                'total_ascent': 0,
                'total_descent': 0,
                'max_slope': 0,
                'avg_slope': 0,
                'steep_segments': [],
                'num_steep_segments': 0,
                'longest_steep_segment': 0,
                'total_steep_distance': 0
            }
            
            slopes = []
            steep_distance = 0
            
            for i in range(1, len(path)):
                # Distance
                dr = path[i][0] - path[i-1][0]
                dc = path[i][1] - path[i-1][1]
                segment_dist = pixel_size * np.sqrt(dr**2 + dc**2)
                stats['total_distance'] += segment_dist
                
                # Elevation change
                elev1 = elev_array[path[i-1][0], path[i-1][1]]
                elev2 = elev_array[path[i][0], path[i][1]]
                elev_change = elev2 - elev1
                
                if elev_change > 0:
                    stats['total_ascent'] += elev_change
                else:
                    stats['total_descent'] += abs(elev_change)
                
                # Slope
                if segment_dist > 0:
                    slope_rad = np.arctan(abs(elev_change) / segment_dist)
                    slope_deg = np.degrees(slope_rad)
                    slopes.append(slope_deg)
                    stats['max_slope'] = max(stats['max_slope'], slope_deg)
                    
                    # Track steep segments
                    if slope_deg > args.steep_threshold:
                        steep_distance += segment_dist
                    elif steep_distance > 0:
                        stats['steep_segments'].append(steep_distance)
                        steep_distance = 0
            
            # Finish last steep segment
            if steep_distance > 0:
                stats['steep_segments'].append(steep_distance)
            
            # Calculate derived stats
            if slopes:
                stats['avg_slope'] = np.mean(slopes)
            stats['distance_ratio'] = stats['total_distance'] / stats['straight_distance'] if stats['straight_distance'] > 0 else 1
            
            if stats['steep_segments']:
                stats['num_steep_segments'] = len(stats['steep_segments'])
                stats['longest_steep_segment'] = max(stats['steep_segments'])
                stats['total_steep_distance'] = sum(stats['steep_segments'])
        else:
            # Use original stats calculation
            stats = pathfinder.calculate_path_stats_with_fatigue(path)
        
        print(f"\nPath found!")
        print(f"  Total distance: {stats['total_distance']:.1f}m")
        print(f"  Direct distance: {stats['straight_distance']:.1f}m")
        print(f"  Distance ratio: {stats['distance_ratio']:.2f}x")
        print(f"  Total ascent: {stats['total_ascent']:.1f}m")
        print(f"  Total descent: {stats['total_descent']:.1f}m")
        print(f"  Average slope: {stats.get('avg_slope', 0):.1f}°")
        print(f"  Maximum slope: {stats.get('max_slope', 0):.1f}°")
        
        if args.sustained_weight > 0 and stats.get('steep_segments'):
            print(f"\nSteep segment analysis:")
            print(f"  Number of segments: {stats['num_steep_segments']}")
            print(f"  Longest segment: {stats['longest_steep_segment']:.1f}m")
            print(f"  Total steep distance: {stats['total_steep_distance']:.1f}m ({100*stats['total_steep_distance']/stats['total_distance']:.1f}%)")
        
        # Prepare path data
        path_array = np.array(path)
        rows = path_array[:, 0]
        cols = path_array[:, 1]
        lons = bounds.west + (cols / elev_array.shape[1]) * (bounds.east - bounds.west)
        lats = bounds.north - (rows / elev_array.shape[0]) * (bounds.north - bounds.south)
        elevations = [elev_array[rows[i], cols[i]] for i in range(len(path))]
        
        # Visualize
        if not args.skip_viz and not args.viz_only:
            print(f"\nGenerating visualization...")
        if args.skip_viz:
            print(f"\nSkipping visualization (requested).")
        elif args.use_terrain:
            # For terrain-aware, use simple visualization
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend to save resources
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            
            # Show elevation
            im = ax.imshow(elev_array, cmap='terrain', origin='upper')
            
            # Plot path
            ax.plot(path_array[:, 1], path_array[:, 0], 'r-', linewidth=2, label='Path')
            ax.plot(path_array[0, 1], path_array[0, 0], 'go', markersize=10, label='Start')
            ax.plot(path_array[-1, 1], path_array[-1, 0], 'ro', markersize=10, label='End')
            
            ax.set_title(f'Terrain-Aware Path (Distance: {stats["total_distance"]:.0f}m, Ratio: {stats["distance_ratio"]:.2f}x)')
            ax.legend()
            plt.colorbar(im, ax=ax, label='Elevation (m)')
            
            plt.tight_layout()
            plt.savefig(args.output)
            plt.close()
            print(f"\nVisualization saved to: {args.output}")
        else:
            output = visualize_path_with_fatigue(elev_array, path, bounds, stats, 
                                               pathfinder, args.output)
            print(f"\nVisualization saved to: {output}")
        
        # Save coordinates file (unless viz-only mode)
        if not args.viz_only:
            coord_file = args.output.replace('.png', '_coords.txt')
            with open(coord_file, 'w') as f:
                f.write("# Latitude,Longitude,Elevation\n")
                for i, (lat, lon) in enumerate(zip(lats, lons)):
                    elev = elev_array[rows[i], cols[i]]
                    f.write(f"{lat:.6f},{lon:.6f},{elev:.1f}\n")
            print(f"Path coordinates saved to: {coord_file}")
        
        # Create GPX file (unless skip-gpx or viz-only mode)
        if not args.skip_gpx and not args.viz_only:
            # Generate GPX filename if not specified
            if args.gpx:
                gpx_filename = args.gpx
            else:
                # Create descriptive filename from parameters
                gpx_filename = f"route_elev{args.elevation_weight}_exp{args.elevation_exponent}_dist{args.distance_weight}"
                if args.sustained_weight > 0:
                    gpx_filename += f"_fatigue{args.sustained_weight}"
                gpx_filename += f"_{stats['distance_ratio']:.1f}x"
                gpx_filename = gpx_filename.replace('.', '_').replace('__', '_') + '.gpx'
            
            # Create route name and description
            route_name = f"Pathfinder Route {stats['distance_ratio']:.1f}x"
            description = (f"Distance: {stats['total_distance']:.0f}m, "
                          f"Ascent: {stats['total_ascent']:.0f}m, "
                          f"Avg slope: {stats.get('avg_slope', 0):.1f}°, "
                          f"Parameters: elev_weight={args.elevation_weight}, "
                          f"elev_exp={args.elevation_exponent}, "
                          f"dist_weight={args.distance_weight}")
            
            if args.sustained_weight > 0:
                description += f", sustained_weight={args.sustained_weight}"
            
            gpx_file = create_gpx(lats, lons, elevations, route_name, description, gpx_filename)
            print(f"GPX file saved to: {gpx_file}")
        
    else:
        print("\nNo path found!")
        print("Try:")
        print("  - Increasing max-slope")
        print("  - Reducing elevation weights")
        print("  - Increasing search margin")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())