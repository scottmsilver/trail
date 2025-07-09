#!/usr/bin/env python3
"""
Command-line interface for elevation pathfinding with customizable parameters.
"""

import argparse
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from elevation import TwoLayerElevationLibrary, Bounds
from elevation_pathfinder_sustained import SustainedSlopePathfinder, visualize_path_with_fatigue


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
    <time>{datetime.utcnow().isoformat()}Z</time>
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
        """
    )
    
    # Required arguments
    parser.add_argument('--start', type=str, required=True,
                       help='Start coordinates as lat,lon')
    parser.add_argument('--goal', type=str, required=True,
                       help='Goal coordinates as lat,lon')
    
    # Penalty weights
    parser.add_argument('--elevation-weight', type=float, default=1.0,
                       help='Weight for elevation change penalty (default: 1.0, try 2-5 for strong avoidance)')
    parser.add_argument('--elevation-exponent', type=float, default=2.0,
                       help='Exponent for elevation penalty (default: 2.0, try 2.5-3.0 for stronger penalty on big changes)')
    parser.add_argument('--distance-weight', type=float, default=0.1,
                       help='Weight for path deviation penalty (default: 0.1, try 0.001-0.01 to allow long detours)')
    
    # Sustained slope parameters
    parser.add_argument('--sustained-weight', type=float, default=0.0,
                       help='Weight for sustained slope penalty (default: 0.0, try 1-3 to break up long climbs)')
    parser.add_argument('--steep-threshold', type=float, default=15.0,
                       help='Slope threshold in degrees for fatigue (default: 15.0)')
    parser.add_argument('--rest-threshold', type=float, default=8.0,
                       help='Slope threshold in degrees for rest (default: 8.0)')
    parser.add_argument('--fatigue-distance', type=float, default=100.0,
                       help='Distance before fatigue kicks in (default: 100.0m)')
    parser.add_argument('--fatigue-exponent', type=float, default=2.0,
                       help='How quickly fatigue accumulates (default: 2.0)')
    
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
                       help='Output filename')
    parser.add_argument('--gpx', type=str, default=None,
                       help='Output GPX filename (if not specified, auto-generated from parameters)')
    
    args = parser.parse_args()
    
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
    
    elev_lib = TwoLayerElevationLibrary(args.data_dir, args.resolution)
    result = elev_lib.load_area(bounds)
    
    if result['status'] != 'success':
        print(f"Failed to load elevation data: {result.get('message', 'Unknown error')}")
        return 1
    
    # Get elevation array
    elev_array, metadata = elev_lib.get_elevation_array(bounds)
    
    # Calculate pixel size
    import numpy as np
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
    print(f"  Max slope: {args.max_slope}°")
    
    # Create pathfinder
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
    
    if path:
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
        
        # Visualize
        output = visualize_path_with_fatigue(elev_array, path, bounds, stats, 
                                           pathfinder, args.output)
        print(f"\nVisualization saved to: {output}")
        
        # Also save a simple path file for further analysis
        import matplotlib.pyplot as plt
        path_array = np.array(path)
        rows = path_array[:, 0]
        cols = path_array[:, 1]
        lons = bounds.west + (cols / elev_array.shape[1]) * (bounds.east - bounds.west)
        lats = bounds.north - (rows / elev_array.shape[0]) * (bounds.north - bounds.south)
        
        # Save coordinates
        coord_file = args.output.replace('.png', '_coords.txt')
        with open(coord_file, 'w') as f:
            f.write("# Latitude,Longitude,Elevation\n")
            for i, (lat, lon) in enumerate(zip(lats, lons)):
                elev = elev_array[rows[i], cols[i]]
                f.write(f"{lat:.6f},{lon:.6f},{elev:.1f}\n")
        print(f"Path coordinates saved to: {coord_file}")
        
        # Create GPX file
        elevations = [elev_array[rows[i], cols[i]] for i in range(len(path))]
        
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