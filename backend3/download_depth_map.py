#!/usr/bin/env python3
"""
Command-line tool to download elevation data and create depth maps for any coordinates.

Usage:
    python download_depth_map.py --lat1 40.6588 --lon1 -111.5780 --lat2 40.6448 --lon2 -111.5595 --resolution 10
    python download_depth_map.py --bounds 40.6448,40.6588,-111.5780,-111.5595 --resolution 3
"""

import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from visualize_elevation import create_depth_map


def main():
    parser = argparse.ArgumentParser(
        description='Download elevation data and create depth maps',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using individual coordinates:
  python download_depth_map.py --lat1 40.6588 --lon1 -111.5780 --lat2 40.6448 --lon2 -111.5595 --resolution 10
  
  # Using bounds (south,north,west,east):
  python download_depth_map.py --bounds 40.6448,40.6588,-111.5780,-111.5595 --resolution 3
  
  # Custom output name:
  python download_depth_map.py --bounds 40.6448,40.6588,-111.5780,-111.5595 --resolution 3 --output my_area.png
        """
    )
    
    # Coordinate options
    coords = parser.add_mutually_exclusive_group(required=True)
    coords.add_argument('--bounds', type=str, 
                       help='Bounds as: south,north,west,east (e.g., 40.6448,40.6588,-111.5780,-111.5595)')
    
    coord_group = coords.add_argument_group('individual coordinates')
    parser.add_argument('--lat1', type=float, help='Start latitude')
    parser.add_argument('--lon1', type=float, help='Start longitude')
    parser.add_argument('--lat2', type=float, help='End latitude')
    parser.add_argument('--lon2', type=float, help='End longitude')
    
    # Other options
    parser.add_argument('--resolution', type=int, choices=[1, 3, 5, 10, 30, 60], 
                       default=10, help='DEM resolution in meters (default: 10)')
    parser.add_argument('--output', type=str, help='Output filename (default: depth_map_<resolution>m.png)')
    parser.add_argument('--data-dir', type=str, default='./elevation_data',
                       help='Directory for elevation data (default: ./elevation_data)')
    
    args = parser.parse_args()
    
    # Parse coordinates
    if args.bounds:
        parts = args.bounds.split(',')
        if len(parts) != 4:
            parser.error("Bounds must be: south,north,west,east")
        try:
            south, north, west, east = map(float, parts)
            # Use corners as start/end points
            start_lat, start_lon = north, west
            end_lat, end_lon = south, east
        except ValueError:
            parser.error("Invalid bounds format")
    else:
        # Check individual coordinates
        if not all([args.lat1 is not None, args.lon1 is not None, 
                   args.lat2 is not None, args.lon2 is not None]):
            parser.error("Must provide either --bounds or all of --lat1, --lon1, --lat2, --lon2")
        start_lat, start_lon = args.lat1, args.lon1
        end_lat, end_lon = args.lat2, args.lon2
    
    # Determine output filename
    if args.output:
        output_file = args.output
    else:
        output_file = f"depth_map_{args.resolution}m.png"
    
    # Create the depth map
    print(f"Downloading elevation data and creating depth map...")
    print(f"Start: {start_lat:.4f}, {start_lon:.4f}")
    print(f"End: {end_lat:.4f}, {end_lon:.4f}")
    print(f"Resolution: {args.resolution}m")
    print(f"Output: {output_file}")
    print(f"Data directory: {args.data_dir}")
    
    try:
        create_depth_map(
            data_dir=args.data_dir,
            start_lat=start_lat,
            start_lon=start_lon,
            end_lat=end_lat,
            end_lon=end_lon,
            resolution=args.resolution,
            output_file=output_file
        )
        print(f"\nSuccess! Created {output_file} and {output_file.replace('.png', '_grayscale.png')}")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()