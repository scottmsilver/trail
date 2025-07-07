#!/usr/bin/env python3
"""
Test different elevation resolutions
"""

from elevation import ElevationLibrary, Bounds

def test_resolution(data_dir: str, resolution: int, test_bounds: Bounds):
    """Test loading data at a specific resolution"""
    print(f"\n{'='*60}")
    print(f"Testing {resolution}m resolution")
    print(f"{'='*60}")
    
    # Create library for this resolution
    lib = ElevationLibrary(data_dir=data_dir, resolution=resolution)
    
    print(f"Tile size: {lib.tile_size}° (~{lib.tile_size * 111:.1f}km)")
    
    # Try to load
    print(f"\nLoading area...")
    try:
        result = lib.load_area(test_bounds)
        print(f"Status: {result['status']}")
        print(f"Tiles downloaded: {result['tiles_downloaded']}")
        
        if result['status'] == 'success':
            # Get a sample point
            lat = (test_bounds.north + test_bounds.south) / 2
            lon = (test_bounds.east + test_bounds.west) / 2
            
            tile_info = lib.get_tile_info(lat, lon)
            if tile_info:
                print(f"\nTile information:")
                print(f"  Shape: {tile_info['shape'][0]}x{tile_info['shape'][1]} pixels")
                print(f"  Pixel size: {tile_info['pixel_size_meters']['x']:.1f}m x {tile_info['pixel_size_meters']['y']:.1f}m")
                print(f"  File size: {tile_info['size_bytes'] / 1024:.1f} KB")
                
                # Test array retrieval
                data, metadata = lib.get_elevation_array(test_bounds)
                print(f"\nArray information:")
                print(f"  Shape: {data.shape}")
                print(f"  Min elevation: {data.min():.1f}m")
                print(f"  Max elevation: {data.max():.1f}m")
                print(f"  Mean elevation: {data.mean():.1f}m")
                
    except Exception as e:
        print(f"Error: {e}")

def main():
    import os
    import sys
    
    # Require data directory
    if len(sys.argv) < 2:
        print("Usage: python test_resolutions.py <data_directory>")
        print("Example: python test_resolutions.py ./elevation_test_data")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    print(f"Using data directory: {os.path.abspath(data_dir)}")
    
    # Define test area (small area in Park City)
    test_bounds = Bounds(
        south=40.650,
        north=40.655,  # Very small area for testing
        west=-111.510,
        east=-111.505
    )
    
    print("\nTesting Multiple Elevation Resolutions")
    print("Test area:", f"{test_bounds.south},{test_bounds.west} to {test_bounds.north},{test_bounds.east}")
    
    # Test different resolutions
    # Note: Not all resolutions may be available for all areas
    resolutions_to_test = [3, 10, 30]
    
    for resolution in resolutions_to_test:
        test_resolution(data_dir, resolution, test_bounds)
    
    # Show directory structure
    print(f"\n{'='*60}")
    print("Directory structure created:")
    print(f"{data_dir}/")
    for res in resolutions_to_test:
        print(f"  └── {res}m/")
        print(f"      ├── tile_index.json")
        print(f"      └── tile_*.tif files")

if __name__ == "__main__":
    main()