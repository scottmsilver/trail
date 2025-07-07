#!/usr/bin/env python3
"""
Test the elevation library
"""

from elevation import ElevationLibrary, Bounds

def main():
    import os
    
    # Use explicit test data directory
    test_data_dir = "./test_elevation_data"
    
    # Create library instance
    lib = ElevationLibrary(data_dir=test_data_dir, resolution=10)
    print(f"Using data directory: {os.path.abspath(test_data_dir)}")
    
    # Define test area (small area in Park City)
    test_bounds = Bounds(
        south=40.65,
        north=40.66,
        west=-111.51,
        east=-111.50
    )
    
    print("Testing Elevation Library")
    print("=" * 50)
    
    # Test 1: Try to get elevation without loading (should fail)
    print("\n1. Testing get without load:")
    try:
        elevation = lib.get_elevation(40.655, -111.505)
        print(f"  ERROR: Should have failed but got: {elevation}")
    except ValueError as e:
        print(f"  ✓ Correctly failed: {e}")
    
    # Test 2: List cached areas (should be empty)
    print("\n2. Listing cached areas:")
    info = lib.list_cached_areas()
    print(f"  Cached tiles: {info['total_tiles']}")
    print(f"  Total size: {info['total_size_mb']:.1f} MB")
    
    # Test 3: Load area
    print(f"\n3. Loading test area:")
    print(f"  Bounds: {test_bounds.south},{test_bounds.west} to {test_bounds.north},{test_bounds.east}")
    result = lib.load_area(test_bounds)
    print(f"  Status: {result['status']}")
    print(f"  Tiles downloaded: {result['tiles_downloaded']}")
    
    # Test 4: Try to get elevation again (should work now)
    print("\n4. Testing get after load:")
    try:
        elevation = lib.get_elevation(40.655, -111.505)
        print(f"  ✓ Elevation: {elevation:.1f} meters")
        
        # Also test tile info
        tile_info = lib.get_tile_info(40.655, -111.505)
        if tile_info:
            print(f"  Tile resolution: {tile_info['resolution_m']}m")
            print(f"  Tile shape: {tile_info['shape']}")
            print(f"  Pixel size: {tile_info['pixel_size_meters']['x']:.1f}m x {tile_info['pixel_size_meters']['y']:.1f}m")
    except ValueError as e:
        print(f"  ERROR: {e}")
    
    # Test 5: Get elevation array
    print("\n5. Testing get elevation array:")
    try:
        data, metadata = lib.get_elevation_array(test_bounds)
        print(f"  ✓ Array shape: {data.shape}")
        print(f"  Min elevation: {data.min():.1f}m")
        print(f"  Max elevation: {data.max():.1f}m")
        print(f"  Pixel size: {metadata['pixel_size_meters']['x']:.1f}m x {metadata['pixel_size_meters']['y']:.1f}m")
        print(f"  Resolution: {metadata['resolution_m']}m")
        print(f"  CRS: {metadata['crs']}")
    except ValueError as e:
        print(f"  ERROR: {e}")
    
    # Test 6: List cached areas again
    print("\n6. Listing cached areas after load:")
    info = lib.list_cached_areas()
    print(f"  Cached tiles: {info['total_tiles']}")
    print(f"  Total size: {info['total_size_mb']:.2f} MB")
    
    # Test 7: Remove area
    print("\n7. Testing remove area:")
    result = lib.remove_area(test_bounds)
    print(f"  Status: {result['status']}")
    print(f"  Tiles removed: {result['tiles_removed']}")
    
    # Test 8: Try to get elevation after removal (should fail)
    print("\n8. Testing get after remove:")
    try:
        elevation = lib.get_elevation(40.655, -111.505)
        print(f"  ERROR: Should have failed but got: {elevation}")
    except ValueError as e:
        print(f"  ✓ Correctly failed: {e}")
    
    # Test 9: List cached areas after removal
    print("\n9. Listing cached areas after remove:")
    info = lib.list_cached_areas()
    print(f"  Cached tiles: {info['total_tiles']}")
    print(f"  Total size: {info['total_size_mb']:.1f} MB")
    
    print("\n" + "=" * 50)
    print("Tests complete!")

if __name__ == "__main__":
    main()