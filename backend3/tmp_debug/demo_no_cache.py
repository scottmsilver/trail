#!/usr/bin/env python3
"""
Demo showing the elevation library without "cache" terminology.
Data is either loaded (statically available) or not.
"""

from elevation import ElevationLibrary, Bounds

def main():
    # Create library instance
    lib = ElevationLibrary(data_dir="./demo_elevation_data", resolution=10)
    
    # Define test area
    test_bounds = Bounds(
        south=40.65,
        north=40.66,
        west=-111.51,
        east=-111.50
    )
    
    print("Elevation Library Demo (No Cache)")
    print("=" * 50)
    
    # 1. Try to get elevation without loading (should fail)
    print("\n1. Attempting to get elevation without loading data:")
    try:
        elevation = lib.get_elevation(40.655, -111.505)
        print(f"  ERROR: Should have failed but got: {elevation}")
    except ValueError as e:
        print(f"  ✓ Correctly failed: {e}")
    
    # 2. List loaded areas (should be empty)
    print("\n2. Listing loaded areas:")
    info = lib.list_loaded_areas()
    print(f"  Loaded tiles: {info['total_tiles']}")
    print(f"  Total size: {info['total_size_mb']:.1f} MB")
    print(f"  Number of areas: {len(info['areas'])}")
    
    # 3. Load area
    print(f"\n3. Loading elevation data for area:")
    print(f"  Bounds: {test_bounds.south},{test_bounds.west} to {test_bounds.north},{test_bounds.east}")
    print("  (This would download from USGS if run for real)")
    
    # 4. List loaded areas again
    print("\n4. After loading, the data is statically available:")
    print("  - Data persists on disk in the data directory")
    print("  - No automatic eviction or expiration")
    print("  - Remains available until explicitly removed")
    
    # 5. Remove data
    print("\n5. Data must be explicitly removed:")
    print("  - Use remove_area() to remove specific areas")
    print("  - Use remove_all() to remove all loaded data")
    
    print("\n" + "=" * 50)
    print("Key points:")
    print("- This is NOT a cache - data is either loaded or not")
    print("- Data persists until explicitly removed")
    print("- No automatic management or eviction")
    print("- Think of it as a static data store, not a cache")

if __name__ == "__main__":
    main()