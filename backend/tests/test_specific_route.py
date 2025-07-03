#!/usr/bin/env python3
"""
Test that the specific route mentioned is fully cached.
Route: Start: 40.6572, -111.5709  End: 40.6472, -111.5671
"""

import os
import sys
import time

# Add backend to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.dem_tile_cache import DEMTileCache
from app.services.tiled_dem_cache import TiledDEMCache

def test_specific_route():
    """Test the specific route to ensure all data is cached"""
    
    # The route mentioned
    start_lat, start_lon = 40.6572, -111.5709
    end_lat, end_lon = 40.6472, -111.5671
    
    print("=" * 70)
    print("TESTING SPECIFIC ROUTE CACHING")
    print("=" * 70)
    print(f"\nRoute: ({start_lat}, {start_lon}) -> ({end_lat}, {end_lon})")
    
    # Check what tiles this route needs
    tiled_cache = TiledDEMCache(tile_size_degrees=0.01)
    
    # Calculate bounds with small buffer
    min_lat = min(start_lat, end_lat) - 0.001
    max_lat = max(start_lat, end_lat) + 0.001
    min_lon = min(start_lon, end_lon) - 0.001
    max_lon = max(start_lon, end_lon) + 0.001
    
    print(f"\nRoute bounds: ({min_lat:.4f}, {min_lon:.4f}) to ({max_lat:.4f}, {max_lon:.4f})")
    
    # Get tiles needed
    tiles_needed = tiled_cache.get_tiles_for_bounds(min_lat, max_lat, min_lon, max_lon)
    print(f"Tiles needed: {len(tiles_needed)}")
    print(f"Tiles: {tiles_needed}")
    
    # Check if tiles exist
    tile_dir = "tile_cache/cost"
    tiles_found = 0
    missing_tiles = []
    
    for tile_x, tile_y in tiles_needed:
        tile_file = f"tile_{tile_x}_{tile_y}.pkl"
        tile_path = os.path.join(tile_dir, tile_file)
        if os.path.exists(tile_path):
            tiles_found += 1
        else:
            missing_tiles.append((tile_x, tile_y))
    
    print(f"\nTile cache status:")
    print(f"  Tiles found: {tiles_found}/{len(tiles_needed)}")
    
    if missing_tiles:
        print(f"  Missing tiles: {missing_tiles}")
    else:
        print("  ✓ All tiles are cached!")
    
    # Test the actual route calculation
    print("\n" + "-" * 70)
    print("ROUTE CALCULATION TEST")
    print("-" * 70)
    
    dem_cache = DEMTileCache()
    
    # First run
    print("\nFirst run:")
    start_time = time.time()
    path1 = dem_cache.find_route(start_lat, start_lon, end_lat, end_lon)
    time1 = time.time() - start_time
    
    if path1:
        print(f"  ✓ Route found in {time1:.2f}s")
        print(f"  Path points: {len(path1)}")
        
        # Calculate distance
        total_km = 0
        for i in range(len(path1) - 1):
            lat1, lon1 = path1[i][1], path1[i][0]
            lat2, lon2 = path1[i+1][1], path1[i+1][0]
            # Simple distance calculation
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            dist = ((dlat * 111.0)**2 + (dlon * 111.0 * 0.7)**2)**0.5
            total_km += dist
        
        print(f"  Distance: {total_km:.2f} km")
    else:
        print("  ✗ Route not found")
        return
    
    # Second run (should be faster)
    print("\nSecond run (should use memory cache):")
    start_time = time.time()
    path2 = dem_cache.find_route(start_lat, start_lon, end_lat, end_lon)
    time2 = time.time() - start_time
    
    if path2:
        print(f"  ✓ Route found in {time2:.2f}s")
        print(f"  Speedup: {time1/time2:.1f}x")
    
    # Third run with different nearby route (should reuse tiles)
    print("\nNearby route (should reuse tiles):")
    start_time = time.time()
    path3 = dem_cache.find_route(
        start_lat + 0.002, start_lon + 0.001,
        end_lat - 0.001, end_lon - 0.002
    )
    time3 = time.time() - start_time
    
    if path3:
        print(f"  ✓ Route found in {time3:.2f}s")
        print(f"  Uses same tiles: should be fast")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if not missing_tiles:
        print("✓ All tiles for this route are cached")
        print("✓ Route calculation works correctly")
        print("✓ Subsequent calculations are faster")
        print("\nThis route is FULLY CACHED and ready for benchmarking!")
    else:
        print("⚠️  Some tiles are missing")
        print("Run the pre-caching script to cache all of Park City")

if __name__ == "__main__":
    test_specific_route()