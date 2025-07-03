#!/usr/bin/env python3
"""
Test if terrain caching is working properly
"""

import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.dem_tile_cache import DEMTileCache
from app.services.obstacle_config import ObstacleConfig
from app.services.path_preferences import PathPreferences


def test_caching():
    """Test terrain caching functionality"""
    
    print("🧪 TESTING TERRAIN CACHING")
    print("="*60)
    
    # Short route for testing
    start_lat, start_lon = 37.7694, -122.4762  # Stow Lake
    end_lat, end_lon = 37.7704, -122.4752      # Very close
    
    print(f"Test route: ({start_lat}, {start_lon}) → ({end_lat}, {end_lon})")
    print("-"*60)
    
    # Initialize cache
    cache = DEMTileCache(
        obstacle_config=ObstacleConfig(),
        path_preferences=PathPreferences()
    )
    
    # First run - should download terrain
    print("\n1️⃣ First run (downloading terrain):")
    start_time = time.time()
    
    try:
        path1 = cache.find_route(start_lat, start_lon, end_lat, end_lon)
        time1 = time.time() - start_time
        
        if path1:
            print(f"✓ Success! Time: {time1:.1f}s, Path points: {len(path1)}")
        else:
            print(f"✗ No path found ({time1:.1f}s)")
            
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return
    
    # Second run - should use cached terrain
    print("\n2️⃣ Second run (should use cache):")
    start_time = time.time()
    
    try:
        path2 = cache.find_route(start_lat, start_lon, end_lat, end_lon)
        time2 = time.time() - start_time
        
        if path2:
            print(f"✓ Success! Time: {time2:.1f}s, Path points: {len(path2)}")
        else:
            print(f"✗ No path found ({time2:.1f}s)")
            
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return
    
    # Third run - definitely should be cached
    print("\n3️⃣ Third run (definitely cached):")
    start_time = time.time()
    
    try:
        path3 = cache.find_route(start_lat, start_lon, end_lat, end_lon)
        time3 = time.time() - start_time
        
        if path3:
            print(f"✓ Success! Time: {time3:.1f}s, Path points: {len(path3)}")
        else:
            print(f"✗ No path found ({time3:.1f}s)")
            
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return
    
    # Analysis
    print("\n📊 ANALYSIS:")
    print("-"*60)
    
    if time1 > 0 and time2 > 0 and time3 > 0:
        speedup2 = time1 / time2
        speedup3 = time1 / time3
        
        print(f"First run:  {time1:.1f}s (baseline)")
        print(f"Second run: {time2:.1f}s ({speedup2:.1f}x faster)")
        print(f"Third run:  {time3:.1f}s ({speedup3:.1f}x faster)")
        
        if speedup2 > 1.5 or speedup3 > 1.5:
            print("\n✅ Caching is working! Subsequent runs are faster.")
        else:
            print("\n⚠️  Caching may not be working properly.")
            print("   Subsequent runs should be significantly faster.")
    
    # Check cache contents
    print(f"\nCache status:")
    print(f"  Terrain cache entries: {len(cache.terrain_cache)}")
    print(f"  Cost surface cache entries: {len(cache.cost_surface_cache)}")
    print(f"  Preprocessing cache entries: {len(cache.preprocessing_cache)}")
    
    # Check disk cache
    dem_dir = 'dem_data'
    if os.path.exists(dem_dir):
        files = [f for f in os.listdir(dem_dir) if f.endswith('.tif')]
        print(f"  DEM files on disk: {len(files)}")
        for f in files[:3]:  # Show first 3
            print(f"    - {f}")
        if len(files) > 3:
            print(f"    ... and {len(files)-3} more")
    
    print("\n" + "="*60)
    print("Test complete!")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    
    test_caching()