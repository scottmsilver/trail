#!/usr/bin/env python3
"""
Simple test of the optimized pathfinding
"""

import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.dem_tile_cache import DEMTileCache
from app.services.obstacle_config import ObstacleConfig
import pytest

@pytest.mark.unit
def test_optimization(mock_dem_cache):
    """Test the optimized pathfinding with a real route"""
    
    # Test coordinates - a moderate distance route
    lat1, lon1 = 37.7749, -122.4194  # San Francisco
    lat2, lon2 = 37.7849, -122.4094  # Slightly northeast
    
    print(f"Testing route from ({lat1}, {lon1}) to ({lat2}, {lon2})")
    print("=" * 60)
    
    # Create cache instance
    obstacle_config = ObstacleConfig()
    cache = DEMTileCache(
        buffer=0.02,
        debug_mode=False,
        obstacle_config=obstacle_config
    )
    
    # Time the route finding
    start_time = time.time()
    
    try:
        path = cache.find_route(lat1, lon1, lat2, lon2)
        
        elapsed = time.time() - start_time
        
        if path:
            print(f"\n✓ Route found successfully!")
            print(f"  Time taken: {elapsed:.3f} seconds")
            print(f"  Path points: {len(path)}")
            
            # Calculate total distance
            total_distance = 0
            for i in range(len(path) - 1):
                p1 = path[i]
                p2 = path[i + 1]
                # Approximate distance
                dx = (p2['lon'] - p1['lon']) * 111000 * 0.866  # cos(37.7 degrees)
                dy = (p2['lat'] - p1['lat']) * 111000
                dist = (dx**2 + dy**2)**0.5
                total_distance += dist
            
            print(f"  Total distance: {total_distance:.0f} meters")
            
            # Show first few waypoints
            print(f"\n  First 3 waypoints:")
            for i, point in enumerate(path[:3]):
                print(f"    {i+1}. Lat: {point['lat']:.6f}, Lon: {point['lon']:.6f}, "
                      f"Elev: {point['elevation']:.1f}m, Slope: {point['slope']}°")
        else:
            print(f"\n✗ No route found")
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        
    print("\n" + "=" * 60)
    
if __name__ == "__main__":
    test_optimization()