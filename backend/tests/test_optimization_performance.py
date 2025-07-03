#!/usr/bin/env python3
"""
Test the optimized pathfinding performance with real routes
"""

import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.dem_tile_cache import DEMTileCache
from app.services.obstacle_config import ObstacleConfig
from app.services.path_preferences import PathPreferences

def test_route_performance(lat1, lon1, lat2, lon2, route_name):
    """Test a single route and measure performance"""
    print(f"\n{'='*60}")
    print(f"Testing route: {route_name}")
    print(f"From: ({lat1}, {lon1}) to ({lat2}, {lon2})")
    print(f"{'='*60}")
    
    # Initialize cache with optimizations
    cache = DEMTileCache(
        obstacle_config=ObstacleConfig(),
        path_preferences=PathPreferences()
    )
    
    # Time the route finding
    start_time = time.time()
    
    try:
        path = cache.find_route(lat1, lon1, lat2, lon2)
        
        elapsed = time.time() - start_time
        
        if path:
            print(f"✓ Route found successfully!")
            print(f"  Time taken: {elapsed:.2f} seconds")
            print(f"  Path points: {len(path)}")
            
            # Calculate approximate distance
            total_distance = 0
            for i in range(len(path) - 1):
                # Simple distance calculation (not accounting for elevation)
                lat_diff = path[i+1]['lat'] - path[i]['lat']
                lon_diff = path[i+1]['lon'] - path[i]['lon']
                # Rough approximation in km
                distance = ((lat_diff * 111)**2 + (lon_diff * 111 * 0.7)**2)**0.5
                total_distance += distance
            
            print(f"  Approximate distance: {total_distance:.2f} km")
            
            # Show elevation range
            elevations = [p['elevation'] for p in path if p.get('elevation') is not None]
            if elevations:
                print(f"  Elevation range: {min(elevations):.0f}m - {max(elevations):.0f}m")
            
            # Show slope statistics
            slopes = [abs(p['slope']) for p in path if p.get('slope') is not None]
            if slopes:
                avg_slope = sum(slopes) / len(slopes)
                max_slope = max(slopes)
                print(f"  Average slope: {avg_slope:.1f}°, Max slope: {max_slope:.1f}°")
                
        else:
            print(f"✗ No route found")
            
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"✗ Error finding route: {str(e)}")
        print(f"  Time before error: {elapsed:.2f} seconds")
    
    return elapsed

def main():
    """Test multiple routes to measure performance"""
    
    # Test routes of varying complexity
    test_routes = [
        # Short urban route (easy)
        {
            'name': 'Short Urban Route (San Francisco)',
            'lat1': 37.7749, 'lon1': -122.4194,
            'lat2': 37.7849, 'lon2': -122.4094,
            'expected_time': 5  # seconds
        },
        # Medium trail route (moderate)
        {
            'name': 'Medium Trail Route (Berkeley Hills)',
            'lat1': 37.8716, 'lon1': -122.2727,
            'lat2': 37.8916, 'lon2': -122.2527,
            'expected_time': 15
        },
        # Long backcountry route (hard)
        {
            'name': 'Long Backcountry Route (Mt. Diablo)',
            'lat1': 37.8816, 'lon1': -121.9146,
            'lat2': 37.9216, 'lon2': -121.8746,
            'expected_time': 30
        },
        # Steep terrain route (very hard)
        {
            'name': 'Steep Terrain Route (Mt. Tamalpais)',
            'lat1': 37.9235, 'lon1': -122.5965,
            'lat2': 37.9435, 'lon2': -122.5765,
            'expected_time': 25
        }
    ]
    
    print("PATHFINDING OPTIMIZATION PERFORMANCE TEST")
    print("=" * 60)
    print("Testing optimized A* pathfinding implementation...")
    
    total_time = 0
    successful_routes = 0
    
    for route in test_routes:
        elapsed = test_route_performance(
            route['lat1'], route['lon1'],
            route['lat2'], route['lon2'],
            route['name']
        )
        
        total_time += elapsed
        
        # Compare with expected time
        if elapsed < route['expected_time'] * 2:  # Allow 2x expected time
            successful_routes += 1
            improvement = (route['expected_time'] - elapsed) / route['expected_time'] * 100
            if improvement > 0:
                print(f"  → {improvement:.1f}% faster than expected!")
            else:
                print(f"  → {-improvement:.1f}% slower than expected")
        else:
            print(f"  → Too slow (expected <{route['expected_time']}s)")
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total routes tested: {len(test_routes)}")
    print(f"Successful routes: {successful_routes}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per route: {total_time/len(test_routes):.2f} seconds")
    
    # Performance grade
    success_rate = successful_routes / len(test_routes) * 100
    if success_rate >= 80:
        print(f"\n✅ PASS: {success_rate:.0f}% success rate")
    else:
        print(f"\n❌ NEEDS IMPROVEMENT: {success_rate:.0f}% success rate")

if __name__ == "__main__":
    main()