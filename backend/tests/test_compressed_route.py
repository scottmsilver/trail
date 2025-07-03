#!/usr/bin/env python3
"""Test compressed pathfinding on the failing route"""

import asyncio
import time
from app.services.trail_finder import TrailFinderService
from app.services.obstacle_config import ObstacleConfig
from app.services.path_preferences import PathPreferences
from app.models.route import Coordinate
from geopy.distance import distance

async def test_compressed_route():
    # Failing route coordinates
    start = Coordinate(lat=40.6566, lon=-111.5701)
    end = Coordinate(lat=40.6286, lon=-111.5689)
    
    print(f"Testing compressed pathfinding")
    print(f"Start: {start.lat}, {start.lon}")
    print(f"End: {end.lat}, {end.lon}")
    
    # Calculate approximate distance
    direct_distance = distance((start.lat, start.lon), (end.lat, end.lon)).km
    print(f"Direct distance: {direct_distance:.2f} km")
    print("=" * 60)
    
    # Use default configuration
    config = ObstacleConfig()
    config.use_continuous_slope = True
    config.slope_profile = 'default'
    
    path_prefs = PathPreferences()
    
    trail_finder = TrailFinderService(
        obstacle_config=config,
        path_preferences=path_prefs
    )
    
    print("\nStarting route search with compressed pathfinding...")
    print("Compression will group similar terrain to reduce search space")
    
    start_time = time.time()
    
    try:
        path, stats = await trail_finder.find_route(start, end, {})
        
        elapsed = time.time() - start_time
        print(f"\nSearch completed in {elapsed:.1f} seconds")
        
        if not path:
            error = stats.get('error', 'Unknown error')
            print(f"Route search failed: {error}")
        else:
            print(f"\nRoute found successfully!")
            print(f"  Total distance: {stats.get('distance_km', 0):.2f} km")
            print(f"  Path points: {len(path)}")
            
            # Analyze path
            if 'path_with_slopes' in stats:
                path_slopes = stats['path_with_slopes']
                slopes = [abs(p.get('slope', 0)) for p in path_slopes if 'slope' in p]
                
                if slopes:
                    import numpy as np
                    print(f"\nSlope statistics:")
                    print(f"  Max slope: {max(slopes):.1f}°")
                    print(f"  Average slope: {np.mean(slopes):.1f}°")
                    print(f"  95th percentile: {np.percentile(slopes, 95):.1f}°")
                    print(f"  Slopes > 30°: {sum(1 for s in slopes if s > 30)}")
                    print(f"  Slopes > 35°: {sum(1 for s in slopes if s > 35)}")
                    
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\nException after {elapsed:.1f} seconds: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_compressed_route())