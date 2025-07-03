#!/usr/bin/env python3
"""Test route with higher resolution DEM"""

import asyncio
import numpy as np
from app.services.trail_finder import TrailFinderService
from app.services.obstacle_config import ObstacleConfig
from app.models.route import Coordinate
import time

async def test_high_resolution():
    # Test coordinates
    start = Coordinate(lat=40.6546, lon=-111.5705)
    end = Coordinate(lat=40.6485, lon=-111.5641)
    
    print(f"Testing route from {start.lat}, {start.lon} to {end.lat}, {end.lon}")
    print("=" * 60)
    
    # Test with city_walker profile (aggressive slope penalties)
    config = ObstacleConfig()
    config.use_continuous_slope = True
    config.slope_profile = 'city_walker'
    
    print("\nProfile: city_walker")
    print("Attempting to use 3m resolution DEM...")
    
    # Time the route finding
    start_time = time.time()
    
    trail_finder = TrailFinderService(obstacle_config=config)
    path, stats = await trail_finder.find_route(start, end, {"userProfile": "city_walker"})
    
    elapsed = time.time() - start_time
    print(f"\nRoute finding took: {elapsed:.1f} seconds")
    
    if not path:
        print(f"No route found: {stats.get('error', 'Unknown error')}")
        return
        
    print(f"\nRoute found:")
    print(f"  Points: {len(path)}")
    print(f"  Distance: {stats.get('distance_km', 0):.2f} km")
    
    # Analyze slopes with higher resolution
    if 'path_with_slopes' in stats:
        path_slopes = stats['path_with_slopes']
        slopes = [abs(p.get('slope', 0)) for p in path_slopes if 'slope' in p]
        
        if slopes:
            print(f"\nSlope statistics:")
            print(f"  Max slope: {max(slopes):.1f}°")
            print(f"  Average slope: {np.mean(slopes):.1f}°")
            print(f"  95th percentile slope: {np.percentile(slopes, 95):.1f}°")
            
            # Count steep segments
            print(f"\nSteep segment counts:")
            thresholds = [15, 20, 25, 30, 35, 40]
            for threshold in thresholds:
                count = sum(1 for s in slopes if s > threshold)
                percentage = (count / len(slopes)) * 100 if slopes else 0
                print(f"  Slopes > {threshold}°: {count} ({percentage:.1f}%)")
            
            # Show distribution
            bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 90]
            hist, _ = np.histogram(slopes, bins=bins)
            print("\nSlope distribution:")
            for i in range(len(bins)-1):
                if hist[i] > 0:
                    percentage = (hist[i] / len(slopes)) * 100
                    print(f"  {bins[i]:2d}-{bins[i+1]:2d}°: {hist[i]:4d} segments ({percentage:5.1f}%)")
            
            # Check resolution from path
            if len(path) > 1:
                from geopy.distance import distance
                distances = []
                for i in range(len(path) - 1):
                    dist = distance(
                        (path[i][1], path[i][0]),  # lat, lon
                        (path[i+1][1], path[i+1][0])
                    ).meters
                    distances.append(dist)
                
                avg_dist = np.mean(distances)
                print(f"\nPath resolution check:")
                print(f"  Average distance between points: {avg_dist:.1f}m")
                print(f"  Expected for 3m resolution: ~3-4m")
                print(f"  Expected for 10m resolution: ~10-14m")

if __name__ == "__main__":
    asyncio.run(test_high_resolution())