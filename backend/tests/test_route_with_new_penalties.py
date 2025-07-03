#!/usr/bin/env python3
"""Test route with new aggressive slope penalties"""

import asyncio
import numpy as np
from app.services.trail_finder import TrailFinderService
from app.services.obstacle_config import ObstacleConfig
from app.models.route import Coordinate

async def test_route():
    # Test coordinates
    start = Coordinate(lat=40.6546, lon=-111.5705)
    end = Coordinate(lat=40.6485, lon=-111.5641)
    
    print(f"Testing route from {start.lat}, {start.lon} to {end.lat}, {end.lon}")
    print("=" * 60)
    
    # Test with different profiles to see improvement
    profiles = [
        ("default", "default"),
        ("city_walker", "city_walker"),
        ("easy", "city_walker"),  # Easy profile now maps to city_walker
    ]
    
    for profile_name, slope_profile in profiles:
        print(f"\nProfile: {profile_name} (slope profile: {slope_profile})")
        print("-" * 40)
        
        config = ObstacleConfig()
        config.use_continuous_slope = True
        config.slope_profile = slope_profile
        
        # Show slope costs
        print("Slope penalties:")
        for slope in [10, 15, 20, 25, 30, 35, 40]:
            cost = config.get_slope_cost_multiplier(slope)
            if cost > 1000:
                print(f"  {slope}°: {cost:.1e}x")
            else:
                print(f"  {slope}°: {cost:.1f}x")
        
        # Find route
        trail_finder = TrailFinderService(obstacle_config=config)
        path, stats = await trail_finder.find_route(start, end, {"userProfile": profile_name})
        
        if not path:
            print(f"\nNo route found: {stats.get('error', 'Unknown error')}")
            continue
            
        print(f"\nRoute found:")
        print(f"  Points: {len(path)}")
        print(f"  Distance: {stats.get('distance_km', 0):.2f} km")
        
        # Analyze slopes
        if 'path_with_slopes' in stats:
            path_slopes = stats['path_with_slopes']
            slopes = [abs(p.get('slope', 0)) for p in path_slopes if 'slope' in p]
            
            if slopes:
                print(f"\nSlope statistics:")
                print(f"  Max slope: {max(slopes):.1f}°")
                print(f"  Average slope: {np.mean(slopes):.1f}°")
                print(f"  Slopes > 20°: {sum(1 for s in slopes if s > 20)}")
                print(f"  Slopes > 25°: {sum(1 for s in slopes if s > 25)}")
                print(f"  Slopes > 30°: {sum(1 for s in slopes if s > 30)}")
                
                # Show distribution
                bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 90]
                hist, _ = np.histogram(slopes, bins=bins)
                print("\n  Slope distribution:")
                for i in range(len(bins)-1):
                    if hist[i] > 0:
                        print(f"    {bins[i]:2d}-{bins[i+1]:2d}°: {hist[i]:3d} segments")

if __name__ == "__main__":
    asyncio.run(test_route())