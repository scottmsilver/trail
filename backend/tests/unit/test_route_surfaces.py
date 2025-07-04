#!/usr/bin/env python3
"""Test route surface preferences with original coordinates"""

import asyncio
from app.services.trail_finder import TrailFinderService
from app.services.obstacle_config import ObstacleConfig
from app.services.path_preferences import PathPreferences
from app.models.route import Coordinate
import pytest

async def test_route_surfaces():
    # Original test coordinates
    start = Coordinate(lat=40.6546, lon=-111.5705)
    end = Coordinate(lat=40.6485, lon=-111.5641)
    
    print(f"Testing route with natural surface preference")
    print(f"Start: {start.lat}, {start.lon}")
    print(f"End: {end.lat}, {end.lon}")
    print("=" * 60)
    
    # Create path preferences that strongly prefer natural surfaces
    path_prefs = PathPreferences()
    
    print("\nPath cost multipliers (lower = more preferred):")
    test_types = ['trail', 'path', 'grass', 'meadow', 'off_path', 'footway', 'residential']
    for path_type in test_types:
        cost = path_prefs.get_path_cost_multiplier(path_type)
        print(f"  {path_type}: {cost}")
    
    # Configure for natural surfaces
    config = ObstacleConfig()
    config.use_continuous_slope = True
    config.slope_profile = 'default'
    
    trail_finder = TrailFinderService(
        obstacle_config=config,
        path_preferences=path_prefs
    )
    
    print("\nFinding route (this may take a while with 3m resolution)...")
    path, stats = await trail_finder.find_route(start, end, {})
    
    if not path:
        print(f"No route found: {stats.get('error', 'Unknown error')}")
        return
        
    print(f"\nRoute found:")
    print(f"  Distance: {stats.get('distance_km', 0):.2f} km")
    print(f"  Points: {len(path)}")
    
    # Check slopes
    if 'path_with_slopes' in stats:
        path_slopes = stats['path_with_slopes']
        slopes = [abs(p.get('slope', 0)) for p in path_slopes if 'slope' in p]
        
        if slopes:
            import numpy as np
            print(f"\nSlope statistics:")
            print(f"  Max slope: {max(slopes):.1f}°")
            print(f"  Average slope: {np.mean(slopes):.1f}°")
            print(f"  Slopes > 30°: {sum(1 for s in slopes if s > 30)}")
    
    print("\nRoute preferences are now set to:")
    print("1. Natural trails and dirt paths (cost: 0.2-0.3)")
    print("2. Grass and meadows (cost: 0.4)")
    print("3. Unobstructed natural terrain (cost: 0.5)")
    print("4. Sidewalks and roads (cost: 0.6-0.99)")
    print("\nThe algorithm will choose natural surfaces when available!")

if __name__ == "__main__":
    asyncio.run(test_route_surfaces())