#!/usr/bin/env python3
"""Test that routes prefer natural surfaces over roads"""

import asyncio
from app.services.trail_finder import TrailFinderService
from app.services.obstacle_config import ObstacleConfig
from app.services.path_preferences import PathPreferences, PathPreferencePresets
from app.models.route import Coordinate
import pytest

async def test_natural_preference():
    # Test area with both roads and natural areas
    # This is an area that likely has parks, trails, and roads
    start = Coordinate(lat=40.7614, lon=-111.8911)  # Liberty Park area in SLC
    end = Coordinate(lat=40.7580, lon=-111.8850)
    
    print("Testing route preference for natural surfaces vs roads")
    print(f"Start: {start.lat}, {start.lon}")
    print(f"End: {end.lat}, {end.lon}")
    print("=" * 60)
    
    # Test different preference profiles
    profiles = [
        ("Default (new)", PathPreferences()),
        ("Trail Seeker", PathPreferencePresets.trail_seeker()),
        ("Urban Walker", PathPreferencePresets.urban_walker()),
        ("Direct Route", PathPreferencePresets.direct_route()),
    ]
    
    for profile_name, path_prefs in profiles:
        print(f"\n{profile_name} Profile:")
        print("-" * 40)
        
        # Show some key preferences
        print("Path cost multipliers:")
        for path_type in ['trail', 'path', 'grass', 'off_path', 'footway', 'residential']:
            cost = path_prefs.get_path_cost_multiplier(path_type)
            print(f"  {path_type}: {cost}")
        
        # Find route
        config = ObstacleConfig()
        config.use_continuous_slope = True
        
        trail_finder = TrailFinderService(
            obstacle_config=config,
            path_preferences=path_prefs
        )
        
        path, stats = await trail_finder.find_route(start, end, {})
        
        if not path:
            print(f"\nNo route found: {stats.get('error', 'Unknown error')}")
            continue
            
        print(f"\nRoute found:")
        print(f"  Distance: {stats.get('distance_km', 0):.2f} km")
        print(f"  Points: {len(path)}")
        
        # Analyze path composition if available
        if 'path_composition' in stats:
            composition = stats['path_composition']
            print("\nPath composition:")
            total_segments = sum(composition.values())
            for path_type, count in sorted(composition.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_segments * 100) if total_segments > 0 else 0
                print(f"  {path_type}: {count} segments ({percentage:.1f}%)")
        
        # Check if we're preferring natural surfaces
        if 'debug_data' in stats and stats['debug_data']:
            debug = stats['debug_data']
            if 'path_types_used' in debug:
                natural_types = ['trail', 'path', 'track', 'grass', 'meadow', 'park']
                road_types = ['residential', 'footway', 'service', 'unclassified']
                
                natural_count = sum(1 for pt in debug['path_types_used'] if pt in natural_types)
                road_count = sum(1 for pt in debug['path_types_used'] if pt in road_types)
                
                print(f"\nSurface preference analysis:")
                print(f"  Natural surface segments: {natural_count}")
                print(f"  Road/paved segments: {road_count}")
                print(f"  Ratio: {natural_count/road_count:.2f}:1" if road_count > 0 else "  All natural!")

if __name__ == "__main__":
    asyncio.run(test_natural_preference())