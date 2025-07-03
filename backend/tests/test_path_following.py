#!/usr/bin/env python
"""Test that routes prefer to follow paths and trails"""
import asyncio
from app.services.trail_finder import TrailFinderService
from app.models.route import Coordinate
from app.services.obstacle_config import ObstacleConfig
from app.services.path_preferences import PathPreferences, PathPreferencePresets

async def test_path_following():
    """Test that the algorithm prefers following paths"""
    # User's coordinates from before
    start = Coordinate(lat=40.6482, lon=-111.5738)
    end = Coordinate(lat=40.6464, lon=-111.5729)
    
    print("PATH FOLLOWING TEST")
    print("="*60)
    print(f"Route: ({start.lat}, {start.lon}) → ({end.lat}, {end.lon})")
    print(f"Distance: ~200m\n")
    
    # Test 1: Route without path preferences (direct route)
    print("1. WITHOUT PATH PREFERENCES (direct route):")
    no_pref_config = ObstacleConfig()
    no_pref = PathPreferences(
        path_costs={'off_path': 1.0},  # No preference
        stick_to_paths=False,
        path_transition_penalty=1.0
    )
    
    service_no_pref = TrailFinderService(
        obstacle_config=no_pref_config,
        path_preferences=no_pref
    )
    path_no_pref, stats_no_pref = await service_no_pref.find_route(start, end, {})
    
    if path_no_pref:
        print(f"   ✓ Route found: {len(path_no_pref)} points")
        print(f"   Distance: {stats_no_pref.get('distance_km', 0):.3f} km")
    else:
        print("   ✗ No route found")
    
    # Test 2: Route WITH path preferences (follows paths)
    print("\n2. WITH PATH PREFERENCES (follows paths):")
    path_pref = PathPreferencePresets.urban_walker()
    
    service_with_pref = TrailFinderService(
        obstacle_config=no_pref_config,
        path_preferences=path_pref
    )
    path_with_pref, stats_with_pref = await service_with_pref.find_route(start, end, {})
    
    if path_with_pref:
        print(f"   ✓ Route found: {len(path_with_pref)} points")
        print(f"   Distance: {stats_with_pref.get('distance_km', 0):.3f} km")
        
        # Compare routes
        if path_no_pref and len(path_with_pref) > len(path_no_pref):
            print("   → Route is longer (follows paths instead of direct line)")
        elif path_no_pref and len(path_with_pref) == len(path_no_pref):
            print("   → Route similar length (may already follow paths)")
    else:
        print("   ✗ No route found")
    
    # Test 3: Different profiles
    print("\n3. PROFILE PREFERENCES:")
    profiles = {
        "Urban Walker": PathPreferencePresets.urban_walker(),
        "Trail Seeker": PathPreferencePresets.trail_seeker(), 
        "Flexible Hiker": PathPreferencePresets.flexible_hiker(),
        "Direct Route": PathPreferencePresets.direct_route()
    }
    
    for name, pref in profiles.items():
        service = TrailFinderService(
            obstacle_config=no_pref_config,
            path_preferences=pref
        )
        path, stats = await service.find_route(start, end, {})
        
        if path:
            print(f"   {name:15} - {len(path)} points, {stats.get('distance_km', 0):.3f} km")
        else:
            print(f"   {name:15} - No route found")
    
    # Summary
    print("\n" + "="*60)
    print("PATH PREFERENCES EXPLAINED:")
    print("• Urban Walker: Strongly prefers sidewalks (cost 0.2) over off-path (2.0)")
    print("• Trail Seeker: Prefers natural trails (0.2) over streets (0.8)")
    print("• Flexible Hiker: Mild preference for paths (0.5-0.7) vs off-path (1.0)")
    print("• Direct Route: Minimal preference - just avoids major roads")
    print("\nThe algorithm naturally follows paths when their cost is lower!")

if __name__ == "__main__":
    asyncio.run(test_path_following())