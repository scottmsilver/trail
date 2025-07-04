#!/usr/bin/env python
"""Test that mimics exactly what the API does"""
import asyncio
from app.services.trail_finder import TrailFinderService
from app.models.route import Coordinate, RouteOptions
from app.services.obstacle_config import ObstacleConfig, ObstaclePresets
from app.main import get_obstacle_config_for_profile

async def test_api_mimick():
    """Test exactly as the API would"""
    # User's coordinates
    start = Coordinate(lat=40.6482, lon=-111.5738)
    end = Coordinate(lat=40.6464, lon=-111.5729)
    
    print(f"Testing route from ({start.lat}, {start.lon}) to ({end.lat}, {end.lon})")
    
    profiles = ["default", "easy", "experienced", "trail_runner", "accessibility"]
    
    for profile in profiles:
        print(f"\n{'='*50}")
        print(f"Testing profile: {profile}")
        print('='*50)
        
        # Get obstacle configuration exactly as API does
        obstacle_config = get_obstacle_config_for_profile(profile)
        
        # Print configuration details
        print(f"OSM tags: {list(obstacle_config.osm_tags.keys())}")
        print(f"Has highways as obstacles: {'highway' in obstacle_config.osm_tags}")
        
        # Create service exactly as API does
        service = TrailFinderService(obstacle_config=obstacle_config)
        
        # Validate
        if not service.validate_route_request(start, end):
            print("✗ Failed validation!")
            continue
            
        # Find route
        try:
            path, stats = await service.find_route(start, end, {})
            
            if path:
                print(f"✓ SUCCESS: {len(path)} points")
                if stats:
                    print(f"  Distance: {stats.get('distance_km', 'N/A')} km")
                    print(f"  Elevation gain: {stats.get('elevation_gain', 'N/A')} m")
            else:
                print(f"✗ FAILED: {stats.get('error', 'No route found')}")
                
            # Also test with debug mode
            debug_service = TrailFinderService(debug_mode=True, obstacle_config=obstacle_config)
            debug_path, debug_stats = await debug_service.find_route(start, end, {})
            
            if debug_path and not path:
                print("  ⚠️  Debug mode found a path but regular mode didn't!")
            elif not debug_path and path:
                print("  ⚠️  Regular mode found a path but debug mode didn't!")
                
        except Exception as e:
            print(f"✗ ERROR: {e}")
    
    # Test with completely custom config
    print(f"\n{'='*50}")
    print("Testing with CUSTOM no-obstacle config")
    print('='*50)
    
    no_obstacle_config = ObstacleConfig(
        osm_tags={},  # No obstacles
        obstacle_costs={'default': 1.0},
        slope_costs=[(0, 1.0), (90, 10.0)]
    )
    
    custom_service = TrailFinderService(obstacle_config=no_obstacle_config)
    custom_path, custom_stats = await custom_service.find_route(start, end, {})
    
    if custom_path:
        print(f"✓ SUCCESS with no obstacles: {len(custom_path)} points")
    else:
        print(f"✗ FAILED even with no obstacles: {custom_stats.get('error', 'Unknown')}")

if __name__ == "__main__":
    asyncio.run(test_api_mimick())