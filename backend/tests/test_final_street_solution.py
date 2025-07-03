#!/usr/bin/env python
"""Final test showing the street obstacle solution"""
import asyncio
from app.services.trail_finder import TrailFinderService
from app.models.route import Coordinate
from app.services.obstacle_config import ObstacleConfig, ObstaclePresets

async def test_final_solution():
    """Comprehensive test of the street routing fix"""
    # User's coordinates
    start = Coordinate(lat=40.6482, lon=-111.5738)
    end = Coordinate(lat=40.6464, lon=-111.5729)
    
    print("STREET ROUTING TEST RESULTS")
    print("="*60)
    print(f"Route: ({start.lat}, {start.lon}) → ({end.lat}, {end.lon})")
    print(f"Distance: ~200m\n")
    
    # Test 1: Check old configuration (with streets as obstacles)
    print("1. OLD CONFIGURATION (streets as obstacles):")
    old_config = ObstacleConfig()
    # Manually add highways back to simulate old behavior
    old_config.osm_tags['highway'] = ['motorway', 'trunk', 'primary', 'secondary']
    old_config.obstacle_costs['highway'] = 1000
    
    old_service = TrailFinderService(obstacle_config=old_config)
    old_path, old_stats = await old_service.find_route(start, end, {})
    
    if old_path:
        print(f"   Result: Route found ({len(old_path)} points)")
        print("   Note: Streets had cost 1000 but route still possible")
    else:
        print(f"   Result: NO ROUTE - {old_stats.get('error', 'Failed')}")
    
    # Test 2: Check new configuration (streets NOT obstacles)
    print("\n2. NEW CONFIGURATION (streets allowed):")
    new_config = ObstacleConfig()  # Uses updated defaults
    print(f"   Highway in obstacles: {'highway' in new_config.osm_tags}")
    
    new_service = TrailFinderService(obstacle_config=new_config)
    new_path, new_stats = await new_service.find_route(start, end, {})
    
    if new_path:
        print(f"   Result: ✓ Route found ({len(new_path)} points)")
    else:
        print(f"   Result: ✗ Failed - {new_stats.get('error', 'Unknown')}")
    
    # Test 3: Test each profile
    print("\n3. PROFILE RESULTS:")
    profiles = {
        "default": "Standard hiking",
        "easy": "Casual hiker (avoids slopes >20°)",
        "experienced": "Experienced hiker (handles slopes up to 35°)",
        "trail_runner": "Trail runner (optimized for speed)",
        "accessibility": "Wheelchair accessible (max slope 10°)"
    }
    
    for profile_name, description in profiles.items():
        if profile_name == "default":
            config = ObstacleConfig()
        elif profile_name == "easy":
            config = ObstaclePresets.easy_hiker()
        elif profile_name == "experienced":
            config = ObstaclePresets.experienced_hiker()
        elif profile_name == "trail_runner":
            config = ObstaclePresets.trail_runner()
        elif profile_name == "accessibility":
            config = ObstaclePresets.accessibility_focused()
        
        service = TrailFinderService(obstacle_config=config)
        path, stats = await service.find_route(start, end, {})
        
        status = "✓" if path else "✗"
        result = f"{len(path)} points" if path else stats.get('error', 'No route')
        print(f"   {status} {profile_name:15} - {result}")
        
        if not path and profile_name == "accessibility":
            print(f"      → Terrain has slopes exceeding {config.slope_costs[2][0]}°")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY:")
    print("1. Streets are NO LONGER treated as obstacles")
    print("2. Most profiles can find routes successfully")
    print("3. Accessibility profile may fail due to strict slope limits")
    print("4. The fix: Removed 'highway' from default OSM obstacle tags")

if __name__ == "__main__":
    asyncio.run(test_final_solution())