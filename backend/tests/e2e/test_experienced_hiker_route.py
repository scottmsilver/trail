#!/usr/bin/env python3
"""Test experienced hiker route that fails"""

import asyncio
from app.services.trail_finder import TrailFinderService
from app.services.path_preferences import PathPreferencePresets

async def test_experienced_hiker_route():
    """Test the failing experienced hiker route"""
    
    # Coordinates that fail
    start = {"lat": 40.6568, "lon": -111.5713}
    end = {"lat": 40.6428, "lon": -111.5777}
    
    print(f"Testing route from {start} to {end}")
    print("Profile: Experienced Hiker (Mountain Goat)")
    
    # Get Mountain Goat preset
    preset = PathPreferencePresets.MOUNTAIN_GOAT
    
    # Create trail finder with Mountain Goat settings
    trail_finder = TrailFinderService(
        obstacle_config=preset['obstacle_config'],
        path_preferences=preset['path_preferences']
    )
    
    try:
        # Calculate route
        print("\nCalculating route...")
        result = await trail_finder.calculate_route(
            start=start,
            end=end,
            options={}
        )
        
        if result and result.get('path'):
            print(f"✓ Route found with {len(result['path'])} waypoints")
            print(f"  Distance: {result['stats']['distance_km']:.2f} km")
            print(f"  Elevation gain: {result['stats']['elevation_gain_m']:.0f} m")
            print(f"  Difficulty: {result['stats']['difficulty']}")
        else:
            print("✗ No route found!")
            
    except Exception as e:
        print(f"✗ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_experienced_hiker_route())