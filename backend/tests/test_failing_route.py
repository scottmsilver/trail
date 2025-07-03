#!/usr/bin/env python3
"""Test why a specific route is failing"""

import asyncio
from app.services.trail_finder import TrailFinderService
from app.services.obstacle_config import ObstacleConfig
from app.services.path_preferences import PathPreferences
from app.models.route import Coordinate

async def test_failing_route():
    # Failing route coordinates
    start = Coordinate(lat=40.6566, lon=-111.5701)
    end = Coordinate(lat=40.6286, lon=-111.5689)
    
    print(f"Testing failing route")
    print(f"Start: {start.lat}, {start.lon}")
    print(f"End: {end.lat}, {end.lon}")
    print("=" * 60)
    
    # Calculate approximate distance
    from geopy.distance import distance
    direct_distance = distance((start.lat, start.lon), (end.lat, end.lon)).km
    print(f"Direct distance: {direct_distance:.2f} km")
    
    # Try with different configurations
    configs = [
        ("Default config", ObstacleConfig(), PathPreferences()),
        ("No steep slope limit", create_lenient_config(), PathPreferences()),
        ("Direct route", ObstacleConfig(), create_direct_preferences()),
    ]
    
    for config_name, obstacle_config, path_prefs in configs:
        print(f"\n{config_name}:")
        print("-" * 40)
        
        obstacle_config.use_continuous_slope = True
        
        trail_finder = TrailFinderService(
            obstacle_config=obstacle_config,
            path_preferences=path_prefs
        )
        
        try:
            path, stats = await trail_finder.find_route(start, end, {})
            
            if not path:
                error = stats.get('error', 'Unknown error')
                print(f"Failed: {error}")
                
                # Check if there's more debug info
                if 'iterations' in stats:
                    print(f"A* iterations: {stats['iterations']}")
                if 'search_area' in stats:
                    print(f"Search area: {stats['search_area']} cells")
            else:
                print(f"Success! Distance: {stats.get('distance_km', 0):.2f} km")
                print(f"Points: {len(path)}")
                
        except Exception as e:
            print(f"Exception: {str(e)}")
            import traceback
            traceback.print_exc()

def create_lenient_config():
    """Create a config with very lenient slope limits"""
    config = ObstacleConfig()
    # Override slope costs to be very lenient
    config.slope_costs = [
        (0, 1.0),
        (20, 2.0),
        (40, 5.0),
        (60, 10.0),
        (80, 100.0),
        (90, 1000.0)
    ]
    return config

def create_direct_preferences():
    """Create preferences for most direct route"""
    prefs = PathPreferences()
    # Make all path types equal cost
    prefs.path_costs = {k: 1.0 for k in prefs.path_costs}
    prefs.stick_to_paths = False
    prefs.path_transition_penalty = 1.0
    return prefs

if __name__ == "__main__":
    asyncio.run(test_failing_route())