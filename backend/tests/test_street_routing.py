#!/usr/bin/env python
"""Test routing that should work but currently fails"""
import asyncio
from app.services.trail_finder import TrailFinderService
from app.models.route import RouteOptions
from app.services.obstacle_config import ObstacleConfig

async def test_street_routing():
    """Test the specific coordinates that fail"""
    # Coordinates provided by user
    start_lat, start_lon = 40.6482, -111.5738
    end_lat, end_lon = 40.6464, -111.5729
    
    print(f"Testing route from ({start_lat}, {start_lon}) to ({end_lat}, {end_lon})")
    print("Distance: ~200m")
    
    # Test with default configuration
    service = TrailFinderService()
    options = RouteOptions()
    
    print("\nTesting with default configuration...")
    route_id = await service.find_route_async(
        start_lat, start_lon, 
        end_lat, end_lon,
        options
    )
    
    # Wait for completion
    max_attempts = 10
    for i in range(max_attempts):
        result = service.get_route_result(route_id)
        if result and result.status != "processing":
            break
        await asyncio.sleep(1)
    
    if result and result.status == "completed":
        print(f"✓ Route found: {len(result.route)} points")
    else:
        print(f"✗ No route found - Status: {result.status if result else 'None'}")
        if result and result.error:
            print(f"  Error: {result.error}")
    
    # Let's also check what obstacles are in the area
    print("\nChecking obstacle configuration...")
    config = ObstacleConfig()
    print("OSM tags being used for obstacles:")
    for category, tags in config.osm_tags.items():
        print(f"  {category}: {tags}")
    
    # Test without any obstacles
    print("\nTesting with NO obstacles...")
    no_obstacle_config = ObstacleConfig(
        osm_tags={},  # No obstacles
        obstacle_costs={"default": 1.0},
        slope_costs=[(0, 1.0), (90, 100.0)]
    )
    
    # Create service with no obstacles
    service_no_obstacles = TrailFinderService()
    # We need to modify the service to use our config
    # Let's trace through the actual obstacle fetching
    
    return result

if __name__ == "__main__":
    asyncio.run(test_street_routing())