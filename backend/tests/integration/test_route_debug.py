#!/usr/bin/env python3
"""Test route finding with debug output for coordinate transformation"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock the problematic import
import sys
import types

# Create mock modules
mock_py3dep = types.ModuleType('py3dep')
sys.modules['py3dep'] = mock_py3dep

# Now we can import the service
from app.services.dem_tile_cache import DEMTileCache
from app.services.obstacle_config import ObstacleConfig
from app.services.path_preferences import PathPreferences
import pytest

@pytest.mark.integration
def test_route():
    """Test route finding with verbose output"""
    lat1, lon1 = 40.6572, -111.5709
    lat2, lon2 = 40.6472, -111.5671
    
    print(f"\n=== Testing route from ({lat1}, {lon1}) to ({lat2}, {lon2}) ===\n")
    
    # Create cache with small buffer
    cache = DEMTileCache(
        buffer=0.02,  # 2km buffer
        obstacle_config=ObstacleConfig(),
        path_preferences=PathPreferences(),
        debug_mode=False
    )
    
    # Try to find the route
    try:
        route = cache.find_route(lat1, lon1, lat2, lon2)
        
        if route:
            print(f"\n✅ Route found with {len(route)} points")
        else:
            print(f"\n❌ No route found")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_route()