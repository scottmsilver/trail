#!/usr/bin/env python
"""Test and debug why streets are blocking routes"""
import numpy as np
from app.services.dem_tile_cache import DEMTileCache
from app.services.obstacle_config import ObstacleConfig
import osmnx as ox
from shapely.geometry import box
import pytest

@pytest.mark.real_data
@pytest.mark.slow
def test_specific_area():
    """Test the specific coordinates where routing fails"""
    # User's coordinates
    start_lat, start_lon = 40.6482, -111.5738
    end_lat, end_lon = 40.6464, -111.5729
    
    print(f"Testing route from ({start_lat}, {start_lon}) to ({end_lat}, {end_lon})")
    
    # Define area with buffer
    buffer = 0.005  # Small buffer for testing
    min_lat = min(start_lat, end_lat) - buffer
    max_lat = max(start_lat, end_lat) + buffer
    min_lon = min(start_lon, end_lon) - buffer
    max_lon = max(start_lon, end_lon) + buffer
    
    # Create bounding box
    bbox = box(min_lon, min_lat, max_lon, max_lat)
    
    # Test 1: Check what OSM features are in the area
    print("\n1. Checking OSM features in the area...")
    config = ObstacleConfig()
    
    try:
        # Fetch features using default tags
        ox.settings.log_console = False
        features = ox.features_from_polygon(bbox, config.osm_tags)
        
        print(f"Found {len(features)} features")
        
        # Group by type
        feature_types = {}
        for idx, row in features.iterrows():
            for tag_type in config.osm_tags.keys():
                if tag_type in row and pd.notna(row[tag_type]):
                    feature_types[tag_type] = feature_types.get(tag_type, 0) + 1
        
        print("Feature breakdown:")
        for ftype, count in feature_types.items():
            print(f"  {ftype}: {count}")
            
        # Check specifically for highways
        if 'highway' in features.columns:
            highways = features[features['highway'].notna()]
            print(f"\nHighways found: {len(highways)}")
            if len(highways) > 0:
                print("Highway types:")
                print(highways['highway'].value_counts())
                
    except Exception as e:
        print(f"Error fetching features: {e}")
    
    # Test 2: Try routing with default config
    print("\n2. Testing with DEFAULT configuration...")
    pathfinder_default = DEMTileCache()
    route_default = pathfinder_default.find_route(start_lat, start_lon, end_lat, end_lon)
    
    if route_default:
        print(f"✓ Route found with default config: {len(route_default)} points")
    else:
        print("✗ No route found with default config")
    
    # Test 3: Try routing with NO highway obstacles
    print("\n3. Testing with STREETS ALLOWED configuration...")
    no_highway_config = ObstacleConfig()
    # Remove highways from obstacle tags
    no_highway_config.osm_tags = {
        'natural': ['water', 'wetland', 'cliff', 'rock', 'scree'],
        'waterway': ['river', 'stream', 'canal'],
        'building': True,
        'barrier': True,
        # Removed: 'highway', 'landuse' (residential), etc.
    }
    # Lower costs for any remaining obstacles
    no_highway_config.obstacle_costs['default'] = 10
    
    pathfinder_no_highway = DEMTileCache(obstacle_config=no_highway_config)
    route_no_highway = pathfinder_no_highway.find_route(start_lat, start_lon, end_lat, end_lon)
    
    if route_no_highway:
        print(f"✓ Route found without highway obstacles: {len(route_no_highway)} points")
    else:
        print("✗ Still no route found")
    
    # Test 4: Try with minimal obstacles
    print("\n4. Testing with MINIMAL obstacles...")
    minimal_config = ObstacleConfig(
        osm_tags={
            'natural': ['water', 'cliff'],
            'building': True,
        },
        obstacle_costs={
            'water': 5000,
            'cliff': np.inf,
            'building': 10000,
            'default': 2
        },
        slope_costs=[(0, 1.0), (45, 10.0), (90, 100.0)]
    )
    
    pathfinder_minimal = DEMTileCache(obstacle_config=minimal_config)
    route_minimal = pathfinder_minimal.find_route(start_lat, start_lon, end_lat, end_lon)
    
    if route_minimal:
        print(f"✓ Route found with minimal obstacles: {len(route_minimal)} points")
    else:
        print("✗ Still no route with minimal obstacles")
        
    # Test 5: Debug mode to see what's blocking
    print("\n5. Running in DEBUG mode to see what's happening...")
    debug_config = ObstacleConfig(
        osm_tags={},  # No obstacles at all
        obstacle_costs={'default': 1.0},
        slope_costs=[(0, 1.0), (90, 1.0)]  # Flat costs
    )
    
    pathfinder_debug = DEMTileCache(debug_mode=True, obstacle_config=debug_config)
    route_debug = pathfinder_debug.find_route(start_lat, start_lon, end_lat, end_lon)
    
    if route_debug:
        print(f"✓ Route found with NO obstacles: {len(route_debug)} points")
    else:
        print("✗ No route even with no obstacles - might be DEM or coordinate issue")
        
    return {
        'default': route_default is not None,
        'no_highway': route_no_highway is not None,
        'minimal': route_minimal is not None,
        'no_obstacles': route_debug is not None
    }

if __name__ == "__main__":
    import pandas as pd  # Import here to avoid issues if not needed elsewhere
    results = test_specific_area()
    
    print("\n" + "="*50)
    print("SUMMARY:")
    print("="*50)
    for config_name, success in results.items():
        status = "✓" if success else "✗"
        print(f"{status} {config_name}: {'SUCCESS' if success else 'FAILED'}")
        
    if not results['no_obstacles']:
        print("\n⚠️  Route fails even with NO obstacles - this suggests:")
        print("  - Coordinates might be outside DEM coverage")
        print("  - DEM download might have failed")
        print("  - Coordinates might be in impassable terrain (water body, etc.)")
    elif not results['default'] and results['no_highway']:
        print("\n✓ Streets/highways are the problem! Need to fix default configuration.")