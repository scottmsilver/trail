#!/usr/bin/env python
"""Debug path following issues"""
import asyncio
from app.services.dem_tile_cache import DEMTileCache
from app.services.obstacle_config import ObstacleConfig
from app.services.path_preferences import PathPreferences
from shapely.geometry import box
import osmnx as ox

async def test_path_debug():
    """Debug what's happening with path fetching"""
    # Test area
    lat1, lon1 = 40.6482, -111.5738
    lat2, lon2 = 40.6464, -111.5729
    
    print("PATH FETCHING DEBUG")
    print("="*60)
    
    # Create simple path preferences
    path_prefs = PathPreferences()
    print(f"Path tags to fetch: {list(path_prefs.preferred_path_tags.keys())}")
    
    # Try to fetch paths directly
    min_lat = min(lat1, lat2) - 0.005
    max_lat = max(lat1, lat2) + 0.005
    min_lon = min(lon1, lon2) - 0.005
    max_lon = max(lon1, lon2) + 0.005
    
    bbox = box(min_lon, min_lat, max_lon, max_lat)
    
    print(f"\nFetching paths in area: ({min_lat:.4f}, {min_lon:.4f}) to ({max_lat:.4f}, {max_lon:.4f})")
    
    try:
        ox.settings.log_console = False
        paths = ox.features_from_polygon(bbox, path_prefs.preferred_path_tags)
        print(f"✓ Found {len(paths)} path features")
        
        if len(paths) > 0:
            # Count by type
            highway_count = paths['highway'].notna().sum() if 'highway' in paths else 0
            leisure_count = paths['leisure'].notna().sum() if 'leisure' in paths else 0
            route_count = paths['route'].notna().sum() if 'route' in paths else 0
            
            print(f"  - Highway paths: {highway_count}")
            print(f"  - Leisure areas: {leisure_count}")
            print(f"  - Routes: {route_count}")
            
            if 'highway' in paths and highway_count > 0:
                print("\nHighway types found:")
                print(paths['highway'].value_counts().head(10))
                
    except Exception as e:
        print(f"✗ Error fetching paths: {e}")
        print("\nThis might mean:")
        print("  - No paths in the area")
        print("  - OSM query issue")
        print("  - Network timeout")
    
    # Test DEM cache with paths
    print("\n" + "-"*60)
    print("Testing DEM cache with path preferences...")
    
    try:
        dem_cache = DEMTileCache(path_preferences=path_prefs)
        path = dem_cache.find_route(lat1, lon1, lat2, lon2)
        
        if path:
            print(f"✓ Route found: {len(path)} points")
        else:
            print("✗ No route found")
            
    except Exception as e:
        print(f"✗ Error in routing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_path_debug())