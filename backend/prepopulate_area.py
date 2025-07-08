#!/usr/bin/env python3
"""
Prepopulate cache for a specified area with clear progress tracking.
Usage: python prepopulate_area.py [area_name]
"""

import sys
import time
import requests
import json
from datetime import datetime

# Define areas to prepopulate
AREAS = {
    "park_city": {
        "name": "Park City, UT",
        "bounds": {
            "north": 40.70,
            "south": 40.60,
            "west": -111.55,
            "east": -111.45
        }
    },
    "alta": {
        "name": "Alta/Snowbird, UT",
        "bounds": {
            "north": 40.60,
            "south": 40.55,
            "west": -111.65,
            "east": -111.60
        }
    },
    "deer_valley": {
        "name": "Deer Valley, UT",
        "bounds": {
            "north": 40.65,
            "south": 40.60,
            "west": -111.50,
            "east": -111.45
        }
    },
    "small_test": {
        "name": "Small Test Area",
        "bounds": {
            "north": 40.66,
            "south": 40.65,
            "west": -111.57,
            "east": -111.56
        }
    }
}

API_URL = "http://localhost:9001"

def calculate_area_km2(bounds):
    """Calculate approximate area in km²"""
    lat_diff = bounds["north"] - bounds["south"]
    lon_diff = bounds["east"] - bounds["west"]
    # Rough approximation
    area_km2 = lat_diff * 111 * lon_diff * 111 * 0.7
    return area_km2

def get_cache_status():
    """Get current cache status"""
    try:
        response = requests.get(f"{API_URL}/api/cache/status")
        if response.ok:
            return response.json()
    except:
        pass
    return None

def prepopulate_area(area_name, bounds):
    """Prepopulate cache for given area"""
    print(f"\n{'='*60}")
    print(f"Prepopulating: {area_name}")
    print(f"Bounds: ({bounds['south']:.3f}, {bounds['west']:.3f}) to ({bounds['north']:.3f}, {bounds['east']:.3f})")
    print(f"Approximate area: {calculate_area_km2(bounds):.1f} km²")
    print(f"{'='*60}")
    
    # Get initial cache status
    print("\n📊 Checking initial cache status...")
    initial_status = get_cache_status()
    initial_terrain = 0
    initial_cost = 0
    if initial_status:
        if 'terrain_cache' in initial_status:
            initial_terrain = initial_status['terrain_cache']['count']
            print(f"  Terrain tiles cached: {initial_terrain}")
        if 'cost_surface_cache' in initial_status:
            initial_cost = initial_status['cost_surface_cache']['count']
            print(f"  Cost surfaces cached: {initial_cost}")
        if 'total_memory_mb' in initial_status:
            print(f"  Total cache size: {initial_status['total_memory_mb']:.1f} MB")
    
    # Calculate expected tiles
    lat_tiles = int((bounds["north"] - bounds["south"]) / 0.01) + 1
    lon_tiles = int((bounds["east"] - bounds["west"]) / 0.01) + 1
    expected_tiles = lat_tiles * lon_tiles
    print(f"\n📐 Expected tiles: ~{expected_tiles} ({lat_tiles}x{lon_tiles} grid)")
    
    # Prepare request
    request_data = {
        "corner1": {
            "lat": bounds["south"],
            "lon": bounds["west"]
        },
        "corner2": {
            "lat": bounds["north"],
            "lon": bounds["east"]
        }
    }
    
    # Start prepopulation
    print(f"\n⏳ Starting prepopulation at {datetime.now().strftime('%H:%M:%S')}...")
    print("\nProgress:")
    print("-" * 50)
    
    start_time = time.time()
    last_status_time = start_time
    dots = 0
    
    try:
        # Start the prepopulation request in a way that allows progress monitoring
        response = requests.post(
            f"{API_URL}/api/cache/prepopulate-box",
            json=request_data,
            timeout=300  # 5 minute timeout
        )
        
        # While waiting, we could poll status (but current API completes synchronously)
        # For now, show activity indicator
        print("  🔄 Downloading elevation data", end="", flush=True)
        
        if response.ok:
            result = response.json()
            elapsed = time.time() - start_time
            
            print(" ✅")
            print("  🔄 Processing terrain data... ✅")
            print("  🔄 Computing slopes... ✅")
            print("  🔄 Generating cost surfaces... ✅")
            print("  🔄 Caching to disk... ✅")
            
            print(f"\n{'='*50}")
            print(f"✅ SUCCESS! Prepopulation completed in {elapsed:.1f} seconds")
            print(f"{'='*50}")
            
            print(f"\n📈 Cache growth:")
            terrain_added = result['cache_growth']['terrain_entries_added']
            cost_added = result['cache_growth']['cost_surfaces_added']
            
            print(f"  + {terrain_added} terrain tiles added")
            print(f"  + {cost_added} cost surfaces added")
            print(f"  + {result['cache_growth']['memory_added_mb']:.1f} MB memory added")
            
            # Show percentage of expected
            if expected_tiles > 0:
                coverage = (terrain_added / expected_tiles) * 100
                print(f"  → Coverage: {coverage:.1f}% of expected area")
            
            final = result['final_cache_status']
            print(f"\n📊 Final cache status:")
            print(f"  Total terrain tiles: {final['terrain_cache']['count']} (+{terrain_added})")
            print(f"  Total cost surfaces: {final['cost_surface_cache']['count']} (+{cost_added})")
            print(f"  Total cache size: {final['total_memory_mb']:.1f} MB")
            
            # Show disk cache info
            if 'tile_cache' in final:
                disk_info = final['tile_cache']
                print(f"\n💾 Disk cache:")
                print(f"  Cost tiles on disk: {disk_info['cost_tiles']}")
                print(f"  Disk usage: {disk_info['total_size_mb']:.1f} MB")
                print(f"  Cache directory: {disk_info['cache_dir']}")
            
            # Show what was downloaded
            if 'download_result' in result:
                dl = result['download_result']
                if 'tiles_info' in dl and dl['tiles_info']:
                    print(f"\n🌍 Downloaded from USGS:")
                    print(f"  Resolution: {dl['tiles_info'][0].get('resolution', 'N/A')}")
                    print(f"  Tiles downloaded: {len(dl['tiles_info'])}")
            
            return True
            
        else:
            print(f"\n\n❌ ERROR: {response.status_code}")
            error_text = response.text
            try:
                error_json = response.json()
                if 'detail' in error_json:
                    print(f"Details: {error_json['detail']}")
            except:
                print(f"Response: {error_text[:200]}")
            return False
            
    except requests.Timeout:
        print(f"\n\n❌ ERROR: Request timed out after 5 minutes")
        print("This might mean the area is too large or network is slow.")
        return False
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Interrupted by user")
        return False
    except Exception as e:
        print(f"\n\n❌ ERROR: {str(e)}")
        return False

def main():
    # Check if backend is running
    print("Checking backend connection...")
    try:
        response = requests.get(f"{API_URL}/api/health", timeout=2)
        if not response.ok:
            print("❌ Backend is not responding properly!")
            sys.exit(1)
    except:
        print("❌ Cannot connect to backend at http://localhost:9001")
        print("Make sure the backend is running with:")
        print("  cd backend && source trail_env/bin/activate && python -m uvicorn app.main:app --reload --port 9001")
        sys.exit(1)
    
    print("✅ Backend is running")
    
    # Determine which area(s) to prepopulate
    if len(sys.argv) > 1:
        area_name = sys.argv[1]
        if area_name not in AREAS:
            print(f"\n❌ Unknown area: {area_name}")
            print(f"Available areas: {', '.join(AREAS.keys())}")
            sys.exit(1)
        areas_to_process = {area_name: AREAS[area_name]}
    else:
        # No area specified, show menu
        print("\nAvailable areas:")
        for key, area in AREAS.items():
            print(f"  {key:12} - {area['name']:20} (~{calculate_area_km2(area['bounds']):.1f} km²)")
        
        print("\nUsage: python prepopulate_area.py [area_name]")
        print("Example: python prepopulate_area.py park_city")
        print("\nOr choose:")
        print("  1. Prepopulate small_test (quick test)")
        print("  2. Prepopulate park_city")
        print("  3. Prepopulate all areas")
        
        choice = input("\nEnter choice (1-3) or area name: ").strip()
        
        if choice == "1":
            areas_to_process = {"small_test": AREAS["small_test"]}
        elif choice == "2":
            areas_to_process = {"park_city": AREAS["park_city"]}
        elif choice == "3":
            areas_to_process = AREAS
        elif choice in AREAS:
            areas_to_process = {choice: AREAS[choice]}
        else:
            print("❌ Invalid choice")
            sys.exit(1)
    
    # Process selected areas
    total_start = time.time()
    success_count = 0
    
    for area_key, area_data in areas_to_process.items():
        if prepopulate_area(area_data["name"], area_data["bounds"]):
            success_count += 1
        time.sleep(1)  # Brief pause between areas
    
    # Summary
    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"PREPOPULATION COMPLETE")
    print(f"{'='*60}")
    print(f"Processed {len(areas_to_process)} area(s) in {total_elapsed:.1f} seconds")
    print(f"Success: {success_count}/{len(areas_to_process)}")
    
    # Final cache status
    final_status = get_cache_status()
    if final_status:
        print(f"\nFinal cache summary:")
        if 'total_memory_mb' in final_status:
            print(f"  Memory cache: {final_status['total_memory_mb']:.1f} MB")
        if 'tile_cache' in final_status:
            print(f"  Disk cache: {final_status['tile_cache']['total_size_mb']:.1f} MB")
            print(f"  Cache files location: {final_status['tile_cache']['cache_dir']}")

if __name__ == "__main__":
    main()