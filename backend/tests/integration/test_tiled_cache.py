#!/usr/bin/env python3
"""Test tiled cache with specific cached tiles"""

import requests
import time
import os
import pytest

# Check what tiles we have
print("Checking cached tiles...")
if os.path.exists("tile_cache/cost"):
    tiles = sorted([f for f in os.listdir("tile_cache/cost") if f.endswith('.pkl')])
    print(f"Found {len(tiles)} cached tiles")
    
    # Extract tile coverage
    tile_coords = []
    for tile in tiles[:10]:  # Sample first 10
        parts = tile.replace('tile_', '').replace('.pkl', '').split('_')
        if len(parts) == 2:
            x, y = int(parts[0]), int(parts[1])
            lat = y / 100.0
            lon = x / 100.0
            tile_coords.append((lat, lon))
    
    # Calculate tile coverage bounds
    if tile_coords:
        lats = [coord[0] for coord in tile_coords]
        lons = [coord[1] for coord in tile_coords]
        
        print(f"Sample tile coverage:")
        print(f"  Lat range: {min(lats):.2f} to {max(lats):.2f}")
        print(f"  Lon range: {min(lons):.2f} to {max(lons):.2f}")
        print(f"  Tiles: {tiles[:5]}...")

# Define a small test route within the cached area
# Based on tiles like (-11157, 4065), this is around lat 40.65, lon -111.57
route = {
    "start": {"lat": 40.652, "lon": -111.566},
    "end": {"lat": 40.654, "lon": -111.564}
}

print(f"\nTesting route:")
print(f"  From: ({route['start']['lat']}, {route['start']['lon']})")
print(f"  To: ({route['end']['lat']}, {route['end']['lon']})")
print(f"  Distance: ~0.3km (should use 1-4 tiles)")

# Run the route
start_time = time.time()

response = requests.post(
    "http://localhost:9001/api/routes/calculate",
    json={
        "start": route["start"],
        "end": route["end"],
        "options": {"userProfile": "moderate_hiker"}
    }
)

if response.status_code != 202:
    print(f"Error: {response.status_code}")
    exit(1)

route_id = response.json()["routeId"]
print(f"Route ID: {route_id}")

# Poll for completion
for i in range(60):  # 30 seconds timeout
    status = requests.get(f"http://localhost:9001/api/routes/{route_id}/status").json()
    
    if status["status"] == "completed":
        elapsed = time.time() - start_time
        # Get route details
        route_data = requests.get(f"http://localhost:9001/api/routes/{route_id}").json()
        distance = route_data["stats"].get("distance_km", 0)
        print(f"\n✓ Completed in {elapsed:.2f}s")
        print(f"  Distance: {distance:.2f}km")
        print(f"\nCheck backend logs for [TILE CACHE HIT] messages!")
        break
    elif status["status"] == "failed":
        print(f"\n✗ Failed: {status.get('message', 'Unknown')}")
        break
    
    time.sleep(0.5)
else:
    print("\n✗ Timeout")

print("\nNow testing the same route again (should use cached result)...")

# Run same route again
start_time2 = time.time()

response2 = requests.post(
    "http://localhost:9001/api/routes/calculate",
    json={
        "start": route["start"],
        "end": route["end"],
        "options": {"userProfile": "moderate_hiker"}
    }
)

if response2.status_code == 202:
    route_id2 = response2.json()["routeId"]
    
    for i in range(60):
        status2 = requests.get(f"http://localhost:9001/api/routes/{route_id2}/status").json()
        
        if status2["status"] == "completed":
            elapsed2 = time.time() - start_time2
            print(f"\n✓ Second run completed in {elapsed2:.2f}s")
            if elapsed > 0:
                speedup = elapsed / elapsed2
                print(f"  Speedup: {speedup:.1f}x")
            break
        elif status2["status"] == "failed":
            print(f"\n✗ Failed: {status2.get('message', 'Unknown')}")
            break
        
        time.sleep(0.5)