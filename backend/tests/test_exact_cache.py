#!/usr/bin/env python3
"""Test cache with exact same coordinates"""

import requests
import time

# Test route - same as what we see in logs
start = {"lat": 40.6531, "lon": -111.5679}
end = {"lat": 40.6021, "lon": -111.5148}

print("Checking initial cache status...")
cache_status = requests.get("http://localhost:9001/api/cache/status").json()
print(f"Terrain cache entries: {cache_status['normal_cache']['terrain_cache']['count']}")
print(f"Cost surface cache entries: {cache_status['normal_cache']['cost_surface_cache']['count']}")

print("\nMaking first request...")
t1 = time.time()
response = requests.post(
    "http://localhost:9001/api/routes/calculate",
    json={
        "start": start,
        "end": end,
        "options": {"userProfile": "moderate_hiker"}
    }
)
route_id = response.json()["routeId"]
print(f"Route ID: {route_id}")

# Wait for completion
completed = False
for i in range(60):  # 30 seconds max
    status = requests.get(f"http://localhost:9001/api/routes/{route_id}/status").json()
    if status["status"] == "completed":
        completed = True
        break
    elif status["status"] == "failed":
        print(f"Route failed: {status.get('message', 'Unknown error')}")
        break
    time.sleep(0.5)

if completed:
    t1_elapsed = time.time() - t1
    print(f"First request completed in: {t1_elapsed:.2f}s")
    
    # Check cache status again
    cache_status = requests.get("http://localhost:9001/api/cache/status").json()
    print(f"\nAfter first request:")
    print(f"Terrain cache entries: {cache_status['normal_cache']['terrain_cache']['count']}")
    print(f"Cost surface cache entries: {cache_status['normal_cache']['cost_surface_cache']['count']}")
    
    print("\nMaking second identical request...")
    t2 = time.time()
    response = requests.post(
        "http://localhost:9001/api/routes/calculate",
        json={
            "start": start,
            "end": end,
            "options": {"userProfile": "moderate_hiker"}
        }
    )
    route_id = response.json()["routeId"]
    print(f"Route ID: {route_id}")
    
    # Wait for completion
    for i in range(60):  # 30 seconds max
        status = requests.get(f"http://localhost:9001/api/routes/{route_id}/status").json()
        if status["status"] == "completed":
            break
        elif status["status"] == "failed":
            print(f"Route failed: {status.get('message', 'Unknown error')}")
            break
        time.sleep(0.5)
    
    t2_elapsed = time.time() - t2
    print(f"Second request completed in: {t2_elapsed:.2f}s")
    
    if t2_elapsed > 0:
        print(f"\nSpeedup: {t1_elapsed/t2_elapsed:.1f}x")
    
print("\nCheck the backend logs to see cache hit/miss patterns!")