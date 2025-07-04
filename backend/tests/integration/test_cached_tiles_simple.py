#!/usr/bin/env python3
"""Simple test using specifically cached tiles"""

import requests
import time
import pytest

# We have tiles like tile_-11157_4065.pkl
# This corresponds to lat 40.65, lon -111.57
print("Testing route with already cached tiles...")
print("Cached tiles include: -11157_4065 (lat 40.65, lon -111.57)")

# Define a tiny route within a single tile
route = {
    "start": {"lat": 40.653, "lon": -111.573},
    "end": {"lat": 40.655, "lon": -111.571}
}

print(f"\nRoute: ({route['start']['lat']}, {route['start']['lon']}) -> ({route['end']['lat']}, {route['end']['lon']})")
print("This should use tile (-11157, 4065) which is already cached!")

# First run
print("\nFirst run:")
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

# Poll for completion
for i in range(120):  # 60 seconds timeout
    status = requests.get(f"http://localhost:9001/api/routes/{route_id}/status").json()
    
    if status["status"] == "completed":
        elapsed = time.time() - start_time
        print(f"✓ Completed in {elapsed:.2f}s")
        print("Check logs for [TILE CACHE HIT] messages!")
        break
    elif status["status"] == "failed":
        print(f"✗ Failed: {status.get('message', 'Unknown')}")
        exit(1)
    
    time.sleep(0.5)

# Second run (should be faster)
print("\nSecond run (same route):")
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
    
    for i in range(120):
        status2 = requests.get(f"http://localhost:9001/api/routes/{route_id2}/status").json()
        
        if status2["status"] == "completed":
            elapsed2 = time.time() - start_time2
            print(f"✓ Completed in {elapsed2:.2f}s")
            print(f"Speedup: {elapsed/elapsed2:.1f}x")
            break
        
        time.sleep(0.5)