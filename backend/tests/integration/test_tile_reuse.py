#!/usr/bin/env python3
"""Test tile reuse - routes in same area should reuse tiles"""

import requests
import time
import pytest

# Two different routes in the same general area
# Both should use some of the same tiles
route1 = {
    "start": {"lat": 40.653, "lon": -111.567},
    "end": {"lat": 40.655, "lon": -111.565}
}

route2 = {
    "start": {"lat": 40.654, "lon": -111.566},
    "end": {"lat": 40.656, "lon": -111.564}
}

print("Testing tile reuse between different routes...")
print("\nRoute 1: Small route to cache some tiles")
print(f"  From: ({route1['start']['lat']}, {route1['start']['lon']})")
print(f"  To: ({route1['end']['lat']}, {route1['end']['lon']})")

# First route - will compute tiles
t1 = time.time()
response = requests.post(
    "http://localhost:9001/api/routes/calculate",
    json={
        "start": route1["start"],
        "end": route1["end"],
        "options": {"userProfile": "moderate_hiker"}
    }
)

if response.status_code == 202:
    route_id = response.json()["routeId"]
    print(f"  Route ID: {route_id}")
    
    # Wait for completion
    while True:
        status = requests.get(f"http://localhost:9001/api/routes/{route_id}/status").json()
        if status["status"] in ["completed", "failed"]:
            break
        time.sleep(0.5)
    
    t1_elapsed = time.time() - t1
    print(f"  Time: {t1_elapsed:.2f}s")

print("\nRoute 2: Nearby route that should reuse tiles")
print(f"  From: ({route2['start']['lat']}, {route2['start']['lon']})")
print(f"  To: ({route2['end']['lat']}, {route2['end']['lon']})")

# Second route - should reuse tiles
t2 = time.time()
response = requests.post(
    "http://localhost:9001/api/routes/calculate",
    json={
        "start": route2["start"],
        "end": route2["end"],
        "options": {"userProfile": "moderate_hiker"}
    }
)

if response.status_code == 202:
    route_id = response.json()["routeId"]
    print(f"  Route ID: {route_id}")
    
    # Wait for completion
    while True:
        status = requests.get(f"http://localhost:9001/api/routes/{route_id}/status").json()
        if status["status"] in ["completed", "failed"]:
            break
        time.sleep(0.5)
    
    t2_elapsed = time.time() - t2
    print(f"  Time: {t2_elapsed:.2f}s")

if t1_elapsed > 0 and t2_elapsed > 0:
    print(f"\nSpeedup from tile reuse: {t1_elapsed/t2_elapsed:.1f}x")
    print("\nCheck backend logs to see:")
    print("  - Route 1: [TILE CACHE MISS] messages")
    print("  - Route 2: [TILE CACHE HIT] messages for overlapping tiles")