#!/usr/bin/env python3
"""Simple test to check cache behavior"""

import requests
import time

# Test route
start = {"lat": 40.6531, "lon": -111.5679}
end = {"lat": 40.6021, "lon": -111.5148}

print("Making first request...")
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

# Wait for completion
while True:
    status = requests.get(f"http://localhost:9001/api/routes/{route_id}/status").json()
    if status["status"] in ["completed", "failed"]:
        break
    time.sleep(0.1)

t1_elapsed = time.time() - t1
print(f"First request took: {t1_elapsed:.2f}s")

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

# Wait for completion
while True:
    status = requests.get(f"http://localhost:9001/api/routes/{route_id}/status").json()
    if status["status"] in ["completed", "failed"]:
        break
    time.sleep(0.1)

t2_elapsed = time.time() - t2
print(f"Second request took: {t2_elapsed:.2f}s")

print(f"\nSpeedup: {t1_elapsed/t2_elapsed:.1f}x")
print("\nCheck the backend logs to see cache hit/miss status!")