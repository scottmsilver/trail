#!/usr/bin/env python3
"""Test one route to see cache behavior"""

import requests
import json

# Simple route
start = {"lat": 40.6531, "lon": -111.5679}
end = {"lat": 40.6521, "lon": -111.5669}  # Very close points

print("Making route request...")
response = requests.post(
    "http://localhost:9001/api/routes/calculate",
    json={
        "start": start,
        "end": end,
        "options": {"userProfile": "moderate_hiker"}
    }
)

print(f"Response: {response.status_code}")
print(f"Data: {json.dumps(response.json(), indent=2)}")

if response.status_code == 202:
    route_id = response.json()["routeId"]
    print(f"\nRoute ID: {route_id}")
    print("Check backend logs for cache behavior!")