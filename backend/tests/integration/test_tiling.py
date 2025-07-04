#!/usr/bin/env python3
"""Test tiling functionality"""

import requests
import json
import pytest

# Test with a route that should fit in a single tile
# 0.01 degrees is about 1.1km, so these points are ~0.15km apart
start = {"lat": 40.6531, "lon": -111.5679}
end = {"lat": 40.6535, "lon": -111.5675}  # Very close, should be in same tile

print("Testing tiled caching with close points...")
print(f"Start: ({start['lat']}, {start['lon']})")
print(f"End: ({end['lat']}, {end['lon']})")
print(f"Distance: ~0.5 km (should fit in one 0.01° tile)")

response = requests.post(
    "http://localhost:9001/api/routes/calculate",
    json={
        "start": start,
        "end": end,
        "options": {"userProfile": "moderate_hiker"}
    }
)

if response.status_code == 202:
    route_id = response.json()["routeId"]
    print(f"\nRoute ID: {route_id}")
    print("\nCheck backend logs for [TILE] prefixed messages!")
    print("Expected to see: [TILE] Need 1 tiles for bounds...")
else:
    print(f"Error: {response.status_code} - {response.text}")