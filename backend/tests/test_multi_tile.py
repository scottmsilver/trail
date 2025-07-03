#!/usr/bin/env python3
"""Test multi-tile composition"""

import requests
import json

# Test with a route that spans multiple tiles
# Each tile is 0.01 degrees, so this spans about 2x2 tiles
start = {"lat": 40.650, "lon": -111.565}
end = {"lat": 40.665, "lon": -111.550}  # ~0.015° lat/lon difference

print("Testing multi-tile composition...")
print(f"Start: ({start['lat']}, {start['lon']})")
print(f"End: ({end['lat']}, {end['lon']})")
print(f"Distance: ~2.1 km (should span 2x2 tiles)")

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
    print("\nCheck backend logs for [TILE] messages!")
    print("Expected:")
    print("  - [TILE] Need 4 tiles for bounds...")
    print("  - [TILE COMPOSE] messages showing composition")
else:
    print(f"Error: {response.status_code} - {response.text}")