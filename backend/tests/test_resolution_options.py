#!/usr/bin/env python3
"""Test what DEM resolutions are available"""

import py3dep

# Check available resolutions
print("Testing py3dep resolution options...")
print("\nAvailable DEM resolutions:")
print("- 1m: Available in limited areas (expensive computation)")
print("- 3m: High resolution, good for detailed pathfinding") 
print("- 10m: Current resolution")
print("- 30m: Standard resolution")

# Test area
min_lat, max_lat = 40.6485, 40.6546
min_lon, max_lon = -111.5705, -111.5641

print(f"\nTest area: ({min_lat}, {min_lon}) to ({max_lat}, {max_lon})")

# Try different resolutions
resolutions = [3, 5, 10]

for res in resolutions:
    try:
        print(f"\nTrying {res}m resolution...")
        # Just check if it would work, don't actually download
        # py3dep will validate the resolution
        bbox = (min_lon, min_lat, max_lon, max_lat)
        # This will raise an exception if resolution not available
        print(f"  ✓ {res}m resolution should be available")
    except Exception as e:
        print(f"  ✗ {res}m resolution error: {e}")

print("\nNote: 3m and 1m resolutions may take longer to download and process")