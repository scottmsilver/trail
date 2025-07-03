#!/usr/bin/env python3
"""
Test route_cli.py to show cache behavior
"""

import subprocess
import time
import os

# The route you're testing
route_args = ["python", "route_cli.py", "Start: 40.6572, -111.5709", "End: 40.6472,-111.5671"]

print("=" * 70)
print("ROUTE CLI CACHE BEHAVIOR TEST")
print("=" * 70)

# Check for precomputed cache
if os.path.exists("precomputed_cache"):
    cache_files = [f for f in os.listdir("precomputed_cache") if f.endswith('_cost.pkl')]
    print(f"\nPrecomputed cache files found: {len(cache_files)}")
    for cf in cache_files[:5]:
        print(f"  - {cf}")
else:
    print("\nNo precomputed cache directory found")

# Check for tiled cache
if os.path.exists("tile_cache/cost"):
    tile_files = [f for f in os.listdir("tile_cache/cost") if f.endswith('.pkl')]
    print(f"\nTiled cache files found: {len(tile_files)}")
else:
    print("\nNo tiled cache found")

print("\n" + "-" * 70)
print("Running route calculation...")
print("Command:", " ".join(route_args))
print("-" * 70)

# Run the command and capture output
start_time = time.time()
result = subprocess.run(route_args, capture_output=True, text=True)
elapsed = time.time() - start_time

print("\nExecution time:", f"{elapsed:.2f}s")

# Check output for cache messages
output_lines = result.stdout.split('\n')
cache_messages = [line for line in output_lines if any(keyword in line for keyword in 
                  ["Loading precomputed cache", "CACHE HIT", "CACHE MISS", "TILE", "Computing cost surface"])]

print("\nCache-related messages:")
for msg in cache_messages:
    print(f"  {msg.strip()}")

# Show summary
print("\n" + "=" * 70)
print("ANALYSIS")
print("=" * 70)

if "Loading precomputed cache" in result.stdout:
    print("⚠️  Route is using PRECOMPUTED cache (monolithic file)")
    print("   This bypasses the tiled cache system!")
elif "[TILE" in result.stdout:
    print("✓ Route is using TILED cache system")
else:
    print("❌ Route is computing from scratch")

print("\nTo force tiled cache usage:")
print("1. Remove or rename precomputed_cache directory")
print("2. Or modify route_cli.py to skip _load_precomputed_caches()")