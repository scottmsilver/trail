#!/usr/bin/env python3
"""Test route_cli.py performance with tiled cache"""

import subprocess
import time

def run_route(iteration):
    """Run route_cli.py and capture timing"""
    cmd = [
        "python", "route_cli.py",
        "Start: 40.6572, -111.5709",
        "End: 40.6472,-111.5671"
    ]
    
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start
    
    # Extract key timing from output
    compute_time = None
    total_time = None
    
    for line in result.stdout.split('\n'):
        if "Computing cost surface..." in line:
            # Next line should have the timing
            continue
        if "✓ Completed in" in line and compute_time is None and "Computing" in prev_line:
            # Extract time from this line
            parts = line.split("✓ Completed in")[1].strip()
            compute_time = parts.split('s')[0]
        if "Total time:" in line:
            parts = line.split("Total time:")[1].strip()
            total_time = parts.split('s')[0]
        prev_line = line
    
    # Also check for specific messages
    using_tiles = "[TILE" in result.stdout
    using_precomputed = "Loading precomputed cache" in result.stdout
    using_tiled_system = "Using tiled cache system" in result.stdout
    
    return {
        'iteration': iteration,
        'elapsed': elapsed,
        'compute_time': compute_time,
        'total_time': total_time,
        'using_tiles': using_tiles,
        'using_precomputed': using_precomputed,
        'using_tiled_system': using_tiled_system
    }

print("ROUTE CLI PERFORMANCE TEST")
print("=" * 60)
print("Route: Start: 40.6572, -111.5709  End: 40.6472,-111.5671")
print("Buffer: 0.02° (~2.2km)")
print("=" * 60)

# Run multiple times
results = []
for i in range(3):
    print(f"\nRun {i+1}...", end='', flush=True)
    result = run_route(i+1)
    results.append(result)
    print(f" {result['elapsed']:.1f}s")
    
    if result['using_tiled_system']:
        print("  ✓ Using tiled cache system")
    if result['compute_time']:
        print(f"  Cost surface computation: {result['compute_time']}")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

times = [r['elapsed'] for r in results]
print(f"Average time: {sum(times)/len(times):.1f}s")
print(f"Best time: {min(times):.1f}s")

if results[0]['elapsed'] > results[-1]['elapsed'] * 1.5:
    speedup = results[0]['elapsed'] / results[-1]['elapsed']
    print(f"Speedup from caching: {speedup:.1f}x")

print("\nCache system status:")
print(f"  Using tiled cache: {'Yes' if results[0]['using_tiled_system'] else 'No'}")
print(f"  Using precomputed: {'Yes' if results[0]['using_precomputed'] else 'No'}")