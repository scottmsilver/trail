# Compressed Pathfinding Implementation

## Problem Solved
Routes failing at 3m resolution due to massive search spaces (15+ million cells).

## Solution
Implemented hierarchical pathfinding that compresses similar terrain into regions.

## How It Works

### 1. Region Creation
- Groups adjacent cells with similar slopes (within 5° difference)
- Similar costs (within 2x ratio)
- Minimum region size: 9 cells
- Creates a graph where nodes are regions instead of individual cells

### 2. Two-Phase Search
- **Phase 1**: High-level A* search on region graph (~100K nodes)
- **Phase 2**: Local A* to refine paths within/between regions

### 3. Automatic Activation
- Triggers when grid distance > 500 cells
- Falls back to regular A* if compressed search fails

## Results

### Test Route: (40.6566, -111.5701) → (40.6286, -111.5689)
- Direct distance: 3.11 km
- Regular A*: **Failed** (10M iterations, timeout)
- Compressed A*: **Success** in 64 seconds!
  - Compression: 15.4M cells → 135K regions (113:1)
  - Route found: 8.25 km, max slope 32.3°

## Performance Benefits
1. **Massive search space reduction** - 100x+ compression typical
2. **Long routes now possible** - Can handle 3+ km routes at 3m resolution
3. **Maintains quality** - Still finds good paths avoiding steep slopes
4. **Automatic** - No user configuration needed

## Implementation Details
- `compressed_pathfinding.py` - Core compression algorithm
- `dem_tile_cache.py` - Integration with existing pathfinding
- Threshold: Grid distance > 500 triggers compression
- Compatible with all existing features (slopes, obstacles, path preferences)

This allows the system to handle much longer routes at high resolution while maintaining the quality benefits of 3m DEM data.