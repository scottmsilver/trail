# Route CLI and Tiled Caching Explanation

## Why route_cli.py Pauses on "Computing cost surface"

The issue is that `route_cli.py` uses a **cache hierarchy** that prioritizes monolithic precomputed cache files over the tiled cache system.

### Cache Hierarchy in route_cli.py:

1. **Precomputed Cache** (First Priority)
   - Loads from `precomputed_cache/` directory
   - Monolithic files like `40.5986,40.7072,-111.6206,-111.5139_cost.pkl`
   - Contains entire cost surface for large area (4359x3249 = 14M cells)
   - **This is what's being used in your case**

2. **Memory Cache** (Second Priority)
   - In-memory cache of previously computed areas

3. **Tiled Cache** (Third Priority)
   - Uses `TiledDEMCache` to compose from smaller tiles
   - Only used when no precomputed cache matches

4. **Compute from Scratch** (Last Resort)
   - Downloads DEM, fetches obstacles, computes cost surface

## The Issue

For your route (`40.6572, -111.5709` to `40.6472, -111.5671`):
- Uses 0.05° buffer = ~5.5km in each direction
- Creates bounds: `40.5972,40.7072,-111.6209,-111.5171`
- Needs 144 tiles (12x12 grid)

But `route_cli.py` finds a precomputed cache file that matches these bounds, so it:
1. Loads the large monolithic file (hence the pause)
2. Never uses the tiled cache system

## Solutions

### Option 1: Remove Precomputed Cache
```bash
mv precomputed_cache precomputed_cache.bak
python route_cli.py "Start: 40.6572, -111.5709" "End: 40.6472,-111.5671"
```

### Option 2: Use Modified CLI
```bash
python route_cli_tiled.py "Start: 40.6572, -111.5709" "End: 40.6472,-111.5671"
```

### Option 3: Reduce Buffer Size
Modify route_cli.py to use smaller buffer (e.g., 0.01° instead of 0.05°) to avoid loading huge areas.

## Performance Comparison

### With Precomputed Cache:
- Loads entire 14M cell cost surface from disk
- Slow initial load but then fast pathfinding
- Uses ~400MB+ memory

### With Tiled Cache:
- Composes only needed tiles
- Much faster initial load
- Uses less memory
- Better for varied routes

## Recommendation

For benchmarking the tiled cache system:
1. Use `route_cli_tiled.py` (disables precomputed cache)
2. Or temporarily rename the `precomputed_cache` directory
3. This will force the system to use the tiled cache

The tiled cache is more efficient for:
- Routes that don't match precomputed areas exactly
- Systems with limited memory
- Scenarios with many different route requests