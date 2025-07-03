#!/usr/bin/env python
import numpy as np
from app.services.dem_tile_cache import DEMTileCache
from app.services.obstacle_config import ObstaclePresets
from rasterio.transform import from_origin

# Create test terrain
rows, cols = 10, 10
dem = np.zeros((rows, cols))

# Create a steep slope from row 0 (0m) to row 9 (90m)
for i in range(rows):
    dem[i, :] = i * 10  # 10m per row

print("DEM (elevation in meters):")
print(dem)

# Setup transform (30m cells)
transform = from_origin(-111.5, 40.5, 30, 30)
obstacle_mask = np.zeros((rows, cols), dtype=bool)

# Calculate slope manually
cell_size_x = transform.a
cell_size_y = -transform.e
dzdx, dzdy = np.gradient(dem, cell_size_x, cell_size_y)
slope_radians = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
slope_degrees = np.degrees(slope_radians)

print("\nSlope (degrees):")
print(slope_degrees)

# Test with accessibility config
access_config = ObstaclePresets.accessibility_focused()
pathfinder = DEMTileCache(obstacle_config=access_config)
cost_surface = pathfinder.compute_cost_surface(dem, transform, obstacle_mask)

print("\nCost surface (accessibility):")
print(cost_surface)

# Check slope cost mapping
print("\nSlope cost mapping for accessibility:")
for slope in [0, 2, 5, 10, 20, 30]:
    cost = access_config.get_slope_cost_multiplier(slope)
    print(f"  {slope}° → cost {cost}")