# Two-Layer Elevation System

A production-ready elevation data system that eliminates tile boundary artifacts while providing efficient data access.

## Overview

The two-layer elevation system solves the tile boundary artifact problem by:
- **Layer 1**: Downloads large seamless regions from py3dep (0.1° x 0.1°)
- **Layer 2**: Creates smaller tiles for efficient queries (size varies by resolution)

## Features

- ✅ No tile boundary artifacts
- ✅ Explicit load/fail behavior (no automatic downloading)
- ✅ Multiple resolution support (1m, 3m, 5m, 10m, 30m, 60m)
- ✅ Efficient tile-based storage
- ✅ Compatible with single-layer API

## Installation

```python
from elevation import TwoLayerElevationLibrary, Bounds
```

## Usage

```python
# Initialize library
lib = TwoLayerElevationLibrary(data_dir="./elevation_data", resolution=10)

# Define area of interest
bounds = Bounds(
    south=40.6448,
    north=40.6588,
    west=-111.5780,
    east=-111.5595
)

# Explicitly load data
result = lib.load_area(bounds)
print(f"Loaded {result['total_tiles']} tiles")

# Query elevation at a point
elevation = lib.get_elevation(40.65, -111.57)
print(f"Elevation: {elevation}m")

# Get elevation array
array, metadata = lib.get_elevation_array(bounds)
print(f"Array shape: {array.shape}")

# List loaded areas
areas = lib.list_loaded_areas()
for area in areas["areas"]:
    print(f"Area: {area['bounds']}, {area['tiles']} tiles")

# Remove data when done
lib.remove_area(bounds)
```

## Key Behavior

**No automatic downloading!** All queries fail if data is not preloaded:

```python
# This will raise ValueError if data not loaded:
elevation = lib.get_elevation(lat, lon)
# ValueError: Elevation data not available at (lat, lon). Please load this area first using load_area().
```

## Visualization

### Quick Visualization
Use `visualize_elevation.py` to create depth maps:

```bash
python visualize_elevation.py
```

This creates:
- `depth_map_10m.png` - Colored terrain map with contours
- `depth_map_10m_grayscale.png` - Grayscale depth map

### Command-Line Download Tool
Use `download_depth_map.py` to download elevation data and create depth maps for any area:

```bash
# Using bounds (south,north,west,east)
python download_depth_map.py --bounds 40.6448,40.6588,-111.5780,-111.5595 --resolution 3

# Using individual coordinates
python download_depth_map.py --lat1 40.6588 --lon1 -111.5780 --lat2 40.6448 --lon2 -111.5595 --resolution 10

# Custom output name
python download_depth_map.py --bounds 40.650,40.670,-111.520,-111.500 --resolution 10 --output park_city_north.png
```

Options:
- `--resolution`: Choose from 1m, 3m, 5m, 10m, 30m, or 60m
- `--output`: Custom filename for the output
- `--data-dir`: Directory for storing elevation data

## Testing

Run tests with pytest:

```bash
# Production tests
pytest tests/test_twolayer_production.py -v

# Comprehensive tests  
pytest tests/test_twolayer_complete.py -v
```

## Files

- `elevation.py` - Main two-layer elevation library
- `visualize_elevation.py` - Visualization tool
- `tests/test_twolayer_production.py` - Production tests
- `tests/test_twolayer_complete.py` - Comprehensive test suite
- `EXPLICIT_LOAD_BEHAVIOR.md` - Documentation on load/fail behavior

## Technical Details

The system works by:
1. Downloading large regions from py3dep into Layer 1
2. Extracting smaller tiles into Layer 2 with proper pixel alignment
3. Merging tiles seamlessly when queried

This eliminates artifacts at tile boundaries while maintaining efficient data access.