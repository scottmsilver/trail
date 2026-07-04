# Trail Pathfinding Data Layers Documentation

## Overview

The trail pathfinding system uses a multi-layered approach to compute optimal hiking routes. This document details each data layer, file format, naming convention, and the data structures contained within.

## Data Layer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Input (GPS Coordinates)             │
└─────────────────────────────┬───────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Bounding Box Calculation                  │
│                    (with configurable buffer)                │
└─────────────────────────────┬───────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      Raw Data Sources                        │
├─────────────────┬───────────────────┬───────────────────────┤
│   Elevation     │    Obstacles      │      Paths            │
│   (USGS 3DEP)   │  (OpenStreetMap)  │  (OpenStreetMap)      │
└─────────────────┴───────────────────┴───────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Processing Layers                         │
├─────────────────┬───────────────────┬───────────────────────┤
│ Smoothed DEM    │  Obstacle Mask    │   Path Raster         │
│ Slope Calculation│  (Boolean Grid)   │  (Type Mapping)       │
└─────────────────┴───────────────────┴───────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      Cost Surface                            │
│              (Combined traversal difficulty)                 │
└─────────────────────────────┬───────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    A* Pathfinding                            │
│                  (Optimal route calculation)                 │
└─────────────────────────────────────────────────────────────┘
```

## File System Structure

```
backend/
├── cache/
│   └── aiohttp_cache.sqlite          # HTTP request cache
│
├── dem_data/                         # Raw elevation files
│   └── dem_[LAT1]_[LAT2]_[LON1]_[LON2].tif
│
├── tile_cache/                       # Processed tile data
│   ├── terrain/                      # Elevation tiles
│   │   └── tile_[X]_[Y].pkl
│   ├── cost/                         # Cost surface tiles
│   │   └── tile_[X]_[Y].pkl
│   └── obstacles/                    # Obstacle/path tiles
│       └── tile_[X]_[Y].pkl
│
└── precomputed_cache/                # Full area caches
    └── [LAT1],[LAT2],[LON1],[LON2]_cost.pkl
```

## Detailed Layer Specifications

### 1. HTTP Cache Layer

**File:** `cache/aiohttp_cache.sqlite`
- **Format:** SQLite database
- **Size:** 600MB-2GB+ (grows over time)
- **Purpose:** Cache raw HTTP responses to minimize API calls
- **Contents:**
  ```sql
  -- Cached responses table structure
  responses:
    - url: TEXT (request URL)
    - response_data: BLOB (compressed response)
    - expiration: TIMESTAMP
    - headers: TEXT (JSON)
  ```

### 2. Digital Elevation Model (DEM) Layer

**Files:** `dem_data/dem_[MIN_LAT]_[MAX_LAT]_[MIN_LON]_[MAX_LON].tif`
- **Example:** `dem_40.640_40.650_-111.580_-111.570.tif`
- **Format:** GeoTIFF
- **Size:** 100-500KB per file
- **Resolution:** 3m, 10m, or 30m (auto-selected based on availability)
- **Coordinate System:** EPSG:4326 (WGS84)
- **Data Structure:**
  ```python
  # GeoTIFF contains:
  {
      'data': 2D numpy array of float32,  # Elevation in meters
      'transform': Affine transform,       # Pixel to coordinate mapping
      'crs': Coordinate Reference System,  # Usually EPSG:4326
      'nodata': -9999,                    # Value for missing data
      'bands': 1                          # Single band for elevation
  }
  ```

### 3. Tile System

All tiles use a 0.01° x 0.01° grid (~1.1km x 1.1km at this latitude).

**Tile Index Calculation:**
```python
tile_x = int(math.floor(longitude * 100))
tile_y = int(math.floor(latitude * 100))
```

#### 3.1 Terrain Tiles

**Files:** `tile_cache/terrain/tile_[X]_[Y].pkl`
- **Example:** `tile_cache/terrain/tile_-11157_4064.pkl`
- **Format:** Pickled Python dictionary
- **Size:** 50-150KB per tile
- **Contents:**
  ```python
  {
      'dem': numpy.ndarray,          # Elevation data (meters)
      'transform': affine.Affine,    # Coordinate transformation
      'crs': rasterio.crs.CRS,       # Coordinate system (EPSG:3857)
      'bounds': tuple,               # (min_lon, min_lat, max_lon, max_lat)
      'resolution': float,           # Meters per pixel
      'shape': tuple                 # (height, width)
  }
  ```

#### 3.2 Cost Surface Tiles

**Files:** `tile_cache/cost/tile_[X]_[Y].pkl`
- **Example:** `tile_cache/cost/tile_-11157_4064.pkl`
- **Format:** Pickled Python dictionary
- **Size:** 100-200KB per tile
- **Contents:**
  ```python
  {
      'cost_surface': numpy.ndarray,    # Traversal cost per cell
      'slope_degrees': numpy.ndarray,   # Terrain slope in degrees
      'indices': numpy.ndarray,         # Spatial index for pathfinding
      'transform': affine.Affine,       # Coordinate transformation
      'crs': rasterio.crs.CRS,          # Coordinate system
      'dem': numpy.ndarray,             # Smoothed elevation data
      'path_raster': numpy.ndarray,     # Path type IDs
      'path_types': dict,               # ID to path type mapping
      'path_raw_tags': dict,            # Original OSM tags
      'tile_x': int,                    # Tile X index
      'tile_y': int,                    # Tile Y index
      'bounds': tuple                   # Geographic bounds
  }
  ```

#### 3.3 Obstacle/Path Tiles

**Files:** `tile_cache/obstacles/tile_[X]_[Y].pkl`
- **Example:** `tile_cache/obstacles/tile_-11157_4064.pkl`
- **Format:** Pickled Python dictionary
- **Size:** 20-100KB per tile
- **Contents:**
  ```python
  {
      'obstacle_mask': numpy.ndarray,   # Boolean mask (True = obstacle)
      'path_raster': numpy.ndarray,     # Integer array of path IDs
      'path_types': dict,               # Mapping of IDs to types
      'path_raw_tags': dict,            # Original OSM feature tags
      'bounds': tuple,                  # Geographic bounds
      'obstacle_types': dict            # Obstacle categories found
  }
  ```

### 4. Precomputed Cache Layer

**Files:** `precomputed_cache/[MIN_LAT],[MAX_LAT],[MIN_LON],[MAX_LON]_cost.pkl`
- **Example:** `precomputed_cache/40.6268,40.6669,-111.5899,-111.5496_cost.pkl`
- **Format:** Pickled Python dictionary
- **Size:** 5-50MB per file
- **Purpose:** Cache entire cost surfaces for frequently accessed areas
- **Contents:**
  ```python
  {
      'cost_surface': numpy.ndarray,    # Complete cost grid
      'indices': numpy.ndarray,         # Spatial indices
      'slope_degrees': numpy.ndarray,   # Slope calculations
      'obstacle_mask': numpy.ndarray,   # Obstacle locations
      'path_raster': numpy.ndarray,     # Path preferences
      'path_types': dict,               # Path type mappings
      'dem': numpy.ndarray,             # Original elevation
      'dem_smoothed': numpy.ndarray,    # Smoothed elevation
      'out_trans': affine.Affine,       # Transform matrix
      'crs': rasterio.crs.CRS,          # Coordinate system
      'bounds': dict                    # Area boundaries
  }
  ```

## Data Processing Pipeline

### 1. Elevation Processing
```
Raw DEM (GeoTIFF) 
    ↓ [Read & Reproject to EPSG:3857]
Reprojected DEM 
    ↓ [Gaussian smoothing σ=1.0]
Smoothed DEM 
    ↓ [Calculate gradients]
Slope Layer (degrees)
```

### 2. Obstacle Processing
```
OSM Vector Data 
    ↓ [Query by bounding box]
GeoDataFrame 
    ↓ [Rasterize to grid]
Obstacle Mask 
    ↓ [Apply 2-pixel buffer]
Buffered Obstacle Mask
```

### 3. Path Processing
```
OSM Path Data 
    ↓ [Query & categorize]
Path Features 
    ↓ [Rasterize with IDs]
Path Raster 
    ↓ [Create type mappings]
Path Preferences Layer
```

### 4. Cost Surface Computation
```python
# Base cost from slope
base_cost = calculate_slope_cost(slope_degrees)

# Apply path preferences
if path_raster[i,j] > 0:
    path_type = path_types[path_raster[i,j]]
    multiplier = PATH_MULTIPLIERS[path_type]
    if slope_degrees[i,j] > 25:
        multiplier = 0.5 + (multiplier * 0.5)
    cost = base_cost * multiplier

# Apply obstacle penalties
if obstacle_mask[i,j]:
    cost = OBSTACLE_COSTS[obstacle_type]
```

## Path Type Categories

| Path Type | Multiplier | Description |
|-----------|------------|-------------|
| track/trail | 0.2 | Hiking trails, dirt paths |
| path/footway | 0.3 | Walking paths |
| cycleway | 0.4 | Bike paths |
| grass/meadow | 0.6 | Natural grass areas |
| park | 0.6 | Park grounds |
| beach/sand | 0.8 | Sandy areas |
| road/street | 0.99 | Paved roads (avoided) |
| off_path | 1.0 | No path present |

## Obstacle Categories

| Obstacle Type | Cost | Description |
|--------------|------|-------------|
| water | 5000 | Lakes, rivers, ponds |
| cliff | ∞ | Vertical rock faces |
| building | 10000 | Structures |
| barrier | 1000 | Walls, fences |
| quarry | 5000 | Mining areas |
| wetland | 100 | Marshes, swamps |

## Slope Cost Function

```python
def calculate_slope_cost(slope_degrees):
    if slope_degrees < 5:
        return 1.0
    elif slope_degrees < 15:
        return 1 + (slope_degrees - 5) * 0.5
    elif slope_degrees < 25:
        return 6 + (slope_degrees - 15) * 2
    elif slope_degrees < 40:
        return 26 + (slope_degrees - 25) * 10
    else:
        return 1000  # Nearly impassable
```

## Cache Usage Hierarchy

1. **Memory Cache**: Active cost surfaces (fastest)
2. **Tile Cache**: Pre-computed 0.01° tiles
3. **Precomputed Cache**: Large area cost surfaces
4. **DEM Cache**: Raw elevation files
5. **HTTP Cache**: Network responses (slowest)

## Performance Characteristics

- **Tile computation**: 100-500ms per tile
- **Tile loading**: 5-20ms from disk
- **Memory usage**: ~1MB per tile in memory
- **Typical route**: Uses 4-20 tiles
- **Cache hit rate**: >90% for popular areas

## Maintenance Notes

- HTTP cache can be cleared without data loss
- DEM files are source data - do not delete
- Tile cache can be regenerated from DEM files
- Precomputed cache is optional but improves performance
- All pickle files use protocol 4 for Python 3.4+ compatibility