# Trail Pathfinding File Structure Guide

## Overview
The system stores data in several cache directories, each serving a specific purpose. Files are named using geographic coordinates to enable quick lookups.

## Directory Structure

```
backend/
├── cache/
│   └── aiohttp_cache.sqlite          # HTTP request cache (651+ MB)
│
├── dem_data/                         # Raw elevation data files
│   └── dem_[LAT1]_[LAT2]_[LON1]_[LON2].tif
│
├── tile_cache/                       # Pre-processed tile data
│   ├── terrain/                      # DEM and transform data
│   ├── cost/                         # Computed cost surfaces
│   └── obstacles/                    # Obstacle and path masks
│
└── precomputed_cache/                # Full area cost surfaces
    └── [LAT1],[LAT2],[LON1],[LON2]_cost.pkl
```

## File Naming Conventions

### 1. DEM Data Files (`dem_data/`)
**Format:** `dem_[MIN_LAT]_[MAX_LAT]_[MIN_LON]_[MAX_LON].tif`

**Example:** `dem_40.640_40.650_-111.580_-111.570.tif`
- Coverage: 40.640°N to 40.650°N, 111.580°W to 111.570°W
- Resolution: ~1.1km x 1.1km (0.01° tiles)
- Content: GeoTIFF with elevation values in meters

### 2. Tile Cache Files (`tile_cache/`)
**Format:** `tile_[X]_[Y].pkl` where X,Y are tile indices

**Example:** `tile_-11157_4064.pkl`
- Tile index calculation: 
  - X = floor(lon * 100)
  - Y = floor(lat * 100)
- Each tile covers 0.01° x 0.01° (~1.1km x 1.1km)

#### Tile Subdirectories:
- **`terrain/`**: Contains DEM, transform, and CRS data
- **`cost/`**: Contains pre-computed cost surfaces
- **`obstacles/`**: Contains obstacle masks and path data

### 3. Precomputed Cache (`precomputed_cache/`)
**Format:** `[MIN_LAT],[MAX_LAT],[MIN_LON],[MAX_LON]_cost.pkl`

**Example:** `40.6268,40.6669,-111.5899,-111.5496_cost.pkl`
- Larger areas with full cost surface pre-computation
- Used for frequently accessed regions

## File Contents

### 1. **aiohttp_cache.sqlite**
SQLite database containing:
- Cached HTTP responses from py3dep (USGS 3DEP)
- Cached OSM queries from osmnx
- Request metadata and expiration times

### 2. **DEM TIF Files**
GeoTIFF format containing:
```python
# Readable with rasterio
{
    'data': 2D array of elevation values (meters),
    'transform': Affine transformation matrix,
    'crs': Coordinate Reference System (usually EPSG:4326),
    'resolution': 3m, 10m, or 30m
}
```

### 3. **Tile PKL Files**
Pickled Python dictionaries containing:

**terrain/ tiles:**
```python
{
    'dem': numpy array (elevation data),
    'transform': affine.Affine object,
    'crs': rasterio.crs.CRS object,
    'bounds': (min_lon, min_lat, max_lon, max_lat)
}
```

**cost/ tiles:**
```python
{
    'cost_surface': numpy array (traversal costs),
    'slope_degrees': numpy array (terrain slopes),
    'indices': numpy array (spatial indices),
    'bounds': (min_lon, min_lat, max_lon, max_lat)
}
```

**obstacles/ tiles:**
```python
{
    'obstacle_mask': boolean numpy array,
    'path_raster': integer numpy array (path IDs),
    'path_types': dict mapping IDs to path types,
    'path_raw_tags': dict of original OSM tags,
    'bounds': (min_lon, min_lat, max_lon, max_lat)
}
```

### 4. **Precomputed Cost Files**
Large pickled dictionaries with everything needed for pathfinding:
```python
{
    'cost_surface': numpy array,
    'indices': numpy array,
    'slope_degrees': numpy array,
    'obstacle_mask': boolean numpy array,
    'path_raster': integer numpy array,
    'path_types': dict,
    'dem': numpy array,
    'out_trans': affine.Affine,
    'crs': rasterio.crs.CRS,
    'bounds': dict with lat/lon bounds
}
```

## Size Estimates

- **DEM files**: ~100-500 KB each (depending on resolution)
- **Tile files**: ~50-200 KB each
- **Precomputed files**: 5-50 MB each (depending on area size)
- **HTTP cache**: Can grow to several GB over time

## Cache Hierarchy

1. **Memory cache** (fastest): Active cost surfaces in RAM
2. **Tile cache**: Pre-processed 0.01° tiles on disk
3. **Precomputed cache**: Large area cost surfaces
4. **DEM cache**: Raw elevation data files
5. **HTTP cache**: Network request results

## Coordinate System Notes

- **Geographic files**: Use decimal degrees (EPSG:4326)
- **Projected data**: Converted to Web Mercator (EPSG:3857) for computation
- **Tile indices**: Integer coordinates based on 0.01° grid

## Example Usage

To find files for a route from (40.6469, -111.5696) to (40.6468, -111.5699):

1. **Bounding box**: 40.6268-40.6669, -111.5899--111.5496
2. **DEM file**: `dem_40.620_40.670_-111.590_-111.540.tif`
3. **Tile indices**: X=-11157, Y=4064
4. **Tile files**: `tile_-11157_4064.pkl` in each subdirectory