# Slope Analysis Layer

A tile-based slope analysis system built on top of the two-layer elevation system. Provides first and second derivatives of terrain elevation.

## Features

- **Slope calculation** - Terrain gradient in degrees
- **Slope change (curvature)** - Rate of change of slope (second derivative)
- **Aspect calculation** - Direction of slope (0-360°, where 0=North)
- **Tile-based storage** - Efficient storage and retrieval
- **Explicit load/fail behavior** - No automatic computation, must explicitly compute areas
- **Multi-resolution support** - Inherits resolution from elevation data

## Installation

```python
from elevation import TwoLayerElevationLibrary, Bounds
from slope_layer import SlopeLayer
```

## Usage

### Basic Usage

```python
# Initialize libraries
elev_lib = TwoLayerElevationLibrary("./elevation_data", resolution=10)
slope_layer = SlopeLayer(elev_lib, "./slope_data")

# Define area of interest
bounds = Bounds(
    south=40.6448,
    north=40.6588,
    west=-111.5780,
    east=-111.5595
)

# Ensure elevation data is loaded
elev_lib.load_area(bounds)

# Compute slopes for the area
result = slope_layer.compute_area(bounds)
print(f"Created {result['tiles_created']} slope tiles")

# Query slope at a point
lat, lon = 40.65, -111.57
slope = slope_layer.get_slope(lat, lon)
print(f"Slope at ({lat}, {lon}): {slope:.1f}°")

# Get slope change (curvature)
slope_change = slope_layer.get_slope_change(lat, lon)
print(f"Slope change: {slope_change:.2f}°/m")

# Get aspect (direction)
aspect = slope_layer.get_aspect(lat, lon)
print(f"Aspect: {aspect:.1f}° from North")
```

### Working with Arrays

```python
# Get slope data for an area
slope_data, metadata = slope_layer.get_slope_array(bounds)

# Access the arrays
slopes = slope_data.slope          # Slope in degrees
curvature = slope_data.slope_change  # Slope change in degrees/meter
aspects = slope_data.aspect        # Direction in degrees (0=N)

# Calculate statistics
print(f"Max slope: {slopes.max():.1f}°")
print(f"Mean slope: {slopes.mean():.1f}°")

# Find steep areas
steep_mask = slopes > 30  # Areas steeper than 30°
steep_percent = 100 * steep_mask.sum() / slopes.size
print(f"Steep terrain: {steep_percent:.1f}%")
```

## Key Behavior

**No automatic computation!** All queries fail if slopes not pre-computed:

```python
# This will raise ValueError if slopes not computed:
slope = slope_layer.get_slope(lat, lon)
# ValueError: Slope data not available at (lat, lon). Please compute this area first using compute_area().
```

## Visualization

Use `visualize_slope.py` to create slope maps:

```bash
python visualize_slope.py
```

This creates:
- `slope_map_10m.png` - Multi-panel visualization with slope, curvature, aspect
- `slope_map_slope_only_10m.png` - Detailed slope map with contours
- `slope_map_aspect_wheel_10m.png` - Aspect map with compass wheel

## Data Management

```python
# List computed areas
areas = slope_layer.list_computed_areas()
for area in areas["areas"]:
    print(f"Area: {area['bounds']}, {area['tiles']} tiles")

# Remove slope data
slope_layer.remove_area(bounds)
```

## Technical Details

### Slope Calculation
- Uses numpy gradient computation on elevation data
- Converts to degrees for intuitive interpretation
- Accounts for latitude-dependent meter/degree conversion

### Slope Change (Curvature)
- Second derivative of elevation
- Indicates terrain convexity/concavity
- Useful for identifying ridges and valleys

### Aspect
- Direction of maximum slope
- 0° = North, 90° = East, 180° = South, 270° = West
- Useful for sun exposure and drainage analysis

### Storage Format
- Multi-band GeoTIFF files
- Band 1: Slope (degrees)
- Band 2: Slope change (degrees/meter)
- Band 3: Aspect (degrees)
- Compressed with DEFLATE

## Example: Trail Analysis

```python
# Analyze trail difficulty
def analyze_trail_difficulty(slope_layer, waypoints):
    difficulties = []
    
    for i in range(len(waypoints) - 1):
        lat1, lon1 = waypoints[i]
        lat2, lon2 = waypoints[i + 1]
        
        # Sample points along segment
        num_samples = 10
        lats = np.linspace(lat1, lat2, num_samples)
        lons = np.linspace(lon1, lon2, num_samples)
        
        segment_slopes = []
        for lat, lon in zip(lats, lons):
            try:
                slope = slope_layer.get_slope(lat, lon)
                segment_slopes.append(slope)
            except ValueError:
                pass
        
        if segment_slopes:
            max_slope = max(segment_slopes)
            if max_slope < 10:
                difficulty = "Easy"
            elif max_slope < 20:
                difficulty = "Moderate"
            elif max_slope < 30:
                difficulty = "Difficult"
            else:
                difficulty = "Expert"
            
            difficulties.append((i, difficulty, max_slope))
    
    return difficulties
```

## Testing

Run tests with pytest:

```bash
python -m pytest tests/test_slope_layer.py -v
```