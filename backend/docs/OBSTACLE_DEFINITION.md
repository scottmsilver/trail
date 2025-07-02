# How Obstacles Are Defined in the Trail Finding System

## Current Implementation

The system defines obstacles through a combination of OpenStreetMap (OSM) data and terrain analysis:

### 1. **OSM-Based Obstacles** (from `fetch_obstacles`)

The system fetches the following features from OpenStreetMap and treats them as obstacles:

```python
tags = {
    'natural': ['water', 'wetland'],              # Lakes, rivers, marshes
    'landuse': ['residential', 'industrial',       # Built-up areas
                'retail', 'construction'],
    'building': True,                              # All buildings
    'leisure': ['park', 'golf_course'],           # Restricted recreational areas
    'highway': ['motorway', 'trunk', 'primary',   # Major roads
                'secondary', 'tertiary'],
    'barrier': True                                # Fences, walls, etc.
}
```

### 2. **Cost Assignment**

The cost surface is calculated as:

```python
# Base cost from terrain slope
cost_surface = 1 + slope * 10

# Obstacles get very high cost
cost_surface[obstacle_mask] = 1000  # Very high but not infinite
```

- **Normal terrain**: Cost = 1 + (slope × 10)
  - Flat ground: ~1.0
  - Moderate slope: ~2-5
  - Steep slope: ~10+
  
- **Obstacles**: Cost = 1000
  - Not infinite (which would make them impassable)
  - But high enough to strongly discourage routing through them

## Obstacle Categories Explained

### Natural Obstacles
- **Water bodies**: Lakes, rivers, ponds, streams
- **Wetlands**: Marshes, swamps, bogs

### Man-made Obstacles
- **Buildings**: All structures
- **Roads**: Major highways that pedestrians shouldn't cross
- **Industrial/Residential areas**: Private property
- **Barriers**: Fences, walls, gates

### Terrain-based Obstacles
- **Steep slopes**: While not fetched from OSM, very steep terrain gets high cost
- **Cliffs**: Would appear as extreme slope values

## Limitations and Improvements

### Current Limitations

1. **Binary obstacle detection**: Features are either obstacles (cost=1000) or not
2. **No obstacle severity levels**: A small stream has same cost as a lake
3. **Missing obstacle types**:
   - Cliffs/drop-offs (only detected via slope)
   - Dense vegetation
   - Seasonal obstacles (snow, flooding)
   - Trail conditions

### Potential Improvements

1. **Graduated Obstacle Costs**:
```python
obstacle_costs = {
    'water': {
        'river': 5000,      # Impassable
        'stream': 100,      # Crossable with difficulty
        'lake': 10000,      # Completely impassable
    },
    'barrier': {
        'fence': 500,       # Might have gates
        'wall': 5000,       # Impassable
    },
    'vegetation': {
        'forest': 2,        # Slight increase
        'dense_forest': 5,  # Harder to traverse
    }
}
```

2. **Dynamic Obstacles**:
- Seasonal water levels
- Snow coverage
- Construction zones
- Trail closures

3. **User-Defined Obstacles**:
- Allow users to mark areas to avoid
- Personal preferences (avoid highways, prefer trails)

4. **Better Cliff Detection**:
```python
# Detect cliffs by sudden elevation changes
elevation_change = np.abs(np.diff(dem, axis=0))
cliff_mask = elevation_change > 10  # 10m drop
cost_surface[cliff_mask] = np.inf  # Truly impassable
```

5. **Trail Preference**:
```python
# Fetch trails from OSM
trails = ox.features_from_polygon(bbox, {'highway': ['path', 'track', 'footway']})
# Reduce cost on trails
trail_mask = rasterize_trails(trails, transform, shape)
cost_surface[trail_mask] *= 0.5  # Prefer trails
```

## Configuration Options

To make obstacles configurable, we could add:

```python
class ObstacleConfig:
    # What to fetch from OSM
    osm_obstacles = {
        'natural': ['water', 'wetland', 'cliff'],
        'landuse': ['military', 'industrial'],
        'boundary': ['protected_area', 'national_park'],
    }
    
    # Cost multipliers
    obstacle_costs = {
        'default': 1000,
        'water': 5000,
        'building': 2000,
        'highway': 500,
    }
    
    # Terrain thresholds
    slope_thresholds = {
        'moderate': (0.3, 2),    # 30% slope = 2x cost
        'steep': (0.5, 10),      # 50% slope = 10x cost  
        'extreme': (0.8, 100),   # 80% slope = 100x cost
    }
    
    # User preferences
    avoid_highways = True
    prefer_trails = True
    max_water_crossing_width = 5  # meters
```

This would allow users to customize what they consider obstacles based on their hiking experience and preferences.