# Path Following System

## Overview

The trail finding application now includes a path preference system that naturally encourages routes to follow roads, sidewalks, and trails rather than cutting directly across terrain. This makes routes more practical and pleasant for hikers.

## How It Works

### Cost-Based Routing
Instead of treating paths as geographic features to follow explicitly, the system uses **cost multipliers**:
- **Paths have lower cost** (e.g., 0.3x for trails, 0.7x for residential streets)
- **Off-path terrain has normal cost** (1.0x)
- The A* algorithm naturally prefers lower-cost paths

### Example Cost Calculation
For a cell with base terrain cost of 2.0:
- On a trail: 2.0 × 0.3 = 0.6 (very attractive)
- On a street: 2.0 × 0.7 = 1.4 (moderately attractive)
- Off-path: 2.0 × 1.0 = 2.0 (normal cost)

## Path Types and Preferences

### Preferred Path Types (fetched from OSM)
```python
'highway': ['footway', 'path', 'track', 'pedestrian', 'steps', 
           'cycleway', 'bridleway', 'trail', 'residential', 
           'living_street', 'service', 'unclassified']
'leisure': ['park', 'nature_reserve', 'garden']
'route': ['hiking', 'foot', 'walking']
```

### Cost Multipliers by Path Type
| Path Type | Cost Multiplier | Description |
|-----------|----------------|-------------|
| trail | 0.25 | Hiking trails - most preferred |
| footway/path | 0.3 | Dedicated walking paths |
| park | 0.4 | Park paths |
| pedestrian | 0.4 | Pedestrian areas |
| residential | 0.7 | Quiet neighborhood streets |
| service | 0.8 | Service roads |
| off_path | 1.0 | Normal terrain |

## Profile Presets

### 1. Urban Walker
- **Use case**: City hiking, prefers sidewalks
- **Path costs**: footway (0.2), residential (0.5), off_path (2.0)
- **Behavior**: Strongly prefers staying on paths

### 2. Trail Seeker  
- **Use case**: Nature hiking, prefers trails
- **Path costs**: trail (0.2), path (0.25), residential (0.8)
- **Behavior**: Seeks natural trails, avoids streets

### 3. Flexible Hiker
- **Use case**: Balanced approach
- **Path costs**: trail (0.5), path (0.6), off_path (1.0)
- **Behavior**: Mild preference for paths

### 4. Direct Route
- **Use case**: Shortest path
- **Path costs**: Minimal preferences
- **Behavior**: Takes most direct route

## Implementation Details

### Components

1. **PathPreferences** (`app/services/path_preferences.py`)
   - Defines path cost multipliers
   - Specifies OSM tags to fetch
   - Configurable transition penalties

2. **DEMTileCache** (updated)
   - Fetches path data from OSM
   - Rasterizes paths to grid
   - Applies cost multipliers during pathfinding

3. **TrailFinderService** (updated)
   - Accepts path preferences
   - Passes to DEM cache

4. **API Integration** (`app/main.py`)
   - Maps user profiles to path preferences
   - Combines with obstacle configuration

### Path Fetching Process

1. **Fetch OSM Data**: Query OpenStreetMap for paths in area
2. **Rasterize Paths**: Convert vector paths to grid cells
3. **Apply Costs**: Multiply terrain costs by path preference
4. **Route Finding**: A* naturally follows lower-cost paths

## Testing

### Test Scripts
- `test_path_following.py` - Basic path preference testing
- `test_path_debug.py` - Debug path fetching
- `test_api_path_following.py` - API integration test

### Test Results
For coordinates (40.6482, -111.5738) → (40.6464, -111.5729):
- **Without preferences**: 9 points, 0.260 km (direct)
- **Urban Walker**: 10 points, 0.300 km (follows streets)
- **Trail Seeker**: 10 points, 0.300 km (follows paths)

## Usage Example

```python
# Create path preferences
path_prefs = PathPreferencePresets.urban_walker()

# Create trail finder with preferences
service = TrailFinderService(
    obstacle_config=obstacle_config,
    path_preferences=path_prefs
)

# Find route - will naturally follow sidewalks
path, stats = await service.find_route(start, end, options)
```

## Benefits

1. **Natural Behavior**: Routes follow paths without explicit constraints
2. **Flexible**: Easy to adjust preferences per user
3. **Realistic**: Produces practical walking routes
4. **Efficient**: Minimal performance impact
5. **Configurable**: Different profiles for different needs

## Future Enhancements

1. **Known Trails**: Integrate official trail databases
2. **Surface Types**: Prefer paved vs unpaved based on profile
3. **Scenic Routes**: Add scenic value to parks and viewpoints
4. **Safety**: Prefer well-lit paths for night hiking
5. **Accessibility**: Surface type preferences for wheelchairs