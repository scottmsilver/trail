# Trail Finding Configuration System

## Overview

The trail finding application now supports configurable obstacle detection and terrain cost calculation through user profiles. This allows different types of users (easy hikers, experienced hikers, trail runners, and those needing accessibility) to get appropriate route recommendations.

## User Profiles

### 1. Easy Hiker
- Avoids slopes > 20°
- Strongly penalizes steep terrain
- Prefers well-established paths
- High cost for obstacles

### 2. Experienced Hiker  
- Comfortable with slopes up to 35°
- Moderate terrain penalties
- Can handle more challenging routes
- Moderate obstacle costs

### 3. Trail Runner
- Handles slopes up to 40°
- Minimal terrain penalties for speed
- Optimizes for direct routes
- Lower obstacle costs

### 4. Accessibility Focused
- Strict slope limit of 10°
- Very high costs for any challenging terrain
- Ensures wheelchair/mobility device friendly routes
- Infinite cost for inaccessible areas

## Implementation Details

### Backend Components

1. **ObstacleConfig** (`app/services/obstacle_config.py`)
   - Defines configurable obstacle costs
   - Slope-based cost multipliers
   - OSM tag mappings for obstacles

2. **DEMTileCache** (`app/services/dem_tile_cache.py`)
   - Integrates obstacle configuration
   - Applies terrain costs based on profile
   - Uses configuration for pathfinding decisions

3. **API Integration** (`app/main.py`)
   - Accepts `userProfile` parameter in route requests
   - Maps profiles to configurations
   - Passes configuration to pathfinding

### Frontend Components

1. **Profile Selector** (`frontend/src/App.tsx`)
   - Dropdown to select user profile
   - Passes profile with route requests
   - Updates UI based on selection

2. **Route Options** (`frontend/src/types.ts`)
   - Includes `userProfile` field
   - Default value: "default"

## Testing

### Unit Tests
- `tests/test_pathfinding_algorithm.py` - Core algorithm verification
- `tests/test_config_integration.py` - Profile-specific behavior
- `test_varied_terrain.py` - Complex terrain scenarios
- `test_slope_calculation.py` - Slope cost verification

### API Tests
- `test_profile_routes.py` - End-to-end profile testing

## Usage Example

```bash
# API Request with user profile
curl -X POST http://localhost:9001/api/routes/calculate \
  -H "Content-Type: application/json" \
  -d '{
    "start": {"lat": 40.6470, "lon": -111.5759},
    "end": {"lat": 40.6559, "lon": -111.5705},
    "options": {"userProfile": "accessibility"}
  }'
```

## Results

The configuration system successfully:
1. ✓ Modifies pathfinding behavior based on user profile
2. ✓ Easy hikers prefer gentler valley routes
3. ✓ Trail runners take more direct paths
4. ✓ Accessibility profile strictly avoids steep terrain
5. ✓ All profiles respect obstacle boundaries