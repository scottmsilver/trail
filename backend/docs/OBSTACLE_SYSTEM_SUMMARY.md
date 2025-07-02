# Trail Finding Obstacle System Summary

## Current Implementation

The trail finding system currently defines obstacles as:

1. **Data Sources**:
   - **OpenStreetMap (OSM)**: Downloads features like water bodies, buildings, roads
   - **Terrain Analysis**: Calculates slope from DEM (Digital Elevation Model)

2. **Obstacle Types from OSM**:
   ```python
   - Water: lakes, rivers, wetlands
   - Buildings: all structures  
   - Roads: major highways
   - Land use: residential, industrial areas
   - Barriers: fences, walls
   - Recreational: golf courses, parks
   ```

3. **Cost Assignment**:
   - **Normal terrain**: Cost = 1 + (slope × 10)
   - **Obstacles**: Cost = 1000 (very high but not infinite)
   - **Result**: Algorithm strongly avoids obstacles but can cross if absolutely necessary

## How It Works

1. **Fetch obstacles** from OpenStreetMap for the hiking area
2. **Rasterize** the vector data onto the same grid as the elevation model
3. **Calculate slope** from the elevation data
4. **Create cost surface**:
   - Base cost from terrain slope
   - High cost (1000) for obstacle cells
5. **A* algorithm** finds path minimizing total cost

## Proposed Enhancement: Configurable Obstacles

The `ObstacleConfig` system would allow:

### 1. **User Profiles**:
- **Easy Hiker**: Avoids all obstacles, prefers gentle slopes
- **Experienced Hiker**: Can cross streams, handle moderate terrain
- **Trail Runner**: Needs smooth terrain, avoids rough ground
- **Accessibility**: Requires wheelchair-accessible paths (max 5° slope)

### 2. **Graduated Costs**:
Instead of binary (obstacle/not obstacle), assign costs based on difficulty:
```
Stream: 200 (crossable with effort)
River: 5000 (uncrossable)
Building: ∞ (absolutely impassable)
Steep slope: 100-1000 (based on angle)
```

### 3. **Feature-Specific Handling**:
- Small streams might be crossable for experienced hikers
- Fences might have gates
- Some "obstacles" are seasonal (snow, flooding)

### 4. **Integration Points**:
To integrate the configuration system:

1. Modify `fetch_obstacles()` to use config OSM tags
2. Update `compute_cost_surface()` to use config cost values
3. Add UI options for selecting user profile
4. Store user preferences for future routes

## Benefits

1. **Personalization**: Routes match user ability and preferences
2. **Safety**: Prevents suggesting dangerous routes to inexperienced users  
3. **Accessibility**: Can generate wheelchair-accessible routes
4. **Flexibility**: Easy to add new obstacle types or adjust costs

## Next Steps

1. Integrate `ObstacleConfig` into `DEMTileCache`
2. Add profile selection to the frontend UI
3. Test with real terrain data
4. Add more sophisticated obstacle detection (cliff edges, seasonal hazards)
5. Allow users to report obstacles or trail conditions