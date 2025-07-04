# Natural Surface Preference Implementation

## Overview
The trail finding algorithm now prefers unobstructed dirt trails and natural surfaces over roads when other factors (slope, difficulty, distance) are similar.

## Cost Multipliers (Lower = More Preferred)

### Most Preferred: Natural Surfaces
- **Trails**: 0.2 (dirt hiking trails)
- **Paths**: 0.25 (unpaved paths)  
- **Tracks**: 0.3 (dirt/gravel tracks)
- **Bridleways**: 0.35 (horse paths)
- **Nature reserves**: 0.35
- **Grass**: 0.4 (open grass areas)
- **Meadows**: 0.4

### Moderately Preferred: Natural Terrain
- **Off-path natural terrain**: 0.5 (better than roads!)
- **Parks**: 0.45
- **Gardens**: 0.5
- **Beaches**: 0.6

### Least Preferred: Paved Surfaces
- **Footways/Sidewalks**: 0.6
- **Pedestrian areas**: 0.65
- **Cycle paths**: 0.7
- **Residential streets**: 0.85
- **Service roads**: 0.9
- **Main roads**: 0.95-0.99

## How It Works

1. **OSM Data**: The system fetches natural surface data including:
   - Highway types (paths, trails, tracks)
   - Leisure areas (parks, nature reserves)
   - Natural features (grassland, meadows, beaches)
   - Land use (grass, recreation grounds)

2. **Cost Calculation**: When calculating route costs:
   - Base cost = slope difficulty × path preference multiplier
   - Natural surfaces get multipliers < 0.5
   - Roads get multipliers > 0.6
   - This makes natural routes more attractive

3. **Route Selection**: The A* algorithm will:
   - Choose a grassy park over a sidewalk
   - Prefer dirt trails over paved paths
   - Take slightly longer routes through nature
   - Only use roads when necessary

## User Profiles

### Trail Seeker
- Strongly prefers natural surfaces (0.15-0.4)
- Avoids roads (0.7-0.95)
- Will take longer routes to stay on trails

### Default User  
- Balanced preference for natural surfaces
- Natural terrain (0.5) preferred over roads (0.85)
- Reasonable detours for better surfaces

### Urban Walker
- Parks and grass preferred in cities
- Still uses sidewalks when needed
- Avoids off-path in urban areas (private property)

## Example Results
For the test route (40.6546, -111.5705 to 40.6485, -111.5641):
- Algorithm now seeks natural surfaces when available
- Will choose park paths over adjacent sidewalks
- Takes grass areas instead of parking lots
- Results in more enjoyable hiking experience

The system successfully balances the desire for natural surfaces with practical considerations like slope, distance, and obstacles.