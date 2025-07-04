# Custom Configuration for Route Finding

## Overview

The trail finding API now supports custom slope penalties and path preferences, allowing users to fine-tune routing behavior beyond the predefined profiles.

## Custom Slope Configuration

### Basic Usage

You can specify custom slope costs by providing a list of slope/cost pairs:

```json
{
  "options": {
    "customSlopeCosts": [
      {"slope_degrees": 0, "cost_multiplier": 1.0},
      {"slope_degrees": 10, "cost_multiplier": 1.5},
      {"slope_degrees": 20, "cost_multiplier": 3.0},
      {"slope_degrees": 30, "cost_multiplier": 10.0}
    ]
  }
}
```

### How It Works

- **Slope values must be in ascending order**
- **Cost multipliers** determine how much to penalize each slope range
- The algorithm interpolates between specified points
- A final point at 90° with infinite cost is automatically added

### Maximum Slope Limit

Set a hard limit on acceptable slopes:

```json
{
  "options": {
    "maxSlope": 15.0  // Routes with slopes >15° will be heavily penalized
  }
}
```

### Examples

#### Wheelchair Accessible (Very Strict)
```json
{
  "customSlopeCosts": [
    {"slope_degrees": 0, "cost_multiplier": 1.0},
    {"slope_degrees": 3, "cost_multiplier": 2.0},
    {"slope_degrees": 5, "cost_multiplier": 10.0}
  ],
  "maxSlope": 5.0
}
```

#### Casual Hiker (Moderate)
```json
{
  "customSlopeCosts": [
    {"slope_degrees": 0, "cost_multiplier": 1.0},
    {"slope_degrees": 10, "cost_multiplier": 1.2},
    {"slope_degrees": 20, "cost_multiplier": 2.0},
    {"slope_degrees": 30, "cost_multiplier": 5.0}
  ]
}
```

#### Mountain Goat (Relaxed)
```json
{
  "customSlopeCosts": [
    {"slope_degrees": 0, "cost_multiplier": 1.0},
    {"slope_degrees": 20, "cost_multiplier": 1.1},
    {"slope_degrees": 40, "cost_multiplier": 1.5},
    {"slope_degrees": 60, "cost_multiplier": 3.0}
  ]
}
```

## Custom Path Preferences

### Basic Usage

Control how much the algorithm prefers different path types:

```json
{
  "options": {
    "customPathCosts": {
      "trail": 0.3,       // Strong preference
      "footway": 0.5,     // Moderate preference
      "residential": 0.8, // Slight preference
      "off_path": 2.0     // Penalty for off-path
    }
  }
}
```

### Path Types

| Path Type | Description | Typical Cost |
|-----------|-------------|--------------|
| `trail` | Hiking trails, nature paths | 0.2 - 0.5 |
| `footway` | Sidewalks, pedestrian paths | 0.3 - 0.6 |
| `path` | General paths | 0.3 - 0.6 |
| `residential` | Neighborhood streets | 0.5 - 0.9 |
| `off_path` | No defined path | 1.0 - 3.0 |

### Cost Multiplier Guidelines

- **< 1.0**: Preferred (lower values = stronger preference)
- **1.0**: Neutral
- **> 1.0**: Avoided (higher values = stronger avoidance)

### Examples

#### Urban Walker
```json
{
  "customPathCosts": {
    "footway": 0.2,     // Strong preference for sidewalks
    "residential": 0.5, // Moderate preference for streets
    "trail": 0.8,       // Slight preference for trails
    "off_path": 3.0     // Strong avoidance of off-path
  }
}
```

#### Trail Enthusiast
```json
{
  "customPathCosts": {
    "trail": 0.1,       // Very strong trail preference
    "path": 0.3,        // Strong path preference
    "footway": 0.9,     // Slight sidewalk preference
    "residential": 2.0, // Avoid streets
    "off_path": 1.5     // Moderate off-path penalty
  }
}
```

## Combined Configuration

You can combine slope and path preferences:

```json
{
  "start": {"lat": 40.6482, "lon": -111.5738},
  "end": {"lat": 40.6464, "lon": -111.5729},
  "options": {
    "userProfile": "default",
    "customSlopeCosts": [
      {"slope_degrees": 0, "cost_multiplier": 1.0},
      {"slope_degrees": 10, "cost_multiplier": 1.5},
      {"slope_degrees": 20, "cost_multiplier": 3.0}
    ],
    "maxSlope": 25.0,
    "customPathCosts": {
      "footway": 0.3,
      "residential": 0.5,
      "off_path": 2.0
    }
  }
}
```

## Integration with User Profiles

Custom configurations override profile defaults:

1. Start with base profile (easy, experienced, etc.)
2. Apply custom slope costs (if provided)
3. Apply max slope limit (if provided)
4. Apply custom path costs (if provided)

## API Examples

### Request with Custom Slope Configuration
```bash
curl -X POST http://localhost:9001/api/routes/calculate \
  -H "Content-Type: application/json" \
  -d '{
    "start": {"lat": 40.6482, "lon": -111.5738},
    "end": {"lat": 40.6464, "lon": -111.5729},
    "options": {
      "customSlopeCosts": [
        {"slope_degrees": 0, "cost_multiplier": 1.0},
        {"slope_degrees": 15, "cost_multiplier": 2.0},
        {"slope_degrees": 30, "cost_multiplier": 10.0}
      ]
    }
  }'
```

### Request with Custom Path Preferences
```bash
curl -X POST http://localhost:9001/api/routes/calculate \
  -H "Content-Type: application/json" \
  -d '{
    "start": {"lat": 40.6482, "lon": -111.5738},
    "end": {"lat": 40.6464, "lon": -111.5729},
    "options": {
      "customPathCosts": {
        "trail": 0.2,
        "footway": 0.4,
        "off_path": 2.5
      }
    }
  }'
```

## Testing Results

For the test coordinates (40.6482, -111.5738) → (40.6464, -111.5729):

| Configuration | Result | Notes |
|--------------|--------|-------|
| Default | ✓ 11 points | Standard routing |
| Strict Slope (max 5°) | ✗ No route | Terrain too steep |
| Relaxed Slope | ✓ 9 points | More direct route |
| Strong Path Preference | ✓ 10 points | Follows paths more |

## Tips for Configuration

### Slope Configuration
- Start with gentle slopes (0-10°) having low costs (1.0-1.5)
- Increase costs exponentially for steeper slopes
- Use `maxSlope` for hard limits (accessibility needs)
- Test with known terrain to calibrate values

### Path Configuration
- Keep preferred paths below 0.5 for strong preference
- Keep avoided paths above 2.0 for strong avoidance
- Balance path costs with slope costs
- Consider the terrain type in your area

### Performance Considerations
- More complex configurations may slightly increase route calculation time
- Very strict constraints may result in no route found
- Start with moderate values and adjust based on results