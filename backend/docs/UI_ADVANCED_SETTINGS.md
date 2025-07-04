# UI Advanced Settings Documentation

## Overview

The Trail Finder UI now includes an Advanced Settings panel that allows users to customize slope penalties and path preferences directly from the interface.

## Features

### 1. Collapsible Panel
- Click "Advanced Settings" to expand/collapse the configuration options
- Settings are organized into two main sections: Slope Configuration and Path Preferences

### 2. Custom Slope Penalties

#### Enable/Disable
- Toggle checkbox to enable custom slope configuration
- When disabled, uses the profile's default settings

#### Slope Configuration Points
- Add multiple slope/cost pairs
- Each point defines:
  - **Slope (degrees)**: The angle of the slope
  - **Cost multiplier**: How much to penalize this slope
- Points must be in ascending order by slope
- Click "×" to remove a point
- Click "+ Add Slope Point" to add more points

#### Maximum Slope Limit
- Optional hard limit on acceptable slopes
- Routes will heavily penalize slopes above this limit
- Toggle checkbox to enable/disable

### 3. Custom Path Preferences

#### Enable/Disable
- Toggle checkbox to enable custom path preferences
- When disabled, uses the profile's default settings

#### Path Type Sliders
- **Trails**: Natural hiking paths
- **Sidewalks**: Footways and pedestrian paths  
- **Streets**: Residential roads
- **Off-path**: Terrain without defined paths

#### Cost Values
- **< 1.0**: Preferred (lower = stronger preference)
- **1.0**: Neutral
- **> 1.0**: Avoided (higher = stronger avoidance)

### 4. Quick Presets

One-click configurations for common scenarios:

#### Wheelchair
- Very strict slope limits (max 5°)
- Rapid cost increase for slopes > 3°
- Suitable for accessibility needs

#### Mountain Goat
- Very relaxed slope penalties
- Comfortable with slopes up to 40°
- For experienced hikers

#### Trail Lover
- Strong preference for natural trails (0.1)
- Avoids streets (2.0)
- Moderate off-path penalty (1.5)

#### City Walker
- Strong preference for sidewalks (0.2)
- Moderate preference for streets (0.4)
- Avoids off-path terrain (3.0)

## User Interface Layout

```
┌─────────────────────────────────────┐
│ User Profile: [Dropdown]            │
├─────────────────────────────────────┤
│ ▼ Advanced Settings                 │
├─────────────────────────────────────┤
│ ☐ Custom Slope Penalties            │
│   ┌─────────┬──────────┬───┐       │
│   │ Slope°  │ Cost ×   │ × │       │
│   ├─────────┼──────────┼───┤       │
│   │ 0       │ 1.0      │   │       │
│   │ 15      │ 2.0      │ × │       │
│   │ 30      │ 5.0      │ × │       │
│   └─────────┴──────────┴───┘       │
│   [+ Add Slope Point]               │
│                                     │
│   ☐ Maximum Slope Limit             │
│   [30] degrees                      │
├─────────────────────────────────────┤
│ ☐ Custom Path Preferences           │
│   Trails      ●━━━━━━━━━ 0.3       │
│   Sidewalks   ━━━●━━━━━━ 0.5       │
│   Streets     ━━━━━●━━━━ 0.7       │
│   Off-path    ━━━━━━━●━━ 1.0       │
├─────────────────────────────────────┤
│ Quick Presets:                      │
│ [Wheelchair] [Mountain Goat]        │
│ [Trail Lover] [City Walker]         │
└─────────────────────────────────────┘
```

## Usage Examples

### Example 1: Wheelchair Accessible Routes
1. Click "Wheelchair" preset
2. System automatically sets:
   - Slope limits: 0°→1.0, 3°→2.0, 5°→10.0
   - Max slope: 5°
3. Routes will avoid any terrain steeper than 5°

### Example 2: Trail Enthusiast
1. Click "Trail Lover" preset
2. Adjust trail preference slider to 0.1 (very strong preference)
3. Set street cost to 2.5 (strong avoidance)
4. Routes will prioritize natural trails over roads

### Example 3: Custom Configuration
1. Enable Custom Slope Penalties
2. Add points: 0°→1.0, 10°→1.2, 25°→3.0
3. Enable Custom Path Preferences
4. Set sidewalks to 0.4, off-path to 2.0
5. Routes will prefer sidewalks and moderate slopes

## Integration with Profiles

- Start with a base profile (Easy, Experienced, etc.)
- Custom settings override the profile defaults
- Can combine profile selection with custom tweaks

## Tips

1. **Start Conservative**: Begin with moderate values and adjust based on results
2. **Test Known Routes**: Try familiar areas to calibrate your preferences
3. **Save Combinations**: Note down settings that work well for your needs
4. **Profile + Custom**: Use profiles as starting points, then fine-tune

## Technical Details

### API Request Format
When using advanced settings, the API receives:
```json
{
  "start": {...},
  "end": {...},
  "options": {
    "userProfile": "default",
    "customSlopeCosts": [
      {"slope_degrees": 0, "cost_multiplier": 1.0},
      {"slope_degrees": 15, "cost_multiplier": 2.0}
    ],
    "maxSlope": 25,
    "customPathCosts": {
      "trail": 0.3,
      "footway": 0.5,
      "residential": 0.8,
      "off_path": 1.5
    }
  }
}
```

### State Management
- Settings are stored in React state
- Updates trigger re-calculation of route options
- Options are passed to API with route requests