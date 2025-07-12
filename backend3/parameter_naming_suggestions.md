# Pathfinder Parameter Naming Suggestions

## Current vs Suggested Names

### Cost/Penalty Parameters (lower = stronger preference)

1. **`--prefer-trails`** → **`--trail-cost-multiplier`**
   - Current: Confusing - sounds like higher = prefer more
   - Suggested: Clear that it's a cost multiplier
   - Default: 0.3 (trails cost 30% of normal terrain)
   - Range: 0.1-2.0 typical (0.1 = strong preference, 2.0 = avoid trails)

2. **`--distance-weight`** → **`--distance-penalty`**
   - Current: Unclear what "weight" means
   - Suggested: Clear that it penalizes longer paths
   - Default: 0.1
   - Lower values allow longer scenic detours

3. **`--elevation-weight`** → **`--climb-penalty`**
   - Current: Ambiguous - weight for what?
   - Suggested: Clear that it penalizes climbing
   - Default: 1.0
   - Lower values = willing to climb more

4. **`--sustained-weight`** → **`--fatigue-penalty-multiplier`**
   - Current: Vague about what's being weighted
   - Suggested: Clear connection to fatigue system
   - Default: 0.0 (disabled)
   - Higher values = sustained climbs penalized more

5. **`--terrain-weight`** → **`--terrain-penalty-scale`**
   - Current: Unclear relationship to terrain costs
   - Suggested: Shows it scales all terrain penalties
   - Default: 1.0
   - Lower values = terrain type matters less

### Direct Effect Parameters (higher = more effect)

These are already well-named:
- `--elevation-exponent` - Clear: higher = steeper slopes penalized more aggressively
- `--steep-threshold` - Clear: angle where fatigue starts
- `--fatigue-distance` - Clear: distance before fatigue becomes significant  
- `--fatigue-exponent` - Clear: how quickly fatigue accelerates
- `--max-slope` - Clear: maximum allowed slope

### Fatigue System Parameters in Detail

The fatigue system uses four interrelated parameters:

1. **`--steep-threshold`** (degrees)
   - Slope angle where fatigue starts accumulating
   - Below: recovery possible
   - Above: fatigue accumulates

2. **`--fatigue-distance`** (meters)
   - Baseline distance of steep climbing before significant fatigue
   - Think: "I can climb steeply for X meters before getting tired"

3. **`--fatigue-exponent`** (power)
   - How fatigue accelerates:
     - 1.0 = linear (steady increase)
     - 2.0 = quadratic (moderate acceleration)
     - 10.0 = cliff-like (fine until threshold, then massive)

4. **`--sustained-weight`** (multiplier)
   - Overall importance of fatigue in path selection
   - 0.0 = disabled
   - Higher = avoid sustained steep sections more

Better names for fatigue parameters:
- `--steep-threshold` → **`--fatigue-slope-threshold`**
- `--fatigue-distance` → **`--fatigue-baseline-distance`**
- `--fatigue-exponent` → **`--fatigue-acceleration`**
- `--sustained-weight` → **`--fatigue-penalty-multiplier`**

## Implementation Note

If renaming, consider:
1. Keep old names as aliases for backward compatibility
2. Add clear help text explaining the "inverse" nature of cost multipliers
3. Consider using 0-1 range for multipliers (0 = no cost, 1 = normal cost)
4. Add validation to prevent confusion (e.g., warn if trail-cost > 2.0)