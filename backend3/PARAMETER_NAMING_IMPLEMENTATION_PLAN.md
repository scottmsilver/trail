# Parameter Naming Consistency Implementation Plan

## Overview
This plan addresses the inconsistent naming of pathfinding parameters, particularly around cost multipliers vs preference weights. The goal is to make parameter names clearly indicate their effect direction and purpose.

## Core Principles

1. **Cost Factors**: Parameters that multiply costs (lower = stronger preference)
   - Named with `_cost_factor` suffix
   - Range: typically 0.1-2.0 (0.1 = 10% cost = strong preference)

2. **Penalty Multipliers**: Parameters that add penalties (higher = more penalty)
   - Named with `_penalty` or `_penalty_multiplier` suffix
   - Range: typically 0.0+ (0 = no penalty)

3. **Exponents**: Parameters that control non-linear scaling
   - Named with `_exponent` suffix
   - Range: typically 1.0-10.0

4. **Thresholds**: Parameters that define cutoff points
   - Named with `_threshold` suffix
   - Units clearly specified (degrees, meters, etc.)

## Phase 1: Parameter Renaming (Backward Compatible)

### 1.1 Cost Factor Parameters (lower = preference)
```python
# Current -> New (with alias)
--prefer-trails -> --trail-cost-factor (alias: --prefer-trails)
--terrain-weight -> --terrain-cost-scale (alias: --terrain-weight)
```

### 1.2 Penalty Parameters (higher = more penalty)
```python
# Current -> New (with alias)
--distance-weight -> --distance-penalty (alias: --distance-weight)
--elevation-weight -> --climb-penalty (alias: --elevation-weight)
--sustained-weight -> --fatigue-penalty-multiplier (alias: --sustained-weight)
```

### 1.3 Already Well-Named Parameters
```python
# Keep as-is
--elevation-exponent
--max-slope
--steep-threshold -> --fatigue-slope-threshold (optional improvement)
--fatigue-distance -> --fatigue-baseline-distance (optional improvement)
--fatigue-exponent -> --fatigue-acceleration (optional improvement)
```

## Phase 2: Code Implementation Steps

### 2.1 Update Argument Parser (pathfinder_cli.py)
```python
# Example implementation pattern:
parser.add_argument(
    '--trail-cost-factor', '--prefer-trails',  # New name first, alias second
    type=float,
    default=0.3,
    help='Cost multiplier for trails (0.1=strong preference, 1.0=neutral, 2.0=avoid)',
    dest='trail_cost_factor'  # Internal variable name
)
```

### 2.2 Update Internal Variable Names
1. Search and replace in elevation_pathfinder_terrain.py:
   - `prefer_trails` -> `trail_cost_factor`
   - `distance_weight` -> `distance_penalty`
   - `elevation_weight` -> `climb_penalty`
   - `sustained_weight` -> `fatigue_penalty_multiplier`

2. Update cost calculation logic to match new semantics

### 2.3 Update Documentation
1. Update all help strings to clarify direction of effect
2. Add validation warnings for unusual values
3. Create migration guide for existing scripts

## Phase 3: Validation and Warnings

### 3.1 Add Parameter Validation
```python
def validate_parameters(args):
    # Warn if cost factors are > 2.0 (likely misunderstanding)
    if args.trail_cost_factor > 2.0:
        logger.warning(
            f"trail-cost-factor={args.trail_cost_factor} is very high. "
            "This will strongly avoid trails. Use 0.1-0.5 to prefer trails."
        )
    
    # Warn if penalties are negative
    if args.distance_penalty < 0:
        raise ValueError("distance-penalty cannot be negative")
```

### 3.2 Add "Did You Mean?" Suggestions
```python
# If user provides old parameter without -- prefix
if 'prefer_trails' in unknown_args:
    print("Did you mean --trail-cost-factor? (formerly --prefer-trails)")
```

## Phase 4: Testing Strategy

### 4.1 Regression Tests
1. Test all old parameter names still work
2. Test new parameter names work identically
3. Test mixed old/new parameter usage

### 4.2 Validation Tests
1. Test warnings for out-of-range values
2. Test error messages for invalid combinations
3. Test help text clarity

### 4.3 Migration Tests
1. Run existing test suite with old parameters
2. Run same tests with new parameters
3. Verify identical results

## Phase 5: Documentation Updates

### 5.1 Update README
- Parameter reference table with old->new mapping
- Clear explanation of cost factors vs penalties
- Visual examples showing effect of different values

### 5.2 Create Migration Guide
- Script to update existing command lines
- Examples of common parameter combinations
- Troubleshooting guide for unexpected behavior

### 5.3 Update Inline Comments
- Clarify cost calculation logic
- Document parameter ranges and effects
- Add examples in code comments

## Implementation Timeline

1. **Week 1**: Implement backward-compatible aliases
2. **Week 2**: Add validation and warnings
3. **Week 3**: Update documentation and tests
4. **Week 4**: Deprecation notices for old names
5. **Future**: Remove old aliases (after deprecation period)

## Success Metrics

1. No breaking changes for existing users
2. New users understand parameters without confusion
3. Reduced support questions about parameter effects
4. Cleaner, more maintainable codebase

## Risk Mitigation

1. **Breaking Changes**: Use aliases to maintain compatibility
2. **User Confusion**: Clear migration guide and warnings
3. **Performance Impact**: No algorithmic changes, only naming
4. **Test Coverage**: Comprehensive testing of both old and new names