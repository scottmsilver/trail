# Explicit Load/Fail Behavior in Two-Layer System

## Overview

The two-layer elevation system implements **strict explicit load/fail behavior**:
- **NO automatic downloading or caching**
- Data must be explicitly loaded with `load_area()`
- All queries fail if data is not preloaded
- No hidden downloads or automatic fetching

## Key Behavior

### 1. Before Loading
```python
# These will FAIL with clear error messages:
lib.get_elevation(lat, lon)
# ValueError: Elevation data not available at (lat, lon). Please load this area first using load_area().

lib.get_elevation_array(bounds)  
# ValueError: Elevation data not available for requested bounds. Missing tiles: [...]. Please load this area first using load_area().
```

### 2. Explicit Loading Required
```python
# Must explicitly load data:
result = lib.load_area(bounds)
# Downloads from py3dep and creates tiles
```

### 3. After Loading
```python
# Now queries work:
elevation = lib.get_elevation(lat, lon)  # Returns elevation value
array, meta = lib.get_elevation_array(bounds)  # Returns array
```

### 4. After Removing
```python
lib.remove_area(bounds)
# Now queries fail again - data is gone
```

## Implementation Details

### Changed Methods

1. **`get_elevation()`**
   - BEFORE: Auto-loaded a small buffer area if tile not found
   - NOW: Immediately fails with clear error message

2. **`get_elevation_array()`**
   - BEFORE: Auto-loaded the requested bounds if tiles missing
   - NOW: Immediately fails with list of missing tiles

### Not a Cache

The `_open_datasets` dictionary is **NOT a cache**:
- It's just keeping rasterio file handles open for performance
- Prevents repeatedly opening/closing the same GeoTIFF files
- Data is either loaded (files exist on disk) or not available
- No automatic fetching based on this "cache"

## Benefits

1. **Predictable Behavior**
   - No surprise downloads
   - Clear when data is/isn't available
   - Explicit control over data loading

2. **Better for Production**
   - No unexpected network calls
   - Deterministic behavior
   - Clear error messages

3. **Matches Single-Layer Philosophy**
   - Both systems require explicit loading
   - No automatic caching
   - User has full control

## Test Results

All tests pass with explicit load/fail behavior:
```
✓ get_elevation fails when not loaded
✓ get_elevation_array fails when not loaded  
✓ After loading, queries work
✓ After removing, queries fail again
```

## Migration Note

This matches the single-layer system behavior:
- Both require explicit `load_area()` calls
- Both fail immediately if data not available
- Both give clear error messages

The two-layer system is a true drop-in replacement with no hidden behaviors.