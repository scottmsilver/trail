# Cache Terminology Removal Summary

## Changes Made

### 1. Method Renamed
- `list_cached_areas()` → `list_loaded_areas()`

### 2. String Updates in elevation.py
- "already cached" → "already loaded"
- "All tiles already cached" → "All tiles already loaded"
- "Remove all cached elevation data" → "Remove all loaded elevation data"
- "Cached elevation data" → "Loaded elevation data"
- "Remove all cached data" → "Remove all loaded data"
- "List cached areas" → "List loaded areas"

### 3. Test Updates
- All test files updated to use `list_loaded_areas()`
- Test method names updated:
  - `test_list_cached_areas_empty` → `test_list_loaded_areas_empty`
  - `test_list_cached_areas_with_data` → `test_list_loaded_areas_with_data`
- Class renamed:
  - `TestCacheListing` → `TestDataListing`

### 4. Test Results
- All 68 tests passing ✅
- No "cache" references in help output ✅
- Error messages don't contain "cache" ✅
- Old `list_cached_areas` method removed ✅

## Verification
Run these commands to verify:
```bash
# Check for any remaining cache references
grep -i "cache" elevation.py

# Run all tests
pytest tests/ -v

# Test CLI help
python elevation.py --data-dir ./test
```

## Key Concept
The elevation library now correctly represents that data is either:
- **Loaded** (statically available on disk)
- **Not loaded** (must be explicitly loaded before use)

There's no implication of caching behavior - data persists until explicitly removed.