# Test Decision Guide: When to Use Real Data

## Quick Decision Tree

```
Is the test validating...
│
├─ Algorithm logic/correctness? → USE MOCK DATA ✅
├─ Component interactions? → USE MOCK DATA ✅
├─ Error handling? → USE MOCK DATA ✅
├─ Performance ratios? → USE MOCK DATA ✅
│
├─ Actual route quality? → USE REAL DATA 🌍
├─ Terrain accuracy? → USE REAL DATA 🌍
├─ Download mechanisms? → USE REAL DATA 🌍
└─ Production validation? → USE REAL DATA 🌍
```

## Examples by Test Type

### ✅ Always Mock (99% of tests)

```python
# Testing algorithm logic
def test_pathfinding_avoids_obstacles(mock_dem_cache):
    """Test that pathfinding correctly avoids impassable terrain"""
    # Mock provides controlled terrain with known obstacles
    
# Testing calculations
def test_slope_cost_calculation(mock_dem_cache):
    """Test slope cost formula correctness"""
    # Mock provides precise slope values to test formula

# Testing cache behavior
def test_cache_hit_improves_performance(mock_dem_cache):
    """Test that cached routes are faster"""
    # Mock eliminates download variability

# Testing error handling
def test_handles_no_path_found(mock_dem_cache):
    """Test graceful handling when no path exists"""
    # Mock can simulate impassable terrain
```

### 🌍 Real Data Only (1% of tests)

```python
@pytest.mark.real_data
@pytest.mark.slow
async def test_park_city_route_quality():
    """Validate actual route quality in Park City terrain"""
    # Need real elevation data to verify route is actually good

@pytest.mark.real_data
async def test_steep_terrain_avoidance():
    """Verify routes actually avoid dangerously steep areas"""
    # Need real terrain to verify safety

@pytest.mark.real_data
async def test_dem_download_retry():
    """Test download retry logic with real service"""
    # Need actual external service interaction
```

## Implementation

### 1. Mark Tests Appropriately

```python
# Fast test (default - no marking needed)
def test_algorithm(mock_dem_cache):
    pass

# Real data test (must be marked)
@pytest.mark.real_data
@pytest.mark.slow
def test_terrain_accuracy():
    pass
```

### 2. Run Tests by Category

```bash
# Development (fast feedback)
pytest -m "not real_data"  # < 5 seconds

# CI/CD Pipeline
pytest -m "not real_data" --cov  # Fast tests with coverage

# Nightly/Weekly validation
pytest -m "real_data" --maxfail=3  # Real terrain validation

# Everything
pytest  # All tests (slow)
```

### 3. Test Organization

```
tests/
├── test_*_fast.py         # Fast tests with mocks
├── test_*.py              # Regular tests (mostly mocked)
├── e2e/
│   └── test_real_*.py     # Real data tests
└── fixtures/
    └── cache_fixtures.py  # Shared mock fixtures
```

## Benefits of This Approach

1. **Fast Development Cycle**
   - 5 seconds vs 2+ minutes per test run
   - No network dependencies
   - Can run offline

2. **Reliable CI/CD**
   - No flaky tests due to network issues
   - Consistent results
   - Fast pipeline execution

3. **Targeted Validation**
   - Real data tests run when needed
   - Focus on actual quality issues
   - Clear separation of concerns

## Common Pitfalls to Avoid

❌ **Don't use real data to test logic**
```python
# Bad: Downloads real data just to test a formula
def test_cost_calculation():
    finder = TrailFinderService()  # Downloads data!
    # ... test cost formula
```

✅ **Do use mocks for logic tests**
```python
# Good: Uses mock data to test formula
def test_cost_calculation(mock_dem_cache):
    # Test with controlled inputs
```

❌ **Don't mock when testing integration with external services**
```python
# Bad: Mocking the very thing we're testing
def test_dem_download():
    with mock.patch('py3dep.get_dem'):  # Don't mock this!
        # ... not actually testing download
```

✅ **Do use real services when that's what you're testing**
```python
# Good: Actually tests the download
@pytest.mark.real_data
def test_dem_download():
    # Real download with retry logic
```

## Summary

- **Default to mocks** - Faster, more reliable, better isolation
- **Use real data sparingly** - Only for quality/accuracy validation  
- **Mark tests clearly** - So others know the intent
- **Separate concerns** - Logic tests vs quality tests

This approach gives you the best of both worlds: fast development with confidence in real-world performance.