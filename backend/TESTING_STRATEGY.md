# Testing Strategy: Mock vs Real Data

## Overview
A clear strategy for deciding when to use mock fixtures vs. real terrain data in tests.

## Test Categories

### 1. Unit Tests (Always Mock)
**Use mock data when testing:**
- Individual functions/methods
- Business logic
- Data transformations
- Error handling
- Cache mechanisms
- Path algorithms (logic, not accuracy)

**Examples:**
```python
# ✅ Good for mocks
def test_slope_calculation_logic(mock_dem_cache)
def test_cache_hit_rate(mock_dem_cache)
def test_coordinate_validation()
def test_cost_surface_generation()
```

### 2. Integration Tests (Usually Mock)
**Use mock data when testing:**
- Component interactions
- Service layer integration
- API endpoint functionality
- Configuration changes
- Performance characteristics

**Examples:**
```python
# ✅ Good for mocks
def test_trail_finder_service_integration(mock_dem_cache)
def test_api_route_endpoint(mock_dem_cache)
def test_different_user_profiles(mock_dem_cache)
```

### 3. End-to-End Tests (Real Data)
**Use real data when testing:**
- Actual route quality
- Real-world terrain handling
- Production-like scenarios
- Data download mechanisms
- External service integration

**Examples:**
```python
# 🌍 Needs real data
@pytest.mark.real_data
def test_park_city_actual_route()
def test_steep_terrain_avoidance()
def test_dem_download_retry_logic()
```

## Decision Framework

### Use Mock Data When:
1. **Testing logic, not data accuracy**
   - Algorithm correctness
   - Business rules
   - Error conditions

2. **Speed is critical**
   - CI/CD pipelines
   - Local development
   - Frequent test runs

3. **Testing specific scenarios**
   - Edge cases
   - Error conditions
   - Controlled inputs

4. **Isolation is important**
   - Unit tests
   - Component boundaries
   - Parallel test execution

### Use Real Data When:
1. **Testing data accuracy**
   - Route quality validation
   - Terrain analysis accuracy
   - Real-world path feasibility

2. **Testing external integrations**
   - DEM data providers
   - API rate limits
   - Network resilience

3. **Performance benchmarking**
   - Real computation times
   - Memory usage patterns
   - Cache effectiveness

4. **User acceptance testing**
   - Actual user scenarios
   - Production validation
   - Route quality assurance

## Implementation Pattern

### 1. Mark Tests Appropriately
```python
import pytest

# Fast test with mocks (default)
def test_pathfinding_logic(mock_dem_cache):
    """Test pathfinding algorithm logic"""
    pass

# Real data test (marked)
@pytest.mark.real_data
@pytest.mark.slow
def test_actual_route_quality():
    """Test actual route quality with real terrain"""
    pass
```

### 2. Organize Test Files
```
tests/
├── unit/                    # Always mock
│   ├── test_algorithms.py
│   └── test_calculations.py
├── integration/             # Usually mock
│   ├── test_services.py
│   └── test_api.py
├── e2e/                    # Real data
│   ├── test_real_routes.py
│   └── test_terrain_accuracy.py
└── fixtures/               # Shared fixtures
    └── cache_fixtures.py
```

### 3. Configure Test Runs
```bash
# Run only fast tests (mocked)
pytest -m "not real_data"

# Run only real data tests
pytest -m "real_data"

# Run specific real data test
pytest -m "real_data" -k "park_city"
```

## Specific Test Scenarios

### Always Mock:
- ✅ Coordinate validation
- ✅ Distance calculations
- ✅ Slope cost formulas
- ✅ Cache storage/retrieval
- ✅ API request/response format
- ✅ Configuration loading
- ✅ Error handling

### Sometimes Mock:
- 🔄 Route calculation (mock for logic, real for quality)
- 🔄 Terrain analysis (mock for algorithm, real for accuracy)
- 🔄 Performance tests (mock for relative, real for absolute)

### Always Real:
- 🌍 DEM data download
- 🌍 Route quality validation
- 🌍 Real terrain classification
- 🌍 Production smoke tests
- 🌍 User acceptance criteria

## Test Data Sets

### Mock Data Scenarios:
```python
# Simple terrain
mock_flat_terrain()      # Testing easy paths
mock_steep_terrain()     # Testing avoidance
mock_mixed_terrain()     # Testing optimization

# Edge cases  
mock_impassable_terrain()  # Testing no-path scenarios
mock_island_terrain()      # Testing disconnected areas
```

### Real Data Locations:
```python
# Standard test routes
PARK_CITY_EASY = ((40.6572, -111.5706), (40.6486, -111.5639))
STEEP_CANYON = ((40.6482, -111.5738), (40.6464, -111.5729))
GOLDEN_GATE = ((37.7694, -122.4862), (37.7754, -122.4584))

# Problem areas for regression testing
KNOWN_DIFFICULT_ROUTES = [...]
```

## Maintenance Strategy

1. **Keep mocks realistic**
   - Update when algorithms change
   - Match real data characteristics
   - Cover edge cases

2. **Minimize real data tests**
   - Only what can't be mocked
   - Run in separate CI job
   - Cache downloaded data

3. **Document test purpose**
   - Why real data is needed
   - What specific aspect is tested
   - Expected outcomes

## CI/CD Configuration

```yaml
# .github/workflows/test.yml
jobs:
  fast-tests:
    name: "Fast Tests (Mocked)"
    runs-on: ubuntu-latest
    steps:
      - run: pytest -m "not real_data"
    
  real-data-tests:
    name: "Real Data Tests"
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    steps:
      - run: pytest -m "real_data" --maxfail=3
```

## Summary

Use mock data by default. Only use real data when testing:
1. Actual route quality
2. Real terrain handling  
3. External service integration
4. Production validation

This keeps tests fast while ensuring quality where it matters.