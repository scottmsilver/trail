# Elevation Library Tests

All tests are organized in the `/tests` directory and can be run with pytest.

## Test Files

### Core Tests
- `test_elevation_simple.py` - Basic loading, removing, and finding functionality
- `test_elevation_basic.py` - Comprehensive tests with mocked py3dep
- `test_elevation_integration.py` - Integration tests with mock tiles
- `test_elevation_metadata.py` - Metadata tracking and persistence tests
- `test_elevation_tiling.py` - Tile calculation and boundary tests
- `test_elevation_mocked.py` - Tests with mocked file operations
- `test_standalone.py` - Tests that don't require any dependencies
- `test_elevation_real_data.py` - Integration tests with real USGS elevation data
- `test_boundary_coordinates.py` - Tests for handling coordinates on tile boundaries

### Non-Test Files
- `check_resolutions.py` - Manual resolution testing script
- `manual_test_elevation.py` - Manual test script

## Running Tests

```bash
# Activate virtual environment (REQUIRED)
source ~/development/trail/trail_env/bin/activate

# Run all tests
pytest

# Run all tests with verbose output
pytest -v

# Run a specific test file
pytest test_elevation_simple.py -v

# Run tests matching a pattern
pytest -k "test_list_loaded_areas" -v

# Run with coverage report
pytest --cov=elevation --cov-report=html

# Run without warnings
pytest -W ignore::DeprecationWarning
```

## Test Summary

- **81 tests passing** ✅ (72 unit tests + 9 real data integration tests)
- **0 tests failing** 🎉

The test suite comprehensively covers:
- Loading and removing elevation data
- Finding data that should/shouldn't exist
- Tile key generation and bounds calculation
- Resolution support (1m, 3m, 5m, 10m, 30m, 60m)
- Metadata tracking and persistence
- Error messages and validation
- Data directory requirements
- No cache terminology (uses "loaded" instead)
- Real elevation data for known peaks (Mount Whitney, Pikes Peak, Mount Rainier, Death Valley)
- Multi-resolution comparison
- Proper cleanup of downloaded data
- Multi-tile areas:
  - Death Valley transect (315 tiles) from Telescope Peak to Badwater Basin
  - Sierra Nevada ridge line (50 tiles) spanning multiple tiles
  - Tile boundary consistency verification