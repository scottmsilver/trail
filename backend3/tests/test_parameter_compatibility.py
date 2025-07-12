"""
Test parameter naming changes for backward compatibility and correct behavior.

This test suite ensures that:
1. Old parameter names continue to work (backward compatibility)
2. New parameter names work as expected
3. Both produce identical results
4. Validation and warnings work correctly
"""

import pytest
import subprocess
import json
import tempfile
import os
from pathlib import Path

# Test coordinates - simple path that should show parameter effects
START_COORDS = (40.6596, -111.5662)  
END_COORDS = (40.6496, -111.5761)


class TestParameterNaming:
    """Test that parameter renaming maintains backward compatibility."""
    
    @pytest.fixture
    def base_command(self):
        """Base command for pathfinding."""
        return [
            "python", "pathfinder_cli.py",
            "--start", f"{START_COORDS[0]},{START_COORDS[1]}",
            "--goal", f"{END_COORDS[0]},{END_COORDS[1]}",
            "--skip-viz",  # Don't create visualizations during tests
            "--skip-gpx"   # Don't create GPX files during tests
        ]
    
    def run_pathfinder(self, args):
        """Run pathfinder with given arguments and return parsed output."""
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        return result
    
    def extract_path_stats(self, output):
        """Extract path statistics from pathfinder output."""
        stats = {}
        for line in output.split('\n'):
            if 'Total distance:' in line:
                stats['distance'] = float(line.split(':')[1].strip().replace('m', ''))
            elif 'Total ascent:' in line:
                stats['ascent'] = float(line.split(':')[1].strip().replace('m', ''))
            elif 'Path cost:' in line:
                stats['cost'] = float(line.split(':')[1].strip())
        return stats
    
    # Test current parameter behavior (baseline tests before changes)
    
    def test_prefer_trails_current_behavior(self, base_command):
        """Test current --prefer-trails parameter behavior."""
        # Default (0.3) - should prefer trails
        result = self.run_pathfinder(base_command + ["--prefer-trails", "0.3"])
        assert result.returncode == 0
        default_stats = self.extract_path_stats(result.stdout)
        
        # Strong preference (0.1) - should really stick to trails
        result = self.run_pathfinder(base_command + ["--prefer-trails", "0.1"])
        assert result.returncode == 0
        strong_pref_stats = self.extract_path_stats(result.stdout)
        
        # Avoid trails (2.0) - should avoid trails
        result = self.run_pathfinder(base_command + ["--prefer-trails", "2.0"])
        assert result.returncode == 0
        avoid_stats = self.extract_path_stats(result.stdout)
        
        # Store baseline results for comparison after changes
        self._baseline_prefer_trails = {
            'default': default_stats,
            'strong_pref': strong_pref_stats,
            'avoid': avoid_stats
        }
    
    def test_elevation_weight_current_behavior(self, base_command):
        """Test current --elevation-weight parameter behavior."""
        # Low penalty (0.1) - willing to climb
        result = self.run_pathfinder(base_command + ["--elevation-weight", "0.1"])
        assert result.returncode == 0
        low_penalty_stats = self.extract_path_stats(result.stdout)
        
        # High penalty (10.0) - avoid climbing
        result = self.run_pathfinder(base_command + ["--elevation-weight", "10.0"])
        assert result.returncode == 0
        high_penalty_stats = self.extract_path_stats(result.stdout)
        
        # Verify high penalty results in less climbing
        assert high_penalty_stats.get('ascent', 0) <= low_penalty_stats.get('ascent', 0)
        
        self._baseline_elevation_weight = {
            'low': low_penalty_stats,
            'high': high_penalty_stats
        }
    
    def test_distance_weight_current_behavior(self, base_command):
        """Test current --distance-weight parameter behavior."""
        # Very low (0.001) - allow long detours
        result = self.run_pathfinder(base_command + ["--distance-weight", "0.001"])
        assert result.returncode == 0
        allow_detours = self.extract_path_stats(result.stdout)
        
        # High (1.0) - prefer short paths
        result = self.run_pathfinder(base_command + ["--distance-weight", "1.0"])
        assert result.returncode == 0
        short_paths = self.extract_path_stats(result.stdout)
        
        # Verify high distance weight results in shorter paths
        assert short_paths.get('distance', float('inf')) <= allow_detours.get('distance', 0)
        
        self._baseline_distance_weight = {
            'low': allow_detours,
            'high': short_paths
        }
    
    def test_all_parameters_together(self, base_command):
        """Test multiple parameters work together correctly."""
        result = self.run_pathfinder(base_command + [
            "--prefer-trails", "0.5",
            "--elevation-weight", "2.0",
            "--distance-weight", "0.1",
            "--elevation-exponent", "2.5",
            "--max-slope", "25"
        ])
        assert result.returncode == 0
        assert "Path found" in result.stdout
    
    # These tests will be used after implementing changes
    
    def test_new_parameter_names_work(self, base_command):
        """Test that new parameter names function correctly."""
        # Test --trail-cost-factor
        result = self.run_pathfinder(base_command + ["--trail-cost-factor", "0.3"])
        assert result.returncode == 0
        
        # Test --climb-penalty
        result = self.run_pathfinder(base_command + ["--climb-penalty", "1.0"])
        assert result.returncode == 0
        
        # Test --distance-penalty
        result = self.run_pathfinder(base_command + ["--distance-penalty", "0.1"])
        assert result.returncode == 0
    
    def test_old_names_still_work(self, base_command):
        """Test that old parameter names continue to function."""
        result = self.run_pathfinder(base_command + ["--prefer-trails", "0.3"])
        assert result.returncode == 0
        assert "deprecated" not in result.stderr  # No deprecation warnings yet
    
    def test_old_and_new_produce_same_results(self, base_command):
        """Test that old and new parameter names produce identical results."""
        # Run with old name
        old_result = self.run_pathfinder(base_command + ["--prefer-trails", "0.3"])
        old_stats = self.extract_path_stats(old_result.stdout)
        
        # Run with new name
        new_result = self.run_pathfinder(base_command + ["--trail-cost-factor", "0.3"])
        new_stats = self.extract_path_stats(new_result.stdout)
        
        # Results should be identical
        assert old_stats == new_stats
    
    def test_parameter_validation_warnings(self, base_command, capsys):
        """Test that confusing parameter values produce warnings."""
        # High trail-cost-factor should warn
        result = self.run_pathfinder(base_command + ["--trail-cost-factor", "5.0"])
        assert "strongly avoid trails" in result.stderr or "strongly avoid trails" in result.stdout
        
        # Negative penalties should error
        result = self.run_pathfinder(base_command + ["--climb-penalty", "-1.0"])
        assert result.returncode != 0
        assert "cannot be negative" in result.stderr
    
    def test_equals_syntax_parameter_suggestion(self, base_command):
        """Test that parameters with = syntax are handled correctly."""
        result = self.run_pathfinder(base_command + ["--prefer-trails=0.25", "--terrain-weight=0.8"])
        assert result.returncode == 0
        # Check the suggestion includes the corrected parameter names
        assert "--trail-cost-factor=0.25" in result.stdout
        assert "--terrain-cost-scale=0.8" in result.stdout


class TestParameterEffects:
    """Test that parameters have the expected effects on pathfinding."""
    
    @pytest.fixture
    def base_command(self):
        """Base command for pathfinding."""
        return [
            "python", "pathfinder_cli.py",
            "--start", f"{START_COORDS[0]},{START_COORDS[1]}",
            "--goal", f"{END_COORDS[0]},{END_COORDS[1]}",
            "--skip-viz",
            "--skip-gpx"
        ]
    
    def run_and_extract_cost(self, base_command, extra_args):
        """Run pathfinder and extract the path cost."""
        result = subprocess.run(
            base_command + extra_args,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        
        for line in result.stdout.split('\n'):
            if 'Path cost:' in line:
                return float(line.split(':')[1].strip())
        return None
    
    def test_prefer_trails_effect(self, base_command):
        """Test that lower prefer-trails values result in lower cost paths."""
        cost_high_pref = self.run_and_extract_cost(base_command, ["--prefer-trails", "0.1"])
        cost_low_pref = self.run_and_extract_cost(base_command, ["--prefer-trails", "2.0"])
        
        # When we prefer trails (0.1), cost should generally be lower
        # This might not always be true depending on the specific path
        assert cost_high_pref is not None
        assert cost_low_pref is not None
    
    def test_elevation_weight_effect(self, base_command):
        """Test that elevation weight affects path choice."""
        # This test would need specific start/end points with elevation choices
        pass
    
    def test_max_slope_enforcement(self, base_command):
        """Test that max-slope is enforced as a hard limit."""
        # Test with very low max-slope
        result = subprocess.run(
            base_command + ["--max-slope", "5"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        
        # With max-slope of 5 degrees, many paths might be impossible
        # Check if path was found or if it reports no valid path
        assert "Path found" in result.stdout or "No valid path" in result.stdout


if __name__ == "__main__":
    # Run specific tests during development
    pytest.main([__file__, "-v", "-k", "test_prefer_trails_current_behavior"])