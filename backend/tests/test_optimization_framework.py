#!/usr/bin/env python3
"""
Route Optimization Testing Framework

This framework:
1. Runs existing routes through the current system to establish baselines
2. Tests various optimizations 
3. Ensures routes remain similar (within tolerance)
4. Measures performance improvements
"""

import asyncio
import time
import json
import numpy as np
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import os

# Add the app directory to the path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.trail_finder import TrailFinderService
from app.services.obstacle_config import ObstaclePresets
from app.services.path_preferences import PathPreferencePresets
from app.models.route import Coordinate


@dataclass
class TestRoute:
    """A test route with metadata"""
    name: str
    start: Coordinate
    end: Coordinate
    profile: str = "default"
    description: str = ""
    expected_distance_km: float = None  # For validation


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run"""
    route_name: str
    optimization: str
    execution_time: float
    path_length: int
    distance_km: float
    elevation_gain: float
    max_slope: float
    path_coords: List[Tuple[float, float]]
    iterations: int = 0
    nodes_explored: int = 0


@dataclass
class RouteComparison:
    """Comparison between two routes"""
    baseline: BenchmarkResult
    optimized: BenchmarkResult
    distance_difference_pct: float
    path_similarity_score: float  # 0-1, 1 = identical
    time_improvement_pct: float
    is_acceptable: bool


class OptimizationFramework:
    """Framework for testing pathfinding optimizations"""
    
    def __init__(self, tolerance_pct: float = 5.0):
        """
        Args:
            tolerance_pct: Maximum acceptable deviation in route distance (%)
        """
        self.tolerance_pct = tolerance_pct
        self.test_routes = self._define_test_routes()
        self.results: Dict[str, List[BenchmarkResult]] = {}
        
    def _define_test_routes(self) -> List[TestRoute]:
        """Define a set of test routes for benchmarking"""
        return [
            # Short urban route
            TestRoute(
                name="urban_short",
                start=Coordinate(lat=40.6482, lon=-111.5738),
                end=Coordinate(lat=40.6464, lon=-111.5729),
                profile="easy",
                description="Short urban route with sidewalks"
            ),
            
            # Medium trail route
            TestRoute(
                name="trail_medium",
                start=Coordinate(lat=40.6568, lon=-111.5713),
                end=Coordinate(lat=40.6428, lon=-111.5777),
                profile="experienced",
                description="Medium trail route with elevation changes"
            ),
            
            # Long backcountry route
            TestRoute(
                name="backcountry_long",
                start=Coordinate(lat=40.6566, lon=-111.5701),
                end=Coordinate(lat=40.6286, lon=-111.5689),
                profile="experienced",
                description="Long backcountry route requiring compression"
            ),
            
            # Steep terrain route
            TestRoute(
                name="steep_terrain",
                start=Coordinate(lat=40.6575, lon=-111.5701),
                end=Coordinate(lat=40.6236, lon=-111.5670),
                profile="trail_runner",
                description="Route through steep terrain"
            ),
            
            # Mixed terrain route
            TestRoute(
                name="mixed_terrain",
                start=Coordinate(lat=40.6465, lon=-111.5754),
                end=Coordinate(lat=40.6554, lon=-111.5715),
                profile="default",
                description="Mixed urban and natural terrain"
            )
        ]
    
    async def run_baseline(self) -> Dict[str, BenchmarkResult]:
        """Run all test routes through the current system to establish baselines"""
        print("\n=== Running Baseline Tests ===")
        baselines = {}
        
        for route in self.test_routes:
            print(f"\nTesting route: {route.name} - {route.description}")
            
            # Set up trail finder with appropriate profile
            if route.profile == "easy":
                obstacle_config = ObstaclePresets.easy_hiker()
                path_preferences = PathPreferencePresets.urban_walker()
            elif route.profile == "experienced":
                obstacle_config = ObstaclePresets.experienced_hiker()
                path_preferences = PathPreferencePresets.trail_seeker()
            elif route.profile == "trail_runner":
                obstacle_config = ObstaclePresets.trail_runner()
                path_preferences = PathPreferencePresets.flexible_hiker()
            else:
                obstacle_config = ObstaclePresets.easy_hiker()
                path_preferences = PathPreferencePresets.flexible_hiker()
            
            trail_finder = TrailFinderService(
                obstacle_config=obstacle_config,
                path_preferences=path_preferences,
                debug_mode=True  # Enable debug mode to get iteration counts
            )
            
            # Time the route calculation
            start_time = time.time()
            path, stats = await trail_finder.find_route(
                route.start,
                route.end,
                {"userProfile": route.profile}
            )
            execution_time = time.time() - start_time
            
            if path:
                result = BenchmarkResult(
                    route_name=route.name,
                    optimization="baseline",
                    execution_time=execution_time,
                    path_length=len(path),
                    distance_km=stats.get("distance_km", 0),
                    elevation_gain=stats.get("elevation_gain_m", 0),
                    max_slope=stats.get("max_slope", 0),
                    path_coords=[(p.lon, p.lat) for p in path],
                    iterations=stats.get("iterations", 0),
                    nodes_explored=stats.get("nodes_explored", 0)
                )
                baselines[route.name] = result
                
                print(f"  ✓ Success: {result.distance_km:.2f}km in {result.execution_time:.2f}s")
                print(f"    Iterations: {result.iterations}, Nodes: {result.nodes_explored}")
            else:
                print(f"  ✗ Failed to find route")
                
        return baselines
    
    async def test_optimization(self, 
                              optimization_name: str,
                              optimization_func: callable,
                              baselines: Dict[str, BenchmarkResult]) -> Dict[str, RouteComparison]:
        """
        Test an optimization against baseline results
        
        Args:
            optimization_name: Name of the optimization
            optimization_func: Function that returns an optimized TrailFinderService
            baselines: Baseline results to compare against
            
        Returns:
            Dictionary of route comparisons
        """
        print(f"\n=== Testing Optimization: {optimization_name} ===")
        comparisons = {}
        
        for route in self.test_routes:
            if route.name not in baselines:
                print(f"\nSkipping {route.name} - no baseline")
                continue
                
            print(f"\nTesting route: {route.name}")
            baseline = baselines[route.name]
            
            # Get optimized trail finder
            trail_finder = optimization_func(route.profile)
            
            # Time the optimized route calculation
            start_time = time.time()
            path, stats = await trail_finder.find_route(
                route.start,
                route.end,
                {"userProfile": route.profile}
            )
            execution_time = time.time() - start_time
            
            if path:
                result = BenchmarkResult(
                    route_name=route.name,
                    optimization=optimization_name,
                    execution_time=execution_time,
                    path_length=len(path),
                    distance_km=stats.get("distance_km", 0),
                    elevation_gain=stats.get("elevation_gain_m", 0),
                    max_slope=stats.get("max_slope", 0),
                    path_coords=[(p.lon, p.lat) for p in path],
                    iterations=stats.get("iterations", 0),
                    nodes_explored=stats.get("nodes_explored", 0)
                )
                
                # Compare with baseline
                comparison = self._compare_routes(baseline, result)
                comparisons[route.name] = comparison
                
                # Print results
                status = "✓" if comparison.is_acceptable else "✗"
                print(f"  {status} Time: {result.execution_time:.2f}s ({comparison.time_improvement_pct:+.1f}%)")
                print(f"    Distance: {result.distance_km:.2f}km ({comparison.distance_difference_pct:+.1f}%)")
                print(f"    Path similarity: {comparison.path_similarity_score:.3f}")
                print(f"    Iterations: {result.iterations} ({result.iterations - baseline.iterations:+d})")
                
                if not comparison.is_acceptable:
                    print(f"    WARNING: Route deviates too much from baseline!")
            else:
                print(f"  ✗ Failed to find route")
                
        return comparisons
    
    def _compare_routes(self, baseline: BenchmarkResult, optimized: BenchmarkResult) -> RouteComparison:
        """Compare two routes and determine if optimization is acceptable"""
        
        # Calculate distance difference
        distance_diff_pct = ((optimized.distance_km - baseline.distance_km) / baseline.distance_km) * 100
        
        # Calculate path similarity using Hausdorff distance
        similarity_score = self._calculate_path_similarity(
            baseline.path_coords, 
            optimized.path_coords
        )
        
        # Calculate time improvement
        time_improvement_pct = ((baseline.execution_time - optimized.execution_time) / baseline.execution_time) * 100
        
        # Determine if acceptable
        is_acceptable = abs(distance_diff_pct) <= self.tolerance_pct and similarity_score > 0.8
        
        return RouteComparison(
            baseline=baseline,
            optimized=optimized,
            distance_difference_pct=distance_diff_pct,
            path_similarity_score=similarity_score,
            time_improvement_pct=time_improvement_pct,
            is_acceptable=is_acceptable
        )
    
    def _calculate_path_similarity(self, path1: List[Tuple[float, float]], 
                                 path2: List[Tuple[float, float]]) -> float:
        """
        Calculate similarity between two paths using modified Hausdorff distance
        Returns a score from 0 to 1, where 1 is identical
        """
        if not path1 or not path2:
            return 0.0
            
        # Convert to numpy arrays
        p1 = np.array(path1)
        p2 = np.array(path2)
        
        # Calculate average distance from each point in path1 to nearest point in path2
        distances1 = []
        for point in p1[::5]:  # Sample every 5th point for efficiency
            dists = np.sqrt(np.sum((p2 - point)**2, axis=1))
            distances1.append(np.min(dists))
            
        # Calculate average distance from each point in path2 to nearest point in path1
        distances2 = []
        for point in p2[::5]:  # Sample every 5th point for efficiency
            dists = np.sqrt(np.sum((p1 - point)**2, axis=1))
            distances2.append(np.min(dists))
            
        # Average Hausdorff distance
        avg_dist = (np.mean(distances1) + np.mean(distances2)) / 2
        
        # Convert to similarity score (assuming distances are in degrees)
        # 0.001 degrees ≈ 111 meters, so we consider paths very similar if avg distance < 0.0001
        similarity = np.exp(-avg_dist * 10000)  # Exponential decay
        
        return min(1.0, similarity)
    
    def save_results(self, filename: str = "optimization_results.json"):
        """Save benchmark results to a JSON file"""
        output = {
            "timestamp": datetime.now().isoformat(),
            "tolerance_pct": self.tolerance_pct,
            "results": {}
        }
        
        for optimization, results in self.results.items():
            output["results"][optimization] = [
                {
                    "route_name": r.route_name,
                    "execution_time": r.execution_time,
                    "distance_km": r.distance_km,
                    "path_length": r.path_length,
                    "iterations": r.iterations,
                    "nodes_explored": r.nodes_explored
                }
                for r in results
            ]
            
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\nResults saved to {filename}")
    
    def print_summary(self, comparisons: Dict[str, Dict[str, RouteComparison]]):
        """Print a summary of all optimization results"""
        print("\n=== OPTIMIZATION SUMMARY ===")
        
        for opt_name, route_comparisons in comparisons.items():
            print(f"\n{opt_name}:")
            
            time_improvements = []
            acceptable_routes = 0
            
            for route_name, comp in route_comparisons.items():
                if comp.is_acceptable:
                    acceptable_routes += 1
                    time_improvements.append(comp.time_improvement_pct)
            
            if time_improvements:
                avg_improvement = np.mean(time_improvements)
                print(f"  Average time improvement: {avg_improvement:.1f}%")
                print(f"  Acceptable routes: {acceptable_routes}/{len(route_comparisons)}")
            else:
                print(f"  No acceptable routes found")


# Example optimization functions
def optimization_1_early_termination(profile: str) -> TrailFinderService:
    """Optimization 1: More aggressive early termination"""
    if profile == "easy":
        obstacle_config = ObstaclePresets.easy_hiker()
        path_preferences = PathPreferencePresets.urban_walker()
    elif profile == "experienced":
        obstacle_config = ObstaclePresets.experienced_hiker()
        path_preferences = PathPreferencePresets.trail_seeker()
    else:
        obstacle_config = ObstaclePresets.easy_hiker()
        path_preferences = PathPreferencePresets.flexible_hiker()
    
    # Modify for more aggressive early termination
    return TrailFinderService(
        obstacle_config=obstacle_config,
        path_preferences=path_preferences,
        debug_mode=True,
        early_termination_factor=1.2  # More aggressive (default might be 1.5)
    )


def optimization_2_bidirectional(profile: str) -> TrailFinderService:
    """Optimization 2: Bidirectional search"""
    if profile == "easy":
        obstacle_config = ObstaclePresets.easy_hiker()
        path_preferences = PathPreferencePresets.urban_walker()
    elif profile == "experienced":
        obstacle_config = ObstaclePresets.experienced_hiker()
        path_preferences = PathPreferencePresets.trail_seeker()
    else:
        obstacle_config = ObstaclePresets.easy_hiker()
        path_preferences = PathPreferencePresets.flexible_hiker()
    
    return TrailFinderService(
        obstacle_config=obstacle_config,
        path_preferences=path_preferences,
        debug_mode=True,
        use_bidirectional=True  # Enable bidirectional search
    )


def optimization_3_cached_heuristics(profile: str) -> TrailFinderService:
    """Optimization 3: Cache heuristic calculations"""
    if profile == "easy":
        obstacle_config = ObstaclePresets.easy_hiker()
        path_preferences = PathPreferencePresets.urban_walker()
    elif profile == "experienced":
        obstacle_config = ObstaclePresets.experienced_hiker()
        path_preferences = PathPreferencePresets.trail_seeker()
    else:
        obstacle_config = ObstaclePresets.easy_hiker()
        path_preferences = PathPreferencePresets.flexible_hiker()
    
    return TrailFinderService(
        obstacle_config=obstacle_config,
        path_preferences=path_preferences,
        debug_mode=True,
        cache_heuristics=True  # Cache expensive heuristic calculations
    )


async def main():
    """Run the optimization framework"""
    framework = OptimizationFramework(tolerance_pct=5.0)
    
    # Run baseline tests
    baselines = await framework.run_baseline()
    framework.results["baseline"] = list(baselines.values())
    
    # Test optimizations
    all_comparisons = {}
    
    # Note: These optimization parameters don't exist yet in TrailFinderService
    # You'll need to implement them based on the results
    
    print("\n" + "="*50)
    print("NOTE: The optimization parameters used here are examples.")
    print("You'll need to implement them in TrailFinderService based on the results.")
    print("="*50)
    
    # For now, let's test with the existing system to establish baselines
    # Once you implement optimizations, uncomment and test:
    
    # optimization_funcs = {
    #     "early_termination": optimization_1_early_termination,
    #     "bidirectional": optimization_2_bidirectional,
    #     "cached_heuristics": optimization_3_cached_heuristics,
    # }
    # 
    # for opt_name, opt_func in optimization_funcs.items():
    #     comparisons = await framework.test_optimization(opt_name, opt_func, baselines)
    #     all_comparisons[opt_name] = comparisons
    #     framework.results[opt_name] = [comp.optimized for comp in comparisons.values()]
    
    # Print summary
    framework.print_summary(all_comparisons)
    
    # Save results
    framework.save_results()


if __name__ == "__main__":
    asyncio.run(main())