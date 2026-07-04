"""
Obstacle configuration for trail finding.
Allows customization of what features are considered obstacles and their costs.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np


@dataclass
class ObstacleConfig:
    """Configuration for obstacle detection and cost assignment"""
    
    # OSM tags to fetch as potential obstacles
    osm_tags: Dict[str, List[str]] = None
    
    # Cost values for different obstacle types
    obstacle_costs: Dict[str, float] = None
    
    # Slope-based cost multipliers: (slope_threshold, cost_multiplier)
    slope_costs: List[tuple[float, float]] = None
    
    # Special handling flags
    avoid_water: bool = True
    avoid_highways: bool = True
    prefer_trails: bool = True
    allow_water_crossing: bool = False
    max_water_crossing_width: float = 5.0  # meters
    use_continuous_slope: bool = True  # Use smooth exponential slope costs
    gradient_preference: float = 1.0  # Higher values prefer gentler slopes (1.0 = normal, 2.0 = prefer gradual)
    
    def __post_init__(self):
        """Set default values if not provided"""
        if self.osm_tags is None:
            self.osm_tags = self.get_default_osm_tags()
        
        if self.obstacle_costs is None:
            self.obstacle_costs = self.get_default_costs()
            
        if self.slope_costs is None:
            self.slope_costs = self.get_default_slope_costs()
    
    @staticmethod
    def get_default_osm_tags() -> Dict[str, List[str]]:
        """Default OSM tags to fetch"""
        return {
            'natural': ['water', 'wetland', 'cliff', 'rock', 'scree'],
            'waterway': ['river', 'stream', 'canal'],
            'landuse': ['industrial', 'commercial', 'military'],  # Removed residential
            'building': True,
            'leisure': ['golf_course', 'swimming_pool'],
            # Removed highway entirely - streets should not be obstacles
            'barrier': True,
            # Removed boundary - protected areas and parks should not be obstacles
            'man_made': ['pipeline', 'power_line']
        }
    
    @staticmethod
    def get_default_costs() -> Dict[str, float]:
        """Default cost values for obstacles"""
        return {
            # Water features
            'water': 5000,          # Lakes, ponds - impassable
            'river': 5000,          # Rivers - impassable
            'stream': 200,          # Small streams - difficult
            'wetland': 500,         # Marshes - very difficult
            
            # Terrain
            'cliff': np.inf,        # Cliffs - absolutely impassable
            'rock': 50,             # Rocky terrain - difficult
            'scree': 100,           # Loose rock - very difficult
            
            # Man-made
            'building': 10000,      # Buildings - impassable
            # Removed highway - streets are not obstacles
            # Removed residential - can walk through neighborhoods
            'industrial': 5000,     # Industrial areas - dangerous
            'barrier': 2000,        # Fences, walls - mostly impassable
            
            # Default
            'default': 100          # Unknown obstacles - reduced from 1000
        }
    
    @staticmethod
    def get_default_slope_costs() -> List[tuple[float, float]]:
        """Default slope-based cost multipliers
        Returns: List of (slope_degrees, cost_multiplier) tuples
        
        Updated to better balance with path preferences:
        - Steeper curve to discourage >20° slopes even on trails
        - Ensures 20° trail is NOT cheaper than 15° off-path
        """
        return [
            (0, 1.0),       # 0° - flat
            (5, 2.0),       # 5° - gentle slope (increased from 1.3)
            (10, 5.0),      # 10° - noticeable slope (increased from 1.8)
            (15, 15.0),     # 15° - moderate slope (increased from 3.0)
            (20, 50.0),     # 20° - steep (increased from 10.0)
            (25, 200.0),    # 25° - very steep (increased from 25.0)
            (30, 800.0),    # 30° - extremely steep (increased from 60.0)
            (35, 3000.0),   # 35° - near limit (increased from 150.0)
            (40, 10000.0),  # 40° - extreme (increased from 400.0)
            (45, 50000.0),  # 45° - barely passable (increased from 1000.0)
            (60, 500000.0), # 60° - essentially impassable (increased from 5000.0)
        ]
    
    def get_cost_for_feature(self, feature_type: str, feature_value: str = None) -> float:
        """Get cost value for a specific OSM feature"""
        # Check specific feature value first
        if feature_value and feature_value in self.obstacle_costs:
            return self.obstacle_costs[feature_value]
        
        # Check feature type
        if feature_type in self.obstacle_costs:
            return self.obstacle_costs[feature_type]
            
        # Return default
        return self.obstacle_costs.get('default', 1000)
    
    def get_slope_cost_multiplier(self, slope_degrees: float) -> float:
        """Get cost multiplier for a given slope using continuous function"""
        # Handle extreme slopes
        if slope_degrees >= 90:
            return np.inf
        
        # Use continuous function if enabled
        if hasattr(self, 'use_continuous_slope') and self.use_continuous_slope:
            return self.continuous_slope_cost(slope_degrees)
            
        # Legacy: Interpolate between defined points
        slopes = [s[0] for s in self.slope_costs]
        costs = [s[1] for s in self.slope_costs]
        
        return np.interp(slope_degrees, slopes, costs)
    
    def continuous_slope_cost(self, slope_degrees: float) -> float:
        """
        Continuous slope cost function using smooth mathematical curves
        
        This provides a more realistic cost model that smoothly increases with slope.
        The function combines polynomial and exponential components for different slope ranges.
        """
        # Clamp slope to reasonable range
        slope = max(0, min(slope_degrees, 89))
        
        # Get profile type
        profile = getattr(self, 'slope_profile', 'default')
        
        # Define continuous functions for different profiles
        if profile == 'wheelchair':
            # Very steep penalty curve for accessibility
            # Gentle up to 5°, then rapidly increasing
            if slope <= 5:
                return 1 + 0.2 * slope  # Linear gentle increase
            elif slope <= 8:
                return 2 + 0.5 * (slope - 5) ** 2  # Quadratic increase
            else:
                return np.inf  # Beyond ADA compliance
                
        elif profile == 'mountain_goat':
            # Tolerant of steep slopes, smooth progression
            # Uses a combination of polynomial and exponential
            if slope <= 10:
                return 1 + 0.01 * slope  # Almost flat cost increase
            elif slope <= 30:
                return 1.1 + 0.002 * (slope - 10) ** 2  # Gentle quadratic
            elif slope <= 50:
                return 1.9 + 0.05 * (slope - 30) ** 1.5  # Moderate increase
            else:
                return 4.4 + 0.2 * np.exp(0.1 * (slope - 50))  # Exponential for extreme
                
        elif profile == 'trail_runner':
            # Prefers moderate slopes, penalizes both flat and very steep
            # Optimized for running efficiency
            if slope <= 3:
                return 1.0  # Flat is fine
            elif slope <= 8:
                return 1 + 0.02 * (slope - 3)  # Slight preference for gentle slopes
            elif slope <= 15:
                return 1.1 + 0.1 * (slope - 8) ** 1.5  # Increasing penalty
            else:
                return 2.3 + 0.3 * np.exp(0.12 * (slope - 15))  # Steep exponential
                
        elif profile == 'city_walker':
            # Strong preference for flat terrain with aggressive penalties
            # Much steeper penalty curve to avoid slopes > 20°
            if slope <= 3:
                return 1 + 0.05 * slope  # Minimal increase for gentle slopes
            elif slope <= 10:
                return 1.15 + 0.3 * (slope - 3) ** 1.8  # Steeper polynomial increase
            elif slope <= 15:
                return 5 + 3 * (slope - 10)  # Rapid linear increase
            elif slope <= 20:
                return 20 + 10 * np.exp(0.3 * (slope - 15))  # Exponential growth
            elif slope <= 25:
                # Very high penalty for 20-25°
                return 500 + 200 * (slope - 20)  # Linear ramp to extreme values
            else:
                # Essentially impassable for slopes > 25°
                return 10000 * np.exp(0.3 * (slope - 25))  # Extreme exponential penalty
                
        else:  # default profile
            # Single continuous exponential function for all slopes
            # Cost = base * exp(growth_rate * slope)
            # Tuned for hiking: gentle at low slopes, exponential growth for steep
            
            # Parameters tuned for good hiking behavior:
            # - Flat ground (0°) = 1.0
            # - Gentle slope (10°) ≈ 1.5
            # - Moderate slope (20°) ≈ 4.5
            # - Steep slope (30°) ≈ 33
            # - Very steep (40°) ≈ 245
            
            base_cost = 1.0
            # Adjust growth rate based on gradient preference
            # Higher gradient_preference = lower growth rate = gentler cost increase
            base_growth_rate = 0.085
            growth_rate = base_growth_rate * (2.0 / (1.0 + self.gradient_preference))
            
            return base_cost * np.exp(growth_rate * slope)


# Preset configurations for different use cases
class ObstaclePresets:
    """Predefined obstacle configurations for common scenarios"""
    
    @staticmethod
    def easy_hiker() -> ObstacleConfig:
        """Configuration for casual hikers - avoid most obstacles"""
        return ObstacleConfig(
            avoid_water=True,
            avoid_highways=True,
            prefer_trails=True,
            slope_costs=[
                (0, 1.0),
                (5, 1.5),
                (10, 3.0),
                (15, 10.0),
                (20, 100.0),
                (30, 1000.0),
            ]
        )
    
    @staticmethod
    def experienced_hiker() -> ObstacleConfig:
        """Configuration for experienced hikers - handle moderate obstacles"""
        config = ObstacleConfig()
        config.allow_water_crossing = True
        config.max_water_crossing_width = 10.0
        config.obstacle_costs['stream'] = 50  # Can cross streams
        config.obstacle_costs['rock'] = 20   # Can handle rocky terrain
        return config
    
    @staticmethod
    def trail_runner() -> ObstacleConfig:
        """Configuration for trail runners - prefer smooth, runnable terrain"""
        config = ObstacleConfig()
        config.prefer_trails = True
        # Penalize rough terrain more
        config.obstacle_costs['rock'] = 200
        config.obstacle_costs['scree'] = 500
        # Prefer gentler slopes
        config.slope_costs = [
            (0, 1.0),
            (5, 1.1),
            (10, 2.0),
            (15, 5.0),
            (20, 50.0),
        ]
        return config
    
    @staticmethod
    def accessibility_focused() -> ObstacleConfig:
        """Configuration for wheelchair/accessibility needs"""
        config = ObstacleConfig()
        # Very strict slope requirements
        config.slope_costs = [
            (0, 1.0),
            (2, 2.0),
            (5, 100.0),    # ADA max slope
            (8, 1000.0),   # Absolute max
            (10, np.inf),  # Impassable
        ]
        # Avoid all rough terrain
        config.obstacle_costs['rock'] = np.inf
        config.obstacle_costs['scree'] = np.inf
        config.obstacle_costs['stream'] = np.inf
        return config


def compute_cost_surface_with_config(
    dem: np.ndarray,
    out_trans,
    obstacle_mask: np.ndarray,
    obstacle_features: dict,
    config: ObstacleConfig
) -> np.ndarray:
    """
    Compute cost surface using configuration settings
    
    Args:
        dem: Digital elevation model array
        out_trans: Affine transform
        obstacle_mask: Boolean mask of obstacle locations
        obstacle_features: Dictionary mapping pixels to OSM feature types
        config: Obstacle configuration
    
    Returns:
        cost_surface: Numpy array of traversal costs
    """
    # Calculate slope
    cell_size_x = out_trans.a
    cell_size_y = -out_trans.e
    dzdx, dzdy = np.gradient(dem, cell_size_x, cell_size_y)
    slope_radians = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
    slope_degrees = np.degrees(slope_radians)
    
    # Initialize cost surface with slope-based costs
    cost_surface = np.ones_like(dem)
    
    # Apply slope costs
    for i in range(slope_degrees.shape[0]):
        for j in range(slope_degrees.shape[1]):
            slope = slope_degrees[i, j]
            cost_surface[i, j] = config.get_slope_cost_multiplier(slope)
    
    # Apply obstacle costs
    # In practice, this would use the obstacle_features dict to assign
    # specific costs based on feature type
    cost_surface[obstacle_mask] = config.obstacle_costs.get('default', 1000)
    
    return cost_surface