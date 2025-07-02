"""
More aggressive slope configuration for better steep slope avoidance
"""

import numpy as np
from app.services.obstacle_config import ObstacleConfig

class AggressiveSlopeConfig(ObstacleConfig):
    """Obstacle configuration with much more aggressive slope penalties"""
    
    def __init__(self, max_acceptable_slope=20):
        super().__init__()
        self.max_acceptable_slope = max_acceptable_slope
        self.use_continuous_slope = True
        self.slope_profile = 'aggressive_custom'
    
    def continuous_slope_cost(self, slope_degrees: float) -> float:
        """
        Very aggressive continuous slope cost function
        
        This creates exponential growth in cost for slopes above threshold
        """
        # Clamp slope to reasonable range
        slope = max(0, min(slope_degrees, 89))
        
        if self.slope_profile == 'aggressive_custom':
            # Extremely aggressive penalties for slopes above threshold
            if slope <= 5:
                return 1 + 0.1 * slope  # Minimal increase for gentle slopes
            elif slope <= 10:
                return 1.5 + 0.5 * (slope - 5)  # Moderate increase
            elif slope <= 15:
                return 4 + 2 * (slope - 10)  # Steeper increase
            elif slope <= self.max_acceptable_slope:
                # Rapid exponential increase approaching max
                return 14 + 10 * np.exp(0.3 * (slope - 15))
            else:
                # Extreme penalty above max acceptable slope
                # This makes slopes above threshold essentially impassable
                return 1000 * np.exp(0.2 * (slope - self.max_acceptable_slope))
        else:
            # Fall back to parent implementation
            return super().continuous_slope_cost(slope_degrees)


def test_aggressive_config():
    """Test the aggressive slope configuration"""
    config = AggressiveSlopeConfig(max_acceptable_slope=20)
    
    print("Aggressive Slope Configuration Test")
    print("Max acceptable slope: 20°")
    print("-" * 40)
    print("Slope | Cost Multiplier")
    print("-" * 40)
    
    test_slopes = [0, 5, 10, 15, 18, 20, 22, 25, 30, 35, 40]
    for slope in test_slopes:
        cost = config.get_slope_cost_multiplier(slope)
        if cost > 10000:
            print(f"{slope:4}° | {cost:.1e} (effectively impassable)")
        else:
            print(f"{slope:4}° | {cost:8.1f}x")
    
    return config

if __name__ == "__main__":
    test_aggressive_config()