#!/usr/bin/env python3
"""Test and visualize continuous slope cost functions"""

import numpy as np
import matplotlib.pyplot as plt
from app.services.obstacle_config import ObstacleConfig

def plot_slope_costs():
    """Plot slope cost functions for different profiles"""
    
    # Create slope range
    slopes = np.linspace(0, 60, 1000)
    
    # Define profiles to test
    profiles = ['default', 'wheelchair', 'mountain_goat', 'trail_runner', 'city_walker']
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    plt.figure(figsize=(12, 8))
    
    # Plot continuous functions
    plt.subplot(2, 1, 1)
    plt.title('Continuous Slope Cost Functions')
    
    for profile, color in zip(profiles, colors):
        config = ObstacleConfig()
        config.use_continuous_slope = True
        config.slope_profile = profile
        
        costs = [config.get_slope_cost_multiplier(s) for s in slopes]
        # Handle infinity for plotting
        costs = [min(c, 20) for c in costs]
        
        plt.plot(slopes, costs, color=color, label=profile, linewidth=2)
    
    plt.xlabel('Slope (degrees)')
    plt.ylabel('Cost Multiplier')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 60)
    plt.ylim(0, 20)
    
    # Plot discrete functions for comparison
    plt.subplot(2, 1, 2)
    plt.title('Discrete Slope Cost Functions (Legacy)')
    
    # Default discrete points
    config = ObstacleConfig()
    config.use_continuous_slope = False
    
    discrete_slopes = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 60]
    discrete_costs = []
    
    for s in discrete_slopes:
        cost = config.get_slope_cost_multiplier(s)
        discrete_costs.append(min(cost, 20))
    
    # Plot interpolated line
    all_costs = [config.get_slope_cost_multiplier(s) for s in slopes]
    all_costs = [min(c, 20) for c in all_costs]
    plt.plot(slopes, all_costs, 'b-', label='Interpolated', linewidth=2)
    plt.plot(discrete_slopes, discrete_costs, 'bo', markersize=8, label='Defined Points')
    
    plt.xlabel('Slope (degrees)')
    plt.ylabel('Cost Multiplier')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 60)
    plt.ylim(0, 20)
    
    plt.tight_layout()
    plt.savefig('slope_cost_comparison.png', dpi=150)
    plt.show()
    
    # Print some sample values
    print("\nSample slope costs (continuous function):")
    print("Profile        5°    10°    15°    20°    30°    40°")
    print("-" * 55)
    
    for profile in profiles:
        config = ObstacleConfig()
        config.use_continuous_slope = True
        config.slope_profile = profile
        
        costs = []
        for slope in [5, 10, 15, 20, 30, 40]:
            cost = config.get_slope_cost_multiplier(slope)
            if cost == np.inf:
                costs.append("  ∞")
            else:
                costs.append(f"{cost:5.2f}")
        
        print(f"{profile:12} {' '.join(costs)}")

if __name__ == "__main__":
    plot_slope_costs()