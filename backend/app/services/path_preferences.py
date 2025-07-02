"""
Path preference system for encouraging following roads and trails
"""
from dataclasses import dataclass
from typing import Dict, Set, Optional
import numpy as np


@dataclass
class PathPreferences:
    """Configuration for path preferences (lower cost = more preferred)"""
    
    # Path type costs (multipliers applied to base terrain cost)
    path_costs: Dict[str, float] = None
    
    # OSM tags for preferred paths to fetch
    preferred_path_tags: Dict[str, list] = None
    
    # Whether to strongly prefer staying on paths once on them
    stick_to_paths: bool = True
    path_transition_penalty: float = 2.0  # Penalty for leaving a path
    
    def __post_init__(self):
        """Set default values if not provided"""
        if self.path_costs is None:
            self.path_costs = self.get_default_path_costs()
            
        if self.preferred_path_tags is None:
            self.preferred_path_tags = self.get_default_path_tags()
    
    @staticmethod
    def get_default_path_tags() -> Dict[str, list]:
        """OSM tags for paths we want to follow"""
        return {
            # All walkable highways in one list
            'highway': ['footway', 'path', 'track', 'pedestrian', 'steps', 
                       'cycleway', 'bridleway', 'trail', 'residential', 
                       'living_street', 'service', 'unclassified'],
            
            # Parks and recreation
            'leisure': ['park', 'nature_reserve', 'garden', 'common', 'playground'],
            
            # Natural surfaces we can walk on
            'natural': ['grassland', 'meadow', 'heath', 'scrub', 'beach', 'sand'],
            
            # Land use that might be walkable
            'landuse': ['grass', 'meadow', 'recreation_ground', 'village_green'],
            
            # Designated routes
            'route': ['hiking', 'foot', 'walking']
        }
    
    @staticmethod
    def get_default_path_costs() -> Dict[str, float]:
        """Cost multipliers for different path types (lower = more preferred)"""
        return {
            # Best: Natural dirt trails and paths (MOST PREFERRED)
            'trail': 0.2,      # Natural trails
            'path': 0.25,      # Dirt paths
            'track': 0.3,      # Dirt/gravel tracks
            'bridleway': 0.35, # Natural horse paths
            
            # Good: Natural terrain without obstacles
            'grass': 0.4,      # Open grass (better than roads!)
            'meadow': 0.4,     # Open meadows
            'nature_reserve': 0.35,
            'park': 0.45,
            'garden': 0.5,
            'beach': 0.6,      # Sandy areas
            
            # Okay: Dedicated pedestrian infrastructure (but still paved)
            'footway': 0.6,    # Sidewalks (less preferred than dirt)
            'pedestrian': 0.65, # Pedestrian areas
            'cycleway': 0.7,   # Bike paths
            'steps': 0.75,     # Stairs
            
            # Less preferred: Roads (necessary evil)
            'living_street': 0.8,
            'residential': 0.85,
            'service': 0.9,
            'unclassified': 0.9,
            'tertiary': 0.95,
            'secondary': 0.97,
            'primary': 0.99,
            
            # Default off-path natural terrain (still better than roads!)
            'off_path': 0.5    # Unobstructed natural terrain preferred over roads
        }
    
    def get_path_cost_multiplier(self, path_type: Optional[str]) -> float:
        """Get cost multiplier for a path type"""
        if path_type is None:
            return self.path_costs.get('off_path', 1.0)
        
        # Check for exact match
        if path_type in self.path_costs:
            return self.path_costs[path_type]
        
        # Default to off-path cost
        return self.path_costs.get('off_path', 1.0)


class PathPreferencePresets:
    """Preset configurations for different use cases"""
    
    @staticmethod
    def urban_walker() -> PathPreferences:
        """Urban areas - prefer sidewalks but take parks/grass when available"""
        return PathPreferences(
            path_costs={
                # Best options in urban areas
                'park': 0.2,        # Parks are best
                'garden': 0.25,
                'path': 0.25,       # Park paths
                'grass': 0.3,       # Grass areas
                
                # Good urban infrastructure
                'footway': 0.4,     # Sidewalks
                'pedestrian': 0.45,
                'steps': 0.5,
                'living_street': 0.6,
                
                # Less ideal but acceptable
                'residential': 0.7,
                'service': 0.8,
                
                # Avoid off-path in urban areas (might be private property)
                'off_path': 1.5
            },
            stick_to_paths=True,
            path_transition_penalty=3.0
        )
    
    @staticmethod
    def trail_seeker() -> PathPreferences:
        """Prefer natural trails and paths over any paved surfaces"""
        return PathPreferences(
            path_costs={
                # Natural surfaces - most preferred
                'trail': 0.15,
                'path': 0.2,
                'track': 0.25,
                'bridleway': 0.3,
                'nature_reserve': 0.2,
                'grass': 0.35,
                'meadow': 0.35,
                'park': 0.3,
                'off_path': 0.4,  # Natural terrain preferred over roads
                
                # Paved surfaces - less preferred
                'footway': 0.7,
                'pedestrian': 0.75,
                'residential': 0.9,
                'service': 0.95,
            },
            stick_to_paths=True,
            path_transition_penalty=2.0
        )
    
    @staticmethod
    def flexible_hiker() -> PathPreferences:
        """Mild preference for paths but willing to go off-trail"""
        return PathPreferences(
            path_costs={
                'trail': 0.5,
                'path': 0.6,
                'footway': 0.6,
                'track': 0.7,
                'residential': 0.8,
                'off_path': 1.0  # No penalty for off-path
            },
            stick_to_paths=False,
            path_transition_penalty=1.2
        )
    
    @staticmethod
    def direct_route() -> PathPreferences:
        """Minimal path preference - just avoid major roads"""
        return PathPreferences(
            path_costs={
                'motorway': 5.0,  # Still avoid highways
                'trunk': 4.0,
                'primary': 1.5,
                'secondary': 1.2,
                'off_path': 1.0
            },
            stick_to_paths=False,
            path_transition_penalty=1.0
        )