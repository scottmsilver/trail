"""
GPX file generator for trail routes
"""
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import xml.etree.ElementTree as ET
from xml.dom import minidom


class GPXGenerator:
    """Generate GPX files from route data"""
    
    @staticmethod
    def create_gpx(
        path_with_slopes: List[Dict[str, Any]], 
        route_name: str = "Trail Route",
        route_description: str = "",
        stats: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a GPX file from path data with elevation and slope information
        
        Args:
            path_with_slopes: List of points with lat, lon, elevation, slope
            route_name: Name for the route
            route_description: Description of the route
            stats: Optional statistics about the route
            
        Returns:
            GPX XML string
        """
        # Create root GPX element with namespace
        gpx_attribs = {
            'version': '1.1',
            'creator': 'Trail Finder',
            'xmlns': 'http://www.topografix.com/GPX/1/1',
            'xmlns:xsi': 'http://www.w3.org/2001/XMLSchema-instance',
            'xsi:schemaLocation': 'http://www.topografix.com/GPX/1/1 http://www.topografix.com/GPX/1/1/gpx.xsd'
        }
        gpx = ET.Element('gpx', gpx_attribs)
        
        # Add metadata
        metadata = ET.SubElement(gpx, 'metadata')
        ET.SubElement(metadata, 'name').text = route_name
        ET.SubElement(metadata, 'desc').text = route_description or f"Trail route with {len(path_with_slopes)} points"
        ET.SubElement(metadata, 'time').text = datetime.now(timezone.utc).isoformat()
        
        # Add route statistics as extensions if available
        if stats:
            extensions = ET.SubElement(metadata, 'extensions')
            if 'distance_km' in stats:
                ET.SubElement(extensions, 'distance').text = f"{stats['distance_km']:.2f} km"
            if 'elevation_gain_m' in stats:
                ET.SubElement(extensions, 'elevation_gain').text = f"{stats['elevation_gain_m']} m"
            if 'max_slope' in stats:
                ET.SubElement(extensions, 'max_slope').text = f"{stats['max_slope']:.1f}°"
            if 'difficulty' in stats:
                ET.SubElement(extensions, 'difficulty').text = stats['difficulty']
        
        # Create track
        trk = ET.SubElement(gpx, 'trk')
        ET.SubElement(trk, 'name').text = route_name
        ET.SubElement(trk, 'type').text = 'Hiking'
        
        # Create track segment
        trkseg = ET.SubElement(trk, 'trkseg')
        
        # Add track points
        for i, point in enumerate(path_with_slopes):
            trkpt = ET.SubElement(trkseg, 'trkpt', {
                'lat': str(point['lat']),
                'lon': str(point['lon'])
            })
            
            # Add elevation if available
            if 'elevation' in point and point['elevation'] is not None:
                ET.SubElement(trkpt, 'ele').text = str(round(point['elevation'], 1))
            
            # Add time (interpolated)
            time = datetime.now(timezone.utc).isoformat()
            ET.SubElement(trkpt, 'time').text = time
            
            # Add extensions for slope and other data
            if 'slope' in point or i > 0:
                extensions = ET.SubElement(trkpt, 'extensions')
                if 'slope' in point:
                    ET.SubElement(extensions, 'slope').text = f"{point['slope']:.1f}"
                    # Add difficulty based on slope
                    difficulty = GPXGenerator._get_slope_difficulty(point['slope'])
                    ET.SubElement(extensions, 'difficulty').text = difficulty
        
        # Convert to pretty-printed string
        rough_string = ET.tostring(gpx, encoding='unicode')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ", encoding='UTF-8').decode('utf-8')
    
    @staticmethod
    def create_simple_gpx(
        path: List[tuple], 
        route_name: str = "Trail Route"
    ) -> str:
        """
        Create a simple GPX file from basic path coordinates
        
        Args:
            path: List of (lon, lat) tuples
            route_name: Name for the route
            
        Returns:
            GPX XML string
        """
        # Convert to path_with_slopes format
        path_with_slopes = []
        for lon, lat in path:
            path_with_slopes.append({
                'lat': lat,
                'lon': lon
            })
        
        return GPXGenerator.create_gpx(path_with_slopes, route_name)
    
    @staticmethod
    def _get_slope_difficulty(slope: float) -> str:
        """Get difficulty rating based on slope"""
        abs_slope = abs(slope)
        if abs_slope < 5:
            return "Easy"
        elif abs_slope < 10:
            return "Moderate"
        elif abs_slope < 15:
            return "Challenging"
        elif abs_slope < 20:
            return "Hard"
        elif abs_slope < 25:
            return "Very Hard"
        else:
            return "Extreme"
    
    @staticmethod
    def create_waypoints_gpx(
        waypoints: List[Dict[str, Any]],
        route_name: str = "Trail Waypoints"
    ) -> str:
        """
        Create a GPX file with waypoints (useful for debugging or key points)
        
        Args:
            waypoints: List of waypoint dicts with lat, lon, name, description
            route_name: Name for the waypoint collection
            
        Returns:
            GPX XML string
        """
        # Create root GPX element
        gpx_attribs = {
            'version': '1.1',
            'creator': 'Trail Finder',
            'xmlns': 'http://www.topografix.com/GPX/1/1',
            'xmlns:xsi': 'http://www.w3.org/2001/XMLSchema-instance',
            'xsi:schemaLocation': 'http://www.topografix.com/GPX/1/1 http://www.topografix.com/GPX/1/1/gpx.xsd'
        }
        gpx = ET.Element('gpx', gpx_attribs)
        
        # Add metadata
        metadata = ET.SubElement(gpx, 'metadata')
        ET.SubElement(metadata, 'name').text = route_name
        ET.SubElement(metadata, 'time').text = datetime.now(timezone.utc).isoformat()
        
        # Add waypoints
        for i, waypoint in enumerate(waypoints):
            wpt = ET.SubElement(gpx, 'wpt', {
                'lat': str(waypoint['lat']),
                'lon': str(waypoint['lon'])
            })
            
            # Add name
            name = waypoint.get('name', f'Waypoint {i+1}')
            ET.SubElement(wpt, 'name').text = name
            
            # Add description if available
            if 'description' in waypoint:
                ET.SubElement(wpt, 'desc').text = waypoint['description']
            
            # Add elevation if available
            if 'elevation' in waypoint:
                ET.SubElement(wpt, 'ele').text = str(round(waypoint['elevation'], 1))
            
            # Add type
            ET.SubElement(wpt, 'type').text = waypoint.get('type', 'Waypoint')
        
        # Convert to pretty-printed string
        rough_string = ET.tostring(gpx, encoding='unicode')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ", encoding='UTF-8').decode('utf-8')