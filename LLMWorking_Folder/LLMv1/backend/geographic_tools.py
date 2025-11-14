"""
Enhanced geographic zone handling with polygon, oval, and complex shape support
"""
from shapely.geometry import Polygon, Point, MultiPolygon
from shapely.ops import transform
import pyproj
from typing import List, Tuple, Dict, Any
import numpy as np
import logging
# Try both import styles to work in different contexts
try:
    # Try relative import first
    from .water_bodies import (
    identify_water_body,
    get_water_body_by_name,
    get_water_bodies_by_type,
    get_all_water_body_names,
    filter_vessels_by_water_body,
    ALL_WATER_BODIES
)
except ImportError:
    # Fall back to direct import
    from water_bodies import (
        identify_water_body,
        get_water_body_by_name,
        get_water_bodies_by_type,
        get_all_water_body_names,
        filter_vessels_by_water_body,
        ALL_WATER_BODIES
    )
    
# Try both import styles to work in different contexts
try:
    # Try relative import first
    from .restricted_zones import (
    identify_restricted_zone,
    get_zone_by_name,
    get_zones_by_type,
    get_zones_by_risk_level,
    get_all_zone_names,
    check_zone_violation,
    get_zone_stats,
    filter_vessels_by_restricted_zone,
    ALL_RESTRICTED_ZONES
)
except ImportError:
    # Fall back to direct import
    from restricted_zones import (
        identify_restricted_zone,
        get_zone_by_name,
        get_zones_by_type,
        get_zones_by_risk_level,
        get_all_zone_names,
        check_zone_violation,
        get_zone_stats,
        filter_vessels_by_restricted_zone,
        ALL_RESTRICTED_ZONES
    )
    
# Try both import styles to work in different contexts
try:
    # Try relative import first
    from .vessel_types import (
    get_vessel_type_name,
    get_vessel_type_category,
    get_vessel_types_by_category,
    get_all_categories,
    filter_dataframe_by_vessel_type,
    filter_dataframe_by_category,
    get_vessel_type_stats,
    get_vessel_type_details,
    is_hazardous_cargo,
    is_commercial_vessel,
    is_special_purpose_vessel,
    VESSEL_TYPES
)
except ImportError:
    # Fall back to direct import
    from vessel_types import (
        get_vessel_type_name,
        get_vessel_type_category,
        get_vessel_types_by_category,
        get_all_categories,
        filter_dataframe_by_vessel_type,
        filter_dataframe_by_category,
        get_vessel_type_stats,
        get_vessel_type_details,
        is_hazardous_cargo,
        is_commercial_vessel,
        is_special_purpose_vessel,
        VESSEL_TYPES
    )

logger = logging.getLogger(__name__)

class GeographicZoneManager:
    """
    Handles complex geographic zones including polygons, ovals, and custom shapes
    """
    
    @staticmethod
    def create_polygon_zone(coordinates: List[Tuple[float, float]], name: str) -> Dict[str, Any]:
        """
        Create a polygon zone from list of (lon, lat) coordinates
        """
        if len(coordinates) < 3:
            raise ValueError("Polygon requires at least 3 points")
        
        # Ensure polygon is closed
        if coordinates[0] != coordinates[-1]:
            coordinates.append(coordinates[0])
        
        polygon = Polygon(coordinates)
        
        return {
            "name": name,
            "type": "polygon",
            "geometry": polygon,
            "coordinates": coordinates,
            "bounds": polygon.bounds,  # (minx, miny, maxx, maxy)
            "area_km2": GeographicZoneManager._calculate_area(polygon)
        }
    
    @staticmethod
    def create_oval_zone(center_lon: float, center_lat: float, 
                         major_axis_nm: float, minor_axis_nm: float,
                         rotation_degrees: float, name: str) -> Dict[str, Any]:
        """
        Create an oval (ellipse) zone
        center_lon, center_lat: center point
        major_axis_nm: length of major axis in nautical miles
        minor_axis_nm: length of minor axis in nautical miles
        rotation_degrees: rotation of major axis from north (0-360)
        """
        # Convert nautical miles to degrees (approximate)
        nm_to_deg_lat = 1.0 / 60.0
        nm_to_deg_lon = nm_to_deg_lat / np.cos(np.radians(center_lat))
        
        # Create ellipse points
        theta = np.linspace(0, 2 * np.pi, 100)
        a = major_axis_nm * nm_to_deg_lon / 2  # semi-major axis in degrees
        b = minor_axis_nm * nm_to_deg_lat / 2  # semi-minor axis in degrees
        
        # Ellipse in standard position
        x = a * np.cos(theta)
        y = b * np.sin(theta)
        
        # Rotate
        rotation_rad = np.radians(rotation_degrees)
        x_rot = x * np.cos(rotation_rad) - y * np.sin(rotation_rad)
        y_rot = x * np.sin(rotation_rad) + y * np.cos(rotation_rad)
        
        # Translate to center
        coords = [(center_lon + x_rot[i], center_lat + y_rot[i]) for i in range(len(theta))]
        
        polygon = Polygon(coords)
        
        return {
            "name": name,
            "type": "oval",
            "geometry": polygon,
            "coordinates": coords,
            "center": (center_lon, center_lat),
            "major_axis_nm": major_axis_nm,
            "minor_axis_nm": minor_axis_nm,
            "rotation": rotation_degrees,
            "bounds": polygon.bounds,
            "area_km2": GeographicZoneManager._calculate_area(polygon)
        }
    
    @staticmethod
    def create_circle_zone(center_lon: float, center_lat: float, 
                          radius_nm: float, name: str) -> Dict[str, Any]:
        """
        Create a circular zone
        """
        # Convert radius to degrees
        nm_to_deg = 1.0 / 60.0
        radius_deg_lat = radius_nm * nm_to_deg
        radius_deg_lon = radius_deg_lat / np.cos(np.radians(center_lat))
        
        # Create circle as polygon with many points
        theta = np.linspace(0, 2 * np.pi, 64)
        coords = [(center_lon + radius_deg_lon * np.cos(t),
                   center_lat + radius_deg_lat * np.sin(t)) for t in theta]
        
        polygon = Polygon(coords)
        
        return {
            "name": name,
            "type": "circle",
            "geometry": polygon,
            "coordinates": coords,
            "center": (center_lon, center_lat),
            "radius_nm": radius_nm,
            "bounds": polygon.bounds,
            "area_km2": GeographicZoneManager._calculate_area(polygon)
        }
    
    @staticmethod
    def create_rectangle_zone(min_lon: float, min_lat: float,
                             max_lon: float, max_lat: float, name: str) -> Dict[str, Any]:
        """
        Create a rectangular zone from bounds
        """
        coords = [
            (min_lon, min_lat),
            (max_lon, min_lat),
            (max_lon, max_lat),
            (min_lon, max_lat),
            (min_lon, min_lat)  # Close the polygon
        ]
        
        return GeographicZoneManager.create_polygon_zone(coords, name)
    
    @staticmethod
    def point_in_zone(lon: float, lat: float, zone: Dict[str, Any]) -> bool:
        """Check if a point is within a zone"""
        point = Point(lon, lat)
        return zone["geometry"].contains(point)
    
    @staticmethod
    def filter_dataframe_by_zone(df, zone: Dict[str, Any], 
                                  lon_col='LON', lat_col='LAT'):
        """Filter pandas/cudf dataframe to points within zone"""
        # Quick bounds check first (fast)
        minx, miny, maxx, maxy = zone["bounds"]
        df_filtered = df[
            (df[lon_col] >= minx) & (df[lon_col] <= maxx) &
            (df[lat_col] >= miny) & (df[lat_col] <= maxy)
        ]
        
        # Then precise polygon check
        mask = df_filtered.apply(
            lambda row: zone["geometry"].contains(Point(row[lon_col], row[lat_col])),
            axis=1
        )
        
        return df_filtered[mask]
    
    @staticmethod
    def _calculate_area(polygon: Polygon) -> float:
        """Calculate area in square kilometers using proper projection"""
        try:
            # Project to equal-area projection for accurate area calculation
            wgs84 = pyproj.CRS('EPSG:4326')
            # Use appropriate UTM zone or equal-area projection
            aeqd = pyproj.CRS('+proj=aeqd +lat_0={} +lon_0={} +x_0=0 +y_0=0'.format(
                polygon.centroid.y, polygon.centroid.x))
            
            project = pyproj.Transformer.from_crs(wgs84, aeqd, always_xy=True).transform
            polygon_proj = transform(project, polygon)
            
            # Area in square meters, convert to km²
            return polygon_proj.area / 1_000_000
        except Exception as e:
            logger.warning(f"Could not calculate exact area: {e}")
            # Fallback to rough estimate
            bounds = polygon.bounds
            width = bounds[2] - bounds[0]
            height = bounds[3] - bounds[1]
            return width * height * 111.32 * 111.32  # Rough km² estimate
    
    @staticmethod
    def named_region_to_zone(region_name: str) -> Dict[str, Any]:
        """Convert named maritime regions to zones"""
        regions = {
            "south china sea": {
                "coords": [(105, 5), (120, 5), (120, 25), (105, 25)],
                "name": "South China Sea"
            },
            "strait of hormuz": {
                "coords": [(55, 25), (57.5, 25), (57.5, 27), (55, 27)],
                "name": "Strait of Hormuz"
            },
            "gulf of aden": {
                "coords": [(43, 11), (51, 11), (51, 13), (43, 13)],
                "name": "Gulf of Aden"
            },
            "malacca strait": {
                "coords": [(99, 1), (104, 1), (104, 6), (99, 6)],
                "name": "Strait of Malacca"
            },
            "suez canal": {
                "coords": [(32.3, 29.9), (32.6, 29.9), (32.6, 31.3), (32.3, 31.3)],
                "name": "Suez Canal"
            },
            "panama canal": {
                "coords": [(-80.0, 8.9), (-79.5, 8.9), (-79.5, 9.4), (-80.0, 9.4)],
                "name": "Panama Canal"
            },
            "english channel": {
                "coords": [(-5, 49), (2, 49), (2, 51), (-5, 51)],
                "name": "English Channel"
            },
            "gibraltar strait": {
                "coords": [(-5.8, 35.8), (-5.0, 35.8), (-5.0, 36.2), (-5.8, 36.2)],
                "name": "Strait of Gibraltar"
            },
            "bosporus": {
                "coords": [(28.9, 40.9), (29.2, 40.9), (29.2, 41.3), (28.9, 41.3)],
                "name": "Bosporus Strait"
            },
            "persian gulf": {
                "coords": [(48, 24), (56, 24), (56, 30), (48, 30)],
                "name": "Persian Gulf"
            },
            "red sea": {
                "coords": [(32, 12), (43, 12), (43, 30), (32, 30)],
                "name": "Red Sea"
            },
            "mediterranean sea": {
                "coords": [(-6, 30), (36, 30), (36, 46), (-6, 46)],
                "name": "Mediterranean Sea"
            },
            # Add more regions as needed
        }
        
        region_key = region_name.lower().strip()
        if region_key in regions:
            region = regions[region_key]
            return GeographicZoneManager.create_polygon_zone(
                region["coords"], 
                region["name"]
            )
        else:
            raise ValueError(f"Unknown region: {region_name}")
    
    @staticmethod
    def get_available_regions() -> List[str]:
        """Get list of predefined region names"""
        return [
            "South China Sea",
            "Strait of Hormuz",
            "Gulf of Aden",
            "Malacca Strait",
            "Suez Canal",
            "Panama Canal",
            "English Channel",
            "Strait of Gibraltar",
            "Bosporus",
            "Persian Gulf",
            "Red Sea",
            "Mediterranean Sea"
        ]
    
    # ============================================================
    # WATER BODY INTEGRATION
    # ============================================================
    
    @staticmethod
    def identify_location(lon: float, lat: float) -> Dict[str, Any]:
        """
        Identify what water body a location is in.
        
        Args:
            lon: Longitude
            lat: Latitude
            
        Returns:
            Water body information or None
        """
        water_body = identify_water_body(lon, lat)
        if water_body:
            logger.info(f"Location ({lon}, {lat}) identified as: {water_body['name']}")
            return water_body
        else:
            logger.warning(f"Location ({lon}, {lat}) not in any defined water body")
            return None
    
    @staticmethod
    def get_water_body_zone(water_body_name: str) -> Dict[str, Any]:
        """
        Get a water body definition as a zone.
        
        Args:
            water_body_name: Name of the water body
            
        Returns:
            Zone dictionary with geometry
        """
        wb = get_water_body_by_name(water_body_name)
        if wb is None:
            raise ValueError(f"Unknown water body: {water_body_name}")
        
        # Convert to Shapely polygon
        coords = [(lon, lat) for lon, lat in wb["polygon"]]
        polygon = Polygon(coords)
        
        return {
            "name": wb["name"],
            "type": wb["type"],
            "parent": wb.get("parent", ""),
            "geometry": polygon,
            "coordinates": wb["polygon"],
            "bounds": polygon.bounds,
            "area_km2": GeographicZoneManager._calculate_area(polygon)
        }
    
    @staticmethod
    def list_all_water_bodies() -> List[str]:
        """
        Get list of all defined water body names.
        
        Returns:
            List of water body names
        """
        return get_all_water_body_names()
    
    @staticmethod
    def list_water_bodies_by_type(water_type: str) -> List[str]:
        """
        Get water bodies of a specific type.
        
        Args:
            water_type: Type (ocean, sea, gulf, bay, strait, canal)
            
        Returns:
            List of water body names
        """
        water_bodies = get_water_bodies_by_type(water_type)
        return [wb["name"] for wb in water_bodies]
    
    @staticmethod
    def filter_by_water_body(df, water_body_name: str):
        """
        Filter a vessel DataFrame by water body.
        
        Args:
            df: DataFrame with LON and LAT columns
            water_body_name: Name of the water body
            
        Returns:
            Filtered DataFrame
        """
        return filter_vessels_by_water_body(df, water_body_name)
    
    @staticmethod
    def get_water_body_stats() -> Dict[str, Any]:
        """
        Get statistics about defined water bodies.
        
        Returns:
            Dictionary with counts by type
        """
        stats = {
            "total": len(ALL_WATER_BODIES),
            "by_type": {}
        }
        
        for water_type in ["ocean", "sea", "gulf", "bay", "strait", "canal"]:
            bodies = get_water_bodies_by_type(water_type)
            stats["by_type"][water_type] = {
                "count": len(bodies),
                "names": [wb["name"] for wb in bodies]
            }
        
        return stats
    
    # ============================================================
    # RESTRICTED ZONES INTEGRATION
    # ============================================================
    
    @staticmethod
    def check_zone_violations(lon: float, lat: float) -> Dict[str, Any]:
        """
        Check if a location violates any restricted zones.
        
        Args:
            lon: Longitude
            lat: Latitude
            
        Returns:
            Violation details with zones and risk levels
        """
        return check_zone_violation(lon, lat)
    
    @staticmethod
    def identify_restricted_zones(lon: float, lat: float) -> List[Dict[str, Any]]:
        """
        Identify all restricted zones at a location.
        
        Args:
            lon: Longitude
            lat: Latitude
            
        Returns:
            List of restricted zone dictionaries
        """
        return identify_restricted_zone(lon, lat)
    
    @staticmethod
    def get_restricted_zone_by_name(zone_name: str) -> Dict[str, Any]:
        """
        Get a restricted zone by name.
        
        Args:
            zone_name: Name of the zone
            
        Returns:
            Zone dictionary or None
        """
        return get_zone_by_name(zone_name)
    
    @staticmethod
    def list_all_restricted_zones() -> List[str]:
        """
        Get list of all restricted zone names.
        
        Returns:
            List of zone names
        """
        return get_all_zone_names()
    
    @staticmethod
    def list_restricted_zones_by_type(zone_type: str) -> List[str]:
        """
        Get restricted zones of a specific type.
        
        Args:
            zone_type: Type (piracy_risk, sanctions, contested, military, 
                            marine_protected, strategic_chokepoint, fishing_restricted)
            
        Returns:
            List of zone names
        """
        zones = get_zones_by_type(zone_type)
        return [z["name"] for z in zones]
    
    @staticmethod
    def list_restricted_zones_by_risk(risk_level: str) -> List[str]:
        """
        Get restricted zones of a specific risk level.
        
        Args:
            risk_level: Risk level (critical, high, medium, low)
            
        Returns:
            List of zone names
        """
        zones = get_zones_by_risk_level(risk_level)
        return [z["name"] for z in zones]
    
    @staticmethod
    def filter_by_restricted_zone(df, zone_name: str):
        """
        Filter a vessel DataFrame by restricted zone.
        
        Args:
            df: DataFrame with LON and LAT columns
            zone_name: Name of the restricted zone
            
        Returns:
            Filtered DataFrame
        """
        return filter_vessels_by_restricted_zone(df, zone_name)
    
    @staticmethod
    def get_restricted_zone_stats() -> Dict[str, Any]:
        """
        Get statistics about defined restricted zones.
        
        Returns:
            Dictionary with counts by type and risk level
        """
        return get_zone_stats()
    
    # ============================================================
    # VESSEL TYPES INTEGRATION
    # ============================================================
    
    @staticmethod
    def get_vessel_type_name(vessel_type_code: int) -> str:
        """
        Get the name of a vessel type by its code.
        
        Args:
            vessel_type_code: AIS vessel type code (20-99)
            
        Returns:
            Vessel type name
        """
        return get_vessel_type_name(vessel_type_code)
    
    @staticmethod
    def get_vessel_type_category(vessel_type_code: int) -> str:
        """
        Get the category of a vessel type.
        
        Args:
            vessel_type_code: AIS vessel type code (20-99)
            
        Returns:
            Category name
        """
        return get_vessel_type_category(vessel_type_code)
    
    @staticmethod
    def get_vessel_types_by_category(category: str) -> List[int]:
        """
        Get all vessel type codes for a category.
        
        Args:
            category: Category name (WIG, Special, HSC, Special Purpose, Passenger, Cargo, Tanker, Other)
            
        Returns:
            List of vessel type codes
        """
        return get_vessel_types_by_category(category)
    
    @staticmethod
    def list_all_vessel_categories() -> List[str]:
        """
        Get list of all vessel categories.
        
        Returns:
            List of category names
        """
        return get_all_categories()
    
    @staticmethod
    def filter_by_vessel_types(df, vessel_type_codes: List[int]):
        """
        Filter a DataFrame by vessel type codes.
        
        Args:
            df: DataFrame with VesselType column
            vessel_type_codes: List of vessel type codes to include
            
        Returns:
            Filtered DataFrame
        """
        return filter_dataframe_by_vessel_type(df, vessel_type_codes)
    
    @staticmethod
    def filter_by_vessel_categories(df, categories: List[str]):
        """
        Filter a DataFrame by vessel categories.
        
        Args:
            df: DataFrame with VesselType column
            categories: List of category names
            
        Returns:
            Filtered DataFrame
        """
        return filter_dataframe_by_category(df, categories)
    
    @staticmethod
    def get_vessel_type_stats() -> Dict[str, Any]:
        """
        Get statistics about defined vessel types.
        
        Returns:
            Dictionary with counts by category
        """
        return get_vessel_type_stats()
    
    @staticmethod
    def get_vessel_type_info(vessel_type_code: int) -> Dict[str, Any]:
        """
        Get full details about a vessel type.
        
        Args:
            vessel_type_code: AIS vessel type code
            
        Returns:
            Dictionary with vessel type details or None
        """
        return get_vessel_type_details(vessel_type_code)
    
    @staticmethod
    def is_hazardous_vessel(vessel_type_code: int) -> bool:
        """
        Check if a vessel type carries hazardous cargo.
        
        Args:
            vessel_type_code: AIS vessel type code
            
        Returns:
            Boolean indicating if vessel carries hazardous materials
        """
        return is_hazardous_cargo(vessel_type_code)
    
    @staticmethod
    def is_commercial(vessel_type_code: int) -> bool:
        """
        Check if a vessel is commercial (cargo/tanker).
        
        Args:
            vessel_type_code: AIS vessel type code
            
        Returns:
            Boolean indicating if vessel is commercial
        """
        return is_commercial_vessel(vessel_type_code)
    
    @staticmethod
    def is_special_purpose(vessel_type_code: int) -> bool:
        """
        Check if a vessel is special purpose.
        
        Args:
            vessel_type_code: AIS vessel type code
            
        Returns:
            Boolean indicating if vessel is special purpose
        """
        return is_special_purpose_vessel(vessel_type_code)

