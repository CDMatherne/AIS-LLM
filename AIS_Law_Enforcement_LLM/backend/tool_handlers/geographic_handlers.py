"""
Geographic Tool Handlers

Handlers for water bodies, restricted zones, vessel types, and geographic zones.
"""

from typing import Dict, Any
from .registry import tool_handler
from .dependencies import get_session_manager, get_analysis_engine
import logging

logger = logging.getLogger(__name__)


async def _get_latest_vessel_position(mmsi: str, session_id: str) -> Dict[str, Any]:
    """
    Helper function to get the latest known position for a vessel.
    
    Args:
        mmsi: Vessel MMSI number
        session_id: User session ID
    
    Returns:
        Dict with 'success', 'longitude', 'latitude', 'timestamp' or 'error'
    """
    session_manager = get_session_manager()
    session = session_manager.get_session(session_id)
    
    if not session:
        return {"success": False, "error": "Session not found"}
    
    # Get timespan from session
    timespan = session.get('analysis_timespan')
    if not timespan:
        return {
            "success": False,
            "error": "No analysis timespan set. Please set a date range first using set_analysis_timespan."
        }
    
    # Get analysis engine and fetch vessel data
    analysis_engine = get_analysis_engine(session_id)
    if not analysis_engine:
        return {"success": False, "error": "Analysis engine not available"}
    
    try:
        # Get vessel tracks for the timespan
        result = await analysis_engine.get_vessel_tracks(
            mmsi_list=[mmsi],
            start_date=timespan["start_date"],
            end_date=timespan["end_date"]
        )
        
        if not result.get("success"):
            return {"success": False, "error": result.get("error", "Could not retrieve vessel data")}
        
        tracks = result.get("tracks", {})
        if mmsi not in tracks:
            return {"success": False, "error": f"No data found for vessel MMSI {mmsi} in the specified date range"}
        
        vessel_track = tracks[mmsi]
        points = vessel_track.get("points", [])
        
        if not points:
            return {"success": False, "error": f"No position data found for vessel MMSI {mmsi}"}
        
        # Get the latest point (last in the sorted list)
        latest_point = points[-1]
        
        return {
            "success": True,
            "longitude": latest_point.get("LON"),
            "latitude": latest_point.get("LAT"),
            "timestamp": latest_point.get("BaseDateTime"),
            "sog": latest_point.get("SOG"),
            "cog": latest_point.get("COG")
        }
    except Exception as e:
        logger.error(f"Error getting vessel position for MMSI {mmsi}: {e}")
        return {"success": False, "error": f"Error retrieving vessel position: {str(e)}"}


@tool_handler("create_geographic_zone")
async def handle_create_zone(tool_input: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    """
    Create a new monitored geographic zone.
    
    Args:
        tool_input: Zone parameters (type, coordinates, name, etc.)
        session_id: User session ID
    
    Returns:
        Created zone details
    """
    session_manager = get_session_manager()
    from geographic_tools import GeographicZoneManager
    
    geo_manager = GeographicZoneManager()
    zone_type = tool_input["zone_type"]
    
    if zone_type == "polygon":
        zone = geo_manager.create_polygon_zone(
            tool_input["coordinates"],
            tool_input["name"]
        )
    elif zone_type == "circle":
        center = tool_input["coordinates"]
        zone = geo_manager.create_circle_zone(
            center[0], center[1],
            tool_input["radius_nm"],
            tool_input["name"]
        )
    elif zone_type == "oval":
        center = tool_input["coordinates"]
        zone = geo_manager.create_oval_zone(
            center[0], center[1],
            tool_input["major_axis_nm"],
            tool_input["minor_axis_nm"],
            tool_input.get("rotation", 0),
            tool_input["name"]
        )
    elif zone_type == "rectangle":
        coords = tool_input["coordinates"]
        zone = geo_manager.create_rectangle_zone(
            coords[0], coords[1], coords[2], coords[3],
            tool_input["name"]
        )
    else:
        return {"success": False, "error": f"Unknown zone type: {zone_type}"}
    
    # Save zone to session
    session_manager.add_zone(session_id, zone)
    
    # Return serializable version
    return {
        "success": True,
        "zone": {
            "name": zone["name"],
            "type": zone["type"],
            "bounds": zone["bounds"],
            "area_km2": zone["area_km2"]
        }
    }


@tool_handler("list_all_water_bodies")
async def handle_list_water_bodies(tool_input: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    """
    List all 24 defined water bodies.
    
    Returns:
        Water bodies organized by type
    """
    from geographic_tools import GeographicZoneManager
    
    geo_manager = GeographicZoneManager()
    water_bodies_stats = geo_manager.get_water_body_stats()
    
    return {
        "success": True,
        "total": water_bodies_stats["total"],
        "by_type": water_bodies_stats["by_type"],
        "message": f"There are {water_bodies_stats['total']} defined water bodies"
    }


@tool_handler("identify_vessel_location")
async def handle_identify_location(tool_input: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    """
    Identify which water body a location or vessel is in.
    
    Args:
        tool_input: Longitude, latitude, optional MMSI
        session_id: User session ID
    
    Returns:
        Water body identification
    """
    from geographic_tools import GeographicZoneManager
    
    geo_manager = GeographicZoneManager()
    
    # Get coordinates from tool_input
    lon = tool_input.get("longitude")
    lat = tool_input.get("latitude")
    
    # If MMSI provided but no coordinates, look up vessel position
    if "mmsi" in tool_input and tool_input["mmsi"] and (lon is None or lat is None):
        position_result = await _get_latest_vessel_position(tool_input["mmsi"], session_id)
        if not position_result.get("success"):
            return {
                "success": False,
                "error": position_result.get("error", "Could not retrieve vessel position"),
                "mmsi": tool_input["mmsi"]
            }
        lon = position_result["longitude"]
        lat = position_result["latitude"]
        logger.info(f"Retrieved position for MMSI {tool_input['mmsi']}: ({lon}, {lat})")
    
    # If still no coordinates, return error
    
    if lon is None or lat is None:
        return {"success": False, "error": "Missing coordinates"}
    
    location_info = geo_manager.identify_location(lon, lat)
    
    if location_info:
        return {
            "success": True,
            "location": {
                "longitude": lon,
                "latitude": lat,
                "water_body": location_info["name"],
                "type": location_info["type"],
                "parent": location_info.get("parent", "")
            },
            "message": f"Location ({lon}, {lat}) is in the {location_info['name']}"
        }
    else:
        return {
            "success": False,
            "location": {
                "longitude": lon,
                "latitude": lat
            },
            "message": f"Location ({lon}, {lat}) is not in any defined water body"
        }


@tool_handler("lookup_geographic_region")
async def handle_lookup_region(tool_input: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    """
    Convert named maritime region to geographic zone.
    
    Args:
        tool_input: Region name
        session_id: User session ID
    
    Returns:
        Zone details for the region
    """
    from geographic_tools import GeographicZoneManager
    
    geo_manager = GeographicZoneManager()
    region_name = tool_input["region_name"]
    
    # Try new water bodies system first
    try:
        zone = geo_manager.get_water_body_zone(region_name)
        return {
            "success": True,
            "region": region_name,
            "zone": {
                "name": zone["name"],
                "type": zone["type"],
                "parent": zone.get("parent", ""),
                "bounds": zone["bounds"],
                "area_km2": zone["area_km2"],
                "coordinates_count": len(zone["coordinates"])
            },
            "message": f"Water body '{zone['name']}' is a {zone['type']} covering approximately {zone['area_km2']:.0f} km²"
        }
    except ValueError:
        # Fall back to old named regions system
        try:
            zone = geo_manager.named_region_to_zone(region_name)
            return {
                "success": True,
                "region": region_name,
                "zone": {
                    "name": zone["name"],
                    "type": zone["type"],
                    "bounds": zone["bounds"],
                    "area_km2": zone["area_km2"]
                }
            }
        except ValueError:
            return {
                "success": False,
                "error": f"Unknown region: {region_name}. Use 'list_all_water_bodies' to see available regions."
            }


@tool_handler("list_vessel_types")
async def handle_list_vessel_types(tool_input: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    """
    List all 58 vessel type codes organized by category.
    
    Args:
        tool_input: Optional category filter
        session_id: User session ID
    
    Returns:
        Vessel types by category
    """
    from geographic_tools import GeographicZoneManager
    
    geo_manager = GeographicZoneManager()
    category = tool_input.get("category", "all")
    
    if category == "all":
        stats = geo_manager.get_vessel_type_stats()
        return {
            "success": True,
            "total_vessel_types": stats["total"],
            "categories": stats["by_category"],
            "message": f"There are {stats['total']} vessel types across 8 categories. Key types: Cargo (70-79), Tanker (80-89), Fishing (30), Law Enforcement (55)"
        }
    else:
        codes = geo_manager.get_vessel_types_by_category(category)
        vessel_list = []
        for code in codes:
            info = geo_manager.get_vessel_type_info(code)
            if info:
                vessel_list.append(f"{code}: {info['name']}")
        
        return {
            "success": True,
            "category": category,
            "vessel_types": vessel_list,
            "count": len(codes),
            "message": f"Category '{category}' contains {len(codes)} vessel types"
        }


@tool_handler("get_vessel_type_info")
async def handle_vessel_type_info(tool_input: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific vessel type code.
    
    Args:
        tool_input: Vessel type code
        session_id: User session ID
    
    Returns:
        Vessel type details
    """
    from geographic_tools import GeographicZoneManager
    
    geo_manager = GeographicZoneManager()
    vessel_type_code = tool_input["vessel_type_code"]
    info = geo_manager.get_vessel_type_info(vessel_type_code)
    
    if info:
        is_hazmat = geo_manager.is_hazardous_vessel(vessel_type_code)
        is_commercial = geo_manager.is_commercial(vessel_type_code)
        is_special = geo_manager.is_special_purpose(vessel_type_code)
        
        return {
            "success": True,
            "code": vessel_type_code,
            "name": info["name"],
            "category": info["category"],
            "description": info["description"],
            "is_hazardous_cargo": is_hazmat,
            "is_commercial": is_commercial,
            "is_special_purpose": is_special,
            "message": f"Vessel type {vessel_type_code}: {info['name']} ({info['category']})"
        }
    else:
        return {
            "success": False,
            "error": f"Unknown vessel type code: {vessel_type_code}. Valid codes are 20-99."
        }


@tool_handler("set_analysis_timespan")
async def handle_set_timespan(tool_input: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    """
    Set or update the date range for this session.
    
    Args:
        tool_input: Start and end dates
        session_id: User session ID
    
    Returns:
        Confirmation of timespan set
    """
    session_manager = get_session_manager()
    from datetime import datetime
    
    start_date = tool_input["start_date"]
    end_date = tool_input["end_date"]
    
    # Validate session exists using proper method
    session = session_manager.get_session(session_id)
    if not session:
        # Provide detailed error for debugging
        available_sessions = list(session_manager.sessions.keys()) if hasattr(session_manager, 'sessions') else []
        logger.error(
            f"Session {session_id} not found. "
            f"Available sessions: {len(available_sessions)} "
            f"({available_sessions[:3] if available_sessions else 'none'}...)"
        )
        return {
            "success": False,
            "error": f"Unable to set timespan - session '{session_id}' not found or expired. Please refresh your session."
        }
    
    # Store in session data
    session['analysis_timespan'] = {
        "start_date": start_date,
        "end_date": end_date,
        "set_at": datetime.now().isoformat()
    }
    logger.info(f"Session {session_id} timespan set: {start_date} to {end_date}")
    
    return {
        "success": True,
        "start_date": start_date,
        "end_date": end_date,
        "message": f"Analysis timespan set to {start_date} through {end_date}. You can now run anomaly analysis for this period."
    }


@tool_handler("get_current_timespan")
async def handle_get_timespan(tool_input: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    """
    Get the current date range for this session.
    
    Args:
        tool_input: None required
        session_id: User session ID
    
    Returns:
        Current timespan if set
    """
    session_manager = get_session_manager()
    
    # Validate session exists using proper method
    session = session_manager.get_session(session_id)
    if not session:
        return {
            "success": False,
            "error": f"Unable to get timespan - session '{session_id}' not found or expired."
        }
    
    timespan = session.get('analysis_timespan')
    if timespan:
        return {
            "success": True,
            "timespan_set": True,
            "start_date": timespan["start_date"],
            "end_date": timespan["end_date"],
            "set_at": timespan.get("set_at", "unknown"),
            "message": f"Current timespan: {timespan['start_date']} to {timespan['end_date']}"
        }
    else:
        return {
            "success": True,
            "timespan_set": False,
            "message": "No timespan has been set yet. Please use set_analysis_timespan to choose dates before running analysis."
        }

