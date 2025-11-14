"""
UI Interaction Tool Handlers

Handlers for triggering frontend popups and dialogs for user input collection.
"""

from typing import Dict, Any
from .registry import tool_handler
import logging

logger = logging.getLogger(__name__)


@tool_handler("request_date_selection")
async def handle_request_date_selection(tool_input: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    """
    Request user to select date range via date picker popup.
    
    Args:
        tool_input: 
            - message: Optional message to display to user
            - default_start: Optional default start date (YYYY-MM-DD)
            - default_end: Optional default end date (YYYY-MM-DD)
            - min_date: Optional minimum selectable date (YYYY-MM-DD)
            - max_date: Optional maximum selectable date (YYYY-MM-DD)
        session_id: User session ID
    
    Returns:
        Result indicating popup was triggered
    """
    message = tool_input.get("message", "Please select the date range for analysis")
    # Default dates: 15-17 October 2024
    default_start = tool_input.get("default_start", "2024-10-15")
    default_end = tool_input.get("default_end", "2024-10-17")
    min_date = tool_input.get("min_date", "2024-10-15")
    max_date = tool_input.get("max_date", "2025-03-30")
    
    logger.info(f"Requesting date selection from user: {message}")
    
    return {
        "success": True,
        "action": "show_date_picker",
        "message": message,
        "default_start": default_start,
        "default_end": default_end,
        "min_date": min_date,
        "max_date": max_date,
        "popup_type": "date_selection"
    }


@tool_handler("request_vessel_type_selection")
async def handle_request_vessel_type_selection(tool_input: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    """
    Request user to select vessel types via checkbox popup.
    
    Args:
        tool_input:
            - message: Optional message to display to user
            - default_types: Optional list of default selected types
            - allow_multiple: Whether to allow multiple selections (default: True)
        session_id: User session ID
    
    Returns:
        Result indicating popup was triggered
    """
    from vessel_types import VESSEL_TYPE_CATEGORIES
    
    message = tool_input.get("message", "Please select vessel types to analyze")
    # Default: VesselType 70 (Cargo)
    default_types = tool_input.get("default_types", ["Cargo"])
    allow_multiple = tool_input.get("allow_multiple", True)
    
    # Build list of available vessel types with descriptions
    vessel_type_options = []
    for category, codes in VESSEL_TYPE_CATEGORIES.items():
        vessel_type_options.append({
            "value": category,
            "label": category,
            "codes": codes,
            "description": f"Vessel types {codes[0]}-{codes[-1]}" if len(codes) > 1 else f"Vessel type {codes[0]}"
        })
    
    logger.info(f"Requesting vessel type selection from user: {message}")
    
    return {
        "success": True,
        "action": "show_vessel_type_picker",
        "message": message,
        "options": vessel_type_options,
        "default_types": default_types,
        "allow_multiple": allow_multiple,
        "popup_type": "vessel_type_selection"
    }


@tool_handler("request_anomaly_selection")
async def handle_request_anomaly_selection(tool_input: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    """
    Request user to select anomaly types via checkbox popup.
    
    Args:
        tool_input:
            - message: Optional message to display to user
            - default_types: Optional list of default selected anomaly types
            - allow_multiple: Whether to allow multiple selections (default: True)
        session_id: User session ID
    
    Returns:
        Result indicating popup was triggered
    """
    message = tool_input.get("message", "Please select anomaly types to detect")
    # Default: Exclude loitering and rendezvous
    default_types = tool_input.get("default_types", [
        "ais_beacon_on",
        "ais_beacon_off", 
        "excessive_travel_distance_fast",
        "cog-heading_inconsistency"
        # Excluded by default: "loitering", "rendezvous"
    ])
    allow_multiple = tool_input.get("allow_multiple", True)
    
    # Define available anomaly types
    anomaly_type_options = [
        {
            "value": "ais_beacon_on",
            "label": "AIS Beacon On",
            "description": "Vessel reappears after being off for extended period"
        },
        {
            "value": "ais_beacon_off",
            "label": "AIS Beacon Off",
            "description": "Vessel disappears for extended period (>6 hours)"
        },
        {
            "value": "excessive_travel_distance_fast",
            "label": "Speed Anomalies",
            "description": "Impossible travel distances between positions"
        },
        {
            "value": "cog-heading_inconsistency",
            "label": "COG/Heading Inconsistency",
            "description": "Large difference between course and heading"
        },
        {
            "value": "loitering",
            "label": "Loitering",
            "description": "Extended presence in small area (24+ hours)"
        },
        {
            "value": "rendezvous",
            "label": "Rendezvous",
            "description": "Multiple vessels meeting at sea"
        }
    ]
    
    logger.info(f"Requesting anomaly type selection from user: {message}")
    
    return {
        "success": True,
        "action": "show_anomaly_picker",
        "message": message,
        "options": anomaly_type_options,
        "default_types": default_types,
        "allow_multiple": allow_multiple,
        "popup_type": "anomaly_selection"
    }


@tool_handler("request_output_selection")
async def handle_request_output_selection(tool_input: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    """
    Request user to select output formats via checkbox popup.
    
    Args:
        tool_input:
            - message: Optional message to display to user
            - default_outputs: Optional list of default selected outputs
            - allow_multiple: Whether to allow multiple selections (default: True)
        session_id: User session ID
    
    Returns:
        Result indicating popup was triggered
    """
    message = tool_input.get("message", "Please select output formats to generate")
    # Default outputs: consolidated events CSV, heatmap, event map
    default_outputs = tool_input.get("default_outputs", [
        "consolidated_events_csv",
        "heatmap",
        "event_map"
    ])
    allow_multiple = tool_input.get("allow_multiple", True)
    
    # Define available output formats
    output_options = [
        {
            "value": "consolidated_events_csv",
            "label": "Consolidated Events CSV",
            "description": "All detected anomalies in CSV format"
        },
        {
            "value": "heatmap",
            "label": "Anomaly Heatmap",
            "description": "Geographic heatmap showing anomaly density"
        },
        {
            "value": "event_map",
            "label": "Event Map",
            "description": "Interactive map with all anomaly locations"
        },
        {
            "value": "statistics_csv",
            "label": "Statistics CSV",
            "description": "Summary statistics in CSV format"
        },
        {
            "value": "excel_report",
            "label": "Excel Report",
            "description": "Comprehensive Excel report with multiple sheets"
        },
        {
            "value": "vessel_tracks",
            "label": "Vessel Track Maps",
            "description": "Individual track maps for vessels with anomalies"
        }
    ]
    
    logger.info(f"Requesting output selection from user: {message}")
    
    return {
        "success": True,
        "action": "show_output_picker",
        "message": message,
        "options": output_options,
        "default_outputs": default_outputs,
        "allow_multiple": allow_multiple,
        "popup_type": "output_selection"
    }

