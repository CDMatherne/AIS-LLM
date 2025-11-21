"""
UI Interaction Tool Handlers

Handlers for triggering frontend popups and dialogs for user input collection.
"""

from typing import Dict, Any
from .registry import tool_handler
import logging
from datetime import datetime
import psutil

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
    # Default dates: January 1, 2021 to today (or recent date)
    from datetime import datetime
    today = datetime.now().strftime('%Y-%m-%d')
    default_start = tool_input.get("default_start", "2021-01-01")
    default_end = tool_input.get("default_end", today)
    min_date = tool_input.get("min_date", "2021-01-01")
    max_date = tool_input.get("max_date", today)  # Allow up to today
    
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
    Shows ALL 58 AIS vessel types grouped by 8 categories.
    
    Args:
        tool_input:
            - message: Optional message to display to user
            - default_types: Optional list of default selected types
            - allow_multiple: Whether to allow multiple selections (default: True)
        session_id: User session ID
    
    Returns:
        Result with comprehensive vessel type list for Claude to present to user
    """
    from ..vessel_types import VESSEL_TYPES, VESSEL_TYPES_BY_CATEGORY
    
    message = tool_input.get("message", "Please select vessel types to analyze from the complete list below")
    default_types = tool_input.get("default_types", ["Cargo"])
    allow_multiple = tool_input.get("allow_multiple", True)
    
    # Build comprehensive list with ALL 58 vessel types grouped by category
    vessel_type_options = []
    category_summary = []
    
    # Sort categories for consistent presentation
    category_order = ['WIG', 'Special', 'HSC', 'Special Purpose', 'Passenger', 'Cargo', 'Tanker', 'Other']
    
    for category in category_order:
        if category not in VESSEL_TYPES_BY_CATEGORY:
            continue
            
        codes = sorted(VESSEL_TYPES_BY_CATEGORY[category])
        
        # Add category header
        category_info = {
            "category": category,
            "code_range": f"{min(codes)}-{max(codes)}",
            "count": len(codes),
            "types": []
        }
        
        # Add each specific vessel type under this category
        for code in codes:
            if code in VESSEL_TYPES:
                details = VESSEL_TYPES[code]
                category_info["types"].append({
                    "code": code,
                    "name": details['name'],
                    "description": details.get('description', '')
                })
        
        vessel_type_options.append(category_info)
        category_summary.append(f"{category} ({min(codes)}-{max(codes)}): {len(codes)} types")
    
    logger.info(f"Requesting vessel type selection from user: {message}")
    logger.info(f"Providing {len(VESSEL_TYPES)} vessel types across {len(category_order)} categories")
    
    return {
        "success": True,
        "action": "show_vessel_type_picker",
        "message": message,
        "vessel_types_by_category": vessel_type_options,
        "category_summary": category_summary,
        "total_vessel_types": len(VESSEL_TYPES),
        "total_categories": len(category_order),
        "default_types": default_types,
        "allow_multiple": allow_multiple,
        "popup_type": "vessel_type_selection",
        "instructions": "Below are ALL 58 AIS vessel types organized by 8 categories. You can select one or multiple types for your analysis."
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
    # Default: Include core 5 types, exclude rendezvous (disabled)
    default_types = tool_input.get("default_types", [
        "ais_beacon_on",
        "ais_beacon_off",
        "excessive_travel_distance_fast",
        "cog-heading_inconsistency",
        "loitering"
        # Excluded/disabled: "rendezvous" (TOO COMPLEX)
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
        }
        # DISABLED - TOO COMPLEX FOR PROCESSING
        # {
        #     "value": "rendezvous",
        #     "label": "Rendezvous",
        #     "description": "Multiple vessels meeting at sea"
        # }
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
    # Default outputs: consolidated events CSV, event map (which includes heatmap)
    default_outputs = tool_input.get("default_outputs", [
        "consolidated_events_csv",
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
            "value": "event_map",
            "label": "Event Map (with Heatmap)",
            "description": "Interactive map with all anomaly locations, clustering, and heatmap overlay"
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


@tool_handler("check_analysis_memory")
async def handle_check_analysis_memory(tool_input: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    """
    Check if system has sufficient memory for analysis.
    Assumes 2GB per day of data.
    
    Args:
        tool_input:
            - start_date: Start date (YYYY-MM-DD)
            - end_date: End date (YYYY-MM-DD)
        session_id: User session ID
    
    Returns:
        Result with memory check status
    """
    try:
        # Parse dates
        start_date = datetime.strptime(tool_input["start_date"], "%Y-%m-%d")
        end_date = datetime.strptime(tool_input["end_date"], "%Y-%m-%d")
        
        # Calculate days
        days = (end_date - start_date).days + 1
        
        # Estimate required memory (2GB per day)
        required_gb = days * 2.0
        required_bytes = required_gb * 1024 * 1024 * 1024
        
        # Get available memory
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024 ** 3)
        
        # Check if sufficient (need 80% of required memory available)
        sufficient = available_gb >= (required_gb * 0.8)
        
        # Calculate recommended max days
        max_days = int(available_gb / 2.0) if not sufficient else days
        
        logger.info(f"Memory check: {days} days requires ~{required_gb:.1f}GB, {available_gb:.1f}GB available, sufficient={sufficient}")
        
        return {
            "success": True,
            "sufficient": sufficient,
            "days_requested": days,
            "required_memory_gb": round(required_gb, 1),
            "available_memory_gb": round(available_gb, 1),
            "total_memory_gb": round(memory.total / (1024 ** 3), 1),
            "memory_percent_used": memory.percent,
            "recommended_max_days": max_days,
            "message": f"System has {available_gb:.1f}GB available. Analysis of {days} days requires ~{required_gb:.1f}GB." + 
                      ("" if sufficient else f" Recommend reducing to {max_days} days or fewer.")
        }
        
    except Exception as e:
        logger.error(f"Error checking memory: {e}", exc_info=True)
        return {
            "success": False,
            "error": f"Failed to check memory: {str(e)}",
            "sufficient": True  # Default to True to not block analysis on error
        }

