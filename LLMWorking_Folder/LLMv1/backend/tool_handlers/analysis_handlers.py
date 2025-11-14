"""
Analysis Tool Handlers

Handlers for anomaly analysis, vessel history, and high-risk vessel identification.
"""

from typing import Dict, Any
from .registry import tool_handler
from .dependencies import get_session_manager, get_analysis_engine
import logging

logger = logging.getLogger(__name__)


@tool_handler("run_anomaly_analysis")
async def handle_anomaly_analysis(tool_input: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    """
    Execute AIS anomaly detection analysis.
    
    Args:
        tool_input: Analysis parameters (dates, zone, anomaly types, mmsi filter, vessel_types)
        session_id: User session ID
    
    Returns:
        Analysis result with anomalies and statistics
    """
    session_manager = get_session_manager()
    analysis_engine = get_analysis_engine(session_id)
    
    if not analysis_engine:
        return {"success": False, "error": "Analysis engine not available or session not found"}
    
    # Get progress callback from session if available
    session = session_manager.get_session(session_id)
    progress_callback = session.get('progress_callback') if session else None
    
    # Get vessel types and mmsi filter
    vessel_types = tool_input.get("vessel_types")
    mmsi_filter = tool_input.get("mmsi_filter")
    
    result = await analysis_engine.run_analysis(
        start_date=tool_input["start_date"],
        end_date=tool_input["end_date"],
        geographic_zone=tool_input.get("geographic_zone"),
        anomaly_types=tool_input.get("anomaly_types", []),
        mmsi_filter=mmsi_filter,
        vessel_types=vessel_types,  # Pass vessel types for filtering
        progress_callback=progress_callback
    )
    
    # Include progress updates in result so LLM can communicate them
    if session:
        progress_updates = session.get('progress_updates', [])
        if progress_updates:
            result['progress_updates'] = progress_updates
            # Clear progress updates after including them
            session['progress_updates'] = []
    
    # Store result in session
    if result['success']:
        session_manager.store_analysis_result(
            session_id,
            result['analysis_id'],
            result
        )
    
    return result


@tool_handler("get_vessel_history")
async def handle_vessel_history(tool_input: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    """
    Retrieve complete movement history for specific vessel(s).
    
    Args:
        tool_input: MMSI list and date range
        session_id: User session ID
    
    Returns:
        Vessel track data
    """
    analysis_engine = get_analysis_engine(session_id)
    
    if not analysis_engine:
        return {"success": False, "error": "Analysis engine not available or session not found"}
    
    result = await analysis_engine.get_vessel_tracks(
        mmsi_list=tool_input["mmsi"],
        start_date=tool_input["start_date"],
        end_date=tool_input["end_date"]
    )
    
    return result


@tool_handler("identify_high_risk_vessels")
async def handle_high_risk_vessels(tool_input: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    """
    Find vessels with most anomalies matching criteria.
    
    Args:
        tool_input: Date range, optional zone, min anomalies, top N
        session_id: User session ID
    
    Returns:
        List of high-risk vessels sorted by anomaly count
    """
    analysis_engine = get_analysis_engine(session_id)
    
    if not analysis_engine:
        return {"success": False, "error": "Analysis engine not available or session not found"}
    
    result = await analysis_engine.get_top_anomaly_vessels(
        start_date=tool_input["start_date"],
        end_date=tool_input["end_date"],
        geographic_zone=tool_input.get("geographic_zone"),
        min_anomalies=tool_input.get("min_anomalies", 1),
        top_n=tool_input.get("top_n", 10)
    )
    
    return result

