"""
Analysis Tool Handlers

Handlers for anomaly analysis, vessel history, and high-risk vessel identification.
"""

from typing import Dict, Any, List, Optional
from .registry import tool_handler
from .dependencies import get_session_manager, get_analysis_engine
import logging

logger = logging.getLogger(__name__)

# Import output generation handlers
from .map_handlers import AllAnomaliesMapHandler
from .chart_handlers import AnomalyTypesChartHandler, TopVesselsChartHandler, AnomaliesByDateChartHandler
from .export_handlers import CSVExportHandler, ExcelExportHandler


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
    
    # Detailed logging for debugging
    logger.info(f"[DEBUG] Tool input received - vessel_types: {vessel_types} (type: {type(vessel_types)})")
    logger.info(f"[DEBUG] Full tool_input keys: {list(tool_input.keys())}")
    if vessel_types is not None:
        logger.info(f"[DEBUG] vessel_types details: value={vessel_types}, type={type(vessel_types)}, is_list={isinstance(vessel_types, list)}, is_str={isinstance(vessel_types, str)}")
    
    try:
        result = await analysis_engine.run_analysis(
            start_date=tool_input["start_date"],
            end_date=tool_input["end_date"],
            geographic_zone=tool_input.get("geographic_zone"),
            anomaly_types=tool_input.get("anomaly_types", []),
            mmsi_filter=mmsi_filter,
            vessel_types=vessel_types,  # Pass vessel types for filtering
            progress_callback=progress_callback,
            session_id=session_id  # NEW: Pass session_id for data caching
        )
        
        # Include progress updates in result so LLM can communicate them
        if session:
            progress_updates = session.get('progress_updates', [])
            if progress_updates:
                result['progress_updates'] = progress_updates
                # Clear progress updates after including them
                session['progress_updates'] = []
        
        # Store result in session only if successful
        if result.get('success', False):
            session_manager.store_analysis_result(
                session_id,
                result.get('analysis_id', 'unknown'),
                result
            )
            
            # Send "Compiling Outputs" message via progress callback if available
            # The progress_callback is stored in session and can send WebSocket messages
            if session:
                session_progress_callback = session.get('progress_callback')
                if session_progress_callback:
                    try:
                        await session_progress_callback("compiling_outputs", "[COMPILING] Compiling outputs...")
                    except Exception as e:
                        logger.warning(f"Failed to send compiling outputs message: {e}")
            
            # NEW: Auto-generate all default outputs
            logger.info(f"Auto-generating complete analysis package for session {session_id}")
            generated_files = await _generate_all_outputs(result, session_id, tool_input)
            result['files_generated'] = generated_files
            result['output_count'] = len(generated_files)
            logger.info(f"Generated {len(generated_files)} output files automatically")
        else:
            # Even if unsuccessful, ensure error message is clear for LLM
            logger.info(f"Analysis failed: {result.get('error', 'Unknown error')}")
            if 'missing_dates' in result:
                logger.info(f"Missing dates: {result['missing_dates']}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in anomaly analysis: {e}", exc_info=True)
        # Return graceful error that won't disrupt conversation
        return {
            'success': False,
            'error': f'An error occurred while analyzing data: {str(e)}. Please try again or select a different date range.',
            'anomalies': [],
            'date_range': {
                'start': tool_input.get("start_date", "unknown"),
                'end': tool_input.get("end_date", "unknown")
            }
        }


# ============================================================
# AUTO-GENERATE ALL OUTPUTS (NEW - Phase 2)
# ============================================================

async def _generate_all_outputs(analysis_result: Dict[str, Any], 
                                session_id: str,
                                tool_input: Dict[str, Any],
                                output_formats: Optional[List[str]] = None) -> List[Dict[str, str]]:
    """
    Generate analysis outputs based on requested formats.
    
    If output_formats is None or empty, generates all default outputs:
    1. All Anomalies Map (HTML) - "event_map"
    2. Anomaly Types Chart (PNG)
    3. Top Vessels Chart (PNG)
    4. Anomalies by Date Chart (PNG)
    5. Anomalies CSV - "consolidated_events_csv"
    6. Statistics CSV
    7. Excel Report
    
    If output_formats is provided, only generates the requested formats.
    Supported format names:
    - "event_map" or "all_anomalies_map" -> All Anomalies Map
    - "consolidated_events_csv" -> Anomalies CSV
    - "statistics_csv" -> Statistics CSV
    - "excel_report" -> Excel Report
    - "anomaly_types_chart" -> Anomaly Types Chart
    - "top_vessels_chart" -> Top Vessels Chart
    - "anomalies_by_date_chart" -> Anomalies by Date Chart
    
    Args:
        analysis_result: Result from run_anomaly_analysis
        session_id: User session ID
        tool_input: Original tool input parameters
        output_formats: Optional list of output format names to generate
    
    Returns:
        List of generated file info dicts
    """
    generated_files = []
    
    # Get session_manager for handler initialization
    session_manager = get_session_manager()
    if not session_manager:
        logger.error("Session manager not available for output generation")
        return generated_files
    
    # Prepare common parameters
    params = {
        'analysis_id': analysis_result.get('analysis_id'),
        'start_date': tool_input.get('start_date'),
        'end_date': tool_input.get('end_date'),
        '_session_id': session_id
    }
    
    # If no output_formats specified, generate all outputs
    if not output_formats:
        output_formats = [
            'event_map',
            'anomaly_types_chart',
            'top_vessels_chart',
            'anomalies_by_date_chart',
            'consolidated_events_csv',
            'statistics_csv',
            'excel_report'
        ]
    
    # Normalize output format names (handle aliases)
    normalized_formats = []
    format_aliases = {
        'event_map': 'event_map',
        'all_anomalies_map': 'event_map',
        'consolidated_events_csv': 'consolidated_events_csv',
        'statistics_csv': 'statistics_csv',
        'excel_report': 'excel_report',
        'anomaly_types_chart': 'anomaly_types_chart',
        'top_vessels_chart': 'top_vessels_chart',
        'anomalies_by_date_chart': 'anomalies_by_date_chart'
    }
    for fmt in output_formats:
        normalized = format_aliases.get(fmt, fmt)
        if normalized not in normalized_formats:
            normalized_formats.append(normalized)
    
    # 1. All Anomalies Map (event_map)
    if 'event_map' in normalized_formats:
        try:
            map_handler = AllAnomaliesMapHandler(session_manager)
            map_result = await map_handler.execute(params, session_id)
            if map_result.get('success'):
                generated_files.append({
                    'type': 'map',
                    'name': 'All Anomalies Map',
                    'path': map_result.get('file_path'),
                    'format': 'html'
                })
                logger.info(f"✓ Generated: All Anomalies Map")
        except Exception as e:
            logger.error(f"Failed to generate All Anomalies Map: {e}")
    
    # 2. Anomaly Types Distribution Chart
    if 'anomaly_types_chart' in normalized_formats:
        try:
            chart_handler = AnomalyTypesChartHandler(session_manager)
            chart_result = await chart_handler.execute(params, session_id)
            if chart_result.get('success'):
                generated_files.append({
                    'type': 'chart',
                    'name': 'Anomaly Types Distribution',
                    'path': chart_result.get('file_path'),
                    'format': 'png'
                })
                logger.info(f"✓ Generated: Anomaly Types Chart")
        except Exception as e:
            logger.error(f"Failed to generate Anomaly Types Chart: {e}")
    
    # 3. Top Vessels Chart
    if 'top_vessels_chart' in normalized_formats:
        try:
            chart_handler = TopVesselsChartHandler(session_manager)
            chart_result = await chart_handler.execute(params, session_id)
            if chart_result.get('success'):
                generated_files.append({
                    'type': 'chart',
                    'name': 'Top Vessels with Anomalies',
                    'path': chart_result.get('file_path'),
                    'format': 'png'
                })
                logger.info(f"✓ Generated: Top Vessels Chart")
        except Exception as e:
            logger.error(f"Failed to generate Top Vessels Chart: {e}")
    
    # 4. Anomalies by Date Chart
    if 'anomalies_by_date_chart' in normalized_formats:
        try:
            chart_handler = AnomaliesByDateChartHandler(session_manager)
            chart_result = await chart_handler.execute(params, session_id)
            if chart_result.get('success'):
                generated_files.append({
                    'type': 'chart',
                    'name': 'Anomalies by Date',
                    'path': chart_result.get('file_path'),
                    'format': 'png'
                })
                logger.info(f"✓ Generated: Anomalies by Date Chart")
        except Exception as e:
            logger.error(f"Failed to generate Anomalies by Date Chart: {e}")
    
    # 5. Anomalies CSV Export (consolidated_events_csv)
    if 'consolidated_events_csv' in normalized_formats:
        try:
            export_handler = CSVExportHandler(session_manager)
            csv_params = {**params, 'export_type': 'anomalies'}
            csv_result = await export_handler.execute(csv_params, session_id)
            if csv_result.get('success'):
                generated_files.append({
                    'type': 'export',
                    'name': 'Anomalies Summary (CSV)',
                    'path': csv_result.get('file_path'),
                    'format': 'csv'
                })
                logger.info(f"✓ Generated: Anomalies CSV")
        except Exception as e:
            logger.error(f"Failed to generate Anomalies CSV: {e}")
    
    # 6. Statistics CSV Export
    if 'statistics_csv' in normalized_formats:
        try:
            export_handler = CSVExportHandler(session_manager)
            stats_params = {**params, 'export_type': 'statistics'}
            stats_result = await export_handler.execute(stats_params, session_id)
            if stats_result.get('success'):
                generated_files.append({
                    'type': 'export',
                    'name': 'Analysis Statistics (CSV)',
                    'path': stats_result.get('file_path'),
                    'format': 'csv'
                })
                logger.info(f"✓ Generated: Statistics CSV")
        except Exception as e:
            logger.error(f"Failed to generate Statistics CSV: {e}")
    
    # 7. Excel Report
    if 'excel_report' in normalized_formats:
        try:
            export_handler = ExcelExportHandler(session_manager)
            excel_result = await export_handler.execute(params, session_id)
            if excel_result.get('success'):
                generated_files.append({
                    'type': 'export',
                    'name': 'Comprehensive Report (Excel)',
                    'path': excel_result.get('file_path'),
                    'format': 'xlsx'
                })
                logger.info(f"✓ Generated: Excel Report")
        except Exception as e:
            logger.error(f"Failed to generate Excel Report: {e}")
    
    requested_count = len(normalized_formats)
    logger.info(f"Output generation complete: {len(generated_files)}/{requested_count} requested outputs succeeded")
    return generated_files


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

