"""
Export Tool Handlers

Handlers for exporting analysis results to various formats.
"""

from typing import Dict, Any
import pandas as pd
from .registry import tool_handler
from .base import ExportToolHandler
import logging

logger = logging.getLogger(__name__)


class CSVExportHandler(ExportToolHandler):
    """Handler for CSV exports"""
    
    async def export_data(self, analysis_result: Dict, params: Dict[str, Any]) -> Dict[str, Any]:
        from export_utils import exporter
        
        export_type = params.get("export_type", "anomalies")
        
        if export_type == "anomalies":
            if not analysis_result.get('anomalies'):
                return {'success': False, 'error': 'No anomalies to export'}
            
            anomalies_df = pd.DataFrame(analysis_result['anomalies'])
            result = exporter.export_anomalies_csv(anomalies_df)
        else:  # statistics
            statistics = analysis_result.get('statistics', {})
            result = exporter.export_statistics_csv(statistics)
        
        return result


class ExcelExportHandler(ExportToolHandler):
    """Handler for Excel exports"""
    
    async def export_data(self, analysis_result: Dict, params: Dict[str, Any]) -> Dict[str, Any]:
        from export_utils import exporter
        
        statistics = analysis_result.get('statistics', {})
        anomalies = analysis_result.get('anomalies', [])
        
        # Handle empty anomalies gracefully
        if anomalies:
            anomalies_df = pd.DataFrame(anomalies)
        else:
            anomalies_df = pd.DataFrame()
            # Add a message to statistics if no anomalies
            if 'total_anomalies' not in statistics:
                statistics['total_anomalies'] = 0
            if 'unique_vessels' not in statistics:
                statistics['unique_vessels'] = 0
        
        result = exporter.export_statistics_excel(statistics, anomalies_df)
        return result


class ReportHandler(ExportToolHandler):
    """Handler for investigation report generation"""
    
    async def export_data(self, analysis_result: Dict, params: Dict[str, Any]) -> Dict[str, Any]:
        """Note: Reports don't follow the standard export pattern, handled separately"""
        return {"success": False, "error": "Use generate_investigation_report handler"}


# Register export handlers
@tool_handler("export_to_csv")
async def handle_csv_export(tool_input: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    """Export analysis results to CSV format."""
    from .dependencies import get_session_manager
    session_manager = get_session_manager()
    handler = CSVExportHandler(session_manager)
    return await handler.execute(tool_input, session_id)


@tool_handler("export_to_excel")
async def handle_excel_export(tool_input: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    """Export analysis results to Excel format with multiple sheets."""
    from .dependencies import get_session_manager
    session_manager = get_session_manager()
    handler = ExcelExportHandler(session_manager)
    return await handler.execute(tool_input, session_id)


@tool_handler("generate_investigation_report")
async def handle_investigation_report(tool_input: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    """Generate a comprehensive investigation report."""
    from .dependencies import get_session_manager
    session_manager = get_session_manager()
    
    session = session_manager.get_session(session_id)
    if not session:
        return {"success": False, "error": "Session not found"}
    
    analysis_engine = session.get('analysis_engine')
    if not analysis_engine:
        return {"success": False, "error": "Analysis engine not available"}
    
    result = await analysis_engine.generate_report(
        analysis_id=tool_input["analysis_id"],
        report_type=tool_input.get("report_type", "summary"),
        include_maps=tool_input.get("include_maps", True),
        include_vessel_details=tool_input.get("include_vessel_details", True)
    )
    
    return result


@tool_handler("list_available_exports")
async def handle_list_exports(tool_input: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    """List all available export files."""
    from export_utils import exporter
    
    exports = exporter.list_exports()
    
    # Add download URLs
    for export in exports:
        export['download_url'] = f"/api/exports/download/{export['filename']}"
    
    return {
        "success": True,
        "exports": exports,
        "count": len(exports)
    }

