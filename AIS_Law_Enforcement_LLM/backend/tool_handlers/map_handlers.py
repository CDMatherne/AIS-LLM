"""
Map Tool Handlers

Handlers for creating various map visualizations using the MapToolHandler base class.
"""

from typing import Dict, Any
import pandas as pd
from .registry import tool_handler
from .base import MapToolHandler
import logging

logger = logging.getLogger(__name__)


class AllAnomaliesMapHandler(MapToolHandler):
    """Handler for all anomalies map with clustering"""
    
    async def create_map(self, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        from map_creator import map_creator
        return map_creator.create_all_anomalies_map(
            anomalies_df=df,
            show_clustering=params.get("show_clustering", True),
            show_heatmap=params.get("show_heatmap", False),
            show_grid=params.get("show_grid", False)
        )


class VesselTrackMapHandler(MapToolHandler):
    """Handler for vessel track maps"""
    
    async def create_map(self, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        from map_creator import map_creator
        mmsi = params.get("mmsi")
        
        if not mmsi:
            return {"success": False, "error": "MMSI is required for vessel track map"}
        
        # Filter data for specific vessel
        vessel_data = df[df['MMSI'] == int(mmsi)] if 'MMSI' in df.columns else pd.DataFrame()
        
        return map_creator.create_vessel_track_map(
            vessel_data=vessel_data,
            mmsi=mmsi,
            anomalies_df=vessel_data
        )


class AnomalyHeatmapHandler(MapToolHandler):
    """Handler for anomaly density heatmaps"""
    
    async def create_map(self, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        from map_creator import map_creator
        return map_creator.create_all_anomalies_map(
            anomalies_df=df,
            output_filename="Anomaly_Heatmap.html",
            show_clustering=False,
            show_heatmap=True,
            show_grid=False,
            title="AIS Anomaly Density Heatmap"
        )


class AnomalyClusterMapHandler(MapToolHandler):
    """Handler for anomaly cluster maps"""
    
    async def create_map(self, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        from map_creator import map_creator
        return map_creator.create_all_anomalies_map(
            anomalies_df=df,
            output_filename="Anomaly_Clusters.html",
            show_clustering=True,
            show_heatmap=False,
            show_grid=params.get("show_grid", False),
            title="AIS Anomaly Clusters"
        )


# Register all map handlers
@tool_handler("create_all_anomalies_map")
async def handle_all_anomalies_map(tool_input: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    """Create an interactive map showing all detected anomalies."""
    from .dependencies import get_session_manager
    session_manager = get_session_manager()
    handler = AllAnomaliesMapHandler(session_manager)
    return await handler.execute(tool_input, session_id)


@tool_handler("create_vessel_track_map")
async def handle_vessel_track_map(tool_input: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    """Create a map showing the movement track of a specific vessel."""
    from .dependencies import get_session_manager
    session_manager = get_session_manager()
    handler = VesselTrackMapHandler(session_manager)
    return await handler.execute(tool_input, session_id)


@tool_handler("create_anomaly_heatmap")
async def handle_anomaly_heatmap(tool_input: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    """Create a density heatmap of anomalies."""
    from .dependencies import get_session_manager
    session_manager = get_session_manager()
    handler = AnomalyHeatmapHandler(session_manager)
    return await handler.execute(tool_input, session_id)


@tool_handler("create_anomaly_cluster_map")
async def handle_anomaly_cluster_map(tool_input: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    """Create a map showing anomaly clusters."""
    from .dependencies import get_session_manager
    session_manager = get_session_manager()
    handler = AnomalyClusterMapHandler(session_manager)
    return await handler.execute(tool_input, session_id)

