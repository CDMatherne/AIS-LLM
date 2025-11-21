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
        try:
            from ..map_creator import map_creator, AISMapCreator
        except ImportError:
            from map_creator import map_creator, AISMapCreator
        
        # Get output directory from session if available
        output_dir = None
        if hasattr(self, 'session_manager'):
            session = self.session_manager.get_session(params.get('_session_id', ''))
            if session and 'output_directory' in session:
                output_dir = session['output_directory']
        
        # Use session-specific output directory if available, otherwise use global instance
        if output_dir:
            map_creator_instance = AISMapCreator(output_dir=output_dir)
            return map_creator_instance.create_all_anomalies_map(
                anomalies_df=df,
                show_clustering=params.get("show_clustering", True),
                show_heatmap=params.get("show_heatmap", True),  # Changed to True - heatmaps always generated (hidden by default)
                show_markers_with_heatmap=params.get("show_markers_with_heatmap", True),
                show_grid=params.get("show_grid", False),
                group_by_day=params.get("group_by_day", True),
                filter_by_anomaly_type=params.get("filter_by_anomaly_type"),
                filter_by_vessel_type=params.get("filter_by_vessel_type"),
                filter_by_mmsi=params.get("filter_by_mmsi")
            )
        else:
            return map_creator.create_all_anomalies_map(
                anomalies_df=df,
                show_clustering=params.get("show_clustering", True),
                show_heatmap=params.get("show_heatmap", True),  # Changed to True - heatmaps always generated (hidden by default)
                show_markers_with_heatmap=params.get("show_markers_with_heatmap", True),
                show_grid=params.get("show_grid", False),
                group_by_day=params.get("group_by_day", True),
                filter_by_anomaly_type=params.get("filter_by_anomaly_type"),
                filter_by_vessel_type=params.get("filter_by_vessel_type"),
                filter_by_mmsi=params.get("filter_by_mmsi")
            )


class VesselTrackMapHandler(MapToolHandler):
    """Handler for vessel track maps"""
    
    async def create_map(self, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create vessel track map. Note: This handler expects anomaly data in df.
        For full track data, use get_vessel_tracks first to get all position data.
        """
        try:
            from ..map_creator import map_creator
        except ImportError:
            from map_creator import map_creator
        mmsi = params.get("mmsi")
        
        if not mmsi:
            return {"success": False, "error": "MMSI is required for vessel track map"}
        
        # Filter data for specific vessel
        vessel_data = df[df['MMSI'] == int(mmsi)] if 'MMSI' in df.columns else pd.DataFrame()
        
        # Get full vessel track data from session if available
        # The handler should ideally receive full track data, not just anomalies
        # For now, use anomaly data as vessel data (limited, but works)
        # In production, this should call get_vessel_tracks to get all position data
        
        return map_creator.create_vessel_track_map(
            vessel_data=vessel_data,
            mmsi=mmsi,
            anomalies_df=vessel_data,
            show_all_points=params.get("show_all_points", True),
            filter_by_anomaly_type=params.get("filter_by_anomaly_type"),
            filter_by_vessel_type=params.get("filter_by_vessel_type")
        )


class AnomalyHeatmapHandler(MapToolHandler):
    """Handler for anomaly density heatmaps - now merged with all anomalies map"""
    
    async def create_map(self, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        # Heatmap is now always included in the all anomalies map
        # This handler just creates the same map (heatmaps are toggleable via layer control)
        try:
            from ..map_creator import map_creator, AISMapCreator
        except ImportError:
            from map_creator import map_creator, AISMapCreator
        
        # Get output directory from session if available
        output_dir = None
        if hasattr(self, 'session_manager'):
            session = self.session_manager.get_session(params.get('_session_id', ''))
            if session and 'output_directory' in session:
                output_dir = session['output_directory']
        
        # Use session-specific output directory if available, otherwise use global instance
        # Note: Heatmaps are now always included but hidden by default
        # Users can toggle them on via the layer control
        if output_dir:
            map_creator_instance = AISMapCreator(output_dir=output_dir)
            return map_creator_instance.create_all_anomalies_map(
                anomalies_df=df,
                output_filename="All Anomalies Map.html",  # Same filename as main map
                show_clustering=params.get("show_clustering", True),
                show_heatmap=False,  # Parameter kept for compatibility but heatmaps are always added
                show_markers_with_heatmap=True,  # Always show markers
                show_grid=False,
                title="AIS Anomalies Map with Heatmap",
                group_by_day=params.get("group_by_day", True),
                filter_by_anomaly_type=params.get("filter_by_anomaly_type"),
                filter_by_vessel_type=params.get("filter_by_vessel_type"),
                filter_by_mmsi=params.get("filter_by_mmsi")
            )
        else:
            return map_creator.create_all_anomalies_map(
                anomalies_df=df,
                output_filename="All Anomalies Map.html",  # Same filename as main map
                show_clustering=params.get("show_clustering", True),
                show_heatmap=False,  # Parameter kept for compatibility but heatmaps are always added
                show_markers_with_heatmap=True,  # Always show markers
                show_grid=False,
                title="AIS Anomalies Map with Heatmap",
                group_by_day=params.get("group_by_day", True),
                filter_by_anomaly_type=params.get("filter_by_anomaly_type"),
                filter_by_vessel_type=params.get("filter_by_vessel_type"),
                filter_by_mmsi=params.get("filter_by_mmsi")
            )


class AnomalyClusterMapHandler(MapToolHandler):
    """Handler for anomaly cluster maps"""
    
    async def create_map(self, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            from ..map_creator import map_creator
        except ImportError:
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

