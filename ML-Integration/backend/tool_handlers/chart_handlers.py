"""
Chart Tool Handlers

Handlers for creating various chart visualizations using the ChartToolHandler base class.
All chart handlers follow a consistent pattern:
1. Get analysis result from session
2. Prepare DataFrame
3. Create specific chart
4. Return with download URL
"""

from typing import Dict, Any
import pandas as pd
from .registry import tool_handler
from .base import ChartToolHandler
import logging

logger = logging.getLogger(__name__)


class AnomalyTypesChartHandler(ChartToolHandler):
    """Handler for anomaly types distribution charts"""
    
    async def create_chart(self, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            from ..chart_creator import chart_creator
        except ImportError:
            from chart_creator import chart_creator
        return chart_creator.create_anomaly_types_distribution(
            anomalies_df=df,
            chart_type=params.get("chart_type", "bar")
        )


class TopVesselsChartHandler(ChartToolHandler):
    """Handler for top vessels by anomaly count charts"""
    
    async def create_chart(self, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            from ..chart_creator import chart_creator
        except ImportError:
            from chart_creator import chart_creator
        return chart_creator.create_top_vessels_chart(
            anomalies_df=df,
            top_n=params.get("top_n", 10)
        )


class AnomaliesByDateChartHandler(ChartToolHandler):
    """Handler for anomalies over time charts"""
    
    async def create_chart(self, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            from ..chart_creator import chart_creator
        except ImportError:
            from chart_creator import chart_creator
        return chart_creator.create_anomalies_by_date_chart(
            anomalies_df=df
        )


class Bar3DChartHandler(ChartToolHandler):
    """Handler for 3D bar charts"""
    
    async def create_chart(self, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            from ..chart_creator import chart_creator
        except ImportError:
            from chart_creator import chart_creator
        return chart_creator.create_3d_bar_chart(
            anomalies_df=df
        )


class ScatterplotChartHandler(ChartToolHandler):
    """Handler for interactive scatterplot charts"""
    
    async def create_chart(self, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            from ..chart_creator import chart_creator
        except ImportError:
            from chart_creator import chart_creator
        return chart_creator.create_scatterplot_interactive(
            anomalies_df=df
        )


class AnomalyTimelineChartHandler(ChartToolHandler):
    """Handler for anomaly timeline charts"""
    
    async def create_chart(self, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            from ..chart_creator import chart_creator
        except ImportError:
            from chart_creator import chart_creator
        return chart_creator.create_anomaly_timeline(
            anomalies_df=df,
            group_by=params.get("group_by", "day")
        )


class VesselTypeDistributionChartHandler(ChartToolHandler):
    """Handler for vessel type distribution charts"""
    
    async def create_chart(self, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            from ..chart_creator import chart_creator
        except ImportError:
            from chart_creator import chart_creator
        return chart_creator.create_vessel_type_distribution(
            anomalies_df=df
        )


class GeographicDistributionChartHandler(ChartToolHandler):
    """Handler for geographic distribution charts"""
    
    async def create_chart(self, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            from ..chart_creator import chart_creator
        except ImportError:
            from chart_creator import chart_creator
        return chart_creator.create_geographic_distribution(
            anomalies_df=df,
            by_zone=params.get("by_zone", True)
        )


# Register all chart handlers
@tool_handler("create_anomaly_types_chart")
async def handle_anomaly_types_chart(tool_input: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    """Create a chart showing distribution of anomaly types."""
    from .dependencies import get_session_manager
    handler = AnomalyTypesChartHandler(get_session_manager())
    return await handler.execute(tool_input, session_id)


@tool_handler("create_top_vessels_chart")
async def handle_top_vessels_chart(tool_input: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    """Create a chart showing vessels with most anomalies."""
    from .dependencies import get_session_manager
    session_manager = get_session_manager()
    handler = TopVesselsChartHandler(session_manager)
    return await handler.execute(tool_input, session_id)


@tool_handler("create_anomalies_by_date_chart")
async def handle_anomalies_by_date_chart(tool_input: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    """Create a chart showing anomalies over time."""
    from .dependencies import get_session_manager
    session_manager = get_session_manager()
    handler = AnomaliesByDateChartHandler(session_manager)
    return await handler.execute(tool_input, session_id)


@tool_handler("create_3d_bar_chart")
async def handle_3d_bar_chart(tool_input: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    """Create a 3D bar chart of anomaly data."""
    from .dependencies import get_session_manager
    session_manager = get_session_manager()
    handler = Bar3DChartHandler(session_manager)
    return await handler.execute(tool_input, session_id)


@tool_handler("create_scatterplot")
async def handle_scatterplot_chart(tool_input: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    """Create an interactive scatterplot of anomalies."""
    from .dependencies import get_session_manager
    session_manager = get_session_manager()
    handler = ScatterplotChartHandler(session_manager)
    return await handler.execute(tool_input, session_id)


@tool_handler("create_anomaly_timeline_chart")
async def handle_anomaly_timeline_chart(tool_input: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    """Create a timeline chart of anomalies."""
    from .dependencies import get_session_manager
    session_manager = get_session_manager()
    handler = AnomalyTimelineChartHandler(session_manager)
    return await handler.execute(tool_input, session_id)


@tool_handler("create_vessel_type_distribution_chart")
async def handle_vessel_type_chart(tool_input: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    """Create a chart showing vessel type distribution."""
    from .dependencies import get_session_manager
    session_manager = get_session_manager()
    handler = VesselTypeDistributionChartHandler(session_manager)
    return await handler.execute(tool_input, session_id)


@tool_handler("create_geographic_distribution_chart")
async def handle_geographic_chart(tool_input: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    """Create a chart showing geographic distribution of anomalies."""
    from .dependencies import get_session_manager
    session_manager = get_session_manager()
    handler = GeographicDistributionChartHandler(session_manager)
    return await handler.execute(tool_input, session_id)

