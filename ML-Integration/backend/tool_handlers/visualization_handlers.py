"""
Visualization Tool Handlers

Handlers for custom visualization creation and execution.
"""

from typing import Dict, Any
import pandas as pd
import os
from .registry import tool_handler
import logging

logger = logging.getLogger(__name__)


@tool_handler("create_custom_visualization")
async def handle_custom_visualization(tool_input: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    """
    Create and execute a custom visualization from user-provided code.
    
    Args:
        tool_input: Code, parameters, name, description, save option
        session_id: User session ID
    
    Returns:
        Visualization result with file path
    """
    from .dependencies import get_session_manager
    session_manager = get_session_manager()
    from visualization_engine import viz_engine
    
    # Get analysis result
    analysis_result = session_manager.get_analysis_result(
        session_id,
        tool_input["analysis_id"]
    )
    
    if not analysis_result:
        return {"error": "Analysis not found"}
    
    # Execute custom visualization
    anomalies_df = pd.DataFrame(analysis_result.get('anomalies', []))
    
    result = viz_engine.create_and_execute(
        code=tool_input["code"],
        data=anomalies_df,
        parameters=tool_input.get("parameters", {}),
        name=tool_input.get("name"),
        description=tool_input.get("description"),
        save_to_registry=tool_input.get("save_to_registry", False)
    )
    
    # Add download URL if successful
    if result.get('success') and 'file_path' in result:
        filename = os.path.basename(result['file_path'])
        result['download_url'] = f"/api/exports/download/{filename}"
    
    return result


@tool_handler("execute_saved_visualization")
async def handle_execute_visualization(tool_input: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    """
    Execute a previously saved visualization.
    
    Args:
        tool_input: Visualization ID, parameters
        session_id: User session ID
    
    Returns:
        Visualization result with file path
    """
    from .dependencies import get_session_manager
    session_manager = get_session_manager()
    from visualization_engine import viz_engine
    
    # Get analysis result
    analysis_result = session_manager.get_analysis_result(
        session_id,
        tool_input["analysis_id"]
    )
    
    if not analysis_result:
        return {"error": "Analysis not found"}
    
    # Execute saved visualization
    anomalies_df = pd.DataFrame(analysis_result.get('anomalies', []))
    
    result = viz_engine.execute_visualization(
        viz_id=tool_input["viz_id"],
        data=anomalies_df,
        parameters=tool_input.get("parameters", {})
    )
    
    # Add download URL if successful
    if result.get('success') and 'file_path' in result.get('result', {}):
        filename = os.path.basename(result['result']['file_path'])
        result['download_url'] = f"/api/exports/download/{filename}"
    
    return result


@tool_handler("list_custom_visualizations")
async def handle_list_visualizations(tool_input: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    """
    List all available custom visualizations.
    
    Args:
        tool_input: Optional visualization type filter
        session_id: User session ID
    
    Returns:
        List of available visualizations
    """
    from visualization_engine import viz_engine
    
    viz_type = tool_input.get("viz_type")
    vizs = viz_engine.registry.list_visualizations(viz_type=viz_type)
    
    return {
        "success": True,
        "visualizations": vizs,
        "count": len(vizs),
        "message": f"Found {len(vizs)} custom visualizations. Use execute_saved_visualization to run them."
    }


@tool_handler("get_visualization_templates")
async def handle_visualization_templates(tool_input: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    """
    Get available visualization templates.
    
    Args:
        tool_input: None required
        session_id: User session ID
    
    Returns:
        List of visualization templates
    """
    from visualization_engine import viz_engine
    
    templates = viz_engine.get_visualization_templates()
    
    return {
        "success": True,
        "templates": templates,
        "count": len(templates),
        "message": "Use these templates as a starting point for create_custom_visualization"
    }

