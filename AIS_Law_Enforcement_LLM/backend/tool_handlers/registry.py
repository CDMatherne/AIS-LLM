"""
Tool Handler Registry

Provides a decorator-based system for registering tool handlers,
replacing the large if/elif chain with a clean dictionary lookup.
"""

from typing import Dict, Callable, Awaitable, Any
import logging

logger = logging.getLogger(__name__)

# Type alias for tool handler functions
ToolHandler = Callable[[Dict[str, Any], str], Awaitable[Dict[str, Any]]]

# Global registry of tool handlers
TOOL_HANDLERS: Dict[str, ToolHandler] = {}


def tool_handler(tool_name: str):
    """
    Decorator to register a tool handler function.
    
    Usage:
        @tool_handler("run_anomaly_analysis")
        async def handle_anomaly_analysis(tool_input: Dict, session_id: str):
            # Handler implementation
            return result
    
    Args:
        tool_name: Name of the tool (must match Claude's tool definition)
    
    Returns:
        Decorator function
    """
    def decorator(func: ToolHandler):
        if tool_name in TOOL_HANDLERS:
            logger.warning(f"Tool handler '{tool_name}' already registered. Overwriting.")
        
        TOOL_HANDLERS[tool_name] = func
        logger.debug(f"Registered tool handler: {tool_name}")
        return func
    
    return decorator


def get_handler(tool_name: str) -> ToolHandler:
    """
    Get a registered tool handler by name.
    
    Args:
        tool_name: Name of the tool
    
    Returns:
        Tool handler function
    
    Raises:
        KeyError: If tool is not registered
    """
    return TOOL_HANDLERS[tool_name]


def list_registered_tools():
    """
    Get a list of all registered tool names.
    
    Returns:
        List of tool names
    """
    return list(TOOL_HANDLERS.keys())


def is_tool_registered(tool_name: str) -> bool:
    """
    Check if a tool is registered.
    
    Args:
        tool_name: Name of the tool
    
    Returns:
        True if registered, False otherwise
    """
    return tool_name in TOOL_HANDLERS

