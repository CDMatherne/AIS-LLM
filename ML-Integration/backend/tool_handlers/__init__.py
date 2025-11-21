"""
Tool Handlers Package

Organized tool handlers for the AIS Law Enforcement LLM Assistant.
Replaces the large if/elif chain in app.py with a registry-based system.
"""

from .registry import tool_handler, TOOL_HANDLERS, get_handler
from .base import ChartToolHandler, MapToolHandler, ExportToolHandler

__all__ = [
    'tool_handler',
    'TOOL_HANDLERS',
    'get_handler',
    'ChartToolHandler',
    'MapToolHandler',
    'ExportToolHandler',
]

