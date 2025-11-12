"""
Base Handler Classes

Provides abstract base classes for common tool handler patterns,
eliminating code duplication in chart, map, and export handlers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)


class BaseToolHandler(ABC):
    """Base class for all tool handlers"""
    
    def __init__(self, session_manager, analysis_engine=None):
        self.session_manager = session_manager
        self.analysis_engine = analysis_engine
    
    @abstractmethod
    async def execute(self, tool_input: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """Execute the tool handler"""
        pass


class ChartToolHandler(BaseToolHandler):
    """
    Base class for chart-creating tool handlers.
    
    Handles the common pattern of:
    1. Getting analysis result
    2. Preparing DataFrame
    3. Creating chart
    4. Adding download URL
    """
    
    async def execute(self, tool_input: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """Execute chart creation"""
        # Get analysis result
        analysis_result = self._get_analysis_result(session_id, tool_input.get("analysis_id"))
        if not analysis_result:
            return {"success": False, "error": "Analysis not found"}
        
        # Prepare DataFrame
        anomalies_df = self._prepare_dataframe(analysis_result)
        if anomalies_df.empty:
            return {"success": False, "error": "No anomalies data to visualize"}
        
        # Create chart (implemented by subclass)
        result = await self.create_chart(anomalies_df, tool_input)
        
        # Add download URL if successful
        return self._add_download_url(result)
    
    @abstractmethod
    async def create_chart(self, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create the specific chart. Implemented by subclasses."""
        pass
    
    def _get_analysis_result(self, session_id: str, analysis_id: str) -> Optional[Dict]:
        """Get analysis result from session manager"""
        return self.session_manager.get_analysis_result(session_id, analysis_id)
    
    def _prepare_dataframe(self, analysis_result: Dict) -> pd.DataFrame:
        """Prepare DataFrame from analysis result"""
        return pd.DataFrame(analysis_result.get('anomalies', []))
    
    def _add_download_url(self, result: Dict) -> Dict:
        """Add download URL to result if file was created"""
        if result.get('success') and 'file_path' in result:
            filename = os.path.basename(result['file_path'])
            result['download_url'] = f"/api/exports/download/{filename}"
        return result


class MapToolHandler(BaseToolHandler):
    """
    Base class for map-creating tool handlers.
    
    Handles the common pattern of:
    1. Getting analysis result
    2. Preparing DataFrame
    3. Creating map
    4. Adding download URL
    """
    
    async def execute(self, tool_input: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """Execute map creation"""
        # Get analysis result
        analysis_result = self._get_analysis_result(session_id, tool_input.get("analysis_id"))
        if not analysis_result:
            return {"success": False, "error": "Analysis not found"}
        
        # Prepare DataFrame
        anomalies_df = self._prepare_dataframe(analysis_result)
        if anomalies_df.empty:
            return {"success": False, "error": "No anomalies data to visualize"}
        
        # Create map (implemented by subclass)
        result = await self.create_map(anomalies_df, tool_input)
        
        # Add download URL if successful
        return self._add_download_url(result)
    
    @abstractmethod
    async def create_map(self, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create the specific map. Implemented by subclasses."""
        pass
    
    def _get_analysis_result(self, session_id: str, analysis_id: str) -> Optional[Dict]:
        """Get analysis result from session manager"""
        return self.session_manager.get_analysis_result(session_id, analysis_id)
    
    def _prepare_dataframe(self, analysis_result: Dict) -> pd.DataFrame:
        """Prepare DataFrame from analysis result"""
        return pd.DataFrame(analysis_result.get('anomalies', []))
    
    def _add_download_url(self, result: Dict) -> Dict:
        """Add download URL to result if file was created"""
        if result.get('success') and 'file_path' in result:
            filename = os.path.basename(result['file_path'])
            result['download_url'] = f"/api/exports/download/{filename}"
        return result


class ExportToolHandler(BaseToolHandler):
    """
    Base class for export tool handlers.
    
    Handles the common pattern of:
    1. Getting analysis result
    2. Exporting data
    3. Adding download URL
    """
    
    async def execute(self, tool_input: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """Execute export"""
        # Get analysis result
        analysis_result = self._get_analysis_result(session_id, tool_input.get("analysis_id"))
        if not analysis_result:
            return {"success": False, "error": "Analysis not found"}
        
        # Export data (implemented by subclass)
        result = await self.export_data(analysis_result, tool_input)
        
        # Add download URL if successful
        return self._add_download_url(result)
    
    @abstractmethod
    async def export_data(self, analysis_result: Dict, params: Dict[str, Any]) -> Dict[str, Any]:
        """Export the data. Implemented by subclasses."""
        pass
    
    def _get_analysis_result(self, session_id: str, analysis_id: str) -> Optional[Dict]:
        """Get analysis result from session manager"""
        return self.session_manager.get_analysis_result(session_id, analysis_id)
    
    def _add_download_url(self, result: Dict) -> Dict:
        """Add download URL to result if file was created"""
        if result.get('success') and 'file_path' in result:
            filename = os.path.basename(result['file_path'])
            result['download_url'] = f"/api/exports/download/{filename}"
        return result

