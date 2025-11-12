"""
Dynamic Visualization Engine
Allows Claude to generate custom visualizations and reports on-the-fly
Successful visualizations are saved and made available to all users
"""
import json
import os
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import folium
from io import BytesIO
import base64

logger = logging.getLogger(__name__)


def get_default_output_directory():
    """Get the default output directory (AISDS_Output in Downloads)"""
    home_dir = Path.home()
    downloads_dir = home_dir / "Downloads"
    output_dir = downloads_dir / "AISDS_Output"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        output_dir = Path("AISDS_Output")
        output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir)


class VisualizationRegistry:
    """
    Registry for storing and managing custom visualizations created by users
    """
    
    def __init__(self, registry_path: str = "visualization_registry.json"):
        self.registry_path = Path(registry_path)
        self.visualizations = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load visualization registry from disk"""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading registry: {e}")
                return {}
        return {}
    
    def _save_registry(self):
        """Save registry to disk"""
        try:
            with open(self.registry_path, 'w') as f:
                json.dump(self.visualizations, f, indent=2)
            logger.info(f"Registry saved with {len(self.visualizations)} visualizations")
        except Exception as e:
            logger.error(f"Error saving registry: {e}")
    
    def add_visualization(self, name: str, code: str, description: str, 
                         viz_type: str, author: str, parameters: Dict[str, Any]) -> str:
        """
        Add a new visualization to the registry
        
        Args:
            name: Unique name for the visualization
            code: Python code for the visualization
            description: What the visualization does
            viz_type: Type (map, chart, report, etc.)
            author: Who created it
            parameters: Required parameters with types
        
        Returns:
            Visualization ID
        """
        viz_id = hashlib.md5(f"{name}{code}".encode()).hexdigest()[:12]
        
        self.visualizations[viz_id] = {
            'id': viz_id,
            'name': name,
            'code': code,
            'description': description,
            'type': viz_type,
            'author': author,
            'parameters': parameters,
            'created': datetime.now().isoformat(),
            'usage_count': 0,
            'rating': 0.0,
            'rating_count': 0,
            'tags': []
        }
        
        self._save_registry()
        logger.info(f"Added visualization: {name} (ID: {viz_id})")
        
        return viz_id
    
    def get_visualization(self, viz_id: str) -> Optional[Dict[str, Any]]:
        """Get visualization by ID"""
        return self.visualizations.get(viz_id)
    
    def get_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get visualization by name"""
        for viz in self.visualizations.values():
            if viz['name'] == name:
                return viz
        return None
    
    def list_visualizations(self, viz_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all visualizations, optionally filtered by type"""
        vizs = list(self.visualizations.values())
        
        if viz_type:
            vizs = [v for v in vizs if v['type'] == viz_type]
        
        # Sort by usage and rating
        vizs.sort(key=lambda x: (x['usage_count'], x['rating']), reverse=True)
        
        return vizs
    
    def increment_usage(self, viz_id: str):
        """Increment usage counter"""
        if viz_id in self.visualizations:
            self.visualizations[viz_id]['usage_count'] += 1
            self._save_registry()
    
    def add_rating(self, viz_id: str, rating: float):
        """Add a rating (1-5 stars)"""
        if viz_id in self.visualizations:
            viz = self.visualizations[viz_id]
            current_total = viz['rating'] * viz['rating_count']
            viz['rating_count'] += 1
            viz['rating'] = (current_total + rating) / viz['rating_count']
            self._save_registry()
    
    def delete_visualization(self, viz_id: str) -> bool:
        """Delete a visualization"""
        if viz_id in self.visualizations:
            del self.visualizations[viz_id]
            self._save_registry()
            return True
        return False


class VisualizationEngine:
    """
    Engine for executing custom visualizations safely
    """
    
    def __init__(self, output_dir: str = None):
        if output_dir is None:
            output_dir = get_default_output_directory()
        self.output_dir = Path(output_dir)
        self.registry = VisualizationRegistry()
        
        # Safe imports allowed in user code
        self.safe_globals = {
            'pd': pd,
            'plt': plt,
            'go': go,
            'px': px,
            'folium': folium,
            'np': __import__('numpy'),
            'datetime': datetime,
            'Path': Path,
            'logger': logger
        }
    
    def execute_visualization(self, viz_id: str, data: pd.DataFrame, 
                             parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a registered visualization
        
        Args:
            viz_id: ID of visualization to execute
            data: Input data (anomalies DataFrame)
            parameters: Parameters for the visualization
        
        Returns:
            Result with file path and metadata
        """
        viz = self.registry.get_visualization(viz_id)
        if not viz:
            return {'success': False, 'error': f'Visualization {viz_id} not found'}
        
        try:
            # Increment usage counter
            self.registry.increment_usage(viz_id)
            
            # Execute the code
            result = self._execute_code(viz['code'], data, parameters)
            
            return {
                'success': True,
                'viz_id': viz_id,
                'viz_name': viz['name'],
                'result': result
            }
            
        except Exception as e:
            logger.error(f"Error executing visualization {viz_id}: {e}")
            return {
                'success': False,
                'error': str(e),
                'viz_id': viz_id
            }
    
    def create_and_execute(self, code: str, data: pd.DataFrame, 
                          parameters: Dict[str, Any],
                          name: Optional[str] = None,
                          description: Optional[str] = None,
                          save_to_registry: bool = False) -> Dict[str, Any]:
        """
        Create and execute a new visualization on-the-fly
        
        Args:
            code: Python code for visualization
            data: Input data
            parameters: Parameters
            name: Name if saving to registry
            description: Description if saving
            save_to_registry: Whether to save successful visualization
        
        Returns:
            Result with file path and metadata
        """
        try:
            # Validate code (basic safety checks)
            if not self._is_code_safe(code):
                return {
                    'success': False,
                    'error': 'Code contains potentially unsafe operations'
                }
            
            # Execute the code
            result = self._execute_code(code, data, parameters)
            
            # If successful and user wants to save it
            if result['success'] and save_to_registry and name:
                viz_type = self._infer_viz_type(code)
                viz_id = self.registry.add_visualization(
                    name=name,
                    code=code,
                    description=description or "User-generated visualization",
                    viz_type=viz_type,
                    author="user",
                    parameters=parameters
                )
                result['viz_id'] = viz_id
                result['saved_to_registry'] = True
            
            return result
            
        except Exception as e:
            logger.error(f"Error in create_and_execute: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _execute_code(self, code: str, data: pd.DataFrame, 
                     parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Safely execute visualization code
        
        The code should define a function called 'generate_visualization'
        that takes (data, parameters, output_dir) and returns a file path
        """
        # Prepare execution environment
        exec_globals = self.safe_globals.copy()
        exec_globals['data'] = data
        exec_globals['parameters'] = parameters
        exec_globals['output_dir'] = self.output_dir
        
        # Execute the code
        exec(code, exec_globals)
        
        # Call the generate_visualization function
        if 'generate_visualization' not in exec_globals:
            raise ValueError("Code must define 'generate_visualization' function")
        
        generate_func = exec_globals['generate_visualization']
        result_path = generate_func(data, parameters, self.output_dir)
        
        # Get file info
        if isinstance(result_path, (str, Path)):
            result_path = Path(result_path)
            if result_path.exists():
                return {
                    'success': True,
                    'file_path': str(result_path),
                    'file_name': result_path.name,
                    'file_size': result_path.stat().st_size,
                    'file_type': result_path.suffix
                }
        
        return {
            'success': False,
            'error': 'No valid output file generated'
        }
    
    def _is_code_safe(self, code: str) -> bool:
        """
        Basic safety check for user code
        """
        # Forbidden operations
        forbidden = [
            'import os',
            'import sys',
            'import subprocess',
            '__import__',
            'eval(',
            'exec(',
            'compile(',
            'open(',  # Direct file operations
            'rm ',
            'del ',
            'system(',
            'popen(',
        ]
        
        code_lower = code.lower()
        for forbidden_op in forbidden:
            if forbidden_op in code_lower:
                logger.warning(f"Code contains forbidden operation: {forbidden_op}")
                return False
        
        return True
    
    def _infer_viz_type(self, code: str) -> str:
        """Infer visualization type from code"""
        code_lower = code.lower()
        
        if 'folium' in code_lower or 'map' in code_lower:
            return 'map'
        elif 'plotly' in code_lower:
            return 'interactive_chart'
        elif 'matplotlib' in code_lower or 'plt' in code_lower:
            return 'static_chart'
        elif 'to_csv' in code_lower or 'to_excel' in code_lower:
            return 'report'
        else:
            return 'custom'
    
    def get_visualization_templates(self) -> Dict[str, str]:
        """
        Get template code examples for different visualization types
        """
        return {
            'matplotlib_chart': '''
def generate_visualization(data, parameters, output_dir):
    """Generate a matplotlib chart"""
    import matplotlib.pyplot as plt
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Your visualization code here
    # Example: Count anomalies by type
    if 'AnomalyType' in data.columns:
        counts = data['AnomalyType'].value_counts()
        ax.bar(counts.index, counts.values)
        ax.set_xlabel('Anomaly Type')
        ax.set_ylabel('Count')
        ax.set_title(parameters.get('title', 'Anomaly Distribution'))
        plt.xticks(rotation=45, ha='right')
    
    # Save
    output_path = output_dir / parameters.get('filename', 'custom_chart.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path
''',
            'plotly_interactive': '''
def generate_visualization(data, parameters, output_dir):
    """Generate an interactive Plotly chart"""
    import plotly.express as px
    
    # Your visualization code here
    # Example: Interactive scatter plot
    if 'LAT' in data.columns and 'LON' in data.columns:
        fig = px.scatter_geo(
            data,
            lat='LAT',
            lon='LON',
            color='AnomalyType' if 'AnomalyType' in data.columns else None,
            hover_data=['MMSI', 'VesselName'] if 'MMSI' in data.columns else None,
            title=parameters.get('title', 'Anomaly Geographic Distribution')
        )
        
        # Save
        output_path = output_dir / parameters.get('filename', 'interactive_chart.html')
        fig.write_html(output_path)
        
        return output_path
    
    return None
''',
            'folium_map': '''
def generate_visualization(data, parameters, output_dir):
    """Generate an interactive Folium map"""
    import folium
    
    # Create map centered on data
    center_lat = data['LAT'].mean() if 'LAT' in data.columns else 0
    center_lon = data['LON'].mean() if 'LON' in data.columns else 0
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=6)
    
    # Add markers
    for idx, row in data.iterrows():
        if 'LAT' in row and 'LON' in row:
            folium.Marker(
                location=[row['LAT'], row['LON']],
                popup=f"MMSI: {row.get('MMSI', 'N/A')}<br>Type: {row.get('AnomalyType', 'N/A')}",
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(m)
    
    # Save
    output_path = output_dir / parameters.get('filename', 'custom_map.html')
    m.save(output_path)
    
    return output_path
''',
            'custom_report': '''
def generate_visualization(data, parameters, output_dir):
    """Generate a custom report"""
    import pandas as pd
    
    # Create report dataframe
    report_data = []
    
    # Add your custom analysis here
    # Example: Summary by vessel
    if 'MMSI' in data.columns:
        vessel_summary = data.groupby('MMSI').agg({
            'AnomalyType': 'count',
            'LAT': 'first',
            'LON': 'first'
        }).rename(columns={'AnomalyType': 'Total_Anomalies'})
        
        # Save to CSV
        output_path = output_dir / parameters.get('filename', 'custom_report.csv')
        vessel_summary.to_csv(output_path)
        
        return output_path
    
    return None
'''
        }


# Global visualization engine instance
viz_engine = VisualizationEngine()

