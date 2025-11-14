"""
Interactive Map Creation using Folium
Creates professional maritime maps for AIS anomaly visualization
"""
import folium
from folium import plugins
from folium.features import DivIcon
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime
import json

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


class AISMapCreator:
    """
    Creates interactive Folium maps for AIS anomaly visualization
    """
    
    # Color scheme for anomaly types
    ANOMALY_COLORS = {
        'AIS_Beacon_Off': 'red',
        'AIS_Beacon_On': 'orange',
        'Speed': 'purple',
        'Course': 'blue',
        'Loitering': 'green',
        'Rendezvous': 'pink',
        'Identity_Spoofing': 'darkred',
        'Zone_Violation': 'darkblue',
        'COG_Heading_Inconsistency': 'lightblue',
        'Excessive_Travel': 'purple',
        'Unknown': 'gray'
    }
    
    # Icons for anomaly types
    ANOMALY_ICONS = {
        'AIS_Beacon_Off': 'ban',
        'AIS_Beacon_On': 'signal',
        'Speed': 'bolt',
        'Course': 'compass',
        'Loitering': 'hourglass',
        'Rendezvous': 'users',
        'Identity_Spoofing': 'user-secret',
        'Zone_Violation': 'exclamation-triangle',
        'COG_Heading_Inconsistency': 'question-circle',
        'Excessive_Travel': 'rocket',
        'Unknown': 'question'
    }
    
    def __init__(self, output_dir: str = None):
        if output_dir is None:
            output_dir = get_default_output_directory()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_all_anomalies_map(self, anomalies_df: pd.DataFrame,
                                 output_filename: str = "All Anomalies Map.html",
                                 show_clustering: bool = True,
                                 show_heatmap: bool = False,
                                 show_grid: bool = False,
                                 title: str = "AIS Anomalies Map") -> Dict[str, Any]:
        """
        Create an interactive map showing all anomalies
        
        Args:
            anomalies_df: DataFrame with anomaly data
            output_filename: Name of output HTML file
            show_clustering: Whether to cluster markers
            show_heatmap: Whether to show heatmap layer
            show_grid: Whether to show lat/long grid
            title: Map title
        
        Returns:
            Dict with success status and file info
        """
        try:
            if anomalies_df.empty:
                return {
                    'success': False,
                    'error': 'No anomalies to map',
                    'file_path': None
                }
            
            # Calculate center point
            center_lat, center_lon = self._calculate_center(anomalies_df)
            
            # Create base map
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=6,
                tiles='OpenStreetMap',
                control_scale=True
            )
            
            # Add title
            title_html = f'''
                <div style="position: fixed; 
                            top: 10px; left: 50px; width: 400px; height: 50px; 
                            background-color: white; border:2px solid grey; z-index:9999; 
                            font-size:16px; font-weight: bold; padding: 10px">
                    {title}<br>
                    <span style="font-size: 12px; font-weight: normal;">
                        {len(anomalies_df)} anomalies detected
                    </span>
                </div>
            '''
            m.get_root().html.add_child(folium.Element(title_html))
            
            # Add different base map options
            folium.TileLayer('OpenStreetMap', name='Street Map').add_to(m)
            folium.TileLayer('Stamen Terrain', name='Terrain').add_to(m)
            folium.TileLayer('Stamen Toner', name='B&W').add_to(m)
            folium.TileLayer('CartoDB positron', name='Light').add_to(m)
            folium.TileLayer('CartoDB dark_matter', name='Dark').add_to(m)
            
            # Add markers
            if show_clustering:
                self._add_clustered_markers(m, anomalies_df)
            else:
                self._add_individual_markers(m, anomalies_df)
            
            # Add heatmap layer if requested
            if show_heatmap:
                self._add_heatmap_layer(m, anomalies_df)
            
            # Add lat/long grid if requested
            if show_grid:
                self._add_latlong_grid(m, anomalies_df)
            
            # Add legend
            self._add_legend(m, anomalies_df)
            
            # Add layer control
            folium.LayerControl().add_to(m)
            
            # Add fullscreen button
            plugins.Fullscreen().add_to(m)
            
            # Add measure control
            plugins.MeasureControl(position='topleft').add_to(m)
            
            # Save map
            # Generate timestamped filename to prevent overwrites
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_name = output_filename.replace('.html', '') if output_filename.endswith('.html') else output_filename
            timestamped_filename = f"{base_name}_{timestamp}.html"
            
            output_path = self.output_dir / timestamped_filename
            m.save(str(output_path))
            
            logger.info(f"Created anomalies map with {len(anomalies_df)} markers at {output_path}")
            
            return {
                'success': True,
                'file_path': str(output_path),
                'file_name': output_filename,
                'file_size': output_path.stat().st_size,
                'anomaly_count': len(anomalies_df),
                'center': [center_lat, center_lon]
            }
            
        except Exception as e:
            logger.error(f"Error creating anomalies map: {e}")
            return {
                'success': False,
                'error': str(e),
                'file_path': None
            }
    
    def create_vessel_track_map(self, vessel_data: pd.DataFrame,
                                mmsi: str,
                                anomalies_df: Optional[pd.DataFrame] = None,
                                output_filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a map showing a single vessel's track with anomalies highlighted
        
        Args:
            vessel_data: DataFrame with vessel position history
            mmsi: Vessel MMSI
            anomalies_df: Optional DataFrame with anomalies for this vessel
            output_filename: Name of output file
        
        Returns:
            Dict with success status and file info
        """
        try:
            if vessel_data.empty:
                return {
                    'success': False,
                    'error': f'No data for vessel {mmsi}',
                    'file_path': None
                }
            
            # Default filename
            if not output_filename:
                output_filename = f"MMSI_{mmsi}_track.html"
            
            # Calculate center
            center_lat, center_lon = self._calculate_center(vessel_data)
            
            # Create map
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=8,
                tiles='OpenStreetMap'
            )
            
            # Add title
            vessel_name = vessel_data.iloc[0].get('VesselName', 'Unknown') if len(vessel_data) > 0 else 'Unknown'
            title_html = f'''
                <div style="position: fixed; 
                            top: 10px; left: 50px; width: 400px; height: 70px; 
                            background-color: white; border:2px solid grey; z-index:9999; 
                            font-size:14px; font-weight: bold; padding: 10px">
                    Vessel Track: {vessel_name}<br>
                    <span style="font-size: 12px; font-weight: normal;">
                        MMSI: {mmsi}<br>
                        {len(vessel_data)} positions tracked
                    </span>
                </div>
            '''
            m.get_root().html.add_child(folium.Element(title_html))
            
            # Draw vessel track
            self._add_vessel_track(m, vessel_data)
            
            # Add anomaly markers if provided
            if anomalies_df is not None and not anomalies_df.empty:
                self._add_individual_markers(m, anomalies_df)
            
            # Add start/end markers
            self._add_start_end_markers(m, vessel_data)
            
            # Save map
            output_path = self.output_dir / "Path_Maps" / output_filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            m.save(str(output_path))
            
            logger.info(f"Created vessel track map for {mmsi} at {output_path}")
            
            return {
                'success': True,
                'file_path': str(output_path),
                'file_name': output_filename,
                'file_size': output_path.stat().st_size,
                'mmsi': mmsi,
                'vessel_name': vessel_name,
                'position_count': len(vessel_data)
            }
            
        except Exception as e:
            logger.error(f"Error creating vessel track map: {e}")
            return {
                'success': False,
                'error': str(e),
                'file_path': None
            }
    
    def _calculate_center(self, df: pd.DataFrame) -> Tuple[float, float]:
        """Calculate center point of data"""
        if 'LAT' in df.columns and 'LON' in df.columns:
            center_lat = df['LAT'].mean()
            center_lon = df['LON'].mean()
            return center_lat, center_lon
        return 0, 0
    
    def _add_clustered_markers(self, m: folium.Map, anomalies_df: pd.DataFrame):
        """Add markers with clustering for performance"""
        marker_cluster = plugins.MarkerCluster(
            name='Anomaly Markers',
            overlay=True,
            control=True,
            icon_create_function=None
        ).add_to(m)
        
        for idx, row in anomalies_df.iterrows():
            if 'LAT' in row and 'LON' in row:
                # Get anomaly type and color
                anomaly_type = row.get('AnomalyType', 'Unknown')
                color = self.ANOMALY_COLORS.get(anomaly_type, 'gray')
                icon = self.ANOMALY_ICONS.get(anomaly_type, 'question')
                
                # Create popup
                popup_html = self._create_popup_html(row)
                
                # Add marker
                folium.Marker(
                    location=[row['LAT'], row['LON']],
                    popup=folium.Popup(popup_html, max_width=300),
                    tooltip=f"{anomaly_type} - MMSI: {row.get('MMSI', 'N/A')}",
                    icon=folium.Icon(
                        color=color,
                        icon=icon,
                        prefix='fa'
                    )
                ).add_to(marker_cluster)
    
    def _add_individual_markers(self, m: folium.Map, anomalies_df: pd.DataFrame):
        """Add individual markers (no clustering)"""
        for idx, row in anomalies_df.iterrows():
            if 'LAT' in row and 'LON' in row:
                anomaly_type = row.get('AnomalyType', 'Unknown')
                color = self.ANOMALY_COLORS.get(anomaly_type, 'gray')
                icon = self.ANOMALY_ICONS.get(anomaly_type, 'question')
                
                popup_html = self._create_popup_html(row)
                
                folium.Marker(
                    location=[row['LAT'], row['LON']],
                    popup=folium.Popup(popup_html, max_width=300),
                    tooltip=f"{anomaly_type}",
                    icon=folium.Icon(color=color, icon=icon, prefix='fa')
                ).add_to(m)
    
    def _add_vessel_track(self, m: folium.Map, vessel_data: pd.DataFrame):
        """Add vessel track as a line"""
        # Sort by timestamp
        if 'BaseDateTime' in vessel_data.columns:
            vessel_data = vessel_data.sort_values('BaseDateTime')
        
        # Create line coordinates
        coordinates = []
        for idx, row in vessel_data.iterrows():
            if 'LAT' in row and 'LON' in row:
                coordinates.append([row['LAT'], row['LON']])
        
        if coordinates:
            # Add polyline
            folium.PolyLine(
                coordinates,
                color='blue',
                weight=2,
                opacity=0.7,
                popup='Vessel Track'
            ).add_to(m)
    
    def _add_start_end_markers(self, m: folium.Map, vessel_data: pd.DataFrame):
        """Add start and end markers for vessel track"""
        if len(vessel_data) > 0:
            # Start marker (green)
            start = vessel_data.iloc[0]
            if 'LAT' in start and 'LON' in start:
                folium.Marker(
                    location=[start['LAT'], start['LON']],
                    popup='Start',
                    tooltip='Track Start',
                    icon=folium.Icon(color='green', icon='play', prefix='fa')
                ).add_to(m)
            
            # End marker (red)
            end = vessel_data.iloc[-1]
            if 'LAT' in end and 'LON' in end:
                folium.Marker(
                    location=[end['LAT'], end['LON']],
                    popup='End',
                    tooltip='Track End',
                    icon=folium.Icon(color='red', icon='stop', prefix='fa')
                ).add_to(m)
    
    def _add_heatmap_layer(self, m: folium.Map, anomalies_df: pd.DataFrame):
        """Add heatmap layer"""
        heat_data = []
        for idx, row in anomalies_df.iterrows():
            if 'LAT' in row and 'LON' in row:
                heat_data.append([row['LAT'], row['LON']])
        
        if heat_data:
            plugins.HeatMap(
                heat_data,
                name='Anomaly Density Heatmap',
                radius=15,
                blur=25,
                max_zoom=13,
                overlay=True,
                control=True
            ).add_to(m)
    
    def _add_latlong_grid(self, m: folium.Map, anomalies_df: pd.DataFrame):
        """Add latitude/longitude grid overlay"""
        # Simple implementation - add grid lines every degree
        if 'LAT' in anomalies_df.columns and 'LON' in anomalies_df.columns:
            min_lat, max_lat = anomalies_df['LAT'].min(), anomalies_df['LAT'].max()
            min_lon, max_lon = anomalies_df['LON'].min(), anomalies_df['LON'].max()
            
            # Round to nearest degree
            min_lat = int(min_lat)
            max_lat = int(max_lat) + 1
            min_lon = int(min_lon)
            max_lon = int(max_lon) + 1
            
            # Add horizontal lines (latitude)
            for lat in range(min_lat, max_lat + 1):
                folium.PolyLine(
                    [[lat, min_lon], [lat, max_lon]],
                    color='gray',
                    weight=1,
                    opacity=0.3,
                    popup=f'Latitude {lat}°'
                ).add_to(m)
            
            # Add vertical lines (longitude)
            for lon in range(min_lon, max_lon + 1):
                folium.PolyLine(
                    [[min_lat, lon], [max_lat, lon]],
                    color='gray',
                    weight=1,
                    opacity=0.3,
                    popup=f'Longitude {lon}°'
                ).add_to(m)
    
    def _add_legend(self, m: folium.Map, anomalies_df: pd.DataFrame):
        """Add legend showing anomaly types"""
        # Get unique anomaly types in the data
        if 'AnomalyType' in anomalies_df.columns:
            anomaly_types = anomalies_df['AnomalyType'].unique()
            
            legend_html = '''
                <div style="position: fixed; 
                            bottom: 50px; right: 50px; width: 220px; 
                            background-color: white; border:2px solid grey; z-index:9999; 
                            font-size:12px; padding: 10px">
                    <p style="margin-top:0; font-weight: bold;">Anomaly Types</p>
            '''
            
            for anomaly_type in sorted(anomaly_types):
                color = self.ANOMALY_COLORS.get(anomaly_type, 'gray')
                count = len(anomalies_df[anomalies_df['AnomalyType'] == anomaly_type])
                legend_html += f'''
                    <p style="margin: 5px 0;">
                        <i class="fa fa-map-marker" style="color:{color}"></i>
                        {anomaly_type} ({count})
                    </p>
                '''
            
            legend_html += '</div>'
            
            m.get_root().html.add_child(folium.Element(legend_html))
    
    def _create_popup_html(self, row: pd.Series) -> str:
        """Create HTML for marker popup"""
        html = '<div style="font-family: Arial; font-size: 12px;">'
        html += f'<b>Anomaly: {row.get("AnomalyType", "Unknown")}</b><br>'
        html += f'<b>MMSI:</b> {row.get("MMSI", "N/A")}<br>'
        
        if 'VesselName' in row and pd.notna(row['VesselName']):
            html += f'<b>Vessel:</b> {row["VesselName"]}<br>'
        
        if 'VesselType' in row and pd.notna(row['VesselType']):
            html += f'<b>Type:</b> {row["VesselType"]}<br>'
        
        if 'BaseDateTime' in row and pd.notna(row['BaseDateTime']):
            html += f'<b>Time:</b> {row["BaseDateTime"]}<br>'
        
        if 'LAT' in row and 'LON' in row:
            html += f'<b>Position:</b> {row["LAT"]:.4f}, {row["LON"]:.4f}<br>'
        
        if 'SOG' in row and pd.notna(row['SOG']):
            html += f'<b>Speed:</b> {row["SOG"]:.1f} knots<br>'
        
        if 'COG' in row and pd.notna(row['COG']):
            html += f'<b>Course:</b> {row["COG"]:.0f}°<br>'
        
        if 'Description' in row and pd.notna(row['Description']):
            html += f'<b>Details:</b> {row["Description"]}<br>'
        
        if 'Confidence' in row and pd.notna(row['Confidence']):
            html += f'<b>Confidence:</b> {row["Confidence"]}%<br>'
        
        html += '</div>'
        return html


# Global map creator instance
map_creator = AISMapCreator()

