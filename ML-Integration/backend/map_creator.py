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
        # 'Rendezvous': 'pink',  # DISABLED - TOO COMPLEX
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
        # 'Rendezvous': 'users',  # DISABLED - TOO COMPLEX
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
                                 show_markers_with_heatmap: bool = True,
                                 show_grid: bool = False,
                                 title: str = "AIS Anomalies Map",
                                 group_by_day: bool = True,
                                 filter_by_anomaly_type: Optional[List[str]] = None,
                                 filter_by_vessel_type: Optional[List[str]] = None,
                                 filter_by_mmsi: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Create an interactive map showing all anomalies
        
        Args:
            anomalies_df: DataFrame with anomaly data
            output_filename: Name of output HTML file
            show_clustering: Whether to cluster markers
            show_heatmap: Whether to show heatmap layer
            show_markers_with_heatmap: Whether to show markers when heatmap is enabled (default: True)
            show_grid: Whether to show lat/long grid
            title: Map title
            group_by_day: Whether to group markers by day and anomaly type (default: True)
            filter_by_anomaly_type: Optional list of anomaly types to show (e.g., ['Speed', 'Course'])
            filter_by_vessel_type: Optional list of vessel type categories (e.g., ['Cargo', 'Tanker'])
            filter_by_mmsi: Optional list of MMSI numbers to filter
        
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
            
            # Apply filters
            filtered_df = anomalies_df.copy()
            original_count = len(filtered_df)
            
            # Filter by anomaly type
            if filter_by_anomaly_type and 'AnomalyType' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['AnomalyType'].isin(filter_by_anomaly_type)]
                logger.info(f"Filtered to anomaly types: {filter_by_anomaly_type} ({len(filtered_df)}/{original_count} remaining)")
            
            # Filter by vessel type
            if filter_by_vessel_type and 'VesselType' in filtered_df.columns:
                from .vessel_types import VESSEL_TYPE_CATEGORIES
                type_codes = []
                for vtype in filter_by_vessel_type:
                    if vtype in VESSEL_TYPE_CATEGORIES:
                        type_codes.extend(VESSEL_TYPE_CATEGORIES[vtype])
                
                if type_codes:
                    filtered_df = filtered_df[filtered_df['VesselType'].isin(type_codes)]
                    logger.info(f"Filtered to vessel types: {filter_by_vessel_type} ({len(filtered_df)}/{original_count} remaining)")
            
            # Filter by MMSI
            if filter_by_mmsi and 'MMSI' in filtered_df.columns:
                # Convert MMSI filter to match DataFrame type
                try:
                    if pd.api.types.is_integer_dtype(filtered_df['MMSI'].dtype):
                        mmsi_filter_converted = [int(m) for m in filter_by_mmsi]
                    else:
                        mmsi_filter_converted = [str(m) for m in filter_by_mmsi]
                    filtered_df = filtered_df[filtered_df['MMSI'].isin(mmsi_filter_converted)]
                    logger.info(f"Filtered to {len(filter_by_mmsi)} MMSIs ({len(filtered_df)}/{original_count} remaining)")
                except (ValueError, TypeError) as e:
                    logger.warning(f"Error converting MMSI filter: {e}")
            
            if filtered_df.empty:
                return {
                    'success': False,
                    'error': 'No anomalies match the filter criteria',
                    'file_path': None
                }
            
            # Calculate center point from filtered data
            center_lat, center_lon = self._calculate_center(filtered_df)
            
            # Create base map with OpenStreetMap tiles
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=6,
                tiles='OpenStreetMap',
                attr='© OpenStreetMap contributors',
                control_scale=True
            )
            
            # Build filter info string
            filter_info_parts = []
            if filter_by_anomaly_type:
                filter_info_parts.append(f"Anomaly Types: {', '.join(filter_by_anomaly_type)}")
            if filter_by_vessel_type:
                filter_info_parts.append(f"Vessel Types: {', '.join(filter_by_vessel_type)}")
            if filter_by_mmsi:
                filter_info_parts.append(f"MMSIs: {len(filter_by_mmsi)} vessels")
            
            filter_info = ""
            if filter_info_parts:
                filter_info = f"<br><span style='font-size: 11px; color: #666;'>Filters: {' | '.join(filter_info_parts)}</span>"
                if len(filtered_df) < original_count:
                    filter_info += f"<br><span style='font-size: 11px; color: #666;'>({len(filtered_df)}/{original_count} anomalies shown)</span>"
            
            # Add title
            title_html = f'''
                <div style="position: fixed; 
                            top: 10px; left: 50px; width: 500px; height: auto; 
                            background-color: white; border:2px solid grey; z-index:9999; 
                            font-size:16px; font-weight: bold; padding: 10px">
                    {title}<br>
                    <span style="font-size: 12px; font-weight: normal;">
                        {len(filtered_df)} anomalies detected{filter_info}
                    </span>
                </div>
            '''
            m.get_root().html.add_child(folium.Element(title_html))
            
            # Add markers and heatmaps with simplified MVP-style structure
            # Use filtered data
            if group_by_day:
                self._add_grouped_markers_and_heatmaps(m, filtered_df, show_clustering, show_heatmap)
            else:
                if show_clustering:
                    self._add_clustered_markers(m, filtered_df)
                else:
                    self._add_individual_markers(m, filtered_df)
                # Add simple heatmap if not grouping by day
                if show_heatmap:
                    self._add_simple_heatmap(m, filtered_df)
            
            # Add lat/long grid if requested (use filtered data)
            if show_grid:
                self._add_latlong_grid(m, filtered_df)
            
            # Add legend (use filtered data)
            self._add_legend(m, filtered_df)
            
            # Add layer control - Folium will automatically categorize:
            # TileLayers → base_layers (radio buttons)
            # FeatureGroups → overlays (checkboxes)
            layer_control = folium.LayerControl(
                collapsed=False,  # Expanded by default
                exclusiveGroups=True  # Enable grouping behavior
            ).add_to(m)
            
            # Add JavaScript to ensure proper layer categorization
            # Force all FeatureGroups to be overlays (checkboxes), not base layers (radio buttons)
            # Only OpenStreetMap should be a base layer (radio button)
            fix_layer_control_js = """
            <script>
            (function() {
                function fixLayerControl() {
                    // Find the layer control
                    var layerControl = document.querySelector('.leaflet-control-layers');
                    if (!layerControl) return false;
                    
                    // Find base layer and overlay sections
                    var baseLayerSection = layerControl.querySelector('.leaflet-control-layers-base');
                    var overlaySection = layerControl.querySelector('.leaflet-control-layers-overlays');
                    
                    if (!baseLayerSection || !overlaySection) return false;
                    
                    var moved = false;
                    
                    // Find all radio buttons in base layers (they should only be OpenStreetMap)
                    var baseLayerInputs = baseLayerSection.querySelectorAll('input[type="radio"]');
                    baseLayerInputs.forEach(function(input) {
                        var label = input.closest('label') || input.parentElement;
                        if (!label) return;
                        
                        var text = label.textContent || label.innerText || '';
                        // If it's not OpenStreetMap, it's a FeatureGroup that should be in overlays
                        if (text && !text.toLowerCase().match(/openstreetmap|open street map/i)) {
                            // Change radio button to checkbox
                            input.type = 'checkbox';
                            input.name = 'overlay'; // Change name to match overlay checkboxes
                            
                            // Move the entire label/input to overlays section
                            var container = label.parentElement || label;
                            if (container && container.parentElement) {
                                container.parentElement.removeChild(container);
                            }
                            overlaySection.appendChild(container);
                            moved = true;
                        }
                    });
                    
                    return moved;
                }
                
                // Try multiple times to catch the layer control after it's fully rendered
                var attempts = 0;
                var maxAttempts = 10;
                var interval = setInterval(function() {
                    attempts++;
                    var moved = fixLayerControl();
                    if (moved || attempts >= maxAttempts) {
                        clearInterval(interval);
                    }
                }, 200);
            })();
            </script>
            """
            m.get_root().html.add_child(folium.Element(fix_layer_control_js))
            
            # Add JavaScript for parent-child layer relationships
            # When a parent (day) is unchecked, hide its children
            # When a parent is checked, show its children
            parent_child_js = """
            <script>
            (function() {
                console.log('[LayerControl] Initializing parent-child relationships...');
                
                function setupParentChildRelationships() {
                    var layerControl = document.querySelector('.leaflet-control-layers');
                    if (!layerControl) {
                        console.log('[LayerControl] Control not found yet...');
                        return false;
                    }
                    
                    var overlaySection = layerControl.querySelector('.leaflet-control-layers-overlays');
                    if (!overlaySection) {
                        console.log('[LayerControl] Overlay section not found yet...');
                        return false;
                    }
                    
                    // Find all labels in overlay section
                    var labels = overlaySection.querySelectorAll('label');
                    console.log('[LayerControl] Found ' + labels.length + ' overlay labels');
                    
                    // Group layers by parent
                    var parentLayers = {};
                    var childLayers = {};
                    
                    labels.forEach(function(label) {
                        var text = label.textContent || label.innerText || '';
                        var input = label.querySelector('input[type="checkbox"]');
                        
                        if (!input) {
                            console.log('[LayerControl] No checkbox in label: ' + text.substring(0, 30));
                            return;
                        }
                        
                        // Trim text BEFORE checking patterns
                        var trimmedText = text.trim();
                        
                        // Check if it's a parent (starts with "All Anomalies (")
                        if (trimmedText.indexOf('All Anomalies (') === 0) {
                            var parentName = trimmedText;
                            console.log('[LayerControl] Found parent: ' + parentName);
                            parentLayers[parentName] = {
                                label: label,
                                input: input,
                                children: []
                            };
                        }
                        // Check if it's a child (contains "└─")
                        else if (trimmedText.indexOf('└─') >= 0) {
                            console.log('[LayerControl] Found potential child: ' + trimmedText.substring(0, 50));
                            
                            // Extract date or "All Data" from child name
                            // Format: "  └─ YYYY-MM-DD - Type" or "  └─ All Data - Type"
                            var dateMatch = trimmedText.match(/└─\\s+(\\d{4}-\\d{2}-\\d{2}|All Data)/);
                            if (dateMatch) {
                                var dateOrAll = dateMatch[1];
                                var parentName;
                                if (dateOrAll === 'All Data') {
                                    parentName = 'All Anomalies (All Days)';
                                } else {
                                    parentName = 'All Anomalies (' + dateOrAll + ')';
                                }
                                console.log('[LayerControl] Child belongs to parent: ' + parentName);
                                childLayers[trimmedText] = {
                                    label: label,
                                    input: input,
                                    parent: parentName
                                };
                            }
                        }
                    });
                    
                    console.log('[LayerControl] Parents: ' + Object.keys(parentLayers).length + 
                                ', Children: ' + Object.keys(childLayers).length);
                    
                    // Link children to parents
                    Object.keys(childLayers).forEach(function(childName) {
                        var child = childLayers[childName];
                        var parentName = child.parent;
                        if (parentLayers[parentName]) {
                            parentLayers[parentName].children.push(child);
                            console.log('[LayerControl] Linked child to: ' + parentName);
                        } else {
                            console.log('[LayerControl] Parent not found for: ' + childName);
                        }
                    });
                    
                    // Add event listeners to parents
                    Object.keys(parentLayers).forEach(function(parentName) {
                        var parent = parentLayers[parentName];
                        var parentInput = parent.input;
                        
                        console.log('[LayerControl] Setting up parent: ' + parentName + 
                                    ' with ' + parent.children.length + ' children');
                        
                        // Remove any existing listeners (in case of re-run)
                        var newInput = parentInput.cloneNode(true);
                        parentInput.parentNode.replaceChild(newInput, parentInput);
                        parentInput = newInput;
                        parent.input = newInput;
                        
                        parentInput.addEventListener('change', function() {
                            var isChecked = parentInput.checked;
                            console.log('[LayerControl] Parent ' + parentName + ' changed to: ' + isChecked);
                            
                            // Show/hide children based on parent state
                            parent.children.forEach(function(child) {
                                if (isChecked) {
                                    // Parent checked - show children
                                    child.label.style.display = '';
                                    console.log('[LayerControl] Showing child');
                                } else {
                                    // Parent unchecked - hide children and uncheck them
                                    child.label.style.display = 'none';
                                    if (child.input.checked) {
                                        console.log('[LayerControl] Unchecking hidden child');
                                        child.input.click(); // Uncheck the layer
                                    }
                                }
                            });
                        });
                        
                        // Initialize children visibility based on parent state
                        setTimeout(function() {
                            var isChecked = parentInput.checked;
                            console.log('[LayerControl] Initializing ' + parentName + ', checked: ' + isChecked);
                            parent.children.forEach(function(child) {
                                if (!isChecked) {
                                    child.label.style.display = 'none';
                                    if (child.input.checked) {
                                        console.log('[LayerControl] Unchecking child during init');
                                        child.input.click(); // Uncheck the layer
                                    }
                                }
                            });
                        }, 100);
                    });
                    
                    console.log('[LayerControl] ✓ Setup complete!');
                    return true;
                }
                
                // Try multiple times with increasing delays to catch the layer control after it's fully rendered
                var attempts = 0;
                var maxAttempts = 20;
                var delay = 500;  // Start with 500ms
                
                function trySetup() {
                    attempts++;
                    console.log('[LayerControl] Attempt ' + attempts + '/' + maxAttempts);
                    var success = setupParentChildRelationships();
                    
                    if (success) {
                        console.log('[LayerControl] Successfully established relationships!');
                    } else if (attempts < maxAttempts) {
                        setTimeout(trySetup, delay);
                    } else {
                        console.log('[LayerControl] Failed to establish relationships after ' + maxAttempts + ' attempts');
                    }
                }
                
                // Start trying after initial page load
                setTimeout(trySetup, 1000);
            })();
            </script>
            """
            m.get_root().html.add_child(folium.Element(parent_child_js))
            
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
            
            logger.info(f"Created anomalies map with {len(filtered_df)} markers at {output_path}")
            
            return {
                'success': True,
                'file_path': str(output_path),
                'file_name': output_filename,
                'file_size': output_path.stat().st_size,
                'anomaly_count': len(filtered_df),
                'original_count': original_count,
                'filters_applied': {
                    'anomaly_type': filter_by_anomaly_type,
                    'vessel_type': filter_by_vessel_type,
                    'mmsi': filter_by_mmsi
                },
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
                                output_filename: Optional[str] = None,
                                show_all_points: bool = True,
                                filter_by_anomaly_type: Optional[List[str]] = None,
                                filter_by_vessel_type: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Create a map showing a single vessel's track with anomalies highlighted
        
        Args:
            vessel_data: DataFrame with vessel position history (all days in analysis period)
            mmsi: Vessel MMSI
            anomalies_df: Optional DataFrame with anomalies for this vessel
            output_filename: Name of output file
            show_all_points: Whether to show all position points along the track
            filter_by_anomaly_type: Optional list of anomaly types to show (e.g., ['Speed', 'Course'])
            filter_by_vessel_type: Optional list of vessel types to filter (not used for single vessel, but kept for consistency)
        
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
            
            # Create map with OpenStreetMap tiles
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=8,
                tiles='OpenStreetMap',
                attr='© OpenStreetMap contributors'
            )
            
            # Add title with filter info
            vessel_name = vessel_data.iloc[0].get('VesselName', 'Unknown') if len(vessel_data) > 0 else 'Unknown'
            vessel_type = vessel_data.iloc[0].get('VesselType', 'Unknown') if len(vessel_data) > 0 else 'Unknown'
            
            # Calculate date range from data
            date_range_str = ""
            if 'BaseDateTime' in vessel_data.columns:
                try:
                    dates = pd.to_datetime(vessel_data['BaseDateTime']).dt.date
                    min_date = dates.min()
                    max_date = dates.max()
                    date_range_str = f"<br>Date Range: {min_date} to {max_date}"
                except:
                    pass
            
            filter_info = ""
            if filter_by_anomaly_type:
                filter_info = f"<br>Anomaly Filter: {', '.join(filter_by_anomaly_type)}"
            
            title_html = f'''
                <div style="position: fixed; 
                            top: 10px; left: 50px; width: 500px; height: auto; 
                            background-color: white; border:2px solid grey; z-index:9999; 
                            font-size:14px; font-weight: bold; padding: 10px">
                    Vessel Track: {vessel_name}<br>
                    <span style="font-size: 12px; font-weight: normal;">
                        MMSI: {mmsi} | Type: {vessel_type}<br>
                        {len(vessel_data)} positions tracked{date_range_str}{filter_info}
                    </span>
                </div>
            '''
            m.get_root().html.add_child(folium.Element(title_html))
            
            # Draw vessel track with all points
            self._add_vessel_track(m, vessel_data, show_all_points=show_all_points)
            
            # Add anomaly markers if provided (with optional filtering)
            if anomalies_df is not None and not anomalies_df.empty:
                filtered_anomalies = anomalies_df.copy()
                
                # Filter by anomaly type if specified
                if filter_by_anomaly_type and 'AnomalyType' in filtered_anomalies.columns:
                    filtered_anomalies = filtered_anomalies[
                        filtered_anomalies['AnomalyType'].isin(filter_by_anomaly_type)
                    ]
                    logger.info(f"Filtered anomalies to types: {filter_by_anomaly_type} ({len(filtered_anomalies)} remaining)")
                
                if not filtered_anomalies.empty:
                    self._add_individual_markers(m, filtered_anomalies)
                else:
                    logger.info(f"No anomalies match the filter criteria for vessel {mmsi}")
            
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
    
    def _add_grouped_markers_and_heatmaps(self, m: folium.Map, anomalies_df: pd.DataFrame, 
                                          show_clustering: bool = True, show_heatmap: bool = False):
        """
        Add markers and heatmaps with simplified MVP-style two-level hierarchy:
        PARENT: Day (or "All Days")
          CHILD: Anomaly Type (contains both markers and heatmap)
        
        Uses visual tree characters (└─) for child layers.
        """
        # Ensure we have BaseDateTime column
        if 'BaseDateTime' not in anomalies_df.columns and 'ReportDate' in anomalies_df.columns:
            anomalies_df = anomalies_df.copy()
            anomalies_df['BaseDateTime'] = pd.to_datetime(anomalies_df['ReportDate'])
        elif 'BaseDateTime' not in anomalies_df.columns:
            logger.warning("No date column found, grouping by anomaly type only")
            self._add_grouped_by_type_only(m, anomalies_df, show_clustering)
            return
        
        # Convert BaseDateTime to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(anomalies_df['BaseDateTime']):
            anomalies_df['BaseDateTime'] = pd.to_datetime(anomalies_df['BaseDateTime'])
        
        # Extract date
        anomalies_df = anomalies_df.copy()
        anomalies_df['Date'] = anomalies_df['BaseDateTime'].dt.date
        
        # Group by date
        dates = sorted(anomalies_df['Date'].unique())
        
        # Get all unique anomaly types across all days
        all_anomaly_types = sorted(anomalies_df['AnomalyType'].unique()) if 'AnomalyType' in anomalies_df.columns else []
        
        # ========================================================================
        # LEVEL 1: PARENT GROUP - "All Days" (All Anomalies)
        # ========================================================================
        all_days_parent = folium.FeatureGroup(
            name='All Anomalies (All Days)',
            overlay=True,  # Visible by default
            control=True
        )
        all_days_parent.add_to(m)
        
        # Add a transparent marker to ensure parent group appears in layer control
        if len(anomalies_df) > 0:
            center_lat, center_lon = self._calculate_center(anomalies_df)
            folium.CircleMarker(
                location=[center_lat, center_lon],
                radius=0.1,
                fill=False,
                color='transparent',
                weight=0,
                opacity=0,
                popup='All Days Group',
                tooltip='All Days'
            ).add_to(all_days_parent)
        
        # ========================================================================
        # LEVEL 2: CHILD GROUPS - Anomaly Types under "All Days"
        # Each child group contains both markers AND heatmap (if enabled)
        # ========================================================================
        for anomaly_type in all_anomaly_types:
            type_df_all = anomalies_df[anomalies_df['AnomalyType'] == anomaly_type]
            color = self.ANOMALY_COLORS.get(anomaly_type, 'gray')
            icon = self.ANOMALY_ICONS.get(anomaly_type, 'question')
            
            # Create CHILD group with visual tree character (└─) and indentation
            # Format: "  └─ All Data - [Anomaly Type]"
            child_group_name = f'  └─ All Data - {anomaly_type}'
            anomaly_type_group = folium.FeatureGroup(
                name=child_group_name,
                overlay=True,  # Visible by default
                control=True
            )
            anomaly_type_group.add_to(m)
            
            # IMPROVEMENT #3: Optimize marker creation with vectorized operations
            # Filter rows with valid coordinates first
            valid_rows = type_df_all[(type_df_all['LAT'].notna()) & (type_df_all['LON'].notna())].copy()
            
            if len(valid_rows) == 0:
                continue
            
            # Pre-compute popup HTMLs and tooltips (vectorized) - Improvement #3.1
            # Use itertuples() instead of iterrows() for better performance
            popup_htmls = []
            tooltips = []
            locations = []
            
            # Batch process popup HTML generation
            for row_tuple in valid_rows.itertuples():
                row_dict = row_tuple._asdict()
                # Convert namedtuple to Series-like dict for popup generation
                row_series = pd.Series({k: v for k, v in row_dict.items() if k != 'Index'})
                popup_htmls.append(self._create_popup_html(row_series))
                mmsi_val = row_dict.get('MMSI', 'N/A')
                tooltips.append(f"{anomaly_type} - MMSI: {mmsi_val}")
                locations.append([row_dict.get('LAT'), row_dict.get('LON')])
            
            # Batch create markers - Improvement #3.2
            if show_clustering:
                cluster_pins = plugins.MarkerCluster(
                    name=f'All Days - {anomaly_type}',
                    overlay=False,
                    control=False
                )
                cluster_pins.add_to(anomaly_type_group)
                
                # Create markers in batch
                for i, location in enumerate(locations):
                    marker = folium.Marker(
                        location=location,
                        popup=folium.Popup(popup_htmls[i], max_width=300),
                        tooltip=tooltips[i],
                        icon=folium.Icon(color=color, icon=icon, prefix='fa')
                    )
                    marker.add_to(cluster_pins)
            else:
                # Create markers in batch
                for i, location in enumerate(locations):
                    marker = folium.Marker(
                        location=location,
                        popup=folium.Popup(popup_htmls[i], max_width=300),
                        tooltip=tooltips[i],
                        icon=folium.Icon(color=color, icon=icon, prefix='fa')
                    )
                    marker.add_to(anomaly_type_group)
            
            # Add heatmap as separate nested control if enabled
            if show_heatmap:
                # Use vectorized operation for heatmap data - Improvement #3.1
                valid_coords = type_df_all[(type_df_all['LAT'].notna()) & (type_df_all['LON'].notna())]
                heat_data_all_type = valid_coords[['LAT', 'LON']].values.tolist()
                
                if heat_data_all_type:
                    # Create separate FeatureGroup for heatmap with nested visual formatting
                    # Format: "    └─ Heatmap" (extra indentation to show nesting under marker group)
                    heatmap_group = folium.FeatureGroup(
                        name=f'    └─ All Data - {anomaly_type} Heatmap',
                        overlay=True,  # Must be True to appear as checkbox (not radio button)
                        show=False,  # Hidden by default (user can toggle independently)
                        control=True
                    )
                    heatmap_group.add_to(m)
                    
                    plugins.HeatMap(
                        heat_data_all_type,
                        name=f'Heatmap - All Data - {anomaly_type}',
                        radius=15,
                        blur=25,
                        max_zoom=13,
                        overlay=False,  # Hidden by default
                        control=False  # Don't show in control, it's part of the heatmap group
                    ).add_to(heatmap_group)
        
        # ========================================================================
        # DAY-SPECIFIC GROUPS: Same two-level hierarchy for each individual day
        # ========================================================================
        for date in dates:
            date_df = anomalies_df[anomalies_df['Date'] == date]
            date_str = date.strftime('%Y-%m-%d')
            
            # LEVEL 1: PARENT GROUP - Individual Day
            day_parent = folium.FeatureGroup(
                name=f'All Anomalies ({date_str})',
                overlay=True,  # Must be True to appear as checkbox (not radio button)
                show=False,  # Hidden by default
                control=True
            )
            day_parent.add_to(m)
            
            # Add a transparent marker to ensure parent group appears in layer control
            if len(date_df) > 0:
                center_lat, center_lon = self._calculate_center(date_df)
                folium.CircleMarker(
                    location=[center_lat, center_lon],
                    radius=0.1,
                    fill=False,
                    color='transparent',
                    weight=0,
                    opacity=0,
                    popup=f'{date_str} Group',
                    tooltip=date_str
                ).add_to(day_parent)
            
            # Group by anomaly type within this day
            if 'AnomalyType' in date_df.columns:
                anomaly_types = sorted(date_df['AnomalyType'].unique())
                
                for anomaly_type in anomaly_types:
                    type_df = date_df[date_df['AnomalyType'] == anomaly_type]
                    color = self.ANOMALY_COLORS.get(anomaly_type, 'gray')
                    icon = self.ANOMALY_ICONS.get(anomaly_type, 'question')
                    
                    # LEVEL 2: CHILD GROUP - Anomaly Type under Day
                    # Format: "  └─ [date] - [Anomaly Type]" (matching MVP style)
                    child_group_name = f'  └─ {date_str} - {anomaly_type}'
                    anomaly_type_group = folium.FeatureGroup(
                        name=child_group_name,
                        overlay=True,  # Must be True to appear as checkbox (not radio button)
                        show=False,  # Hidden by default (under parent)
                        control=True
                    )
                    anomaly_type_group.add_to(m)
                    
                    # IMPROVEMENT #3: Optimize marker creation with vectorized operations
                    # Filter rows with valid coordinates first
                    valid_rows = type_df[(type_df['LAT'].notna()) & (type_df['LON'].notna())].copy()
                    
                    if len(valid_rows) == 0:
                        continue
                    
                    # Pre-compute popup HTMLs and tooltips (vectorized) - Improvement #3.1
                    popup_htmls = []
                    tooltips = []
                    locations = []
                    
                    # Batch process popup HTML generation
                    for row_tuple in valid_rows.itertuples():
                        row_dict = row_tuple._asdict()
                        # Convert namedtuple to Series-like dict for popup generation
                        row_series = pd.Series({k: v for k, v in row_dict.items() if k != 'Index'})
                        popup_htmls.append(self._create_popup_html(row_series))
                        mmsi_val = row_dict.get('MMSI', 'N/A')
                        tooltips.append(f"{anomaly_type} - MMSI: {mmsi_val} - {date_str}")
                        locations.append([row_dict.get('LAT'), row_dict.get('LON')])
                    
                    # Batch create markers - Improvement #3.2
                    if show_clustering:
                        cluster_pins = plugins.MarkerCluster(
                            name=f'{date_str} - {anomaly_type}',
                            overlay=False,
                            control=False
                        )
                        cluster_pins.add_to(anomaly_type_group)
                        
                        # Create markers in batch
                        for i, location in enumerate(locations):
                            marker = folium.Marker(
                                location=location,
                                popup=folium.Popup(popup_htmls[i], max_width=300),
                                tooltip=tooltips[i],
                                icon=folium.Icon(color=color, icon=icon, prefix='fa')
                            )
                            marker.add_to(cluster_pins)
                    else:
                        # Create markers in batch
                        for i, location in enumerate(locations):
                            marker = folium.Marker(
                                location=location,
                                popup=folium.Popup(popup_htmls[i], max_width=300),
                                tooltip=tooltips[i],
                                icon=folium.Icon(color=color, icon=icon, prefix='fa')
                            )
                            marker.add_to(anomaly_type_group)
                    
                    # Add heatmap as separate nested control if enabled
                    if show_heatmap:
                        # Use vectorized operation for heatmap data - Improvement #3.1
                        valid_coords = type_df[(type_df['LAT'].notna()) & (type_df['LON'].notna())]
                        heat_data_day_type = valid_coords[['LAT', 'LON']].values.tolist()
                        
                        if heat_data_day_type:
                            # Create separate FeatureGroup for heatmap with nested visual formatting
                            # Format: "    └─ [date] - [anomaly_type] Heatmap" (extra indentation to show nesting)
                            heatmap_group = folium.FeatureGroup(
                                name=f'    └─ {date_str} - {anomaly_type} Heatmap',
                                overlay=True,  # Must be True to appear as checkbox (not radio button)
                                show=False,  # Hidden by default (user can toggle independently)
                                control=True
                            )
                            heatmap_group.add_to(m)
                            
                            plugins.HeatMap(
                                heat_data_day_type,
                                name=f'Heatmap - {date_str} - {anomaly_type}',
                                radius=15,
                                blur=25,
                                max_zoom=13,
                                overlay=False,  # Hidden by default
                                control=False  # Don't show in control, it's part of the heatmap group
                            ).add_to(heatmap_group)
    
    def _add_simple_heatmap(self, m: folium.Map, anomalies_df: pd.DataFrame):
        """Add a simple heatmap layer when not grouping by day"""
        heat_data = []
        for idx, row in anomalies_df.iterrows():
            if 'LAT' in row and 'LON' in row:
                heat_data.append([row['LAT'], row['LON']])
        
        if heat_data:
            plugins.HeatMap(
                heat_data,
                name='Heatmap',
                radius=15,
                blur=25,
                max_zoom=13,
                overlay=True,
                control=True
            ).add_to(m)
    
    def _add_grouped_by_type_only(self, m: folium.Map, anomalies_df: pd.DataFrame, show_clustering: bool = True):
        """Add markers grouped by anomaly type only (when no date available)"""
        if 'AnomalyType' not in anomalies_df.columns:
            # Fallback to regular markers
            if show_clustering:
                self._add_clustered_markers(m, anomalies_df)
            else:
                self._add_individual_markers(m, anomalies_df)
            return
        
        anomaly_types = anomalies_df['AnomalyType'].unique()
        
        for anomaly_type in sorted(anomaly_types):
            type_df = anomalies_df[anomalies_df['AnomalyType'] == anomaly_type]
            color = self.ANOMALY_COLORS.get(anomaly_type, 'gray')
            icon = self.ANOMALY_ICONS.get(anomaly_type, 'question')
            
            type_group = folium.FeatureGroup(
                name=f'{anomaly_type} ({len(type_df)})',
                overlay=True,
                control=True
            )
            type_group.add_to(m)
            
            if show_clustering:
                marker_cluster = plugins.MarkerCluster(
                    name=f'{anomaly_type}',
                    overlay=False,
                    control=False
                )
                marker_cluster.add_to(type_group)
                
                for idx, row in type_df.iterrows():
                    if 'LAT' in row and 'LON' in row:
                        popup_html = self._create_popup_html(row)
                        folium.Marker(
                            location=[row['LAT'], row['LON']],
                            popup=folium.Popup(popup_html, max_width=300),
                            tooltip=f"{anomaly_type} - MMSI: {row.get('MMSI', 'N/A')}",
                            icon=folium.Icon(color=color, icon=icon, prefix='fa')
                        ).add_to(marker_cluster)
            else:
                for idx, row in type_df.iterrows():
                    if 'LAT' in row and 'LON' in row:
                        popup_html = self._create_popup_html(row)
                        folium.Marker(
                            location=[row['LAT'], row['LON']],
                            popup=folium.Popup(popup_html, max_width=300),
                            tooltip=f"{anomaly_type} - MMSI: {row.get('MMSI', 'N/A')}",
                            icon=folium.Icon(color=color, icon=icon, prefix='fa')
                        ).add_to(type_group)
    
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
    
    def _add_vessel_track(self, m: folium.Map, vessel_data: pd.DataFrame, show_all_points: bool = True):
        """Add vessel track as a line with optional markers for all points"""
        # Sort by timestamp
        if 'BaseDateTime' in vessel_data.columns:
            vessel_data = vessel_data.sort_values('BaseDateTime')
        
        # Create line coordinates
        coordinates = []
        for idx, row in vessel_data.iterrows():
            if 'LAT' in row and 'LON' in row:
                coordinates.append([row['LAT'], row['LON']])
        
        if coordinates:
            # Add polyline for the track
            folium.PolyLine(
                coordinates,
                color='blue',
                weight=3,
                opacity=0.7,
                popup='Vessel Track',
                tooltip=f'Track with {len(coordinates)} points'
            ).add_to(m)
            
            # Add markers for all points if requested
            if show_all_points:
                # Create a FeatureGroup for track points
                track_points_group = folium.FeatureGroup(
                    name='Track Points',
                    overlay=True,
                    control=True
                )
                track_points_group.add_to(m)
                
                # Add a marker for each point (use smaller, semi-transparent markers)
                point_num = 0
                for idx, row in vessel_data.iterrows():
                    if 'LAT' in row and 'LON' in row:
                        point_num += 1
                        # Create popup with point details
                        point_popup = f"Point {point_num}/{len(vessel_data)}<br>"
                        if 'BaseDateTime' in row and pd.notna(row['BaseDateTime']):
                            point_popup += f"Time: {row['BaseDateTime']}<br>"
                        if 'SOG' in row and pd.notna(row['SOG']):
                            point_popup += f"Speed: {row['SOG']:.1f} knots<br>"
                        if 'COG' in row and pd.notna(row['COG']):
                            point_popup += f"Course: {row['COG']:.0f}°"
                        
                        folium.CircleMarker(
                            location=[row['LAT'], row['LON']],
                            radius=3,
                            popup=folium.Popup(point_popup, max_width=200),
                            tooltip=f"Point {point_num}",
                            color='blue',
                            fillColor='lightblue',
                            fillOpacity=0.6,
                            weight=1
                        ).add_to(track_points_group)
    
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
    
    def _add_heatmap_layer(self, m: folium.Map, anomalies_df: pd.DataFrame, group_by_day: bool = True):
        """
        Add heatmap layers with hierarchical grouping:
        PARENT: Day (or "All Days")
          CHILD: Anomaly Type
            GRANDCHILD: Heatmap (contains heatmap visualization)
        """
        # Ensure we have BaseDateTime column
        if 'BaseDateTime' not in anomalies_df.columns and 'ReportDate' in anomalies_df.columns:
            anomalies_df = anomalies_df.copy()
            anomalies_df['BaseDateTime'] = pd.to_datetime(anomalies_df['ReportDate'])
        
        if 'BaseDateTime' not in anomalies_df.columns:
            # No date column, create single "All Days" heatmap (hidden by default)
            # IMPROVEMENT #3: Use vectorized operation
            valid_coords = anomalies_df[(anomalies_df['LAT'].notna()) & (anomalies_df['LON'].notna())]
            heat_data_all = valid_coords[['LAT', 'LON']].values.tolist() if len(valid_coords) > 0 else []
            
            if heat_data_all:
                plugins.HeatMap(
                    heat_data_all,
                    name='🔥 All Days - Heatmap',
                    radius=15,
                    blur=25,
                    max_zoom=13,
                    overlay=False,  # Hidden by default - user can toggle on
                    control=True
                ).add_to(m)
            return
        
        # Convert BaseDateTime to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(anomalies_df['BaseDateTime']):
            anomalies_df = anomalies_df.copy()
            anomalies_df['BaseDateTime'] = pd.to_datetime(anomalies_df['BaseDateTime'])
        
        # Extract date
        anomalies_df = anomalies_df.copy()
        anomalies_df['Date'] = anomalies_df['BaseDateTime'].dt.date
        
        # Group by date
        dates = sorted(anomalies_df['Date'].unique())
        
        # Get all unique anomaly types for "All Days" groups
        all_anomaly_types = sorted(anomalies_df['AnomalyType'].unique()) if 'AnomalyType' in anomalies_df.columns else []
        
        # ========================================================================
        # LEVEL 2: CHILD GROUPS - Anomaly Types under "All Days"
        # LEVEL 3: GRANDCHILD GROUPS - Heatmaps under each Anomaly Type
        # ========================================================================
        for anomaly_type in all_anomaly_types:
            type_df_all = anomalies_df[anomalies_df['AnomalyType'] == anomaly_type]
            # IMPROVEMENT #3: Use vectorized operation for heatmap data
            valid_coords = type_df_all[(type_df_all['LAT'].notna()) & (type_df_all['LON'].notna())]
            heat_data_all_type = valid_coords[['LAT', 'LON']].values.tolist() if len(valid_coords) > 0 else []
            
            if heat_data_all_type:
                # LEVEL 3: Create GRANDCHILD group (Heatmap) under CHILD (Anomaly Type)
                # Note: Parent "All Days" already exists from _add_grouped_markers
                plugins.HeatMap(
                    heat_data_all_type,
                    name=f'🔥 All Days - {anomaly_type} - Heatmap ({len(type_df_all)})',
                    radius=15,
                    blur=25,
                    max_zoom=13,
                    overlay=False,  # Hidden by default - user can toggle on
                    control=True
                ).add_to(m)
        
        # ========================================================================
        # DAY-SPECIFIC HEATMAPS: Same hierarchy for each individual day
        # ========================================================================
        if group_by_day:
            try:
                for date in dates:
                    date_df = anomalies_df[anomalies_df['Date'] == date]
                    date_str = date.strftime('%Y-%m-%d')
                    
                    # Create heatmap for each anomaly type in this day
                    if 'AnomalyType' in date_df.columns:
                        for anomaly_type in sorted(date_df['AnomalyType'].unique()):
                            type_df = date_df[date_df['AnomalyType'] == anomaly_type]
                            # IMPROVEMENT #3: Use vectorized operation for heatmap data
                            valid_coords = type_df[(type_df['LAT'].notna()) & (type_df['LON'].notna())]
                            heat_data_day_type = valid_coords[['LAT', 'LON']].values.tolist() if len(valid_coords) > 0 else []
                            
                            if heat_data_day_type:
                                # LEVEL 3: Create GRANDCHILD group (Heatmap) under CHILD (Anomaly Type)
                                # Note: Parent Day and Child Anomaly Type already exist from _add_grouped_markers
                                plugins.HeatMap(
                                    heat_data_day_type,
                                    name=f'🔥 {date_str} - {anomaly_type} - Heatmap ({len(type_df)})',
                                    radius=15,
                                    blur=25,
                                    max_zoom=13,
                                    overlay=False,  # Hidden by default
                                    control=True
                                ).add_to(m)
                    else:
                        # No anomaly type, create single heatmap for the day
                        # IMPROVEMENT #3: Use vectorized operation for heatmap data
                        valid_coords = date_df[(date_df['LAT'].notna()) & (date_df['LON'].notna())]
                        heat_data_day = valid_coords[['LAT', 'LON']].values.tolist() if len(valid_coords) > 0 else []
                        
                        if heat_data_day:
                            plugins.HeatMap(
                                heat_data_day,
                                name=f'🔥 {date_str} - Heatmap ({len(date_df)})',
                                radius=15,
                                blur=25,
                                max_zoom=13,
                                overlay=False,  # Hidden by default
                                control=True
                            ).add_to(m)
            except Exception as e:
                logger.warning(f"Could not create day-grouped heatmaps: {e}")
    
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
    
    def create_filtered_path_maps(self, vessel_tracks: Dict[str, pd.DataFrame],
                                  anomalies_df: pd.DataFrame,
                                  mmsi_filter: Optional[List[str]] = None,
                                  vessel_type_filter: Optional[List[str]] = None,
                                  anomaly_type_filter: Optional[List[str]] = None,
                                  output_dir: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Create filtered path maps for multiple vessels
        
        Args:
            vessel_tracks: Dict mapping MMSI to DataFrame with full position data for all days
            anomalies_df: DataFrame with all anomalies
            mmsi_filter: Optional list of MMSIs to include
            vessel_type_filter: Optional list of vessel type categories (e.g., ['Cargo', 'Tanker'])
            anomaly_type_filter: Optional list of anomaly types to show
            output_dir: Optional output directory (defaults to self.output_dir)
        
        Returns:
            Dict mapping MMSI to map creation results
        """
        results = {}
        
        # Filter vessels by MMSI if specified
        if mmsi_filter:
            vessel_tracks = {mmsi: df for mmsi, df in vessel_tracks.items() if mmsi in mmsi_filter}
        
        # Filter by vessel type if specified
        if vessel_type_filter:
            from .vessel_types import VESSEL_TYPE_CATEGORIES
            type_codes = []
            for vtype in vessel_type_filter:
                if vtype in VESSEL_TYPE_CATEGORIES:
                    type_codes.extend(VESSEL_TYPE_CATEGORIES[vtype])
            
            filtered_tracks = {}
            for mmsi, df in vessel_tracks.items():
                if 'VesselType' in df.columns and df['VesselType'].isin(type_codes).any():
                    filtered_tracks[mmsi] = df
            vessel_tracks = filtered_tracks
        
        # Create maps for each vessel
        for mmsi, vessel_df in vessel_tracks.items():
            # Get anomalies for this vessel
            vessel_anomalies = anomalies_df[anomalies_df['MMSI'] == int(mmsi)] if 'MMSI' in anomalies_df.columns else pd.DataFrame()
            
            # Skip vessels with no position data
            if vessel_df.empty or ('LAT' not in vessel_df.columns or 'LON' not in vessel_df.columns):
                logger.warning(f"Skipping vessel {mmsi}: No valid position data")
                continue
            
            # Create map with filters
            result = self.create_vessel_track_map(
                vessel_data=vessel_df,
                mmsi=mmsi,
                anomalies_df=vessel_anomalies if not vessel_anomalies.empty else None,
                show_all_points=True,
                filter_by_anomaly_type=anomaly_type_filter,
                filter_by_vessel_type=vessel_type_filter
            )
            
            results[mmsi] = result
        
        return results
    
    def _add_legend(self, m: folium.Map, anomalies_df: pd.DataFrame):
        """Add legend showing anomaly types"""
        # Get unique anomaly types in the data
        if 'AnomalyType' in anomalies_df.columns:
            anomaly_types = anomalies_df['AnomalyType'].unique()
            
            legend_html = '''
                <div style="position: fixed; 
                            bottom: 50px; left: 50px; width: 220px; 
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

