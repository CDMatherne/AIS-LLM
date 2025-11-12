"""
SEED VISUALIZATIONS - Reference File

⚠️ IMPORTANT: This file is NOT imported by the application.
It is a seed/reference file containing example visualizations.

Purpose:
- Contains 3 example visualizations for law enforcement use cases
- Serves as a reference for creating custom visualizations
- Can be used to seed the visualization registry

Usage:
- Copy visualization code into the registry via API or LLM interface
- See INSTALLATION_INSTRUCTIONS at bottom of file

Based on actual output data from Output/AIS_Anomalies_Summary.csv
These 3 visualizations are ready to be added to the visualization registry.
Each one solves a real law enforcement need.
"""

# ============================================================================
# SEED VISUALIZATION 1: Course Anomaly Analysis
# ============================================================================

SEED_1_NAME = "course_anomaly_tracker"
SEED_1_DESCRIPTION = "Interactive timeline showing course anomalies with heading vs COG discrepancies. Helps identify vessels that may be trying to evade tracking by reporting false headings."
SEED_1_TYPE = "interactive_chart"
SEED_1_CATEGORY = "Law Enforcement"

SEED_1_CODE = '''
def generate_visualization(data, parameters, output_dir):
    """
    Course Anomaly Tracker - Interactive timeline with heading/COG analysis
    
    Specifically designed for law enforcement to identify vessels with 
    suspicious heading discrepancies that may indicate evasion tactics.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import pandas as pd
    import os
    
    # Filter for course anomalies
    course_data = data[data['AnomalyType'] == 'Course'].copy()
    
    if course_data.empty:
        raise ValueError("No course anomalies found in data")
    
    # Ensure numeric types
    course_data['CourseHeadingDiff'] = pd.to_numeric(course_data['CourseHeadingDiff'], errors='coerce')
    course_data['COG'] = pd.to_numeric(course_data['COG'], errors='coerce')
    course_data['Heading'] = pd.to_numeric(course_data['Heading'], errors='coerce')
    course_data['BaseDateTime'] = pd.to_datetime(course_data['BaseDateTime'])
    
    # Remove any rows with invalid data
    course_data = course_data.dropna(subset=['CourseHeadingDiff', 'COG', 'Heading'])
    
    # Sort by time
    course_data = course_data.sort_values('BaseDateTime')
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            'Course vs Heading Discrepancy Over Time',
            'Distribution of Heading-COG Differences'
        ),
        row_heights=[0.6, 0.4],
        vertical_spacing=0.12
    )
    
    # Top plot: Timeline scatter
    # Color by severity (larger difference = more suspicious)
    severity = course_data['CourseHeadingDiff'].abs()
    
    fig.add_trace(
        go.Scatter(
            x=course_data['BaseDateTime'],
            y=course_data['CourseHeadingDiff'],
            mode='markers',
            marker=dict(
                size=10,
                color=severity,
                colorscale='Reds',
                showscale=True,
                colorbar=dict(
                    title="Discrepancy<br>(degrees)",
                    x=1.15,
                    len=0.5,
                    y=0.75
                ),
                line=dict(width=1, color='DarkSlateGray')
            ),
            text=[
                f"<b>{row['VesselName']}</b><br>" +
                f"MMSI: {row['MMSI']}<br>" +
                f"Heading: {row['Heading']:.1f}°<br>" +
                f"COG: {row['COG']:.1f}°<br>" +
                f"Diff: {row['CourseHeadingDiff']:.1f}°<br>" +
                f"Time: {row['BaseDateTime']}"
                for idx, row in course_data.iterrows()
            ],
            hovertemplate='%{text}<extra></extra>',
            name='Course Anomalies'
        ),
        row=1, col=1
    )
    
    # Add threshold lines
    fig.add_hline(y=45, line_dash="dash", line_color="orange", 
                 annotation_text="High Risk Threshold", row=1, col=1)
    fig.add_hline(y=-45, line_dash="dash", line_color="orange", row=1, col=1)
    fig.add_hline(y=90, line_dash="dash", line_color="red", 
                 annotation_text="Critical", row=1, col=1)
    fig.add_hline(y=-90, line_dash="dash", line_color="red", row=1, col=1)
    
    # Bottom plot: Histogram
    fig.add_trace(
        go.Histogram(
            x=course_data['CourseHeadingDiff'],
            nbinsx=30,
            marker=dict(
                color='steelblue',
                line=dict(color='white', width=1)
            ),
            name='Distribution'
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_yaxes(title_text="Heading - COG Difference (degrees)", row=1, col=1)
    fig.update_xaxes(title_text="Heading - COG Difference (degrees)", row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)
    
    fig.update_layout(
        title=dict(
            text=parameters.get('title', 
                '<b>Course Anomaly Analysis - Law Enforcement Report</b><br>' +
                '<sub>Vessels with suspicious heading/COG discrepancies</sub>'),
            x=0.5,
            xanchor='center'
        ),
        showlegend=False,
        height=800,
        hovermode='closest',
        template='plotly_white'
    )
    
    # Add summary annotation
    summary = f"""
    <b>Summary:</b>
    Total Course Anomalies: {len(course_data)}
    Unique Vessels: {course_data['MMSI'].nunique()}
    Max Discrepancy: {course_data['CourseHeadingDiff'].abs().max():.1f}°
    Vessels >90° diff: {len(course_data[course_data['CourseHeadingDiff'].abs() > 90])}
    """
    
    fig.add_annotation(
        text=summary,
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        xanchor='left', yanchor='top',
        showarrow=False,
        bgcolor="lightyellow",
        bordercolor="black",
        borderwidth=1,
        font=dict(size=10, family="monospace")
    )
    
    # Save
    output_file = os.path.join(output_dir, 
        parameters.get('filename', 'course_anomaly_tracker.html'))
    fig.write_html(output_file)
    
    return output_file
'''


# ============================================================================
# SEED VISUALIZATION 2: Vessel Risk Heat Matrix
# ============================================================================

SEED_2_NAME = "vessel_risk_heat_matrix"
SEED_2_DESCRIPTION = "Heat matrix showing vessels (rows) vs anomaly types (columns) with risk scoring. Instantly identifies the highest risk vessels requiring immediate investigation."
SEED_2_TYPE = "static_chart"
SEED_2_CATEGORY = "Law Enforcement"

SEED_2_CODE = '''
def generate_visualization(data, parameters, output_dir):
    """
    Vessel Risk Heat Matrix - Prioritize investigations
    
    Creates a heat matrix with risk scoring to help law enforcement
    prioritize which vessels require immediate attention.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    import os
    
    # Create pivot table of vessel vs anomaly type
    pivot = data.groupby(['MMSI', 'AnomalyType']).size().unstack(fill_value=0)
    
    # Calculate risk score for each vessel
    # Different anomaly types have different weights
    risk_weights = {
        'Zone_Violation': 3,
        'Course': 2,
        'Speed': 2,
        'Loitering': 3,
        'Rendezvous': 4,
        'Identity_Spoofing': 5,
        'AIS_Beacon_Off': 4,
        'AIS_Beacon_On': 2,
        'COG_Heading_Inconsistency': 2,
        'Excessive_Travel': 2
    }
    
    # Calculate weighted risk scores
    risk_scores = pd.Series(0, index=pivot.index)
    for anomaly_type in pivot.columns:
        weight = risk_weights.get(anomaly_type, 1)
        risk_scores += pivot[anomaly_type] * weight
    
    # Get top N vessels by risk score
    top_n = parameters.get('top_n', 20)
    top_vessels = risk_scores.nlargest(top_n).index
    pivot_top = pivot.loc[top_vessels]
    
    # Get vessel names
    vessel_labels = []
    for mmsi in top_vessels:
        vessel_data = data[data['MMSI'] == mmsi]
        name = vessel_data['VesselName'].iloc[0] if 'VesselName' in vessel_data.columns else 'Unknown'
        risk = int(risk_scores.loc[mmsi])
        vessel_labels.append(f"{name[:20]:<20} (MMSI: {mmsi}) [Risk: {risk}]")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, max(12, top_n * 0.5)))
    
    # Create heatmap
    sns.heatmap(
        pivot_top, 
        annot=True, 
        fmt='d', 
        cmap='YlOrRd',
        cbar_kws={'label': 'Anomaly Count'},
        linewidths=1,
        linecolor='white',
        ax=ax,
        vmin=0,
        vmax=pivot_top.values.max()
    )
    
    # Customize
    ax.set_yticklabels(vessel_labels, rotation=0, fontsize=9, fontfamily='monospace')
    ax.set_xlabel('Anomaly Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Vessel (Name, MMSI, Risk Score)', fontsize=12, fontweight='bold')
    ax.set_title(
        parameters.get('title', 
            f'TOP {top_n} HIGH-RISK VESSELS - LAW ENFORCEMENT PRIORITY MATRIX\\n' +
            'Color intensity = anomaly count | Risk Score = weighted severity'),
        fontsize=14, fontweight='bold', pad=20
    )
    
    # Rotate x labels
    plt.xticks(rotation=45, ha='right')
    
    # Add risk score color bar on the right
    risk_scores_top = risk_scores.loc[top_vessels].values
    risk_colors = plt.cm.Reds(risk_scores_top / risk_scores_top.max())
    
    # Add risk indicators
    for i, (idx, vessel) in enumerate(pivot_top.iterrows()):
        # Add risk indicator box at the end of each row
        rect = plt.Rectangle((len(pivot_top.columns) + 0.1, i), 0.3, 0.8, 
                             facecolor=risk_colors[i], edgecolor='black', linewidth=2)
        ax.add_patch(rect)
    
    plt.tight_layout()
    
    # Save
    output_file = os.path.join(output_dir, 
        parameters.get('filename', 'vessel_risk_heat_matrix.png'))
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_file
'''


# ============================================================================
# SEED VISUALIZATION 3: Zone Violation Geographic Intelligence Map
# ============================================================================

SEED_3_NAME = "zone_violation_intel_map"
SEED_3_DESCRIPTION = "Intelligence-focused map showing zone violations with vessel tracks, risk indicators, and zone boundaries. Essential for maritime law enforcement operations."
SEED_3_TYPE = "map"
SEED_3_CATEGORY = "Law Enforcement"

SEED_3_CODE = '''
def generate_visualization(data, parameters, output_dir):
    """
    Zone Violation Intelligence Map
    
    Creates a professional intelligence map showing zone violations
    with detailed vessel information and risk assessment.
    """
    import folium
    from folium import plugins
    import pandas as pd
    import os
    
    # Filter for zone violations
    zone_data = data[data['AnomalyType'] == 'Zone_Violation'].copy()
    
    if zone_data.empty:
        raise ValueError("No zone violations found in data")
    
    # Calculate center
    center_lat = zone_data['LAT'].mean()
    center_lon = zone_data['LON'].mean()
    
    # Create map with dark theme for operations
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=parameters.get('zoom', 7),
        tiles='CartoDB dark_matter'
    )
    
    # Add alternative base maps
    folium.TileLayer('OpenStreetMap', name='Street View').add_to(m)
    folium.TileLayer('Stamen Terrain', name='Terrain').add_to(m)
    folium.TileLayer('CartoDB positron', name='Light').add_to(m)
    
    # Draw zone boundaries
    if 'ZoneName' in zone_data.columns:
        zones = zone_data.groupby('ZoneName').first()
        
        for zone_name, zone_info in zones.iterrows():
            if pd.notna(zone_info.get('ZoneLatMin')):
                # Draw zone rectangle
                bounds = [
                    [zone_info['ZoneLatMin'], zone_info['ZoneLonMin']],
                    [zone_info['ZoneLatMax'], zone_info['ZoneLonMax']]
                ]
                
                folium.Rectangle(
                    bounds=bounds,
                    color='red',
                    fill=True,
                    fillColor='red',
                    fillOpacity=0.1,
                    weight=3,
                    popup=f'<b>Restricted Zone: {zone_name}</b>',
                    tooltip=f'Zone: {zone_name}'
                ).add_to(m)
    
    # Count violations per vessel
    vessel_violation_counts = zone_data.groupby('MMSI').size()
    
    # Define risk levels
    def get_risk_level(count):
        if count >= 10:
            return 'CRITICAL', 'darkred', 15
        elif count >= 5:
            return 'HIGH', 'red', 12
        elif count >= 3:
            return 'MEDIUM', 'orange', 10
        else:
            return 'LOW', 'yellow', 8
    
    # Add vessel markers
    marker_cluster = plugins.MarkerCluster(name='Violation Points').add_to(m)
    
    for mmsi, vessel_group in zone_data.groupby('MMSI'):
        violation_count = len(vessel_group)
        risk_level, color, size = get_risk_level(violation_count)
        
        # Get vessel info
        vessel_info = vessel_group.iloc[0]
        vessel_name = vessel_info.get('VesselName', 'Unknown')
        vessel_type = vessel_info.get('VesselType', 'Unknown')
        
        # Create detailed popup
        popup_html = f"""
        <div style="font-family: Arial; width: 300px;">
            <h4 style="margin: 0; color: {color};">⚠️ ZONE VIOLATION - {risk_level} RISK</h4>
            <hr>
            <table style="width: 100%; font-size: 11px;">
                <tr><td><b>Vessel:</b></td><td>{vessel_name}</td></tr>
                <tr><td><b>MMSI:</b></td><td>{mmsi}</td></tr>
                <tr><td><b>Type:</b></td><td>{vessel_type}</td></tr>
                <tr><td><b>IMO:</b></td><td>{vessel_info.get('IMO', 'N/A')}</td></tr>
                <tr><td><b>Flag:</b></td><td>{vessel_info.get('CallSign', 'N/A')}</td></tr>
                <tr><td><b>Violations:</b></td><td><b>{violation_count}</b></td></tr>
            </table>
            <hr>
            <p style="font-size: 10px; margin: 5px 0;">
                <b>Recent Violations:</b><br>
        """
        
        # Add up to 5 most recent violations
        for idx, row in vessel_group.head(5).iterrows():
            popup_html += f"• {row['BaseDateTime']} - {row.get('ZoneName', 'Unknown Zone')}<br>"
        
        popup_html += """
            </p>
        </div>
        """
        
        # Add marker for latest position
        latest = vessel_group.iloc[-1]
        
        folium.CircleMarker(
            location=[latest['LAT'], latest['LON']],
            radius=size,
            popup=folium.Popup(popup_html, max_width=320),
            tooltip=f"{vessel_name} - {violation_count} violations ({risk_level} RISK)",
            color='black',
            fillColor=color,
            fillOpacity=0.8,
            weight=2
        ).add_to(marker_cluster)
        
        # Draw vessel track if multiple points
        if len(vessel_group) > 1:
            track_coords = [[row['LAT'], row['LON']] for idx, row in vessel_group.iterrows()]
            folium.PolyLine(
                track_coords,
                color=color,
                weight=2,
                opacity=0.6,
                popup=f"{vessel_name} - Track"
            ).add_to(m)
    
    # Add heatmap layer
    heat_data = [[row['LAT'], row['LON']] for idx, row in zone_data.iterrows()]
    plugins.HeatMap(
        heat_data,
        name='Violation Density',
        radius=20,
        blur=30,
        max_zoom=10,
        overlay=True,
        control=True,
        show=False
    ).add_to(m)
    
    # Add title with statistics
    title_html = f"""
    <div style="position: fixed; 
                top: 10px; left: 50px; 
                width: 500px; height: auto; 
                background-color: rgba(0, 0, 0, 0.8); 
                border: 3px solid red;
                z-index: 9999; 
                padding: 10px;
                color: white;
                font-family: Arial;">
        <h3 style="margin: 0; color: #ff4444;">⚠️ ZONE VIOLATION INTELLIGENCE MAP</h3>
        <hr style="margin: 5px 0; border-color: #ff4444;">
        <table style="width: 100%; font-size: 12px; color: white;">
            <tr><td>Total Violations:</td><td><b>{len(zone_data)}</b></td></tr>
            <tr><td>Unique Vessels:</td><td><b>{zone_data['MMSI'].nunique()}</b></td></tr>
            <tr><td>Critical Risk Vessels:</td><td><b style="color: #ff0000;">{len(vessel_violation_counts[vessel_violation_counts >= 10])}</b></td></tr>
            <tr><td>High Risk Vessels:</td><td><b style="color: #ff6600;">{len(vessel_violation_counts[vessel_violation_counts >= 5])}</b></td></tr>
            <tr><td>Date Range:</td><td>{zone_data['BaseDateTime'].min()} to {zone_data['BaseDateTime'].max()}</td></tr>
        </table>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Add legend
    legend_html = """
    <div style="position: fixed; 
                bottom: 50px; right: 50px;
                width: 200px;
                background-color: rgba(255, 255, 255, 0.9);
                border: 2px solid black;
                z-index: 9999;
                padding: 10px;
                font-size: 11px;">
        <p style="margin: 0; font-weight: bold;">Risk Levels</p>
        <hr style="margin: 5px 0;">
        <p style="margin: 2px;"><span style="color: darkred;">●</span> CRITICAL (10+ violations)</p>
        <p style="margin: 2px;"><span style="color: red;">●</span> HIGH (5-9 violations)</p>
        <p style="margin: 2px;"><span style="color: orange;">●</span> MEDIUM (3-4 violations)</p>
        <p style="margin: 2px;"><span style="color: yellow;">●</span> LOW (1-2 violations)</p>
        <hr style="margin: 5px 0;">
        <p style="margin: 2px; font-size: 9px;">Red boxes = Restricted zones</p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add fullscreen
    plugins.Fullscreen().add_to(m)
    
    # Add measurement tool
    plugins.MeasureControl(position='topleft').add_to(m)
    
    # Save
    output_file = os.path.join(output_dir, 
        parameters.get('filename', 'zone_violation_intel_map.html'))
    m.save(output_file)
    
    return output_file
'''


# ============================================================================
# Installation Instructions
# ============================================================================

INSTALLATION_INSTRUCTIONS = """
To add these seed visualizations to your system:

METHOD 1 - Via API (Recommended):
----------------------------------
```python
import requests

session_id = "your_session_id"
analysis_id = "your_analysis_id"

# Add Seed 1
requests.post('http://localhost:8000/api/visualizations/create', json={
    'session_id': session_id,
    'analysis_id': analysis_id,
    'code': SEED_1_CODE,
    'name': SEED_1_NAME,
    'description': SEED_1_DESCRIPTION,
    'save_to_registry': True
})

# Repeat for SEED_2 and SEED_3
```

METHOD 2 - Via Claude Chat:
----------------------------
Once connected to Claude, say:
"Claude, please save this visualization code to the registry"
[paste SEED_1_CODE]
Name: course_anomaly_tracker
Description: [paste SEED_1_DESCRIPTION]

METHOD 3 - Direct Registry Addition:
------------------------------------
Add directly to visualization_engine.py's registry initialization.

Testing:
--------
After adding, test with:
"Claude, show me the course anomaly tracker for my last analysis"
"Claude, create a vessel risk heat matrix with top 15 vessels"
"Claude, generate the zone violation intelligence map"
"""

if __name__ == "__main__":
    print("=" * 80)
    print("AIS LAW ENFORCEMENT LLM - SEED VISUALIZATIONS")
    print("=" * 80)
    print()
    print("3 Professional Law Enforcement Visualizations Ready for Deployment")
    print()
    print(f"1. {SEED_1_NAME}")
    print(f"   Type: {SEED_1_TYPE}")
    print(f"   {SEED_1_DESCRIPTION}")
    print()
    print(f"2. {SEED_2_NAME}")
    print(f"   Type: {SEED_2_TYPE}")
    print(f"   {SEED_2_DESCRIPTION}")
    print()
    print(f"3. {SEED_3_NAME}")
    print(f"   Type: {SEED_3_TYPE}")
    print(f"   {SEED_3_DESCRIPTION}")
    print()
    print("=" * 80)
    print(INSTALLATION_INSTRUCTIONS)
    print("=" * 80)

