"""
Chart Creation for AIS Anomaly Analysis
Creates professional charts and visualizations using matplotlib and plotly
"""
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Set style for matplotlib
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


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


class AISChartCreator:
    """
    Creates professional charts and visualizations for AIS analysis
    """
    
    def __init__(self, output_dir: str = None):
        if output_dir is None:
            output_dir = get_default_output_directory()
        self.output_dir = Path(output_dir)
        self.charts_dir = self.output_dir / "Charts"
        self.charts_dir.mkdir(parents=True, exist_ok=True)
    
    def create_anomaly_types_distribution(self, anomalies_df: pd.DataFrame,
                                         chart_type: str = "bar",
                                         output_filename: str = "anomaly_types_distribution.png") -> Dict[str, Any]:
        """
        Create a chart showing the distribution of anomaly types
        
        Args:
            anomalies_df: DataFrame with anomaly data
            chart_type: 'bar', 'pie', or 'both'
            output_filename: Name of output file
        
        Returns:
            Dict with success status and file info
        """
        try:
            if anomalies_df.empty or 'AnomalyType' not in anomalies_df.columns:
                return {
                    'success': False,
                    'error': 'No anomaly type data available',
                    'file_path': None
                }
            
            # Count anomalies by type
            type_counts = anomalies_df['AnomalyType'].value_counts()
            
            if chart_type in ['bar', 'both']:
                # Create bar chart
                fig, ax = plt.subplots(figsize=(12, 8))
                
                colors = plt.cm.Set3(np.linspace(0, 1, len(type_counts)))
                bars = ax.bar(range(len(type_counts)), type_counts.values, color=colors)
                
                ax.set_xlabel('Anomaly Type', fontsize=12, fontweight='bold')
                ax.set_ylabel('Count', fontsize=12, fontweight='bold')
                ax.set_title('Distribution of Anomaly Types', fontsize=14, fontweight='bold', pad=20)
                ax.set_xticks(range(len(type_counts)))
                ax.set_xticklabels(type_counts.index, rotation=45, ha='right')
                
                # Add value labels on bars
                for i, (bar, count) in enumerate(zip(bars, type_counts.values)):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(count)}',
                           ha='center', va='bottom', fontweight='bold')
                
                # Add percentage labels
                total = type_counts.sum()
                for i, (bar, count) in enumerate(zip(bars, type_counts.values)):
                    height = bar.get_height()
                    percentage = (count / total) * 100
                    ax.text(bar.get_x() + bar.get_width()/2., height/2,
                           f'{percentage:.1f}%',
                           ha='center', va='center', fontsize=9, color='white', fontweight='bold')
                
                plt.tight_layout()
                
                # Generate timestamped filename to prevent overwrites
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                base_name = output_filename.replace('.png', '') if output_filename.endswith('.png') else output_filename
                timestamped_filename = f"{base_name}_{timestamp}.png"
                
                # Save
                output_path = self.charts_dir / timestamped_filename
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Created anomaly types distribution chart at {output_path}")
                
                return {
                    'success': True,
                    'file_path': str(output_path),
                    'file_name': timestamped_filename,
                    'file_size': output_path.stat().st_size,
                    'chart_type': 'bar',
                    'anomaly_types': len(type_counts),
                    'total_anomalies': int(total)
                }
            
            elif chart_type == 'pie':
                # Create pie chart
                fig, ax = plt.subplots(figsize=(10, 10))
                
                colors = plt.cm.Set3(np.linspace(0, 1, len(type_counts)))
                wedges, texts, autotexts = ax.pie(type_counts.values, 
                                                   labels=type_counts.index,
                                                   autopct='%1.1f%%',
                                                   colors=colors,
                                                   startangle=90)
                
                # Make percentage text bold
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                    autotext.set_fontsize(10)
                
                ax.set_title('Distribution of Anomaly Types', fontsize=14, fontweight='bold', pad=20)
                
                plt.tight_layout()
                
                # Generate timestamped filename to prevent overwrites
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                base_name = output_filename.replace('.png', '') if output_filename.endswith('.png') else output_filename
                timestamped_filename = f"{base_name}_{timestamp}.png"
                
                # Save
                output_path = self.charts_dir / timestamped_filename
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Created anomaly types pie chart at {output_path}")
                
                return {
                    'success': True,
                    'file_path': str(output_path),
                    'file_name': output_filename,
                    'file_size': output_path.stat().st_size,
                    'chart_type': 'pie',
                    'anomaly_types': len(type_counts),
                    'total_anomalies': int(type_counts.sum())
                }
            
        except Exception as e:
            logger.error(f"Error creating anomaly types chart: {e}")
            return {
                'success': False,
                'error': str(e),
                'file_path': None
            }
    
    def create_top_vessels_chart(self, anomalies_df: pd.DataFrame,
                                 top_n: int = 10,
                                 output_filename: str = "top_vessels_with_anomalies.png") -> Dict[str, Any]:
        """
        Create a bar chart showing vessels with most anomalies
        
        Args:
            anomalies_df: DataFrame with anomaly data
            top_n: Number of top vessels to show
            output_filename: Name of output file
        
        Returns:
            Dict with success status and file info
        """
        try:
            if anomalies_df.empty or 'MMSI' not in anomalies_df.columns:
                return {
                    'success': False,
                    'error': 'No vessel data available',
                    'file_path': None
                }
            
            # Count anomalies by vessel
            vessel_counts = anomalies_df['MMSI'].value_counts().head(top_n)
            
            # Get vessel names if available
            vessel_labels = []
            for mmsi in vessel_counts.index:
                vessel_data = anomalies_df[anomalies_df['MMSI'] == mmsi]
                if 'VesselName' in vessel_data.columns and pd.notna(vessel_data['VesselName'].iloc[0]):
                    vessel_name = vessel_data['VesselName'].iloc[0]
                    vessel_labels.append(f"{vessel_name}\n(MMSI: {mmsi})")
                else:
                    vessel_labels.append(f"MMSI: {mmsi}")
            
            # Create chart
            fig, ax = plt.subplots(figsize=(14, 8))
            
            colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(vessel_counts)))
            bars = ax.barh(range(len(vessel_counts)), vessel_counts.values, color=colors)
            
            ax.set_yticks(range(len(vessel_counts)))
            ax.set_yticklabels(vessel_labels, fontsize=10)
            ax.set_xlabel('Number of Anomalies', fontsize=12, fontweight='bold')
            ax.set_title(f'Top {top_n} Vessels with Most Anomalies', fontsize=14, fontweight='bold', pad=20)
            
            # Add value labels
            for i, (bar, count) in enumerate(zip(bars, vessel_counts.values)):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2.,
                       f' {int(count)}',
                       ha='left', va='center', fontweight='bold', fontsize=10)
            
            # Invert y-axis so #1 is on top
            ax.invert_yaxis()
            
            plt.tight_layout()
            
            # Generate timestamped filename to prevent overwrites
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_name = output_filename.replace('.png', '') if output_filename.endswith('.png') else output_filename
            timestamped_filename = f"{base_name}_{timestamp}.png"
            
            # Save
            output_path = self.charts_dir / timestamped_filename
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Created top vessels chart at {output_path}")
            
            return {
                'success': True,
                'file_path': str(output_path),
                'file_name': output_filename,
                'file_size': output_path.stat().st_size,
                'vessels_shown': len(vessel_counts),
                'top_vessel_mmsi': str(vessel_counts.index[0]),
                'top_vessel_anomalies': int(vessel_counts.values[0])
            }
            
        except Exception as e:
            logger.error(f"Error creating top vessels chart: {e}")
            return {
                'success': False,
                'error': str(e),
                'file_path': None
            }
    
    def create_anomalies_by_date_chart(self, anomalies_df: pd.DataFrame,
                                      output_filename: str = "anomalies_by_date.png") -> Dict[str, Any]:
        """
        Create a time series chart showing anomalies over time
        
        Args:
            anomalies_df: DataFrame with anomaly data
            output_filename: Name of output file
        
        Returns:
            Dict with success status and file info
        """
        try:
            if anomalies_df.empty or 'BaseDateTime' not in anomalies_df.columns:
                return {
                    'success': False,
                    'error': 'No temporal data available',
                    'file_path': None
                }
            
            # Convert to datetime
            anomalies_df = anomalies_df.copy()
            anomalies_df['BaseDateTime'] = pd.to_datetime(anomalies_df['BaseDateTime'])
            anomalies_df['Date'] = anomalies_df['BaseDateTime'].dt.date
            
            # Count by date
            date_counts = anomalies_df.groupby('Date').size().reset_index(name='Count')
            date_counts['Date'] = pd.to_datetime(date_counts['Date'])
            
            # Create chart
            fig, ax = plt.subplots(figsize=(14, 8))
            
            ax.plot(date_counts['Date'], date_counts['Count'], 
                   marker='o', linewidth=2, markersize=6, color='#2E86AB')
            ax.fill_between(date_counts['Date'], date_counts['Count'], alpha=0.3, color='#2E86AB')
            
            ax.set_xlabel('Date', fontsize=12, fontweight='bold')
            ax.set_ylabel('Number of Anomalies', fontsize=12, fontweight='bold')
            ax.set_title('Anomalies Detected Over Time', fontsize=14, fontweight='bold', pad=20)
            
            # Rotate x labels
            plt.xticks(rotation=45, ha='right')
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            stats_text = f"Total: {date_counts['Count'].sum()}\n"
            stats_text += f"Peak: {date_counts['Count'].max()} ({date_counts.loc[date_counts['Count'].idxmax(), 'Date'].strftime('%Y-%m-%d')})\n"
            stats_text += f"Avg: {date_counts['Count'].mean():.1f}/day"
            
            ax.text(0.02, 0.98, stats_text,
                   transform=ax.transAxes,
                   fontsize=10,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            
            # Generate timestamped filename to prevent overwrites
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_name = output_filename.replace('.png', '') if output_filename.endswith('.png') else output_filename
            timestamped_filename = f"{base_name}_{timestamp}.png"
            
            # Save
            output_path = self.charts_dir / timestamped_filename
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Created anomalies by date chart at {output_path}")
            
            return {
                'success': True,
                'file_path': str(output_path),
                'file_name': output_filename,
                'file_size': output_path.stat().st_size,
                'date_range': {
                    'start': date_counts['Date'].min().strftime('%Y-%m-%d'),
                    'end': date_counts['Date'].max().strftime('%Y-%m-%d')
                },
                'total_anomalies': int(date_counts['Count'].sum()),
                'peak_day': date_counts.loc[date_counts['Count'].idxmax(), 'Date'].strftime('%Y-%m-%d'),
                'peak_count': int(date_counts['Count'].max())
            }
            
        except Exception as e:
            logger.error(f"Error creating anomalies by date chart: {e}")
            return {
                'success': False,
                'error': str(e),
                'file_path': None
            }
    
    def create_3d_bar_chart(self, anomalies_df: pd.DataFrame,
                           output_filename: str = "anomalies_3d_bar_chart.png") -> Dict[str, Any]:
        """
        Create a 3D bar chart showing anomalies by type and date
        
        Args:
            anomalies_df: DataFrame with anomaly data
            output_filename: Name of output file
        
        Returns:
            Dict with success status and file info
        """
        try:
            if anomalies_df.empty:
                return {
                    'success': False,
                    'error': 'No data available',
                    'file_path': None
                }
            
            # Prepare data
            anomalies_df = anomalies_df.copy()
            if 'BaseDateTime' in anomalies_df.columns:
                anomalies_df['BaseDateTime'] = pd.to_datetime(anomalies_df['BaseDateTime'])
                anomalies_df['Date'] = anomalies_df['BaseDateTime'].dt.date
            else:
                return {
                    'success': False,
                    'error': 'No temporal data available',
                    'file_path': None
                }
            
            if 'AnomalyType' not in anomalies_df.columns:
                return {
                    'success': False,
                    'error': 'No anomaly type data available',
                    'file_path': None
                }
            
            # Create pivot table
            pivot = anomalies_df.groupby(['Date', 'AnomalyType']).size().unstack(fill_value=0)
            
            # Create 3D plot
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Prepare data for 3D bars
            x_data = []
            y_data = []
            z_data = []
            colors = []
            
            color_map = plt.cm.Set3(np.linspace(0, 1, len(pivot.columns)))
            
            for i, date in enumerate(pivot.index):
                for j, anomaly_type in enumerate(pivot.columns):
                    x_data.append(i)
                    y_data.append(j)
                    z_data.append(pivot.loc[date, anomaly_type])
                    colors.append(color_map[j])
            
            # Plot bars
            ax.bar3d(x_data, y_data, np.zeros(len(z_data)), 
                    0.8, 0.8, z_data, 
                    color=colors, alpha=0.8, shade=True)
            
            # Labels
            ax.set_xlabel('Date', fontsize=10, fontweight='bold')
            ax.set_ylabel('Anomaly Type', fontsize=10, fontweight='bold')
            ax.set_zlabel('Count', fontsize=10, fontweight='bold')
            ax.set_title('3D Distribution of Anomalies by Type and Date', 
                        fontsize=12, fontweight='bold', pad=20)
            
            # Set ticks
            ax.set_xticks(range(len(pivot.index)))
            ax.set_xticklabels([str(d) for d in pivot.index], rotation=45, ha='right', fontsize=8)
            ax.set_yticks(range(len(pivot.columns)))
            ax.set_yticklabels(pivot.columns, fontsize=8)
            
            plt.tight_layout()
            
            # Generate timestamped filename to prevent overwrites
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_name = output_filename.replace('.png', '') if output_filename.endswith('.png') else output_filename
            timestamped_filename = f"{base_name}_{timestamp}.png"
            
            # Save
            output_path = self.charts_dir / timestamped_filename
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Created 3D bar chart at {output_path}")
            
            return {
                'success': True,
                'file_path': str(output_path),
                'file_name': output_filename,
                'file_size': output_path.stat().st_size,
                'dimensions': {
                    'dates': len(pivot.index),
                    'anomaly_types': len(pivot.columns)
                }
            }
            
        except Exception as e:
            logger.error(f"Error creating 3D bar chart: {e}")
            return {
                'success': False,
                'error': str(e),
                'file_path': None
            }
    
    def create_scatterplot_interactive(self, anomalies_df: pd.DataFrame,
                                      output_filename: str = "Anomalies_Scatterplot.html") -> Dict[str, Any]:
        """
        Create an interactive Plotly scatterplot showing geographic distribution
        
        Args:
            anomalies_df: DataFrame with anomaly data
            output_filename: Name of output file
        
        Returns:
            Dict with success status and file info
        """
        try:
            if anomalies_df.empty or 'LAT' not in anomalies_df.columns or 'LON' not in anomalies_df.columns:
                return {
                    'success': False,
                    'error': 'No geographic data available',
                    'file_path': None
                }
            
            # Prepare data
            df = anomalies_df.copy()
            
            # Create color mapping for anomaly types
            if 'AnomalyType' in df.columns:
                color_discrete_map = {
                    'AIS_Beacon_Off': '#d62728',
                    'AIS_Beacon_On': '#ff7f0e',
                    'Speed': '#9467bd',
                    'Course': '#1f77b4',
                    'Loitering': '#2ca02c',
                    'Rendezvous': '#e377c2',
                    'Identity_Spoofing': '#8c564b',
                    'Zone_Violation': '#17becf',
                    'COG_Heading_Inconsistency': '#bcbd22',
                    'Excessive_Travel': '#9467bd'
                }
                
                # Create scatter plot
                fig = px.scatter_geo(df,
                                    lat='LAT',
                                    lon='LON',
                                    color='AnomalyType',
                                    hover_data=['MMSI', 'VesselName', 'BaseDateTime'] if 'VesselName' in df.columns else ['MMSI', 'BaseDateTime'],
                                    color_discrete_map=color_discrete_map,
                                    title='Geographic Distribution of AIS Anomalies')
            else:
                # No anomaly type - simple scatter
                fig = px.scatter_geo(df,
                                    lat='LAT',
                                    lon='LON',
                                    hover_data=['MMSI', 'BaseDateTime'],
                                    title='Geographic Distribution of AIS Anomalies')
            
            # Update layout
            fig.update_geos(
                projection_type="natural earth",
                showcoastlines=True,
                coastlinecolor="RebeccaPurple",
                showland=True,
                landcolor="LightGreen",
                showocean=True,
                oceancolor="LightBlue"
            )
            
            fig.update_layout(
                height=700,
                title_font_size=16,
                title_font_family="Arial Black",
                showlegend=True
            )
            
            # Save
            output_path = self.charts_dir / output_filename
            fig.write_html(str(output_path))
            
            logger.info(f"Created interactive scatterplot at {output_path}")
            
            return {
                'success': True,
                'file_path': str(output_path),
                'file_name': output_filename,
                'file_size': output_path.stat().st_size,
                'anomaly_count': len(df),
                'is_interactive': True
            }
            
        except Exception as e:
            logger.error(f"Error creating scatterplot: {e}")
            return {
                'success': False,
                'error': str(e),
                'file_path': None
            }


# Global chart creator instance
chart_creator = AISChartCreator()

