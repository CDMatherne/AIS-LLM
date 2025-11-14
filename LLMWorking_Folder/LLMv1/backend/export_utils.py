"""
Export utilities for AIS anomaly data
Handles CSV, Excel, and file management
"""
import pandas as pd
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging
import uuid

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


class AISExporter:
    """
    Handles exporting AIS anomaly data to various formats
    """
    
    def __init__(self, output_directory: str = None):
        """
        Initialize exporter
        
        Args:
            output_directory: Base directory for exports (defaults to AISDS_Output in Downloads)
        """
        if output_directory is None:
            output_directory = get_default_output_directory()
        self.output_directory = Path(output_directory)
        self.ensure_output_directories()
    
    def ensure_output_directories(self):
        """Create output directory structure"""
        directories = [
            self.output_directory,
            self.output_directory / "Charts",
            self.output_directory / "Path_Maps",
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured directory exists: {directory}")
    
    def export_anomalies_csv(self, anomalies_df: pd.DataFrame, 
                            filename: str = "AIS_Anomalies_Summary.csv") -> Dict[str, Any]:
        """
        Export anomalies to CSV file matching SFD.py format
        
        Args:
            anomalies_df: DataFrame with anomaly data
            filename: Output filename
        
        Returns:
            Dict with success status and file path
        """
        try:
            if anomalies_df.empty:
                logger.warning("No anomalies to export")
                return {
                    'success': False,
                    'error': 'No anomalies to export',
                    'file_path': None
                }
            
            # Ensure required columns exist
            required_columns = ['MMSI', 'AnomalyType', 'BaseDateTime', 'LAT', 'LON']
            missing_columns = [col for col in required_columns if col not in anomalies_df.columns]
            
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return {
                    'success': False,
                    'error': f'Missing columns: {missing_columns}',
                    'file_path': None
                }
            
            # Sort by timestamp and MMSI
            if 'BaseDateTime' in anomalies_df.columns:
                anomalies_df = anomalies_df.sort_values(['BaseDateTime', 'MMSI'])
            
            # Select and order columns for export (similar to SFD.py)
            export_columns = []
            for col in ['MMSI', 'VesselName', 'VesselType', 'AnomalyType', 
                       'BaseDateTime', 'LAT', 'LON', 'SOG', 'COG', 'Heading',
                       'Description', 'Confidence', 'Severity', 'Details']:
                if col in anomalies_df.columns:
                    export_columns.append(col)
            
            export_df = anomalies_df[export_columns].copy()
            
            # Format datetime column
            if 'BaseDateTime' in export_df.columns:
                export_df['BaseDateTime'] = pd.to_datetime(export_df['BaseDateTime']).dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Generate timestamped filename to prevent overwrites
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            if filename is None:
                base_name = "consolidated_events"
            else:
                base_name = filename.replace('.csv', '') if filename.endswith('.csv') else filename
            timestamped_filename = f"{base_name}_{timestamp}.csv"
            
            # Export to CSV
            file_path = self.output_directory / timestamped_filename
            export_df.to_csv(file_path, index=False)
            
            logger.info(f"Exported {len(export_df)} anomalies to {file_path}")
            
            return {
                'success': True,
                'file_path': str(file_path),
                'record_count': len(export_df),
                'file_size_bytes': file_path.stat().st_size,
                'columns': list(export_df.columns)
            }
            
        except Exception as e:
            logger.error(f"Error exporting CSV: {e}")
            return {
                'success': False,
                'error': str(e),
                'file_path': None
            }
    
    def export_statistics_csv(self, statistics: Dict[str, Any], 
                             filename: str = "Analysis_Statistics.csv") -> Dict[str, Any]:
        """
        Export analysis statistics to CSV
        
        Args:
            statistics: Dictionary with statistical data
            filename: Output filename
        
        Returns:
            Dict with success status and file path
        """
        try:
            # Create statistics dataframe
            stats_data = []
            
            # Basic statistics
            if 'total_anomalies' in statistics:
                stats_data.append({'Metric': 'Total Anomalies', 'Value': statistics['total_anomalies']})
            if 'unique_vessels' in statistics:
                stats_data.append({'Metric': 'Unique Vessels with Anomalies', 'Value': statistics['unique_vessels']})
            if 'date_range' in statistics:
                stats_data.append({'Metric': 'Date Range Start', 'Value': statistics['date_range'].get('start', 'N/A')})
                stats_data.append({'Metric': 'Date Range End', 'Value': statistics['date_range'].get('end', 'N/A')})
            
            # Anomaly types breakdown
            if 'anomaly_types' in statistics:
                stats_data.append({'Metric': '--- Anomaly Types ---', 'Value': ''})
                for anomaly_type, count in statistics['anomaly_types'].items():
                    stats_data.append({'Metric': f'  {anomaly_type}', 'Value': count})
            
            # Severity distribution
            if 'severity_distribution' in statistics:
                stats_data.append({'Metric': '--- Severity Distribution ---', 'Value': ''})
                for severity, count in statistics['severity_distribution'].items():
                    stats_data.append({'Metric': f'  {severity}', 'Value': count})
            
            # Processing info
            if 'processing_backend' in statistics:
                stats_data.append({'Metric': 'Processing Backend', 'Value': statistics['processing_backend']})
            if 'total_records' in statistics:
                stats_data.append({'Metric': 'Total Records Processed', 'Value': statistics['total_records']})
            
            stats_df = pd.DataFrame(stats_data)
            
            # Generate timestamped filename to prevent overwrites
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            if filename is None:
                base_name = "statistics"
            else:
                base_name = filename.replace('.csv', '') if filename.endswith('.csv') else filename
            timestamped_filename = f"{base_name}_{timestamp}.csv"
            
            # Export to CSV
            file_path = self.output_directory / timestamped_filename
            stats_df.to_csv(file_path, index=False)
            
            logger.info(f"Exported statistics to {file_path}")
            
            return {
                'success': True,
                'file_path': str(file_path),
                'metric_count': len(stats_data),
                'file_size_bytes': file_path.stat().st_size
            }
            
        except Exception as e:
            logger.error(f"Error exporting statistics CSV: {e}")
            return {
                'success': False,
                'error': str(e),
                'file_path': None
            }
    
    def export_statistics_excel(self, statistics: Dict[str, Any], anomalies_df: pd.DataFrame,
                               filename: str = None) -> Dict[str, Any]:
        """
        Export comprehensive statistics to Excel with multiple sheets
        
        Args:
            statistics: Dictionary with statistical data
            anomalies_df: DataFrame with anomaly data
            filename: Output filename
        
        Returns:
            Dict with success status and file path
        """
        try:
            # Generate timestamped filename to prevent overwrites
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            if filename is None:
                base_name = "analysis_report"
            else:
                base_name = filename.replace('.xlsx', '').replace('.xls', '') if filename.endswith(('.xlsx', '.xls')) else filename
            timestamped_filename = f"{base_name}_{timestamp}.xlsx"
            
            file_path = self.output_directory / timestamped_filename
            
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                # Sheet 1: Summary Statistics
                summary_data = []
                summary_data.append({'Metric': 'Analysis Summary', 'Value': ''})
                summary_data.append({'Metric': 'Generated', 'Value': datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
                summary_data.append({'Metric': 'Total Anomalies', 'Value': statistics.get('total_anomalies', 0)})
                summary_data.append({'Metric': 'Unique Vessels', 'Value': statistics.get('unique_vessels', 0)})
                
                if 'date_range' in statistics:
                    summary_data.append({'Metric': 'Date Range', 'Value': f"{statistics['date_range'].get('start')} to {statistics['date_range'].get('end')}"})
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Sheet 2: Anomaly Types
                if 'anomaly_types' in statistics and statistics['anomaly_types']:
                    types_df = pd.DataFrame([
                        {'Anomaly Type': k, 'Count': v, 'Percentage': f"{v/statistics['total_anomalies']*100:.1f}%"}
                        for k, v in statistics['anomaly_types'].items()
                    ])
                    types_df = types_df.sort_values('Count', ascending=False)
                    types_df.to_excel(writer, sheet_name='Anomaly Types', index=False)
                
                # Sheet 3: Top Vessels
                if not anomalies_df.empty and 'MMSI' in anomalies_df.columns:
                    vessel_counts = anomalies_df.groupby('MMSI').size().reset_index(name='Anomaly Count')
                    vessel_counts = vessel_counts.sort_values('Anomaly Count', ascending=False).head(20)
                    
                    # Add vessel names if available
                    if 'VesselName' in anomalies_df.columns:
                        vessel_names = anomalies_df.groupby('MMSI')['VesselName'].first()
                        vessel_counts['Vessel Name'] = vessel_counts['MMSI'].map(vessel_names)
                        vessel_counts = vessel_counts[['MMSI', 'Vessel Name', 'Anomaly Count']]
                    
                    vessel_counts.to_excel(writer, sheet_name='Top Vessels', index=False)
                
                # Sheet 4: Temporal Distribution (if date info available)
                if not anomalies_df.empty and 'BaseDateTime' in anomalies_df.columns:
                    anomalies_df['Date'] = pd.to_datetime(anomalies_df['BaseDateTime']).dt.date
                    date_counts = anomalies_df.groupby('Date').size().reset_index(name='Anomaly Count')
                    date_counts = date_counts.sort_values('Date')
                    date_counts.to_excel(writer, sheet_name='By Date', index=False)
            
            logger.info(f"Exported Excel statistics to {file_path}")
            
            return {
                'success': True,
                'file_path': str(file_path),
                'file_size_bytes': file_path.stat().st_size,
                'sheets': ['Summary', 'Anomaly Types', 'Top Vessels', 'By Date']
            }
            
        except Exception as e:
            logger.error(f"Error exporting Excel: {e}")
            return {
                'success': False,
                'error': str(e),
                'file_path': None
            }
    
    def get_export_file_info(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Get information about an exported file
        
        Args:
            filename: Name of the file
        
        Returns:
            Dict with file information or None if not found
        """
        file_path = self.output_directory / filename
        
        if not file_path.exists():
            return None
        
        stat = file_path.stat()
        
        return {
            'filename': filename,
            'path': str(file_path),
            'size_bytes': stat.st_size,
            'size_mb': stat.st_size / (1024 * 1024),
            'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
            'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
        }
    
    def list_exports(self) -> List[Dict[str, Any]]:
        """
        List all export files in the output directory
        
        Returns:
            List of file information dictionaries
        """
        exports = []
        
        for file_path in self.output_directory.glob('*.*'):
            if file_path.is_file():
                exports.append(self.get_export_file_info(file_path.name))
        
        return [e for e in exports if e is not None]
    
    def cleanup_old_exports(self, max_age_days: int = 30) -> int:
        """
        Remove export files older than specified days
        
        Args:
            max_age_days: Maximum age in days
        
        Returns:
            Number of files deleted
        """
        deleted_count = 0
        current_time = datetime.now().timestamp()
        max_age_seconds = max_age_days * 24 * 3600
        
        for file_path in self.output_directory.rglob('*.*'):
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime
                if file_age > max_age_seconds:
                    try:
                        file_path.unlink()
                        deleted_count += 1
                        logger.info(f"Deleted old export: {file_path}")
                    except Exception as e:
                        logger.error(f"Failed to delete {file_path}: {e}")
        
        return deleted_count


# Global exporter instance (singleton pattern) - can be imported by other modules
exporter = AISExporter()
