#!/usr/bin/env python3
"""
Advanced Analysis Module for SFD Project

This module provides advanced analysis capabilities on cached AIS data
from previous analysis runs.
"""

import os
import sys
import glob
import logging
import configparser
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import json
import re
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import traceback
import platform
import subprocess

# Check for tkcalendar
try:
    from tkcalendar import DateEntry
    TKCALENDAR_AVAILABLE = True
except ImportError:
    TKCALENDAR_AVAILABLE = False


# Try to import optional dependencies
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from openpyxl import Workbook
    from openpyxl.chart import BarChart, LineChart, Reference
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False

try:
    import folium
    from folium.plugins import MarkerCluster, HeatMap
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Set up logging
logger = logging.getLogger("Advanced_Analysis")


# ============================================================================
# HELPER FUNCTIONS AND VALIDATION
# ============================================================================

def get_config_path(config_path='config.ini'):
    """Get the absolute path to config.ini, resolving relative to script directory."""
    if os.path.isabs(config_path):
        return config_path
    
    # Get the script directory
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        script_dir = os.path.dirname(sys.executable)
    else:
        # Running as script
        script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try script directory first
    abs_config_path = os.path.join(script_dir, config_path)
    if os.path.exists(abs_config_path):
        return abs_config_path
    
    # Try current working directory as fallback
    if os.path.exists(config_path):
        return os.path.abspath(config_path)
    
    # Return the script directory path even if it doesn't exist (for error messages)
    return abs_config_path


def validate_config(config_path='config.ini'):
    """Validate that config.ini exists and contains required sections."""
    # Resolve config path relative to script directory
    config_path = get_config_path(config_path)
    
    if not os.path.exists(config_path):
        return False, f"Config file not found: {config_path}"
    
    try:
        config = configparser.ConfigParser()
        config.read(config_path)
        
        required_sections = ['DEFAULT']
        missing_sections = [s for s in required_sections if s not in config]
        
        if missing_sections:
            return False, f"Missing required config sections: {', '.join(missing_sections)}"
        
        output_dir = config.get('DEFAULT', 'output_directory', fallback=None)
        if not output_dir:
            output_dir = config.get('Paths', 'output_directory', fallback=None)
        
        if not output_dir:
            return False, "Output directory not specified in config.ini"
        
        return True, None
        
    except Exception as e:
        return False, f"Error reading config file: {str(e)}"


def validate_data_availability(run_info):
    """Validate that required data is available for analysis."""
    warnings = []
    
    output_dir = run_info.get('output_directory')
    if not output_dir:
        return False, "Output directory not specified", warnings
    
    if not os.path.exists(output_dir):
        warnings.append(f"Output directory does not exist: {output_dir}")
        try:
            os.makedirs(output_dir, exist_ok=True)
            warnings.append(f"Created output directory: {output_dir}")
        except Exception as e:
            return False, f"Cannot create output directory: {e}", warnings
    
    cache_dir = get_cache_dir()
    if not os.path.exists(cache_dir):
        warnings.append(f"Cache directory does not exist: {cache_dir}")
    
    cache_files = find_cache_files_for_date_range(
        run_info.get('start_date', ''),
        run_info.get('end_date', ''),
        run_info.get('ship_types', []),
        cache_dir
    )
    
    if not cache_files:
        warnings.append("No cached data files found for the specified date range")
    
    anomaly_summary_path = os.path.join(output_dir, "AIS_Anomalies_Summary.csv")
    if not os.path.exists(anomaly_summary_path):
        warnings.append("Anomaly summary file not found. Some features may be limited.")
    
    return True, None, warnings


def get_memory_usage():
    """Get current memory usage information."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return {
            'rss_mb': mem_info.rss / (1024 * 1024),
            'vms_mb': mem_info.vms / (1024 * 1024),
            'percent': process.memory_percent()
        }
    except ImportError:
        return {'rss_mb': 'N/A', 'vms_mb': 'N/A', 'percent': 'N/A'}
    except Exception:
        return {'rss_mb': 'N/A', 'vms_mb': 'N/A', 'percent': 'N/A'}


def log_memory_usage(operation_name=""):
    """Log current memory usage."""
    mem_usage = get_memory_usage()
    logger.info(f"Memory usage {operation_name}: RSS={mem_usage.get('rss_mb', 'N/A')} MB, "
               f"Percent={mem_usage.get('percent', 'N/A')}%")


def cleanup_large_dataframe(df, keep_columns=None):
    """Clean up a DataFrame to reduce memory usage."""
    if df is None or df.empty:
        return df
    
    original_size = df.memory_usage(deep=True).sum() / (1024 * 1024)
    
    for col in df.columns:
        if df[col].dtype == 'object':
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio < 0.5:
                try:
                    df[col] = df[col].astype('category')
                except:
                    pass
    
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    if keep_columns:
        df = df[keep_columns]
    
    new_size = df.memory_usage(deep=True).sum() / (1024 * 1024)
    logger.info(f"DataFrame memory reduced from {original_size:.2f} MB to {new_size:.2f} MB")
    
    return df


def format_file_size(size_bytes):
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def get_all_vessel_types():
    """Get list of all possible vessel types (20-99) as defined in GUI."""
    return list(range(20, 100))


def get_vessel_type_name(vessel_type):
    """Get the human-readable name for a vessel type number."""
    vessel_type_names = {
        20: 'Wing in ground (WIG), all ships of this type',
        21: 'Wing in ground (WIG), Hazardous category A',
        22: 'Wing in ground (WIG), Hazardous category B',
        23: 'Wing in ground (WIG), Hazardous category C',
        24: 'Wing in ground (WIG), Hazardous category D',
        30: 'Fishing',
        31: 'Towing',
        32: 'Towing: length exceeds 200m or breadth exceeds 25m',
        33: 'Dredging or underwater ops',
        34: 'Diving ops',
        35: 'Military ops',
        36: 'Sailing',
        37: 'Pleasure Craft',
        38: 'Reserved',
        39: 'Reserved',
        40: 'High speed craft (HSC), all ships of this type',
        41: 'High speed craft (HSC), Hazardous category A',
        42: 'High speed craft (HSC), Hazardous category B',
        43: 'High speed craft (HSC), Hazardous category C',
        44: 'High speed craft (HSC), Hazardous category D',
        45: 'High speed craft (HSC), Reserved',
        46: 'High speed craft (HSC), Reserved',
        47: 'High speed craft (HSC), Reserved',
        48: 'High speed craft (HSC), Reserved',
        49: 'High speed craft (HSC), No additional information',
        50: 'Pilot Vessel',
        51: 'Search and Rescue vessel',
        52: 'Tug',
        53: 'Port Tender',
        54: 'Anti-pollution equipment',
        55: 'Law Enforcement',
        56: 'Spare - Local Vessel',
        57: 'Spare - Local Vessel',
        58: 'Medical Transport',
        59: 'Noncombatant ship (RR Resolution No. 18)',
        60: 'Passenger, all ships of this type',
        61: 'Passenger, Hazardous category A',
        62: 'Passenger, Hazardous category B',
        63: 'Passenger, Hazardous category C',
        64: 'Passenger, Hazardous category D',
        69: 'Passenger, No additional information',
        70: 'Cargo, all ships of this type',
        71: 'Cargo, Hazardous category A',
        72: 'Cargo, Hazardous category B',
        73: 'Cargo, Hazardous category C',
        74: 'Cargo, Hazardous category D',
        79: 'Cargo, No additional information',
        80: 'Tanker, all ships of this type',
        81: 'Tanker, Hazardous category A',
        82: 'Tanker, Hazardous category B',
        83: 'Tanker, Hazardous category C',
        84: 'Tanker, Hazardous category D',
        89: 'Tanker, No additional information',
        90: 'Other, all ships of this type',
        91: 'Other, Hazardous category A',
        92: 'Other, Hazardous category B',
        93: 'Other, Hazardous category C',
        94: 'Other, Hazardous category D',
        99: 'Other, No additional information'
    }
    return vessel_type_names.get(int(vessel_type), f'Vessel Type {int(vessel_type)}')


def get_all_anomaly_types():
    """Get list of all possible anomaly types as defined in GUI."""
    return [
        "AIS Beacon Off",
        "AIS Beacon On",
        "Excessive Travel Distance (Fast)",
        "Excessive Travel Distance (Slow)",
        "Course over Ground-Heading Inconsistency",
        "Loitering",
        "Rendezvous",
        "Identity Spoofing",
        "Zone Violations"
    ]


def map_anomaly_type_gui_to_data(gui_name):
    """Map GUI anomaly type name to data anomaly type name."""
    mapping = {
        "AIS Beacon Off": "AIS_Beacon_Off",
        "AIS Beacon On": "AIS_Beacon_On",
        "Excessive Travel Distance (Fast)": "Speed",
        "Excessive Travel Distance (Slow)": "Speed",
        "Course over Ground-Heading Inconsistency": "Course",
        "Loitering": "Loitering",
        "Rendezvous": "Rendezvous",
        "Identity Spoofing": "Identity_Spoofing",
        "Zone Violations": "Zone_Violation"
    }
    return mapping.get(gui_name, gui_name)


def map_anomaly_type_data_to_gui(data_name):
    """Map data anomaly type name to GUI anomaly type name."""
    mapping = {
        "AIS_Beacon_Off": "AIS Beacon Off",
        "AIS_Beacon_On": "AIS Beacon On",
        "Speed": "Excessive Travel Distance (Fast)",
        "Course": "Course over Ground-Heading Inconsistency",
        "Loitering": "Loitering",
        "Rendezvous": "Rendezvous",
        "Identity_Spoofing": "Identity Spoofing",
        "Zone_Violation": "Zone Violations"
    }
    return mapping.get(data_name, data_name)


def get_available_vessel_types(df):
    """Get list of available vessel types from data."""
    if df.empty or 'VesselType' not in df.columns:
        return []
    return sorted(df['VesselType'].dropna().unique().tolist())


def get_available_anomaly_types(anomaly_df):
    """Get list of available anomaly types from data."""
    if anomaly_df.empty or 'AnomalyType' not in anomaly_df.columns:
        return []
    return sorted(anomaly_df['AnomalyType'].dropna().unique().tolist())


def validate_mmsi(mmsi_value):
    """Validate MMSI value."""
    try:
        mmsi = int(mmsi_value)
        if mmsi < 100000000 or mmsi > 999999999:
            return False, None, "MMSI must be a 9-digit number"
        return True, mmsi, None
    except (ValueError, TypeError):
        return False, None, "MMSI must be a valid number"


def validate_date_range(start_date, end_date):
    """Validate date range."""
    try:
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        if start > end:
            return False, "Start date must be before end date"
        
        if (end - start).days > 365:
            return False, "Date range cannot exceed 365 days"
        
        return True, None
    except ValueError as e:
        return False, f"Invalid date format: {e}"


def check_dependencies():
    """Check if all required dependencies are available."""
    dependencies = {
        'pandas': False, 'numpy': False, 'folium': False, 'plotly': False,
        'matplotlib': False, 'sklearn': False, 'openpyxl': False,
    }
    
    for dep in dependencies.keys():
        try:
            __import__(dep)
            dependencies[dep] = True
        except ImportError:
            dependencies[dep] = False
    
    return dependencies


def get_dependency_warnings():
    """Get warnings about missing optional dependencies."""
    warnings = []
    deps = check_dependencies()
    
    optional_deps = {
        'plotly': 'Interactive visualizations will use basic HTML tables',
        'folium': 'Map generation will not be available',
        'sklearn': 'Vessel clustering will not be available',
        'openpyxl': 'Excel export will use CSV format instead',
    }
    
    for dep, message in optional_deps.items():
        if not deps.get(dep, False):
            warnings.append(f"{dep}: {message}")
    
    return warnings


# ============================================================================
# DATA ACCESS LAYER
# ============================================================================

def get_cache_dir():
    """Get the directory to use for caching data."""
    home_dir = os.path.expanduser("~")
    cache_dir = os.path.join(home_dir, ".ais_data_cache")
    return cache_dir


def get_last_run_info(config_path='config.ini'):
    """Extract information about the last analysis run from config.ini."""
    # Resolve config path relative to script directory
    config_path = get_config_path(config_path)
    
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        return None
    
    config = configparser.ConfigParser()
    config.read(config_path)
    
    def get_config_value(section, key, fallback=None, value_type='str'):
        try:
            if value_type == 'float':
                return config.getfloat(section, key, fallback=fallback)
            elif value_type == 'int':
                return config.getint(section, key, fallback=fallback)
            elif value_type == 'boolean':
                return config.getboolean(section, key, fallback=fallback)
            else:
                return config.get(section, key, fallback=fallback)
        except (configparser.NoSectionError, configparser.NoOptionError):
            return fallback
    
    output_dir = get_config_value('DEFAULT', 'output_directory', 
                                  fallback=get_config_value('Paths', 'output_directory', 
                                                          fallback='C:/AIS_Data/Output'))
    
    start_date = get_config_value('DEFAULT', 'start_date', fallback='2024-10-15')
    end_date = get_config_value('DEFAULT', 'end_date', fallback='2024-10-17')
    
    data_dir = get_config_value('DEFAULT', 'data_directory',
                               fallback=get_config_value('Paths', 'data_directory', fallback=''))
    
    ship_types_str = get_config_value('SHIP_FILTERS', 'selected_ship_types', fallback='')
    if ship_types_str:
        try:
            ship_types = [int(t.strip()) for t in ship_types_str.split(',') if t.strip()]
        except ValueError:
            ship_types = []
    else:
        ship_types = []
    
    anomaly_types = {
        'ais_beacon_off': get_config_value('ANOMALY_TYPES', 'ais_beacon_off', fallback=False, value_type='boolean'),
        'ais_beacon_on': get_config_value('ANOMALY_TYPES', 'ais_beacon_on', fallback=False, value_type='boolean'),
        'excessive_travel_distance_fast': get_config_value('ANOMALY_TYPES', 'excessive_travel_distance_fast', fallback=False, value_type='boolean'),
        'excessive_travel_distance_slow': get_config_value('ANOMALY_TYPES', 'excessive_travel_distance_slow', fallback=False, value_type='boolean'),
        'cog-heading_inconsistency': get_config_value('ANOMALY_TYPES', 'cog-heading_inconsistency', fallback=False, value_type='boolean'),
        'loitering': get_config_value('ANOMALY_TYPES', 'loitering', fallback=False, value_type='boolean'),
        'rendezvous': get_config_value('ANOMALY_TYPES', 'rendezvous', fallback=False, value_type='boolean'),
        'identity_spoofing': get_config_value('ANOMALY_TYPES', 'identity_spoofing', fallback=False, value_type='boolean'),
        'zone_violations': get_config_value('ANOMALY_TYPES', 'zone_violations', fallback=False, value_type='boolean'),
    }
    
    return {
        'output_directory': output_dir,
        'start_date': start_date,
        'end_date': end_date,
        'ship_types': ship_types,
        'anomaly_types': anomaly_types,
        'data_directory': data_dir
    }


def find_cache_files_for_date_range(start_date, end_date, ship_types, cache_dir=None):
    """Find all cache files that match the date range and ship types."""
    if cache_dir is None:
        cache_dir = get_cache_dir()
    
    if not os.path.exists(cache_dir):
        logger.warning(f"Cache directory does not exist: {cache_dir}")
        return []
    
    cache_files = glob.glob(os.path.join(cache_dir, "*.parquet"))
    
    try:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d').date()
        end_dt = datetime.strptime(end_date, '%Y-%m-%d').date()
    except ValueError as e:
        logger.error(f"Invalid date format: {e}")
        return []
    
    matching_files = []
    
    for cache_file in cache_files:
        try:
            df_sample = pd.read_parquet(cache_file, nrows=1)
            
            if 'MMSI' not in df_sample.columns or 'BaseDateTime' not in df_sample.columns:
                continue
            
            df_check = pd.read_parquet(cache_file)
            if 'BaseDateTime' in df_check.columns:
                df_check['BaseDateTime'] = pd.to_datetime(df_check['BaseDateTime'])
                file_dates = df_check['BaseDateTime'].dt.date.unique()
                
                if any(start_dt <= d <= end_dt for d in file_dates):
                    matching_files.append(cache_file)
            
        except Exception as e:
            logger.debug(f"Error checking cache file {cache_file}: {e}")
            continue
    
    logger.info(f"Found {len(matching_files)} matching cache files for date range {start_date} to {end_date}")
    return matching_files


def load_cached_data_for_date_range(start_date, end_date, ship_types, cache_dir=None):
    """Load all cached parquet files matching the date range and ship types."""
    cache_files = find_cache_files_for_date_range(start_date, end_date, ship_types, cache_dir)
    
    if not cache_files:
        logger.warning("No matching cache files found")
        return pd.DataFrame()
    
    dataframes = []
    for cache_file in cache_files:
        try:
            df = pd.read_parquet(cache_file)
            
            if ship_types and 'VesselType' in df.columns:
                df = df[df['VesselType'].isin(ship_types)]
            
            if 'BaseDateTime' in df.columns:
                df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'])
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                end_dt = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)
                df = df[(df['BaseDateTime'] >= start_dt) & (df['BaseDateTime'] < end_dt)]
            
            if not df.empty:
                dataframes.append(df)
                logger.info(f"Loaded {len(df)} records from {os.path.basename(cache_file)}")
        except Exception as e:
            logger.error(f"Error loading cache file {cache_file}: {e}")
            continue
    
    if not dataframes:
        logger.warning("No data loaded from cache files")
        return pd.DataFrame()
    
    combined_df = pd.concat(dataframes, ignore_index=True)
    logger.info(f"Total records loaded: {len(combined_df)}")
    return combined_df


def load_anomaly_summary(output_dir):
    """Load the AIS_Anomalies_Summary.csv file if it exists."""
    summary_path = os.path.join(output_dir, "AIS_Anomalies_Summary.csv")
    
    if not os.path.exists(summary_path):
        logger.warning(f"Anomaly summary file not found: {summary_path}")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(summary_path)
        logger.info(f"Loaded {len(df)} anomalies from summary file")
        return df
    except Exception as e:
        logger.error(f"Error loading anomaly summary: {e}")
        return pd.DataFrame()


# ============================================================================
# ADVANCED ANALYSIS CLASS
# ============================================================================

class AdvancedAnalysis:
    """Main class for Advanced Analysis functionality."""
    
    def __init__(self, parent_window, output_directory=None, config_path='config.ini'):
        """Initialize Advanced Analysis with enhanced validation."""
        self.parent_window = parent_window
        # Resolve config path relative to script directory
        self.config_path = get_config_path(config_path)
        
        is_valid, error_msg = validate_config(self.config_path)
        if not is_valid:
            raise ValueError(f"Config validation failed: {error_msg}")
        
        self.run_info = get_last_run_info(self.config_path)
        
        if self.run_info is None:
            raise ValueError("Could not load last run information from config.ini")
        
        self.output_directory = output_directory or self.run_info['output_directory']
        os.makedirs(self.output_directory, exist_ok=True)
        
        is_valid, error_msg, warnings = validate_data_availability(self.run_info)
        if not is_valid:
            logger.warning(f"Data validation warning: {error_msg}")
        
        for warning in warnings:
            logger.warning(warning)
        
        dep_warnings = get_dependency_warnings()
        for warning in dep_warnings:
            logger.warning(f"Dependency: {warning}")
        
        self._cached_data = None
        self._anomaly_data = None
        
        logger.info(f"Advanced Analysis initialized with output directory: {self.output_directory}")
        log_memory_usage("after initialization")
    
    def load_cached_data(self, force_reload=False):
        """Load cached data from the last run with enhanced error handling."""
        if self._cached_data is not None and not force_reload:
            return self._cached_data
        
        logger.info("Loading cached data...")
        log_memory_usage("before loading cached data")
        
        try:
            self._cached_data = load_cached_data_for_date_range(
                self.run_info['start_date'],
                self.run_info['end_date'],
                self.run_info['ship_types']
            )
            
            if not self._cached_data.empty:
                self._cached_data = cleanup_large_dataframe(self._cached_data)
                log_memory_usage("after loading cached data")
                logger.info(f"Loaded {len(self._cached_data)} records")
            else:
                logger.warning("No cached data loaded")
            
            return self._cached_data
            
        except MemoryError:
            logger.error("Insufficient memory to load cached data")
            logger.info("Try processing a smaller date range or enable data filtering")
            raise
        except Exception as e:
            logger.error(f"Error loading cached data: {e}")
            raise
    
    def load_anomaly_data(self, force_reload=False):
        """Load anomaly summary data."""
        if self._anomaly_data is not None and not force_reload:
            return self._anomaly_data
        
        logger.info("Loading anomaly data...")
        self._anomaly_data = load_anomaly_summary(self.output_directory)
        return self._anomaly_data
    
    # ========================================================================
    # TAB 1: ADDITIONAL OUTPUTS
    # ========================================================================
    
    def export_full_dataset(self, output_path=None, chunk_size=100000):
        """Export the complete analysis dataset to CSV format."""
        try:
            logger.info("Exporting full dataset to CSV...")
            log_memory_usage("before export")
            
            df = self.load_cached_data()
            
            if df.empty:
                logger.error("No data available to export")
                return None
            
            if output_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = os.path.join(self.output_directory, f"Full_Dataset_{timestamp}.csv")
            
            if len(df) > chunk_size:
                logger.info(f"Large dataset detected ({len(df)} records). Writing in chunks...")
                # Write header first
                df.head(0).to_csv(output_path, index=False, mode='w')
                # Write data in chunks
                for i in range(0, len(df), chunk_size):
                    chunk = df.iloc[i:i+chunk_size]
                    chunk.to_csv(output_path, index=False, mode='a', header=False)
                    logger.info(f"Written {min(i+chunk_size, len(df))}/{len(df)} records...")
            else:
                df.to_csv(output_path, index=False)
            
            file_size = os.path.getsize(output_path)
            logger.info(f"Full dataset exported to: {output_path} ({format_file_size(file_size)})")
            log_memory_usage("after export")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error exporting full dataset: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def generate_summary_report(self, output_path=None):
        """Create a summary report with key findings and statistics."""
        try:
            logger.info("Generating summary report...")
            
            df = self.load_cached_data()
            anomaly_df = self.load_anomaly_data()
            
            if df.empty:
                logger.error("No data available for summary report")
                return None
            
            if output_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = os.path.join(self.output_directory, f"Summary_Report_{timestamp}.html")
            
            report_lines = []
            report_lines.append("<html><head><title>AIS Analysis Summary Report</title>")
            report_lines.append("<style>body { font-family: Arial, sans-serif; margin: 20px; }")
            report_lines.append("h1 { color: #003366; } h2 { color: #0066cc; }")
            report_lines.append("table { border-collapse: collapse; width: 100%; margin: 20px 0; }")
            report_lines.append("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
            report_lines.append("th { background-color: #003366; color: white; }</style></head><body>")
            
            report_lines.append("<h1>AIS Shipping Fraud Detection - Summary Report</h1>")
            report_lines.append(f"<p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
            report_lines.append(f"<p><strong>Analysis Period:</strong> {self.run_info['start_date']} to {self.run_info['end_date']}</p>")
            
            report_lines.append("<h2>Data Overview</h2>")
            report_lines.append("<table>")
            report_lines.append(f"<tr><th>Metric</th><th>Value</th></tr>")
            report_lines.append(f"<tr><td>Total Records</td><td>{len(df):,}</td></tr>")
            if 'MMSI' in df.columns:
                report_lines.append(f"<tr><td>Unique Vessels (MMSI)</td><td>{df['MMSI'].nunique():,}</td></tr>")
            else:
                report_lines.append(f"<tr><td>Unique Vessels (MMSI)</td><td>N/A</td></tr>")
            
            if 'VesselType' in df.columns:
                report_lines.append(f"<tr><td>Vessel Types Analyzed</td><td>{df['VesselType'].nunique()}</td></tr>")
            
            if 'VesselName' in df.columns:
                named_vessels = df[df['VesselName'].notna()]['MMSI'].nunique() if 'MMSI' in df.columns else 0
                report_lines.append(f"<tr><td>Vessels with Names</td><td>{named_vessels:,}</td></tr>")
            
            report_lines.append("</table>")
            
            if not anomaly_df.empty:
                report_lines.append("<h2>Anomaly Summary</h2>")
                report_lines.append("<table>")
                report_lines.append(f"<tr><th>Metric</th><th>Value</th></tr>")
                report_lines.append(f"<tr><td>Total Anomalies Detected</td><td>{len(anomaly_df):,}</td></tr>")
                
                if 'AnomalyType' in anomaly_df.columns:
                    anomaly_counts = anomaly_df['AnomalyType'].value_counts()
                    report_lines.append("<tr><td colspan='2'><strong>Anomalies by Type:</strong></td></tr>")
                    for anomaly_type, count in anomaly_counts.items():
                        report_lines.append(f"<tr><td>&nbsp;&nbsp;{anomaly_type}</td><td>{count:,}</td></tr>")
                
                if 'MMSI' in anomaly_df.columns:
                    vessels_with_anomalies = anomaly_df['MMSI'].nunique()
                    report_lines.append(f"<tr><td>Vessels with Anomalies</td><td>{vessels_with_anomalies:,}</td></tr>")
                
                report_lines.append("</table>")
            
            if 'VesselType' in df.columns:
                report_lines.append("<h2>Vessel Type Distribution</h2>")
                vessel_type_counts = df['VesselType'].value_counts().head(10)
                report_lines.append("<table>")
                report_lines.append("<tr><th>Vessel Type</th><th>Record Count</th></tr>")
                for vessel_type, count in vessel_type_counts.items():
                    report_lines.append(f"<tr><td>{vessel_type}</td><td>{count:,}</td></tr>")
                report_lines.append("</table>")
            
            if not anomaly_df.empty and 'MMSI' in anomaly_df.columns:
                report_lines.append("<h2>Top 10 Vessels by Anomaly Count</h2>")
                top_vessels = anomaly_df['MMSI'].value_counts().head(10)
                report_lines.append("<table>")
                report_lines.append("<tr><th>MMSI</th><th>Anomaly Count</th></tr>")
                for mmsi, count in top_vessels.items():
                    report_lines.append(f"<tr><td>{mmsi}</td><td>{count:,}</td></tr>")
                report_lines.append("</table>")
            
            report_lines.append("<h2>Key Findings</h2>")
            report_lines.append("<ul>")
            
            if not anomaly_df.empty:
                report_lines.append(f"<li>Detected {len(anomaly_df):,} anomalies across {anomaly_df['MMSI'].nunique() if 'MMSI' in anomaly_df.columns else 'N/A'} vessels</li>")
            
            if 'VesselType' in df.columns:
                most_common_type = df['VesselType'].mode()[0] if not df['VesselType'].mode().empty else 'N/A'
                report_lines.append(f"<li>Most common vessel type: {most_common_type}</li>")
            
            report_lines.append(f"<li>Analysis covered {len(df):,} total AIS records</li>")
            report_lines.append("</ul>")
            
            report_lines.append("</body></html>")
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_lines))
            
            logger.info(f"Summary report generated: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating summary report: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def export_vessel_statistics(self, output_path=None):
        """Export vessel-specific statistics to Excel format."""
        try:
            logger.info("Exporting vessel statistics...")
            
            df = self.load_cached_data()
            anomaly_df = self.load_anomaly_data()
            
            if df.empty:
                logger.error("No data available for vessel statistics")
                return None
            
            if output_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = os.path.join(self.output_directory, f"Vessel_Statistics_{timestamp}.xlsx")
            
            if 'MMSI' not in df.columns:
                logger.error("MMSI column not found in dataset")
                return None
            
            vessel_stats = []
            
            unique_mmsis = df['MMSI'].unique()
            logger.info(f"Processing statistics for {len(unique_mmsis)} vessels...")
            
            for idx, mmsi in enumerate(unique_mmsis):
                if (idx + 1) % 100 == 0:
                    logger.info(f"Processing vessel {idx + 1}/{len(unique_mmsis)}...")
                
                vessel_data = df[df['MMSI'] == mmsi]
                vessel_anomalies = anomaly_df[anomaly_df['MMSI'] == mmsi] if not anomaly_df.empty and 'MMSI' in anomaly_df.columns else pd.DataFrame()
                
                stats = {
                    'MMSI': mmsi,
                    'Total_Records': len(vessel_data),
                    'Anomaly_Count': len(vessel_anomalies),
                }
                
                if 'VesselName' in vessel_data.columns:
                    names = vessel_data['VesselName'].dropna().unique()
                    stats['VesselName'] = names[0] if len(names) > 0 else 'Unknown'
                else:
                    stats['VesselName'] = 'Unknown'
                
                if 'VesselType' in vessel_data.columns:
                    types = vessel_data['VesselType'].unique()
                    stats['VesselType'] = types[0] if len(types) > 0 else 'Unknown'
                else:
                    stats['VesselType'] = 'Unknown'
                
                if 'BaseDateTime' in vessel_data.columns:
                    vessel_data['BaseDateTime'] = pd.to_datetime(vessel_data['BaseDateTime'])
                    stats['First_Seen'] = vessel_data['BaseDateTime'].min()
                    stats['Last_Seen'] = vessel_data['BaseDateTime'].max()
                    stats['Days_Active'] = (stats['Last_Seen'] - stats['First_Seen']).days + 1
                
                if 'SOG' in vessel_data.columns:
                    speeds = vessel_data['SOG'].dropna()
                    if len(speeds) > 0:
                        stats['Avg_Speed'] = speeds.mean()
                        stats['Max_Speed'] = speeds.max()
                        stats['Min_Speed'] = speeds.min()
                
                if 'LAT' in vessel_data.columns and 'LON' in vessel_data.columns:
                    stats['Min_Latitude'] = vessel_data['LAT'].min()
                    stats['Max_Latitude'] = vessel_data['LAT'].max()
                    stats['Min_Longitude'] = vessel_data['LON'].min()
                    stats['Max_Longitude'] = vessel_data['LON'].max()
                
                if not vessel_anomalies.empty and 'AnomalyType' in vessel_anomalies.columns:
                    anomaly_types = vessel_anomalies['AnomalyType'].value_counts().to_dict()
                    for anomaly_type, count in anomaly_types.items():
                        stats[f'Anomaly_{anomaly_type}'] = count
                
                vessel_stats.append(stats)
            
            if not vessel_stats:
                logger.error("No vessel statistics generated")
                return None
            
            stats_df = pd.DataFrame(vessel_stats)
            
            try:
                if OPENPYXL_AVAILABLE:
                    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                        stats_df.to_excel(writer, sheet_name='Vessel Statistics', index=False)
                        
                        summary_data = {
                            'Metric': ['Total Vessels', 'Vessels with Anomalies', 'Total Anomalies', 'Total Records'],
                            'Value': [
                                len(stats_df),
                                len(stats_df[stats_df['Anomaly_Count'] > 0]) if 'Anomaly_Count' in stats_df.columns else 0,
                                stats_df['Anomaly_Count'].sum() if 'Anomaly_Count' in stats_df.columns else 0,
                                stats_df['Total_Records'].sum() if 'Total_Records' in stats_df.columns else 0
                            ]
                        }
                        summary_df = pd.DataFrame(summary_data)
                        summary_df.to_excel(writer, sheet_name='Summary', index=False)
                    
                    logger.info(f"Vessel statistics exported to: {output_path}")
                    return output_path
                else:
                    raise ImportError("openpyxl not available")
            except (ImportError, Exception) as e:
                csv_path = output_path.replace('.xlsx', '.csv')
                stats_df.to_csv(csv_path, index=False)
                logger.warning(f"Excel export failed ({str(e)}). Exported to CSV instead: {csv_path}")
                return csv_path
                
        except Exception as e:
            logger.error(f"Error exporting vessel statistics: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def generate_anomaly_timeline(self, output_path=None):
        """Create a timeline visualization of anomalies."""
        try:
            logger.info("Generating anomaly timeline...")
            
            anomaly_df = self.load_anomaly_data()
            
            if anomaly_df.empty:
                logger.error("No anomaly data available for timeline")
                return None
            
            if output_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = os.path.join(self.output_directory, f"Anomaly_Timeline_{timestamp}.html")
            
            if 'Date' in anomaly_df.columns:
                anomaly_df['DateTime'] = pd.to_datetime(anomaly_df['Date'])
            elif 'BaseDateTime' in anomaly_df.columns:
                anomaly_df['DateTime'] = pd.to_datetime(anomaly_df['BaseDateTime'])
            else:
                logger.error("No date column found in anomaly data")
                return None
            
            if PLOTLY_AVAILABLE:
                fig = go.Figure()
                
                if 'AnomalyType' in anomaly_df.columns:
                    for anomaly_type in anomaly_df['AnomalyType'].unique():
                        type_data = anomaly_df[anomaly_df['AnomalyType'] == anomaly_type]
                        timeline_data = type_data.groupby(type_data['DateTime'].dt.date).size()
                        
                        fig.add_trace(go.Scatter(
                            x=timeline_data.index,
                            y=timeline_data.values,
                            mode='lines+markers',
                            name=anomaly_type,
                            line=dict(width=2)
                        ))
                else:
                    timeline_data = anomaly_df.groupby(anomaly_df['DateTime'].dt.date).size()
                    fig.add_trace(go.Scatter(
                        x=timeline_data.index,
                        y=timeline_data.values,
                        mode='lines+markers',
                        name='All Anomalies',
                        line=dict(width=2)
                    ))
                
                fig.update_layout(
                    title='Anomaly Timeline',
                    xaxis_title='Date',
                    yaxis_title='Number of Anomalies',
                    hovermode='x unified',
                    template='plotly_white'
                )
                
                fig.write_html(output_path)
                logger.info(f"Anomaly timeline generated: {output_path}")
                return output_path
            else:
                logger.warning("Plotly not available. Creating simple timeline table.")
                timeline_data = anomaly_df.groupby(anomaly_df['DateTime'].dt.date).size().reset_index()
                timeline_data.columns = ['Date', 'Anomaly_Count']
                
                html_content = ["<html><head><title>Anomaly Timeline</title></head><body>"]
                html_content.append("<h1>Anomaly Timeline</h1>")
                html_content.append("<table border='1'><tr><th>Date</th><th>Anomaly Count</th></tr>")
                
                for _, row in timeline_data.iterrows():
                    html_content.append(f"<tr><td>{row['Date']}</td><td>{row['Anomaly_Count']}</td></tr>")
                
                html_content.append("</table></body></html>")
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(html_content))
                
                logger.info(f"Anomaly timeline (table) generated: {output_path}")
                return output_path
                
        except Exception as e:
            logger.error(f"Error generating anomaly timeline: {e}")
            logger.error(traceback.format_exc())
            return None


    # ========================================================================
    # TAB 2: FURTHER ANALYSIS  
    # ========================================================================
    
    def correlation_analysis(self, vessel_types, anomaly_types, output_path=None):
        """Perform correlation analysis between vessel types and anomaly types."""
        try:
            logger.info(f"Performing correlation analysis: Vessel Types={vessel_types}, Anomaly Types={anomaly_types}")
            
            df = self.load_cached_data()
            anomaly_df = self.load_anomaly_data()
            
            if df.empty or anomaly_df.empty:
                logger.error("Insufficient data for correlation analysis")
                return None
            
            if vessel_types and 'VesselType' in df.columns:
                # Convert vessel types to int for proper comparison, and convert DataFrame column to int
                vessel_types_int = [int(vt) for vt in vessel_types]
                df['VesselType'] = pd.to_numeric(df['VesselType'], errors='coerce').astype('Int64')
                df_filtered = df[df['VesselType'].isin(vessel_types_int)]
                vessel_mmsis = df_filtered['MMSI'].unique()
                anomaly_df_filtered = anomaly_df[anomaly_df['MMSI'].isin(vessel_mmsis)]
            else:
                anomaly_df_filtered = anomaly_df
            
            if anomaly_types and 'AnomalyType' in anomaly_df_filtered.columns:
                anomaly_df_filtered = anomaly_df_filtered[anomaly_df_filtered['AnomalyType'].isin(anomaly_types)]
            
            results = {}
            
            if vessel_types and 'VesselType' in df.columns:
                # Ensure VesselType is int for proper comparison
                df['VesselType'] = pd.to_numeric(df['VesselType'], errors='coerce').astype('Int64')
                vessel_types_int = [int(vt) for vt in vessel_types]
                vessel_counts = df[df['VesselType'].isin(vessel_types_int)]['VesselType'].value_counts()
                results['vessel_type_counts'] = vessel_counts.to_dict()
            
            if 'AnomalyType' in anomaly_df_filtered.columns:
                anomaly_counts = anomaly_df_filtered['AnomalyType'].value_counts()
                results['anomaly_type_counts'] = anomaly_counts.to_dict()
            
            if 'VesselType' in df.columns and 'AnomalyType' in anomaly_df_filtered.columns:
                # Ensure VesselType is int for proper merging
                df['VesselType'] = pd.to_numeric(df['VesselType'], errors='coerce').astype('Int64')
                mmsi_vessel_map = df[['MMSI', 'VesselType']].drop_duplicates()
                anomaly_with_vessel = anomaly_df_filtered.merge(mmsi_vessel_map, on='MMSI', how='left')
                
                if not anomaly_with_vessel.empty:
                    # Convert VesselType to int for crosstab
                    anomaly_with_vessel['VesselType'] = pd.to_numeric(anomaly_with_vessel['VesselType'], errors='coerce').astype('Int64')
                    crosstab = pd.crosstab(anomaly_with_vessel['VesselType'], 
                                          anomaly_with_vessel['AnomalyType'])
                    results['crosstab'] = crosstab
            
            if output_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = os.path.join(self.output_directory, f"Correlation_Analysis_{timestamp}.html")
            
            if PLOTLY_AVAILABLE:
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Vessel Type Distribution', 'Anomaly Type Distribution', 
                                  'Cross-Tabulation Heatmap', 'Summary Statistics'),
                    specs=[[{"type": "bar"}, {"type": "bar"}],
                          [{"type": "heatmap"}, {"type": "table"}]]
                )
                
                if 'vessel_type_counts' in results:
                    fig.add_trace(
                        go.Bar(x=list(results['vessel_type_counts'].keys()),
                              y=list(results['vessel_type_counts'].values()),
                              name='Vessel Types'),
                        row=1, col=1
                    )
                
                if 'anomaly_type_counts' in results:
                    fig.add_trace(
                        go.Bar(x=list(results['anomaly_type_counts'].keys()),
                              y=list(results['anomaly_type_counts'].values()),
                              name='Anomaly Types'),
                        row=1, col=2
                    )
                
                if 'crosstab' in results:
                    crosstab = results['crosstab']
                    fig.add_trace(
                        go.Heatmap(z=crosstab.values,
                                  x=crosstab.columns,
                                  y=crosstab.index,
                                  colorscale='Viridis'),
                        row=2, col=1
                    )
                
                fig.update_layout(height=800, title_text="Correlation Analysis Results")
                fig.write_html(output_path)
            else:
                html_content = ["<html><head><title>Correlation Analysis</title></head><body>"]
                html_content.append("<h1>Correlation Analysis Results</h1>")
                
                if 'crosstab' in results:
                    html_content.append("<h2>Cross-Tabulation</h2>")
                    html_content.append(results['crosstab'].to_html())
                
                html_content.append("</body></html>")
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(html_content))
            
            logger.info(f"Correlation analysis completed: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error in correlation analysis: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def temporal_pattern_analysis(self, output_path=None):
        """Analyze temporal patterns including hourly/daily distributions."""
        try:
            logger.info("Performing temporal pattern analysis...")
            
            anomaly_df = self.load_anomaly_data()
            
            if anomaly_df.empty:
                logger.error("No anomaly data for temporal analysis")
                return None
            
            if 'BaseDateTime' in anomaly_df.columns:
                anomaly_df['DateTime'] = pd.to_datetime(anomaly_df['BaseDateTime'])
            elif 'Date' in anomaly_df.columns:
                anomaly_df['DateTime'] = pd.to_datetime(anomaly_df['Date'])
            else:
                logger.error("No datetime column found")
                return None
            
            anomaly_df['Hour'] = anomaly_df['DateTime'].dt.hour
            anomaly_df['DayOfWeek'] = anomaly_df['DateTime'].dt.day_name()
            anomaly_df['Date'] = anomaly_df['DateTime'].dt.date
            
            hourly_dist = anomaly_df['Hour'].value_counts().sort_index()
            daily_dist = anomaly_df.groupby('Date').size()
            dayofweek_dist = anomaly_df['DayOfWeek'].value_counts()
            
            if output_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = os.path.join(self.output_directory, f"Temporal_Patterns_{timestamp}.html")
            
            if PLOTLY_AVAILABLE:
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Hourly Distribution', 'Daily Trend', 
                                  'Day of Week Distribution', 'Anomaly Type by Hour'),
                    specs=[[{"type": "bar"}, {"type": "scatter"}],
                          [{"type": "bar"}, {"type": "bar"}]]
                )
                
                fig.add_trace(
                    go.Bar(x=hourly_dist.index, y=hourly_dist.values, name='Hourly'),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(x=daily_dist.index, y=daily_dist.values, 
                              mode='lines+markers', name='Daily'),
                    row=1, col=2
                )
                
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                dayofweek_ordered = pd.Series([dayofweek_dist.get(d, 0) for d in day_order], index=day_order)
                fig.add_trace(
                    go.Bar(x=dayofweek_ordered.index, y=dayofweek_ordered.values, name='Day of Week'),
                    row=2, col=1
                )
                
                if 'AnomalyType' in anomaly_df.columns:
                    anomaly_by_hour = pd.crosstab(anomaly_df['Hour'], anomaly_df['AnomalyType'])
                    for col in anomaly_by_hour.columns:
                        fig.add_trace(
                            go.Bar(x=anomaly_by_hour.index, y=anomaly_by_hour[col], 
                                  name=col),
                            row=2, col=2
                        )
                
                fig.update_layout(height=800, title_text="Temporal Pattern Analysis")
                fig.write_html(output_path)
            else:
                html_content = ["<html><head><title>Temporal Patterns</title></head><body>"]
                html_content.append("<h1>Temporal Pattern Analysis</h1>")
                html_content.append("<h2>Hourly Distribution</h2>")
                html_content.append(hourly_dist.to_frame('Count').to_html())
                html_content.append("</body></html>")
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(html_content))
            
            logger.info(f"Temporal pattern analysis completed: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error in temporal pattern analysis: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def vessel_behavior_clustering(self, vessel_types=None, output_path=None, n_clusters=5):
        """Apply clustering algorithms to identify similar vessel behaviors.
        
        Args:
            vessel_types: List of vessel type integers to filter by. If None, uses all vessels.
            output_path: Output file path
            n_clusters: Number of clusters to create
        """
        try:
            logger.info("Performing vessel behavior clustering...")
            
            df = self.load_cached_data()
            
            if df.empty:
                logger.error("No data for clustering")
                return None
            
            # Filter by vessel types if specified
            if vessel_types is not None and len(vessel_types) > 0:
                if 'VesselType' in df.columns:
                    # Convert VesselType to int for comparison, handling NaN values
                    df['VesselType'] = df['VesselType'].astype('Int64')
                    vessel_types_int = [int(vt) for vt in vessel_types]
                    df = df[df['VesselType'].isin(vessel_types_int)]
                    logger.info(f"Filtered to {len(vessel_types)} vessel type(s): {vessel_types}")
                    
                    if df.empty:
                        logger.warning(f"No data found for selected vessel types: {vessel_types}")
                        logger.info("This may be because these vessel types are not in the current dataset.")
                        logger.info("The analysis will continue with available data, or you may need to run analysis with these vessel types included.")
                else:
                    logger.warning("VesselType column not found, cannot filter by vessel types")
            
            if df.empty:
                logger.error("No data available for clustering")
                return None
            
            vessel_features = []
            
            for mmsi in df['MMSI'].unique():
                vessel_data = df[df['MMSI'] == mmsi]
                
                features = {
                    'MMSI': mmsi,
                    'avg_speed': vessel_data['SOG'].mean() if 'SOG' in vessel_data.columns else 0,
                    'max_speed': vessel_data['SOG'].max() if 'SOG' in vessel_data.columns else 0,
                    'total_records': len(vessel_data),
                    'lat_range': vessel_data['LAT'].max() - vessel_data['LAT'].min() if 'LAT' in vessel_data.columns else 0,
                    'lon_range': vessel_data['LON'].max() - vessel_data['LON'].min() if 'LON' in vessel_data.columns else 0,
                }
                
                if 'VesselType' in vessel_data.columns:
                    features['vessel_type'] = vessel_data['VesselType'].mode()[0] if not vessel_data['VesselType'].mode().empty else 0
                
                vessel_features.append(features)
            
            features_df = pd.DataFrame(vessel_features)
            
            feature_cols = [col for col in features_df.columns if col != 'MMSI']
            X = features_df[feature_cols].fillna(0)
            
            if not SKLEARN_AVAILABLE:
                logger.error("scikit-learn not available for clustering")
                return None
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            features_df['Cluster'] = kmeans.fit_predict(X)
            
            if output_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = os.path.join(self.output_directory, f"Vessel_Clustering_{timestamp}.html")
            
            if PLOTLY_AVAILABLE and 'lat_range' in feature_cols and 'lon_range' in feature_cols:
                fig = px.scatter(features_df, x='lat_range', y='lon_range', 
                                color='Cluster', size='total_records',
                                hover_data=['MMSI', 'avg_speed'],
                                title='Vessel Behavior Clusters')
                fig.write_html(output_path)
            else:
                features_df.to_csv(output_path.replace('.html', '.csv'), index=False)
                output_path = output_path.replace('.html', '.csv')
            
            logger.info(f"Clustering completed: {output_path}")
            return output_path
                
        except Exception as e:
            logger.error(f"Error in vessel behavior clustering: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def anomaly_frequency_analysis(self, output_path=None):
        """Analyze frequency and distribution of different anomaly types."""
        try:
            logger.info("Performing anomaly frequency analysis...")
            
            anomaly_df = self.load_anomaly_data()
            
            if anomaly_df.empty:
                logger.error("No anomaly data for frequency analysis")
                return None
            
            if 'AnomalyType' in anomaly_df.columns:
                freq_dist = anomaly_df['AnomalyType'].value_counts()
                total = len(anomaly_df)
                rel_freq = freq_dist / total * 100
                
                if output_path is None:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    output_path = os.path.join(self.output_directory, f"Anomaly_Frequency_{timestamp}.html")
                
                if PLOTLY_AVAILABLE:
                    fig = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=('Absolute Frequency', 'Relative Frequency (%)'),
                        specs=[[{"type": "bar"}, {"type": "bar"}]]
                    )
                    
                    fig.add_trace(
                        go.Bar(x=freq_dist.index, y=freq_dist.values, name='Count'),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Bar(x=rel_freq.index, y=rel_freq.values, name='Percentage'),
                        row=1, col=2
                    )
                    
                    fig.update_layout(height=500, title_text="Anomaly Frequency Analysis")
                    fig.write_html(output_path)
                else:
                    freq_df = pd.DataFrame({
                        'AnomalyType': freq_dist.index,
                        'Count': freq_dist.values,
                        'Percentage': rel_freq.values
                    })
                    csv_path = output_path.replace('.html', '.csv')
                    freq_df.to_csv(csv_path, index=False)
                    output_path = csv_path
                
                logger.info(f"Anomaly frequency analysis completed: {output_path}")
                return output_path
            else:
                logger.error("AnomalyType column not found")
                return None
                
        except Exception as e:
            logger.error(f"Error in anomaly frequency analysis: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def create_custom_chart(self, chart_type, x_column=None, y_column=None, color_column=None, 
                           group_by=None, aggregation='count', title=None, output_path=None):
        """Create a custom chart based on user selections.
        
        Args:
            chart_type: Type of chart ('bar', 'scatter', 'line', 'pie', 'stacked_bar', 'timeline', 'histogram', 'box')
            x_column: Column name for x-axis
            y_column: Column name for y-axis (if applicable)
            color_column: Column name for color grouping
            group_by: Column name to group by
            aggregation: Aggregation function ('count', 'sum', 'mean', 'max', 'min')
            title: Chart title
            output_path: Output file path
        """
        try:
            logger.info(f"Creating custom chart: type={chart_type}, x={x_column}, y={y_column}")
            
            df = self.load_cached_data()
            anomaly_df = self.load_anomaly_data()
            
            # Determine which dataset to use
            if chart_type in ['timeline'] or (x_column and x_column in anomaly_df.columns):
                data_df = anomaly_df.copy()
            else:
                data_df = df.copy()
            
            if data_df.empty:
                logger.error("No data available for chart creation")
                return None
            
            if output_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = os.path.join(self.output_directory, f"Custom_Chart_{chart_type}_{timestamp}.html")
            
            if not PLOTLY_AVAILABLE:
                logger.error("Plotly not available for chart creation")
                return None
            
            # Prepare data based on chart type
            if group_by and group_by in data_df.columns:
                if aggregation == 'count':
                    chart_data = data_df.groupby(group_by).size().reset_index(name='Count')
                    x_col = group_by
                    y_col = 'Count'
                elif aggregation == 'sum' and y_column:
                    chart_data = data_df.groupby(group_by)[y_column].sum().reset_index()
                    x_col = group_by
                    y_col = y_column
                elif aggregation == 'mean' and y_column:
                    chart_data = data_df.groupby(group_by)[y_column].mean().reset_index()
                    x_col = group_by
                    y_col = y_column
                elif aggregation == 'max' and y_column:
                    chart_data = data_df.groupby(group_by)[y_column].max().reset_index()
                    x_col = group_by
                    y_col = y_column
                elif aggregation == 'min' and y_column:
                    chart_data = data_df.groupby(group_by)[y_column].min().reset_index()
                    x_col = group_by
                    y_col = y_column
                else:
                    chart_data = data_df.groupby(group_by).size().reset_index(name='Count')
                    x_col = group_by
                    y_col = 'Count'
            else:
                chart_data = data_df.copy()
                x_col = x_column
                y_col = y_column
            
            # Create chart based on type
            fig = None
            
            if chart_type == 'bar':
                if y_col:
                    fig = px.bar(chart_data, x=x_col, y=y_col, color=color_column, 
                                title=title or f"Bar Chart: {x_col} vs {y_col}")
                else:
                    value_counts = chart_data[x_col].value_counts() if x_col else pd.Series()
                    fig = go.Figure(data=[go.Bar(x=value_counts.index, y=value_counts.values)])
                    fig.update_layout(title=title or f"Bar Chart: {x_col}", xaxis_title=x_col, yaxis_title='Count')
            
            elif chart_type == 'scatter':
                if x_col and y_col:
                    fig = px.scatter(chart_data, x=x_col, y=y_col, color=color_column,
                                    title=title or f"Scatter Plot: {x_col} vs {y_col}")
                else:
                    logger.error("Scatter plot requires both x and y columns")
                    return None
            
            elif chart_type == 'line':
                if x_col and y_col:
                    fig = px.line(chart_data, x=x_col, y=y_col, color=color_column,
                                title=title or f"Line Chart: {x_col} vs {y_col}")
                else:
                    logger.error("Line chart requires both x and y columns")
                    return None
            
            elif chart_type == 'pie':
                if y_col:
                    fig = px.pie(chart_data, names=x_col, values=y_col, 
                               title=title or f"Pie Chart: {x_col}")
                else:
                    value_counts = chart_data[x_col].value_counts() if x_col else pd.Series()
                    fig = px.pie(values=value_counts.values, names=value_counts.index,
                                title=title or f"Pie Chart: {x_col}")
            
            elif chart_type == 'stacked_bar':
                if x_col and y_col and color_column:
                    fig = px.bar(chart_data, x=x_col, y=y_col, color=color_column,
                                title=title or f"Stacked Bar Chart: {x_col} by {color_column}",
                                barmode='stack')
                else:
                    logger.error("Stacked bar chart requires x, y, and color columns")
                    return None
            
            elif chart_type == 'timeline':
                if 'BaseDateTime' in chart_data.columns or 'Date' in chart_data.columns:
                    date_col = 'BaseDateTime' if 'BaseDateTime' in chart_data.columns else 'Date'
                    chart_data[date_col] = pd.to_datetime(chart_data[date_col])
                    chart_data['Date'] = chart_data[date_col].dt.date
                    
                    if group_by and group_by in chart_data.columns:
                        timeline_data = chart_data.groupby(['Date', group_by]).size().reset_index(name='Count')
                        fig = px.bar(timeline_data, x='Date', y='Count', color=group_by,
                                    title=title or f"Timeline: {group_by} over time",
                                    barmode='stack')
                    else:
                        timeline_data = chart_data.groupby('Date').size().reset_index(name='Count')
                        fig = px.bar(timeline_data, x='Date', y='Count',
                                    title=title or f"Timeline Chart")
                else:
                    logger.error("Timeline chart requires a date column (BaseDateTime or Date)")
                    return None
            
            elif chart_type == 'histogram':
                if x_col:
                    fig = px.histogram(chart_data, x=x_col, color=color_column,
                                      title=title or f"Histogram: {x_col}")
                else:
                    logger.error("Histogram requires an x column")
                    return None
            
            elif chart_type == 'box':
                if x_col and y_col:
                    fig = px.box(chart_data, x=x_col, y=y_col, color=color_column,
                                title=title or f"Box Plot: {x_col} vs {y_col}")
                else:
                    logger.error("Box plot requires both x and y columns")
                    return None
            
            else:
                logger.error(f"Unsupported chart type: {chart_type}")
                return None
            
            if fig:
                fig.update_layout(height=600)
                fig.write_html(output_path)
                logger.info(f"Custom chart created: {output_path}")
                return output_path
            else:
                logger.error("Failed to create chart")
                return None
                
        except Exception as e:
            logger.error(f"Error creating custom chart: {e}")
            logger.error(traceback.format_exc())
            return None
    
    # ========================================================================
    # TAB 3: MAPPING TOOLS
    # ========================================================================
    
    def create_full_spectrum_map(self, show_pins=True, show_heatmap=True, output_path=None):
        """Create a comprehensive map showing all anomalies."""
        try:
            if not FOLIUM_AVAILABLE:
                logger.error("folium not available for map creation")
                return None
            
            logger.info("Creating full spectrum anomaly map...")
            
            df = self.load_cached_data()
            anomaly_df = self.load_anomaly_data()
            
            if anomaly_df.empty:
                logger.error("No anomaly data for map")
                return None
            
            if output_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = os.path.join(self.output_directory, f"Full_Spectrum_Map_{timestamp}.html")
            
            center_lat = anomaly_df['LAT'].mean() if 'LAT' in anomaly_df.columns else 0
            center_lon = anomaly_df['LON'].mean() if 'LON' in anomaly_df.columns else 0
            m = folium.Map(location=[center_lat, center_lon], zoom_start=6)
            
            if show_heatmap:
                valid_anomalies = anomaly_df.dropna(subset=['LAT', 'LON'])
                if not valid_anomalies.empty:
                    heat_data = [[row['LAT'], row['LON'], 1] for _, row in valid_anomalies.iterrows()]
                    HeatMap(heat_data, radius=15, blur=10, max_zoom=1).add_to(m)
            
            if show_pins:
                marker_cluster = MarkerCluster().add_to(m)
                
                valid_anomalies = anomaly_df.dropna(subset=['LAT', 'LON'])
                for _, row in valid_anomalies.iterrows():
                    popup_text = f"MMSI: {row.get('MMSI', 'N/A')}<br>"
                    if 'AnomalyType' in row:
                        popup_text += f"Type: {row['AnomalyType']}<br>"
                    if 'BaseDateTime' in row:
                        popup_text += f"Time: {row['BaseDateTime']}"
                    
                    folium.Marker(
                        location=[row['LAT'], row['LON']],
                        popup=folium.Popup(popup_text, max_width=200)
                    ).add_to(marker_cluster)
            
            folium.LayerControl().add_to(m)
            
            m.save(output_path)
            logger.info(f"Full spectrum map created: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating full spectrum map: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def create_vessel_map(self, mmsi, map_type='path', output_path=None):
        """Create maps focused on specific vessels by MMSI."""
        try:
            if not FOLIUM_AVAILABLE:
                logger.error("folium not available for map creation")
                return None
            
            logger.info(f"Creating {map_type} map for vessel {mmsi}...")
            
            df = self.load_cached_data()
            anomaly_df = self.load_anomaly_data()
            
            if df.empty:
                logger.error("No data available for vessel map")
                return None
            
            # Convert MMSI to same type as in dataframe for proper comparison
            if 'MMSI' in df.columns:
                # Try to match the type in the dataframe
                sample_mmsi = df['MMSI'].iloc[0] if len(df) > 0 else None
                if sample_mmsi is not None:
                    if isinstance(sample_mmsi, (int, np.integer)):
                        mmsi = int(mmsi)
                    elif isinstance(sample_mmsi, (str, np.str_)):
                        mmsi = str(mmsi)
                # Also ensure dataframe MMSI is consistent type
                df['MMSI'] = df['MMSI'].astype(type(mmsi))
            
            vessel_data = df[df['MMSI'] == mmsi].copy()
            
            if vessel_data.empty:
                logger.error(f"No data found for vessel {mmsi}")
                return None
            
            # Filter anomalies for this vessel
            if not anomaly_df.empty and 'MMSI' in anomaly_df.columns:
                # Ensure MMSI types match
                anomaly_df['MMSI'] = anomaly_df['MMSI'].astype(type(mmsi))
                vessel_anomalies = anomaly_df[anomaly_df['MMSI'] == mmsi].copy()
            else:
                vessel_anomalies = pd.DataFrame()
            
            if output_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = os.path.join(self.output_directory, f"Vessel_{mmsi}_{map_type}_{timestamp}.html")
            
            center_lat = vessel_data['LAT'].mean()
            center_lon = vessel_data['LON'].mean()
            m = folium.Map(location=[center_lat, center_lon], zoom_start=8)
            
            if map_type == 'path':
                # Sort by datetime if available for proper path ordering
                if 'BaseDateTime' in vessel_data.columns:
                    vessel_data = vessel_data.sort_values('BaseDateTime')
                
                path_coords = [[row['LAT'], row['LON']] for _, row in vessel_data.iterrows() 
                              if pd.notna(row['LAT']) and pd.notna(row['LON'])]
                if path_coords:
                    folium.PolyLine(path_coords, color='blue', weight=3, opacity=0.7).add_to(m)
                    folium.Marker(path_coords[0], popup='Start', icon=folium.Icon(color='green')).add_to(m)
                    folium.Marker(path_coords[-1], popup='End', icon=folium.Icon(color='red')).add_to(m)
                    logger.info(f"Path map: Added {len(path_coords)} points for vessel {mmsi}")
                else:
                    logger.warning(f"No valid coordinates found for path map")
            
            elif map_type == 'anomaly':
                if not vessel_anomalies.empty:
                    marker_count = 0
                    for _, row in vessel_anomalies.iterrows():
                        if pd.notna(row.get('LAT')) and pd.notna(row.get('LON')):
                            popup_text = f"MMSI: {mmsi}<br>Anomaly: {row.get('AnomalyType', 'Unknown')}"
                            if 'BaseDateTime' in row:
                                popup_text += f"<br>Time: {row['BaseDateTime']}"
                            folium.Marker(
                                [row['LAT'], row['LON']],
                                popup=folium.Popup(popup_text, max_width=200),
                                icon=folium.Icon(color='red', icon='exclamation-sign')
                            ).add_to(m)
                            marker_count += 1
                    logger.info(f"Anomaly map: Added {marker_count} anomaly markers for vessel {mmsi}")
                else:
                    logger.warning(f"No anomalies found for vessel {mmsi}")
            
            elif map_type == 'heatmap':
                heat_data = [[row['LAT'], row['LON'], 1] for _, row in vessel_data.iterrows()
                            if pd.notna(row['LAT']) and pd.notna(row['LON'])]
                if heat_data:
                    HeatMap(heat_data, radius=15, blur=10).add_to(m)
                    logger.info(f"Heatmap: Added {len(heat_data)} heat points for vessel {mmsi}")
                else:
                    logger.warning(f"No valid coordinates found for heatmap")
            
            m.save(output_path)
            logger.info(f"Vessel map created: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating vessel map: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def get_top_vessels_by_anomaly(self, anomaly_type=None, limit=10):
        """Get top vessels by anomaly count for a specific anomaly type."""
        try:
            anomaly_df = self.load_anomaly_data()
            
            if anomaly_df.empty:
                return pd.DataFrame()
            
            if anomaly_type and 'AnomalyType' in anomaly_df.columns:
                filtered = anomaly_df[anomaly_df['AnomalyType'] == anomaly_type]
            else:
                filtered = anomaly_df
            
            if 'MMSI' in filtered.columns:
                top_vessels = filtered['MMSI'].value_counts().head(limit).reset_index()
                top_vessels.columns = ['MMSI', 'Anomaly_Count']
                return top_vessels
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error getting top vessels: {e}")
            return pd.DataFrame()
    
    def create_filtered_map(self, map_type='path', vessel_types=None, anomaly_types=None, vessel_mmsi=None, output_path=None):
        """Create filtered maps for specific anomaly types and/or vessel types, or specific vessel + anomaly type."""
        try:
            if not FOLIUM_AVAILABLE:
                logger.error("folium not available for map creation")
                return None
            
            logger.info(f"Creating filtered {map_type} map: vessel_types={vessel_types}, anomaly_types={anomaly_types}, vessel_mmsi={vessel_mmsi}")
            
            df = self.load_cached_data()
            anomaly_df = self.load_anomaly_data()
            
            # Filter by vessel MMSI if specified
            if vessel_mmsi:
                df = df[df['MMSI'] == vessel_mmsi].copy()
                anomaly_df = anomaly_df[anomaly_df['MMSI'] == vessel_mmsi].copy() if not anomaly_df.empty and 'MMSI' in anomaly_df.columns else pd.DataFrame()
            
            # Filter by vessel types if specified
            if vessel_types and 'VesselType' in df.columns:
                # Convert vessel types to int for proper comparison
                vessel_types_int = [int(vt) for vt in vessel_types]
                df['VesselType'] = pd.to_numeric(df['VesselType'], errors='coerce').astype('Int64')
                df = df[df['VesselType'].isin(vessel_types_int)].copy()
                vessel_mmsis = df['MMSI'].unique()
                anomaly_df = anomaly_df[anomaly_df['MMSI'].isin(vessel_mmsis)].copy() if not anomaly_df.empty and 'MMSI' in anomaly_df.columns else pd.DataFrame()
            
            # Filter by anomaly types if specified
            if anomaly_types and 'AnomalyType' in anomaly_df.columns:
                anomaly_df = anomaly_df[anomaly_df['AnomalyType'].isin(anomaly_types)].copy()
                if vessel_mmsi is None:  # Only filter df if not already filtered by MMSI
                    vessel_mmsis = anomaly_df['MMSI'].unique()
                    df = df[df['MMSI'].isin(vessel_mmsis)].copy()
            
            if df.empty:
                logger.error("No data found for filtered map")
                return None
            
            if output_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filter_str = f"filtered_{map_type}_{timestamp}"
                if vessel_mmsi:
                    filter_str = f"vessel_{vessel_mmsi}_{map_type}_{timestamp}"
                output_path = os.path.join(self.output_directory, f"{filter_str}.html")
            
            center_lat = df['LAT'].mean() if 'LAT' in df.columns else 0
            center_lon = df['LON'].mean() if 'LON' in df.columns else 0
            m = folium.Map(location=[center_lat, center_lon], zoom_start=8)
            
            if map_type == 'path':
                # Group by MMSI and draw paths
                for mmsi in df['MMSI'].unique():
                    vessel_data = df[df['MMSI'] == mmsi]
                    path_coords = [[row['LAT'], row['LON']] for _, row in vessel_data.iterrows() 
                                  if pd.notna(row['LAT']) and pd.notna(row['LON'])]
                    if path_coords:
                        folium.PolyLine(path_coords, color='blue', weight=2, opacity=0.7, 
                                       popup=f"Vessel {mmsi}").add_to(m)
            
            elif map_type == 'anomaly':
                if not anomaly_df.empty:
                    marker_cluster = MarkerCluster().add_to(m)
                    for _, row in anomaly_df.iterrows():
                        if pd.notna(row.get('LAT')) and pd.notna(row.get('LON')):
                            popup_text = f"MMSI: {row.get('MMSI', 'N/A')}<br>"
                            popup_text += f"Anomaly: {row.get('AnomalyType', 'Unknown')}<br>"
                            if 'BaseDateTime' in row:
                                popup_text += f"Time: {row['BaseDateTime']}"
                            folium.Marker(
                                [row['LAT'], row['LON']],
                                popup=folium.Popup(popup_text, max_width=200),
                                icon=folium.Icon(color='red', icon='exclamation-sign')
                            ).add_to(marker_cluster)
            
            elif map_type == 'heatmap':
                if not anomaly_df.empty:
                    heat_data = [[row['LAT'], row['LON'], 1] for _, row in anomaly_df.iterrows()
                                if pd.notna(row.get('LAT')) and pd.notna(row.get('LON'))]
                else:
                    heat_data = [[row['LAT'], row['LON'], 1] for _, row in df.iterrows()
                                if pd.notna(row['LAT']) and pd.notna(row['LON'])]
                if heat_data:
                    HeatMap(heat_data, radius=15, blur=10).add_to(m)
            
            m.save(output_path)
            logger.info(f"Filtered map created: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating filtered map: {e}")
            logger.error(traceback.format_exc())
            return None
    
    # ========================================================================
    # TAB 4: VESSEL-SPECIFIC ANALYSIS
    # ========================================================================
    
    def extended_time_analysis(self, mmsi, additional_days_start, additional_days_end, output_path=None):
        """Analyze additional days of data for a specific vessel."""
        try:
            logger.info(f"Performing extended time analysis for vessel {mmsi}...")
            
            is_valid, error_msg = validate_date_range(additional_days_start, additional_days_end)
            if not is_valid:
                logger.error(f"Invalid date range: {error_msg}")
                return None
            
            original_df = self.load_cached_data()
            original_vessel = original_df[original_df['MMSI'] == mmsi] if not original_df.empty else pd.DataFrame()
            
            extended_df = load_cached_data_for_date_range(
                additional_days_start,
                additional_days_end,
                self.run_info['ship_types']
            )
            extended_vessel = extended_df[extended_df['MMSI'] == mmsi] if not extended_df.empty else pd.DataFrame()
            
            if original_vessel.empty and extended_vessel.empty:
                logger.error(f"No data found for vessel {mmsi}")
                return None
            
            combined_vessel = pd.concat([original_vessel, extended_vessel], ignore_index=True) if not extended_vessel.empty else original_vessel
            
            stats = {
                'MMSI': mmsi,
                'Original_Period_Records': len(original_vessel),
                'Extended_Period_Records': len(extended_vessel),
                'Total_Records': len(combined_vessel),
            }
            
            if 'BaseDateTime' in combined_vessel.columns:
                combined_vessel['BaseDateTime'] = pd.to_datetime(combined_vessel['BaseDateTime'])
                stats['First_Seen'] = combined_vessel['BaseDateTime'].min()
                stats['Last_Seen'] = combined_vessel['BaseDateTime'].max()
                stats['Total_Days'] = (stats['Last_Seen'] - stats['First_Seen']).days + 1
            
            if output_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = os.path.join(self.output_directory, f"Extended_Analysis_{mmsi}_{timestamp}.html")
            
            html_content = ["<html><head><title>Extended Time Analysis</title></head><body>"]
            html_content.append(f"<h1>Extended Time Analysis for Vessel {mmsi}</h1>")
            html_content.append("<h2>Statistics</h2>")
            html_content.append("<table border='1'><tr><th>Metric</th><th>Value</th></tr>")
            
            for key, value in stats.items():
                html_content.append(f"<tr><td>{key}</td><td>{value}</td></tr>")
            
            html_content.append("</table></body></html>")
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(html_content))
            
            logger.info(f"Extended time analysis completed: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error in extended time analysis: {e}")
            logger.error(traceback.format_exc())
            return None


# ============================================================================
# GUI INTERFACE CLASS
# ============================================================================

class ProgressDialog:
    """Progress dialog for long-running operations."""
    
    def __init__(self, parent, title="Processing", message="Please wait..."):
        self.parent = parent
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(title)
        self.dialog.geometry("400x150")
        self.dialog.transient(parent)
        self.dialog.resizable(False, False)
        
        self.message_var = tk.StringVar(value=message)
        ttk.Label(self.dialog, textvariable=self.message_var, 
                 font=("Arial", 10)).pack(pady=10)
        
        self.progress = ttk.Progressbar(self.dialog, mode="indeterminate", length=300)
        self.progress.pack(pady=10, padx=20, fill=tk.X)
        self.progress.start()
        
        self.status_var = tk.StringVar(value="")
        ttk.Label(self.dialog, textvariable=self.status_var, 
                 font=("Arial", 9)).pack(pady=5)
        
        self.dialog.update_idletasks()
        width = self.dialog.winfo_width()
        height = self.dialog.winfo_height()
        x = (self.dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (height // 2)
        self.dialog.geometry(f"{width}x{height}+{x}+{y}")
        
        self.dialog.grab_set()
        self.dialog.focus_set()
    
    def update_message(self, message):
        self.message_var.set(message)
        self.dialog.update()
    
    def update_status(self, status):
        self.status_var.set(status)
        self.dialog.update()
    
    def close(self):
        self.progress.stop()
        self.dialog.destroy()


class AdvancedAnalysisGUI:
    """GUI interface for Advanced Analysis features."""
    
    def __init__(self, parent_window, output_directory=None, config_path='config.ini'):
        self.parent_window = parent_window
        try:
            # Resolve config path relative to script directory
            resolved_config_path = get_config_path(config_path)
            self.analysis = AdvancedAnalysis(parent_window, output_directory, resolved_config_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize Advanced Analysis:\n{str(e)}")
            raise
        
        self.window = tk.Toplevel(parent_window)
        self.window.title("Advanced Analytical Tools")
        self.window.geometry("900x700")
        self.window.minsize(800, 600)
        
        main_frame = ttk.Frame(self.window, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(main_frame, text="Advanced Analytical Tools", 
                 font=("Arial", 16, "bold")).pack(anchor=tk.W)
        ttk.Label(main_frame, 
                 text="Perform additional analysis on the previously generated dataset",
                 font=("Arial", 11)).pack(anchor=tk.W, pady=(0, 15))
        
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self._create_tab1_additional_outputs()
        self._create_tab2_further_analysis()
        self._create_tab3_mapping_tools()
        self._create_tab4_vessel_analysis()
        
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self.window, textvariable=self.status_var, relief="sunken", anchor=tk.W).pack(side=tk.BOTTOM, fill=tk.X)
    
    def _create_tab1_additional_outputs(self):
        """Create Tab 1: Additional Outputs"""
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="Additional Outputs")
        
        ttk.Label(tab, text="Generate Additional Outputs from Dataset", 
                 font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(0, 10))
        
        frame1 = ttk.Frame(tab)
        frame1.pack(fill=tk.X, pady=5)
        ttk.Button(frame1, text="Export Full Dataset to CSV", width=30,
                  command=self._export_full_dataset).pack(side=tk.LEFT, padx=5)
        ttk.Label(frame1, text="Export the complete analysis dataset to CSV format", 
                 wraplength=500).pack(side=tk.LEFT, padx=5)
        
        frame2 = ttk.Frame(tab)
        frame2.pack(fill=tk.X, pady=5)
        ttk.Button(frame2, text="Generate Summary Report", width=30,
                  command=self._generate_summary_report).pack(side=tk.LEFT, padx=5)
        ttk.Label(frame2, text="Create a summary report with key findings and statistics", 
                 wraplength=500).pack(side=tk.LEFT, padx=5)
        
        frame3 = ttk.Frame(tab)
        frame3.pack(fill=tk.X, pady=5)
        ttk.Button(frame3, text="Export Vessel Statistics", width=30,
                  command=self._export_vessel_statistics).pack(side=tk.LEFT, padx=5)
        ttk.Label(frame3, text="Export vessel-specific statistics to Excel format", 
                 wraplength=500).pack(side=tk.LEFT, padx=5)
        
        frame4 = ttk.Frame(tab)
        frame4.pack(fill=tk.X, pady=5)
        ttk.Button(frame4, text="Generate Anomaly Timeline", width=30,
                  command=self._generate_anomaly_timeline).pack(side=tk.LEFT, padx=5)
        ttk.Label(frame4, text="Create a timeline visualization of anomalies", 
                 wraplength=500).pack(side=tk.LEFT, padx=5)
    
    def _create_tab2_further_analysis(self):
        """Create Tab 2: Further Analysis"""
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="Further Analysis")
        
        ttk.Label(tab, text="Further Analysis Tools", 
                 font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(0, 10))
        
        frame1 = ttk.Frame(tab)
        frame1.pack(fill=tk.X, pady=5)
        ttk.Button(frame1, text="Anomaly Correlation Analysis", width=30,
                  command=self._correlation_analysis_dialog).pack(side=tk.LEFT, padx=5)
        ttk.Label(frame1, text="Analyze correlations between vessel types and anomaly types", 
                 wraplength=500).pack(side=tk.LEFT, padx=5)
        
        frame2 = ttk.Frame(tab)
        frame2.pack(fill=tk.X, pady=5)
        ttk.Button(frame2, text="Temporal Pattern Analysis", width=30,
                  command=self._temporal_pattern_analysis).pack(side=tk.LEFT, padx=5)
        ttk.Label(frame2, text="Analyze patterns over time, including hourly/daily distributions", 
                 wraplength=500).pack(side=tk.LEFT, padx=5)
        
        frame3 = ttk.Frame(tab)
        frame3.pack(fill=tk.X, pady=5)
        ttk.Button(frame3, text="Vessel Behavior Clustering", width=30,
                  command=self._vessel_behavior_clustering).pack(side=tk.LEFT, padx=5)
        ttk.Label(frame3, text="Apply clustering algorithms to identify similar vessel behaviors", 
                 wraplength=500).pack(side=tk.LEFT, padx=5)
        
        frame4 = ttk.Frame(tab)
        frame4.pack(fill=tk.X, pady=5)
        ttk.Button(frame4, text="Anomaly Frequency Analysis", width=30,
                  command=self._anomaly_frequency_analysis).pack(side=tk.LEFT, padx=5)
        ttk.Label(frame4, text="Analyze frequency and distribution of different anomaly types", 
                 wraplength=500).pack(side=tk.LEFT, padx=5)
        
        frame5 = ttk.Frame(tab)
        frame5.pack(fill=tk.X, pady=5)
        ttk.Button(frame5, text="Create Custom Chart", width=30,
                  command=self._create_custom_chart_dialog).pack(side=tk.LEFT, padx=5)
        ttk.Label(frame5, text="Create custom charts with various types (bar, scatter, pie, etc.)", 
                 wraplength=500).pack(side=tk.LEFT, padx=5)
    
    def _create_tab3_mapping_tools(self):
        """Create Tab 3: Mapping Tools"""
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="Mapping Tools")
        
        ttk.Label(tab, text="Advanced Mapping Tools", 
                 font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(0, 10))
        
        frame1 = ttk.Frame(tab)
        frame1.pack(fill=tk.X, pady=5)
        ttk.Button(frame1, text="Full Spectrum Anomaly Map", width=30,
                  command=self._full_spectrum_map_dialog).pack(side=tk.LEFT, padx=5)
        ttk.Label(frame1, text="Create a comprehensive map showing all anomalies", 
                 wraplength=500).pack(side=tk.LEFT, padx=5)
        
        frame2 = ttk.Frame(tab)
        frame2.pack(fill=tk.X, pady=5)
        ttk.Label(frame2, text="MMSI:").pack(side=tk.LEFT, padx=5)
        self.vessel_mmsi_var = tk.StringVar()
        ttk.Entry(frame2, textvariable=self.vessel_mmsi_var, width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame2, text="Create Vessel Map", width=25,
                  command=self._create_vessel_map_dialog).pack(side=tk.LEFT, padx=5)
        
        frame3 = ttk.Frame(tab)
        frame3.pack(fill=tk.X, pady=5)
        ttk.Label(frame3, text="Top 10 Vessels by Anomaly Type:").pack(anchor=tk.W)
        listbox_frame = ttk.Frame(frame3)
        listbox_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.top_vessels_listbox = tk.Listbox(listbox_frame, height=10)
        scrollbar = ttk.Scrollbar(listbox_frame, orient=tk.VERTICAL, command=self.top_vessels_listbox.yview)
        self.top_vessels_listbox.configure(yscrollcommand=scrollbar.set)
        self.top_vessels_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Add right-click context menu for copy
        self.top_vessels_menu = tk.Menu(self.window, tearoff=0)
        self.top_vessels_menu.add_command(label="Copy MMSI", command=self._copy_mmsi_from_listbox)
        self.top_vessels_listbox.bind("<Button-3>", self._show_listbox_menu)
        self._populate_top_vessels()
        
        frame4 = ttk.Frame(tab)
        frame4.pack(fill=tk.X, pady=5)
        ttk.Button(frame4, text="Create Filtered Map", width=30,
                  command=self._create_filtered_map_dialog).pack(side=tk.LEFT, padx=5)
        ttk.Label(frame4, text="Create path/anomaly/heat-maps with filters", 
                 wraplength=500).pack(side=tk.LEFT, padx=5)
    
    def _create_tab4_vessel_analysis(self):
        """Create Tab 4: Vessel-Specific Analysis"""
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="Vessel-Specific Analysis")
        
        ttk.Label(tab, text="Vessel-Specific Analysis Tools", 
                 font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(0, 10))
        
        frame1 = ttk.Frame(tab)
        frame1.pack(fill=tk.X, pady=5)
        ttk.Label(frame1, text="MMSI:").pack(side=tk.LEFT, padx=5)
        self.extended_mmsi_var = tk.StringVar()
        ttk.Entry(frame1, textvariable=self.extended_mmsi_var, width=15).pack(side=tk.LEFT, padx=5)
        ttk.Label(frame1, text="Additional Date Range:").pack(side=tk.LEFT, padx=5)
        
        # Use DateEntry if available, otherwise use text entry
        if TKCALENDAR_AVAILABLE:
            ttk.Label(frame1, text="Start:").pack(side=tk.LEFT, padx=2)
            self.extended_start_picker = DateEntry(frame1, width=12, background='darkblue',
                                                   foreground='white', borderwidth=2, date_pattern='y-mm-dd',
                                                   state='readonly')
            self.extended_start_picker.pack(side=tk.LEFT, padx=2)
            
            ttk.Label(frame1, text="End:").pack(side=tk.LEFT, padx=2)
            self.extended_end_picker = DateEntry(frame1, width=12, background='darkblue',
                                                 foreground='white', borderwidth=2, date_pattern='y-mm-dd',
                                                 state='readonly')
            self.extended_end_picker.pack(side=tk.LEFT, padx=2)
            self.extended_start_var = None
            self.extended_end_var = None
        else:
            self.extended_start_var = tk.StringVar()
            self.extended_end_var = tk.StringVar()
            ttk.Entry(frame1, textvariable=self.extended_start_var, width=12).pack(side=tk.LEFT, padx=2)
            ttk.Label(frame1, text="to").pack(side=tk.LEFT)
            ttk.Entry(frame1, textvariable=self.extended_end_var, width=12).pack(side=tk.LEFT, padx=2)
            self.extended_start_picker = None
            self.extended_end_picker = None
        
        ttk.Button(frame1, text="Analyze", width=20,
                  command=self._extended_time_analysis).pack(side=tk.LEFT, padx=5)
        
        frame2 = ttk.Frame(tab)
        frame2.pack(fill=tk.X, pady=5)
        ttk.Button(frame2, text="AI Predicted Path (Coming Soon)", width=30,
                  command=lambda: messagebox.showinfo("Coming Soon", 
                  "AI path prediction feature will be available in a future release.")).pack(side=tk.LEFT, padx=5)
    
    def _export_full_dataset(self):
        progress = ProgressDialog(self.window, "Exporting Dataset", "Exporting full dataset to CSV...")
        try:
            self.status_var.set("Exporting full dataset...")
            result = self.analysis.export_full_dataset()
            progress.close()
            if result:
                self.status_var.set(f"Exported to: {result}")
                file_size = os.path.getsize(result)
                messagebox.showinfo("Success", 
                                  f"Full dataset exported successfully!\n\n"
                                  f"Location: {result}\n"
                                  f"Size: {format_file_size(file_size)}")
            else:
                self.status_var.set("Export failed")
                messagebox.showerror("Error", "Failed to export full dataset. Check logs for details.")
        except Exception as e:
            progress.close()
            self.status_var.set("Export failed")
            logger.error(f"Error in export: {e}")
            messagebox.showerror("Error", f"Export failed: {str(e)}")
    
    def _generate_summary_report(self):
        progress = ProgressDialog(self.window, "Generating Report", "Generating summary report...")
        try:
            self.status_var.set("Generating summary report...")
            result = self.analysis.generate_summary_report()
            progress.close()
            if result:
                self.status_var.set(f"Report generated: {result}")
                messagebox.showinfo("Success", f"Summary report generated:\n{result}")
            else:
                self.status_var.set("Report generation failed")
                messagebox.showerror("Error", "Failed to generate summary report. Check logs for details.")
        except Exception as e:
            progress.close()
            self.status_var.set("Report generation failed")
            logger.error(f"Error generating report: {e}")
            messagebox.showerror("Error", f"Report generation failed: {str(e)}")
    
    def _export_vessel_statistics(self):
        progress = ProgressDialog(self.window, "Exporting Statistics", "Exporting vessel statistics...")
        try:
            self.status_var.set("Exporting vessel statistics...")
            result = self.analysis.export_vessel_statistics()
            progress.close()
            if result:
                self.status_var.set(f"Statistics exported: {result}")
                messagebox.showinfo("Success", f"Vessel statistics exported to:\n{result}")
            else:
                self.status_var.set("Export failed")
                messagebox.showerror("Error", "Failed to export vessel statistics. Check logs for details.")
        except Exception as e:
            progress.close()
            self.status_var.set("Export failed")
            logger.error(f"Error exporting statistics: {e}")
            messagebox.showerror("Error", f"Export failed: {str(e)}")
    
    def _generate_anomaly_timeline(self):
        progress = ProgressDialog(self.window, "Generating Timeline", "Generating anomaly timeline...")
        try:
            self.status_var.set("Generating anomaly timeline...")
            result = self.analysis.generate_anomaly_timeline()
            progress.close()
            if result:
                self.status_var.set(f"Timeline generated: {result}")
                messagebox.showinfo("Success", f"Anomaly timeline generated:\n{result}")
            else:
                self.status_var.set("Timeline generation failed")
                messagebox.showerror("Error", "Failed to generate anomaly timeline. Check logs for details.")
        except Exception as e:
            progress.close()
            self.status_var.set("Timeline generation failed")
            logger.error(f"Error generating timeline: {e}")
            messagebox.showerror("Error", f"Timeline generation failed: {str(e)}")
    
    def _correlation_analysis_dialog(self):
        """Create dialog for correlation analysis selection"""
        dialog = tk.Toplevel(self.window)
        dialog.title("Anomaly Correlation Analysis")
        dialog.geometry("600x500")
        
        try:
            df = self.analysis.load_cached_data()
            anomaly_df = self.analysis.load_anomaly_data()
            
            # Get available types from data - convert to int for proper comparison
            if 'VesselType' in df.columns:
                # Convert to int, handling NaN values
                available_vessel_types_raw = df['VesselType'].dropna().unique()
                available_vessel_types = sorted([int(vt) for vt in available_vessel_types_raw if pd.notna(vt)])
            else:
                available_vessel_types = []
            
            available_anomaly_types_data = sorted(anomaly_df['AnomalyType'].unique().tolist()) if not anomaly_df.empty and 'AnomalyType' in anomaly_df.columns else []
            
            # Map data anomaly types to GUI names
            available_anomaly_types_gui = [map_anomaly_type_data_to_gui(at) for at in available_anomaly_types_data]
            
            # Get all possible types
            all_vessel_types = get_all_vessel_types()
            all_anomaly_types = get_all_anomaly_types()
            
            # Create scrollable frames
            canvas = tk.Canvas(dialog)
            scrollbar = ttk.Scrollbar(dialog, orient="vertical", command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas)
            
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
            
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            
            ttk.Label(scrollable_frame, text="Select Vessel Types (up to 2):", font=("Arial", 10, "bold")).pack(pady=5)
            vessel_frame = ttk.Frame(scrollable_frame)
            vessel_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
            
            vessel_vars = {}
            vessel_checkboxes = []
            for vtype in all_vessel_types:
                var = tk.BooleanVar()
                vessel_vars[vtype] = var
                is_available = vtype in available_vessel_types
                vessel_name = get_vessel_type_name(vtype)
                display_text = f"Type {vtype}: {vessel_name}"
                cb = ttk.Checkbutton(vessel_frame, text=display_text, variable=var, state='normal' if is_available else 'disabled')
                cb.pack(anchor=tk.W)
                vessel_checkboxes.append((vtype, var, is_available))
            
            ttk.Label(scrollable_frame, text="Select Anomaly Types (up to 2):", font=("Arial", 10, "bold")).pack(pady=5)
            anomaly_frame = ttk.Frame(scrollable_frame)
            anomaly_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
            
            anomaly_vars = {}
            anomaly_checkboxes = []
            for atype in all_anomaly_types:
                var = tk.BooleanVar()
                anomaly_vars[atype] = var
                is_available = atype in available_anomaly_types_gui
                cb = ttk.Checkbutton(anomaly_frame, text=atype, variable=var, state='normal' if is_available else 'disabled')
                cb.pack(anchor=tk.W)
                anomaly_checkboxes.append((atype, var, is_available))
            
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            
            def run_analysis():
                selected_vessel_types = [vtype for vtype, var, available in vessel_checkboxes if var.get() and available][:2]
                selected_anomaly_types_gui = [atype for atype, var, available in anomaly_checkboxes if var.get() and available][:2]
                
                # Map GUI names back to data names
                selected_anomaly_types = [map_anomaly_type_gui_to_data(at) for at in selected_anomaly_types_gui]
                
                if not selected_vessel_types and not selected_anomaly_types:
                    messagebox.showerror("Error", "Please select at least one vessel type or anomaly type")
                    return
                
                dialog.destroy()
                self.status_var.set("Performing correlation analysis...")
                result = self.analysis.correlation_analysis(selected_vessel_types, selected_anomaly_types)
                if result:
                    self.status_var.set(f"Analysis complete: {result}")
                    messagebox.showinfo("Success", f"Correlation analysis complete:\n{result}")
                else:
                    self.status_var.set("Analysis failed")
                    messagebox.showerror("Error", "Failed to perform correlation analysis")
            
            button_frame = ttk.Frame(dialog)
            button_frame.pack(pady=10)
            ttk.Button(button_frame, text="Run Analysis", command=run_analysis).pack()
            
        except Exception as e:
            logger.error(f"Error creating correlation dialog: {e}")
            logger.error(traceback.format_exc())
            messagebox.showerror("Error", f"Error creating correlation analysis dialog: {e}")
            dialog.destroy()
    
    def _temporal_pattern_analysis(self):
        progress = ProgressDialog(self.window, "Analyzing Patterns", "Analyzing temporal patterns...")
        try:
            self.status_var.set("Analyzing temporal patterns...")
            result = self.analysis.temporal_pattern_analysis()
            progress.close()
            if result:
                self.status_var.set(f"Analysis complete: {result}")
                messagebox.showinfo("Success", f"Temporal pattern analysis complete:\n{result}")
            else:
                self.status_var.set("Analysis failed")
                messagebox.showerror("Error", "Failed to perform temporal pattern analysis. Check logs for details.")
        except Exception as e:
            progress.close()
            self.status_var.set("Analysis failed")
            logger.error(f"Error in temporal analysis: {e}")
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
    
    def _vessel_behavior_clustering(self):
        """Create dialog for vessel behavior clustering with vessel type selection"""
        dialog = tk.Toplevel(self.window)
        dialog.title("Vessel Behavior Clustering")
        dialog.geometry("600x600")
        
        try:
            df = self.analysis.load_cached_data()
            
            # Get available types from data - convert to int for proper comparison
            if 'VesselType' in df.columns:
                # Convert to int, handling NaN values
                available_vessel_types_raw = df['VesselType'].dropna().unique()
                available_vessel_types = sorted([int(vt) for vt in available_vessel_types_raw if pd.notna(vt)])
            else:
                available_vessel_types = []
            
            # Get all possible types
            all_vessel_types = get_all_vessel_types()
            
            # Create scrollable frames
            canvas = tk.Canvas(dialog)
            scrollbar = ttk.Scrollbar(dialog, orient="vertical", command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas)
            
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
            
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            
            ttk.Label(scrollable_frame, text="Select Vessel Types for Clustering", 
                     font=("Arial", 12, "bold")).pack(pady=10)
            ttk.Label(scrollable_frame, 
                     text="Select one or more vessel types to analyze. All selected types will be included.",
                     wraplength=550).pack(pady=5)
            
            # Number of clusters
            ttk.Label(scrollable_frame, text="Number of Clusters:", font=("Arial", 10, "bold")).pack(pady=(10, 5))
            n_clusters_var = tk.StringVar(value='5')
            clusters_frame = ttk.Frame(scrollable_frame)
            clusters_frame.pack(pady=5)
            ttk.Label(clusters_frame, text="Clusters:").pack(side=tk.LEFT, padx=5)
            clusters_entry = ttk.Entry(clusters_frame, textvariable=n_clusters_var, width=10)
            clusters_entry.pack(side=tk.LEFT, padx=5)
            
            # Vessel type selection
            ttk.Label(scrollable_frame, text="Vessel Types:", font=("Arial", 10, "bold")).pack(pady=(10, 5))
            
            vessel_vars = {}
            vessel_checkboxes = []
            
            for vtype in all_vessel_types:
                var = tk.BooleanVar()
                vessel_vars[vtype] = var
                vessel_name = get_vessel_type_name(vtype)
                
                # Check if this type is in the current data
                is_in_data = vtype in available_vessel_types
                
                # Make all types selectable, but indicate which are in data
                if is_in_data:
                    display_text = f"Type {vtype}: {vessel_name} [In Data]"
                else:
                    display_text = f"Type {vtype}: {vessel_name} [Not in Current Data]"
                
                # All types are enabled for selection
                cb = ttk.Checkbutton(scrollable_frame, text=display_text, variable=var, 
                                   state='normal')
                cb.pack(anchor=tk.W, padx=20, pady=2)
                vessel_checkboxes.append((vtype, var, True))  # All are available for selection
            
            canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            def perform_clustering():
                selected_types = [vtype for vtype, var, is_available in vessel_checkboxes 
                                if var.get()]
                
                if not selected_types:
                    messagebox.showwarning("Warning", "No vessel types selected. Please select at least one vessel type.")
                    return
                
                try:
                    n_clusters = int(n_clusters_var.get())
                    if n_clusters < 2:
                        messagebox.showerror("Error", "Number of clusters must be at least 2")
                        return
                except ValueError:
                    messagebox.showerror("Error", "Number of clusters must be a valid integer")
                    return
                
                dialog.destroy()
                progress = ProgressDialog(self.window, "Clustering", "Performing vessel clustering...")
                try:
                    self.status_var.set("Performing vessel clustering...")
                    result = self.analysis.vessel_behavior_clustering(
                        vessel_types=selected_types if selected_types else None,
                        n_clusters=n_clusters
                    )
                    progress.close()
                    if result:
                        self.status_var.set(f"Clustering complete: {result}")
                        messagebox.showinfo("Success", f"Vessel clustering complete:\n{result}")
                    else:
                        self.status_var.set("Clustering failed")
                        messagebox.showerror("Error", "Failed to perform vessel clustering. Check logs for details.")
                except Exception as e:
                    progress.close()
                    self.status_var.set("Clustering failed")
                    logger.error(f"Error in clustering: {e}")
                    logger.error(traceback.format_exc())
                    messagebox.showerror("Error", f"Clustering failed: {str(e)}")
            
            button_frame = ttk.Frame(dialog)
            button_frame.pack(pady=10)
            ttk.Button(button_frame, text="Perform Clustering", command=perform_clustering).pack(side=tk.LEFT, padx=5)
            ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
            
        except Exception as e:
            logger.error(f"Error creating clustering dialog: {e}")
            logger.error(traceback.format_exc())
            messagebox.showerror("Error", f"Error creating clustering dialog: {e}")
            dialog.destroy()
    
    def _anomaly_frequency_analysis(self):
        progress = ProgressDialog(self.window, "Analyzing Frequency", "Analyzing anomaly frequency...")
        try:
            self.status_var.set("Analyzing anomaly frequency...")
            result = self.analysis.anomaly_frequency_analysis()
            progress.close()
            if result:
                self.status_var.set(f"Analysis complete: {result}")
                messagebox.showinfo("Success", f"Anomaly frequency analysis complete:\n{result}")
            else:
                self.status_var.set("Analysis failed")
                messagebox.showerror("Error", "Failed to perform anomaly frequency analysis. Check logs for details.")
        except Exception as e:
            progress.close()
            self.status_var.set("Analysis failed")
            logger.error(f"Error in frequency analysis: {e}")
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
    
    def _create_custom_chart_dialog(self):
        """Create dialog for custom chart creation"""
        dialog = tk.Toplevel(self.window)
        dialog.title("Create Custom Chart")
        dialog.geometry("700x650")
        
        try:
            df = self.analysis.load_cached_data()
            anomaly_df = self.analysis.load_anomaly_data()
            
            # Get available columns from both datasets
            df_columns = sorted([col for col in df.columns if col not in ['MMSI', 'LAT', 'LON']]) if not df.empty else []
            anomaly_columns = sorted([col for col in anomaly_df.columns if col not in ['MMSI', 'LAT', 'LON']]) if not anomaly_df.empty else []
            all_columns = sorted(list(set(df_columns + anomaly_columns)))
            
            # Chart type selection
            ttk.Label(dialog, text="Chart Type:", font=("Arial", 10, "bold")).pack(pady=5, anchor=tk.W, padx=10)
            chart_type_var = tk.StringVar(value='bar')
            chart_types = [
                ('Bar Chart', 'bar'),
                ('Scatter Plot', 'scatter'),
                ('Line Chart', 'line'),
                ('Pie Chart', 'pie'),
                ('Stacked Bar Chart', 'stacked_bar'),
                ('Timeline', 'timeline'),
                ('Histogram', 'histogram'),
                ('Box Plot', 'box')
            ]
            
            chart_frame = ttk.Frame(dialog)
            chart_frame.pack(fill=tk.X, padx=10, pady=5)
            # Arrange chart types in a grid for better layout
            row = 0
            col = 0
            for text, value in chart_types:
                ttk.Radiobutton(chart_frame, text=text, variable=chart_type_var, value=value).grid(row=row, column=col, padx=5, pady=2, sticky=tk.W)
                col += 1
                if col > 3:  # 4 columns per row
                    col = 0
                    row += 1
            
            # X-axis column
            ttk.Label(dialog, text="X-Axis Column:", font=("Arial", 10, "bold")).pack(pady=(10, 5), anchor=tk.W, padx=10)
            x_column_var = tk.StringVar()
            x_combo = ttk.Combobox(dialog, textvariable=x_column_var, values=all_columns, width=40, state='readonly')
            x_combo.pack(padx=10, pady=5, anchor=tk.W)
            
            # Y-axis column (for charts that need it)
            ttk.Label(dialog, text="Y-Axis Column (optional):", font=("Arial", 10, "bold")).pack(pady=(10, 5), anchor=tk.W, padx=10)
            y_column_var = tk.StringVar()
            y_combo = ttk.Combobox(dialog, textvariable=y_column_var, values=all_columns, width=40, state='readonly')
            y_combo.pack(padx=10, pady=5, anchor=tk.W)
            
            # Color/Group by column
            ttk.Label(dialog, text="Color/Group By Column (optional):", font=("Arial", 10, "bold")).pack(pady=(10, 5), anchor=tk.W, padx=10)
            color_column_var = tk.StringVar()
            color_combo = ttk.Combobox(dialog, textvariable=color_column_var, values=all_columns, width=40, state='readonly')
            color_combo.pack(padx=10, pady=5, anchor=tk.W)
            
            # Group by column
            ttk.Label(dialog, text="Group By Column (optional):", font=("Arial", 10, "bold")).pack(pady=(10, 5), anchor=tk.W, padx=10)
            group_by_var = tk.StringVar()
            group_combo = ttk.Combobox(dialog, textvariable=group_by_var, values=all_columns, width=40, state='readonly')
            group_combo.pack(padx=10, pady=5, anchor=tk.W)
            
            # Aggregation method
            ttk.Label(dialog, text="Aggregation (if grouping):", font=("Arial", 10, "bold")).pack(pady=(10, 5), anchor=tk.W, padx=10)
            aggregation_var = tk.StringVar(value='count')
            agg_frame = ttk.Frame(dialog)
            agg_frame.pack(fill=tk.X, padx=10, pady=5)
            for agg in ['count', 'sum', 'mean', 'max', 'min']:
                ttk.Radiobutton(agg_frame, text=agg.capitalize(), variable=aggregation_var, value=agg).pack(side=tk.LEFT, padx=5)
            
            # Chart title
            ttk.Label(dialog, text="Chart Title (optional):", font=("Arial", 10, "bold")).pack(pady=(10, 5), anchor=tk.W, padx=10)
            title_var = tk.StringVar()
            title_entry = ttk.Entry(dialog, textvariable=title_var, width=50)
            title_entry.pack(padx=10, pady=5, anchor=tk.W)
            
            def update_requirements():
                """Update UI based on selected chart type"""
                chart_type = chart_type_var.get()
                
                # Enable/disable fields based on chart type
                if chart_type in ['scatter', 'line', 'box', 'stacked_bar']:
                    y_combo.config(state='readonly')
                else:
                    y_combo.config(state='disabled')
                    y_column_var.set('')
                
                if chart_type == 'timeline':
                    x_combo.config(state='disabled')
                    x_column_var.set('')
                else:
                    x_combo.config(state='readonly')
            
            chart_type_var.trace('w', lambda *args: update_requirements())
            update_requirements()
            
            def create_chart():
                chart_type = chart_type_var.get()
                x_column = x_column_var.get() if x_column_var.get() else None
                y_column = y_column_var.get() if y_column_var.get() else None
                color_column = color_column_var.get() if color_column_var.get() else None
                group_by = group_by_var.get() if group_by_var.get() else None
                aggregation = aggregation_var.get()
                title = title_var.get() if title_var.get() else None
                
                # Validation
                if chart_type in ['scatter', 'line', 'box', 'stacked_bar'] and not y_column:
                    messagebox.showerror("Error", f"{chart_type.capitalize()} chart requires a Y-axis column")
                    return
                
                if chart_type != 'timeline' and not x_column:
                    messagebox.showerror("Error", "Please select an X-axis column")
                    return
                
                if chart_type == 'stacked_bar' and not color_column:
                    messagebox.showerror("Error", "Stacked bar chart requires a color/group by column")
                    return
                
                dialog.destroy()
                progress = ProgressDialog(self.window, "Creating Chart", f"Creating {chart_type} chart...")
                try:
                    self.status_var.set(f"Creating {chart_type} chart...")
                    result = self.analysis.create_custom_chart(
                        chart_type=chart_type,
                        x_column=x_column,
                        y_column=y_column,
                        color_column=color_column,
                        group_by=group_by,
                        aggregation=aggregation,
                        title=title
                    )
                    progress.close()
                    if result:
                        self.status_var.set(f"Chart created: {result}")
                        messagebox.showinfo("Success", f"Chart created:\n{result}")
                    else:
                        self.status_var.set("Chart creation failed")
                        messagebox.showerror("Error", "Failed to create chart. Check logs for details.")
                except Exception as e:
                    progress.close()
                    self.status_var.set("Chart creation failed")
                    logger.error(f"Error creating chart: {e}")
                    logger.error(traceback.format_exc())
                    messagebox.showerror("Error", f"Chart creation failed: {str(e)}")
            
            button_frame = ttk.Frame(dialog)
            button_frame.pack(pady=20)
            ttk.Button(button_frame, text="Create Chart", command=create_chart).pack(side=tk.LEFT, padx=5)
            ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
            
        except Exception as e:
            logger.error(f"Error creating chart dialog: {e}")
            logger.error(traceback.format_exc())
            messagebox.showerror("Error", f"Error creating chart dialog: {e}")
            dialog.destroy()
    
    def _full_spectrum_map_dialog(self):
        dialog = tk.Toplevel(self.window)
        dialog.title("Full Spectrum Map Options")
        
        show_pins_var = tk.BooleanVar(value=True)
        show_heatmap_var = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(dialog, text="Show Anomaly Pins", variable=show_pins_var).pack(pady=5)
        ttk.Checkbutton(dialog, text="Show Heatmap Overlay", variable=show_heatmap_var).pack(pady=5)
        
        def create_map():
            dialog.destroy()
            progress = ProgressDialog(self.window, "Creating Map", "Creating full spectrum map...")
            try:
                self.status_var.set("Creating full spectrum map...")
                result = self.analysis.create_full_spectrum_map(
                    show_pins=show_pins_var.get(),
                    show_heatmap=show_heatmap_var.get()
                )
                progress.close()
                if result:
                    self.status_var.set(f"Map created: {result}")
                    messagebox.showinfo("Success", f"Map created:\n{result}")
                else:
                    self.status_var.set("Map creation failed")
                    messagebox.showerror("Error", "Failed to create map. Check logs for details.")
            except Exception as e:
                progress.close()
                self.status_var.set("Map creation failed")
                logger.error(f"Error creating map: {e}")
                messagebox.showerror("Error", f"Map creation failed: {str(e)}")
        
        ttk.Button(dialog, text="Create Map", command=create_map).pack(pady=10)
    
    def _create_vessel_map_dialog(self):
        mmsi_str = self.vessel_mmsi_var.get()
        if not mmsi_str:
            messagebox.showerror("Error", "Please enter an MMSI number")
            return
        
        try:
            mmsi = int(mmsi_str)
        except ValueError:
            messagebox.showerror("Error", "MMSI must be a number")
            return
        
        dialog = tk.Toplevel(self.window)
        dialog.title("Vessel Map Type")
        
        map_type_var = tk.StringVar(value='path')
        ttk.Radiobutton(dialog, text="Path Map", variable=map_type_var, value='path').pack(pady=5)
        ttk.Radiobutton(dialog, text="Anomaly Map", variable=map_type_var, value='anomaly').pack(pady=5)
        ttk.Radiobutton(dialog, text="Heatmap", variable=map_type_var, value='heatmap').pack(pady=5)
        
        def create_map():
            dialog.destroy()
            progress = ProgressDialog(self.window, "Creating Map", f"Creating {map_type_var.get()} map...")
            try:
                self.status_var.set(f"Creating {map_type_var.get()} map for vessel {mmsi}...")
                result = self.analysis.create_vessel_map(mmsi, map_type_var.get())
                progress.close()
                if result:
                    self.status_var.set(f"Map created: {result}")
                    messagebox.showinfo("Success", f"Map created:\n{result}")
                else:
                    self.status_var.set("Map creation failed")
                    messagebox.showerror("Error", "Failed to create map. Check logs for details.")
            except Exception as e:
                progress.close()
                self.status_var.set("Map creation failed")
                logger.error(f"Error creating map: {e}")
                messagebox.showerror("Error", f"Map creation failed: {str(e)}")
        
        ttk.Button(dialog, text="Create Map", command=create_map).pack(pady=10)
    
    def _populate_top_vessels(self):
        """Populate the top vessels listbox"""
        try:
            anomaly_df = self.analysis.load_anomaly_data()
            if not anomaly_df.empty and 'AnomalyType' in anomaly_df.columns:
                for anomaly_type in sorted(anomaly_df['AnomalyType'].unique()):
                    top_vessels = self.analysis.get_top_vessels_by_anomaly(anomaly_type, 10)
                    if not top_vessels.empty:
                        # Map data name to GUI name for display
                        gui_name = map_anomaly_type_data_to_gui(anomaly_type)
                        self.top_vessels_listbox.insert(tk.END, f"{gui_name}:")
                        for _, row in top_vessels.iterrows():
                            self.top_vessels_listbox.insert(tk.END, f"  MMSI {row['MMSI']}: {row['Anomaly_Count']} anomalies")
        except Exception as e:
            logger.error(f"Error populating top vessels: {e}")
    
    def _show_listbox_menu(self, event):
        """Show context menu for listbox"""
        try:
            selection = self.top_vessels_listbox.curselection()
            if selection:
                self.top_vessels_menu.post(event.x_root, event.y_root)
        except Exception as e:
            logger.error(f"Error showing listbox menu: {e}")
    
    def _copy_mmsi_from_listbox(self):
        """Copy MMSI from selected listbox item to clipboard"""
        try:
            selection = self.top_vessels_listbox.curselection()
            if selection:
                item_text = self.top_vessels_listbox.get(selection[0])
                # Extract MMSI from text like "  MMSI 123456789: 5 anomalies"
                match = re.search(r'MMSI\s+(\d+)', item_text)
                if match:
                    mmsi = match.group(1)
                    self.window.clipboard_clear()
                    self.window.clipboard_append(mmsi)
                    self.status_var.set(f"Copied MMSI {mmsi} to clipboard")
                else:
                    messagebox.showwarning("Warning", "Could not extract MMSI from selected item")
        except Exception as e:
            logger.error(f"Error copying MMSI: {e}")
            messagebox.showerror("Error", f"Failed to copy MMSI: {e}")
    
    def _create_filtered_map_dialog(self):
        """Create dialog for filtered map creation"""
        dialog = tk.Toplevel(self.window)
        dialog.title("Create Filtered Map")
        dialog.geometry("500x600")
        
        try:
            df = self.analysis.load_cached_data()
            anomaly_df = self.analysis.load_anomaly_data()
            
            # Get available types from data - convert to int for proper comparison
            if 'VesselType' in df.columns:
                # Convert to int, handling NaN values
                available_vessel_types_raw = df['VesselType'].dropna().unique()
                available_vessel_types = sorted([int(vt) for vt in available_vessel_types_raw if pd.notna(vt)])
            else:
                available_vessel_types = []
            
            available_anomaly_types_data = sorted(anomaly_df['AnomalyType'].unique().tolist()) if not anomaly_df.empty and 'AnomalyType' in anomaly_df.columns else []
            available_anomaly_types_gui = [map_anomaly_type_data_to_gui(at) for at in available_anomaly_types_data]
            
            # Map type selection
            ttk.Label(dialog, text="Map Type:", font=("Arial", 10, "bold")).pack(pady=5)
            map_type_var = tk.StringVar(value='path')
            ttk.Radiobutton(dialog, text="Path Map", variable=map_type_var, value='path').pack(anchor=tk.W, padx=20)
            ttk.Radiobutton(dialog, text="Anomaly Map", variable=map_type_var, value='anomaly').pack(anchor=tk.W, padx=20)
            ttk.Radiobutton(dialog, text="Heatmap", variable=map_type_var, value='heatmap').pack(anchor=tk.W, padx=20)
            
            # Vessel MMSI (optional)
            ttk.Label(dialog, text="Vessel MMSI (optional):", font=("Arial", 10, "bold")).pack(pady=5)
            vessel_mmsi_var = tk.StringVar()
            ttk.Entry(dialog, textvariable=vessel_mmsi_var, width=20).pack(pady=5)
            
            # Vessel types selection
            ttk.Label(dialog, text="Vessel Types (optional):", font=("Arial", 10, "bold")).pack(pady=5)
            vessel_frame = ttk.Frame(dialog)
            vessel_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
            
            vessel_vars = {}
            vessel_checkboxes = []
            for vtype in available_vessel_types:
                var = tk.BooleanVar()
                vessel_vars[vtype] = var
                vessel_name = get_vessel_type_name(vtype)
                display_text = f"Type {vtype}: {vessel_name}"
                cb = ttk.Checkbutton(vessel_frame, text=display_text, variable=var)
                cb.pack(anchor=tk.W)
                vessel_checkboxes.append((vtype, var))
            
            # Anomaly types selection
            ttk.Label(dialog, text="Anomaly Types (optional):", font=("Arial", 10, "bold")).pack(pady=5)
            anomaly_frame = ttk.Frame(dialog)
            anomaly_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
            
            anomaly_vars = {}
            anomaly_checkboxes = []
            for atype in available_anomaly_types_gui:
                var = tk.BooleanVar()
                anomaly_vars[atype] = var
                cb = ttk.Checkbutton(anomaly_frame, text=atype, variable=var)
                cb.pack(anchor=tk.W)
                anomaly_checkboxes.append((atype, var))
            
            def create_map():
                vessel_mmsi = None
                mmsi_str = vessel_mmsi_var.get().strip()
                if mmsi_str:
                    try:
                        vessel_mmsi = int(mmsi_str)
                    except ValueError:
                        messagebox.showerror("Error", "MMSI must be a number")
                        return
                
                selected_vessel_types = [vtype for vtype, var in vessel_checkboxes if var.get()]
                selected_anomaly_types_gui = [atype for atype, var in anomaly_checkboxes if var.get()]
                selected_anomaly_types = [map_anomaly_type_gui_to_data(at) for at in selected_anomaly_types_gui]
                
                if not vessel_mmsi and not selected_vessel_types and not selected_anomaly_types:
                    messagebox.showerror("Error", "Please select at least one filter (vessel MMSI, vessel types, or anomaly types)")
                    return
                
                dialog.destroy()
                progress = ProgressDialog(self.window, "Creating Map", "Creating filtered map...")
                try:
                    self.status_var.set("Creating filtered map...")
                    result = self.analysis.create_filtered_map(
                        map_type=map_type_var.get(),
                        vessel_types=selected_vessel_types if selected_vessel_types else None,
                        anomaly_types=selected_anomaly_types if selected_anomaly_types else None,
                        vessel_mmsi=vessel_mmsi
                    )
                    progress.close()
                    if result:
                        self.status_var.set(f"Map created: {result}")
                        messagebox.showinfo("Success", f"Filtered map created:\n{result}")
                    else:
                        self.status_var.set("Map creation failed")
                        messagebox.showerror("Error", "Failed to create filtered map. Check logs for details.")
                except Exception as e:
                    progress.close()
                    self.status_var.set("Map creation failed")
                    logger.error(f"Error creating filtered map: {e}")
                    messagebox.showerror("Error", f"Map creation failed: {str(e)}")
            
            ttk.Button(dialog, text="Create Map", command=create_map).pack(pady=10)
            
        except Exception as e:
            logger.error(f"Error creating filtered map dialog: {e}")
            logger.error(traceback.format_exc())
            messagebox.showerror("Error", f"Error creating filtered map dialog: {e}")
            dialog.destroy()
    
    def _extended_time_analysis(self):
        mmsi_str = self.extended_mmsi_var.get()
        
        # Get dates from DateEntry if available, otherwise from text entry
        if TKCALENDAR_AVAILABLE and self.extended_start_picker is not None:
            start_date = self.extended_start_picker.get_date().strftime('%Y-%m-%d')
            end_date = self.extended_end_picker.get_date().strftime('%Y-%m-%d')
        else:
            start_date = self.extended_start_var.get() if self.extended_start_var else None
            end_date = self.extended_end_var.get() if self.extended_end_var else None
        
        if not mmsi_str:
            messagebox.showerror("Error", "Please enter an MMSI number")
            return
        
        if not start_date or not end_date:
            messagebox.showerror("Error", "Please select both start and end dates")
            return
        
        try:
            mmsi = int(mmsi_str)
        except ValueError:
            messagebox.showerror("Error", "MMSI must be a number")
            return
        
        progress = ProgressDialog(self.window, "Extended Analysis", f"Analyzing vessel {mmsi}...")
        try:
            self.status_var.set(f"Performing extended analysis for vessel {mmsi}...")
            result = self.analysis.extended_time_analysis(mmsi, start_date, end_date)
            progress.close()
            if result:
                self.status_var.set(f"Analysis complete: {result}")
                messagebox.showinfo("Success", f"Extended analysis complete:\n{result}")
            else:
                self.status_var.set("Analysis failed")
                messagebox.showerror("Error", "Failed to perform extended analysis. Check logs for details.")
        except Exception as e:
            progress.close()
            self.status_var.set("Analysis failed")
            logger.error(f"Error in extended analysis: {e}")
            logger.error(traceback.format_exc())
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")


# ============================================================================
# CLI INTERFACE FUNCTIONS
# ============================================================================

def cli_export_full_dataset(output_dir=None, config_path='config.ini'):
    """CLI function to export full dataset"""
    resolved_config_path = get_config_path(config_path)
    analysis = AdvancedAnalysis(None, output_dir, resolved_config_path)
    return analysis.export_full_dataset()

def cli_generate_summary_report(output_dir=None, config_path='config.ini'):
    """CLI function to generate summary report"""
    resolved_config_path = get_config_path(config_path)
    analysis = AdvancedAnalysis(None, output_dir, resolved_config_path)
    return analysis.generate_summary_report()
