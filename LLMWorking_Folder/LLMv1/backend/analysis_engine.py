"""
AIS Analysis Engine with GPU Support
Wraps SFD.py functionality with GPU acceleration
"""
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import uuid
import asyncio

# Import GPU support
from .gpu_support import (
    GPU_AVAILABLE, GPU_TYPE, GPU_BACKEND,
    convert_to_gpu_dataframe, convert_to_cpu_dataframe,
    get_gpu_info, get_processing_backend
)

logger = logging.getLogger(__name__)

# Try to import GPU libraries for haversine calculations
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

def normalize_angle_difference(angle_diff):
    """
    Normalize angle difference to be between -180 and 180 degrees.
    
    Args:
        angle_diff (float): The angle difference in degrees
        
    Returns:
        float: The normalized angle difference
    """
    while angle_diff > 180:
        angle_diff -= 360
    while angle_diff < -180:
        angle_diff += 360
    return angle_diff

def haversine_vectorized(df, use_gpu=None):
    """
    Vectorized implementation of the Haversine formula with GPU support when available.
    
    Args:
        df (DataFrame): DataFrame containing LAT1, LON1, LAT2, LON2 columns
        use_gpu (bool, optional): Whether to use GPU acceleration. If None, uses GPU_AVAILABLE.
        
    Returns:
        Series: Distances in nautical miles
    """
    # Earth radius in nautical miles
    r = 3440.1
    
    # Determine if we should use GPU
    should_use_gpu = (use_gpu is not False) and GPU_AVAILABLE and CUPY_AVAILABLE
    
    if should_use_gpu and cp is not None:
        # GPU implementation with cupy
        lat1_rad = cp.radians(cp.asarray(df['LAT1'].values))
        lon1_rad = cp.radians(cp.asarray(df['LON1'].values))
        lat2_rad = cp.radians(cp.asarray(df['LAT2'].values))
        lon2_rad = cp.radians(cp.asarray(df['LON2'].values))
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = cp.sin(dlat/2)**2 + cp.cos(lat1_rad) * cp.cos(lat2_rad) * cp.sin(dlon/2)**2
        c = 2 * cp.arcsin(cp.sqrt(a))
        
        result = cp.asnumpy(c * r)
        return pd.Series(result, index=df.index)
    else:
        # CPU implementation with numpy
        lat1_rad = np.radians(df['LAT1'])
        lon1_rad = np.radians(df['LON1'])
        lat2_rad = np.radians(df['LAT2'])
        lon2_rad = np.radians(df['LON2'])
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return pd.Series(c * r, index=df.index)


class AISAnalysisEngine:
    """
    AIS analysis engine with GPU acceleration support
    """
    
    def __init__(self, data_connector):
        self.data_connector = data_connector
        self.gpu_available = GPU_AVAILABLE
        self.gpu_type = GPU_TYPE
        self.gpu_backend = GPU_BACKEND
        
        # Log GPU status
        if self.gpu_available:
            gpu_info = get_gpu_info()
            logger.info(f"Analysis engine initialized with {gpu_info['type']} {gpu_info['backend']} GPU acceleration")
            logger.info(f"Found {gpu_info['device_count']} GPU device(s): {', '.join(gpu_info['device_names'])}")
        else:
            logger.info("Analysis engine initialized with CPU processing")
    
    async def run_analysis(self, 
                          start_date: str, 
                          end_date: str,
                          geographic_zone: Optional[Dict[str, Any]] = None,
                          anomaly_types: List[str] = None,
                          mmsi_filter: Optional[List[str]] = None,
                          vessel_types: Optional[List[str]] = None,
                          progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Run anomaly detection analysis with GPU acceleration
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            geographic_zone: Optional geographic zone to filter
            anomaly_types: List of anomaly types to detect
            mmsi_filter: Optional list of MMSIs to filter
            progress_callback: Optional callback function(status_message, stage) for progress updates
        
        Returns:
            Analysis results with anomalies and statistics
        """
        analysis_id = str(uuid.uuid4())
        logger.info(f"Starting analysis {analysis_id} from {start_date} to {end_date}")
        
        async def _progress(stage: str, message: str, data: Optional[Dict] = None):
            """Helper to send progress updates"""
            if progress_callback:
                try:
                    # Progress callback is async, await it
                    if asyncio.iscoroutinefunction(progress_callback):
                        await progress_callback(stage, message, data)
                    else:
                        progress_callback(stage, message, data)
                except Exception as e:
                    logger.warning(f"Progress callback error: {e}")
        
        try:
            # Stage 1: Finding files
            await _progress("finding_files", f"[SEARCHING] Searching for data files from {start_date} to {end_date}...")
            
            # Get available dates to show file count
            available_dates = self.data_connector.get_available_dates()
            # Convert start_date and end_date to datetime for comparison
            from datetime import datetime
            start_dt = datetime.strptime(start_date, '%Y-%m-%d') if isinstance(start_date, str) else start_date
            end_dt = datetime.strptime(end_date, '%Y-%m-%d') if isinstance(end_date, str) else end_date
            
            date_range_dates = [
                d for d in available_dates 
                if start_dt <= d <= end_dt
            ]
            file_count = len(date_range_dates)
            
            await _progress("files_found", f"[FOUND] Found {file_count} data file(s) for the requested date range", {
                "file_count": file_count,
                "date_range": {"start": start_date, "end": end_date}
            })
            
            # Stage 2: Loading data (uses cache if available for much faster loading)
            await _progress("loading_data", f"[LOADING] Loading data from {file_count} file(s)...")
            df = self.data_connector.load_date_range(
                start_date, 
                end_date,
                vessel_types=vessel_types,  # Pass for cache optimization
                mmsi_filter=mmsi_filter  # Pass for cache optimization
            )
            
            if df.empty:
                await _progress("error", "[ERROR] No data found for specified date range")
                return {
                    'success': False,
                    'error': 'No data found for specified date range',
                    'analysis_id': analysis_id,
                    'anomalies': []
                }
            
            total_records = len(df)
            logger.info(f"Loaded {total_records} AIS records")
            
            # Stage 3: Data loaded
            await _progress("data_loaded", f"[LOADED] Loaded {total_records:,} AIS records", {
                "total_records": total_records
            })
            
            # Stage 4: Counting vessels
            unique_vessels = df['MMSI'].nunique() if 'MMSI' in df.columns else 0
            await _progress("counting_vessels", f"[COUNTING] Analyzing vessel data: Found {unique_vessels:,} unique vessels", {
                "unique_vessels": unique_vessels,
                "total_records": total_records
            })
            
            # Stage 5: Creating dataframe
            await _progress("creating_dataframe", f"[CREATING] Creating analysis dataframe...")
            
            # Convert to GPU if available
            if self.gpu_available:
                logger.info(f"Converting data to GPU ({self.gpu_type} {self.gpu_backend})")
                await _progress("gpu_conversion", f"[GPU] Converting to GPU ({self.gpu_type} {self.gpu_backend}) for faster processing...")
                df_gpu = convert_to_gpu_dataframe(df)
            else:
                df_gpu = df
            
            # Apply vessel type filter if provided
            if vessel_types and 'VesselType' in df_gpu.columns:
                from vessel_types import VESSEL_TYPE_CATEGORIES
                type_codes = []
                for vtype in vessel_types:
                    if vtype in VESSEL_TYPE_CATEGORIES:
                        type_codes.extend(VESSEL_TYPE_CATEGORIES[vtype])
                
                if type_codes:
                    original_count = len(df_gpu)
                    df_gpu = df_gpu[df_gpu['VesselType'].isin(type_codes)]
                    filtered_vessels = df_gpu['MMSI'].nunique() if 'MMSI' in df_gpu.columns else 0
                    await _progress("vessel_filter", f"[FILTERED] Filtered to {filtered_vessels:,} vessels of selected type(s): {', '.join(vessel_types)}", {
                        "vessel_types": vessel_types,
                        "filtered_vessels": filtered_vessels,
                        "filtered_records": len(df_gpu),
                        "original_records": original_count
                    })
            
            # Apply MMSI filter
            if mmsi_filter:
                df_gpu = df_gpu[df_gpu['MMSI'].isin(mmsi_filter)]
                logger.info(f"Filtered to {len(df_gpu)} records for {len(mmsi_filter)} vessels")
                await _progress("mmsi_filter", f"[FILTERED] Filtered to {len(mmsi_filter)} specific vessel(s)", {
                    "mmsi_count": len(mmsi_filter),
                    "filtered_records": len(df_gpu)
                })
            
            # Apply geographic filter
            if geographic_zone:
                from geographic_tools import GeographicZoneManager
                await _progress("geographic_filter", f"[FILTERING] Applying geographic zone filter...")
                df_gpu = GeographicZoneManager.filter_dataframe_by_zone(df_gpu, geographic_zone)
                logger.info(f"Filtered to {len(df_gpu)} records in geographic zone")
                await _progress("geographic_filtered", f"[FILTERED] Geographic filter applied: {len(df_gpu):,} records in zone", {
                    "filtered_records": len(df_gpu)
                })
            
            # Stage 6: Running anomaly detection
            await _progress("detecting_anomalies", f"[ANALYZING] Running anomaly detection analysis...")
            
            # Run anomaly detection (pass progress callback for updates during long operations)
            anomalies = await self._detect_anomalies(df_gpu, anomaly_types, progress_callback=_progress)
            
            # Convert back to pandas if needed
            if self.gpu_available:
                await _progress("converting_cpu", f"[CONVERTING] Converting results back to CPU format...")
                anomalies = convert_to_cpu_dataframe(anomalies)
            
            anomaly_count = len(anomalies) if not anomalies.empty else 0
            await _progress("anomalies_detected", f"[DETECTED] Anomaly detection complete: Found {anomaly_count:,} anomalies", {
                "anomaly_count": anomaly_count
            })
            
            # Stage 7: Generating statistics
            await _progress("generating_statistics", f"[STATS] Generating analysis statistics and reports...")
            stats = self._generate_statistics(anomalies, df_gpu)
            
            # Stage 8: Preparing final results
            await _progress("preparing_results", f"[PREPARING] Preparing final analysis results...")
            
            result = {
                'success': True,
                'analysis_id': analysis_id,
                'total_records': len(df),
                'filtered_records': len(df_gpu),
                'anomalies_found': len(anomalies),
                'anomalies': anomalies.to_dict('records') if not anomalies.empty else [],
                'statistics': stats,
                'processing_backend': f"{self.gpu_type} {self.gpu_backend}" if self.gpu_available else "CPU",
                'date_range': {
                    'start': start_date,
                    'end': end_date
                },
                'filters': {
                    'geographic_zone': geographic_zone is not None,
                    'mmsi_filter': mmsi_filter is not None,
                    'anomaly_types': anomaly_types
                }
            }
            
            # Stage 9: Complete
            await _progress("complete", f"[COMPLETE] Analysis complete! Found {anomaly_count:,} anomalies from {unique_vessels:,} vessels", {
                "anomaly_count": anomaly_count,
                "unique_vessels": unique_vessels,
                "total_records": total_records
            })
            
            logger.info(f"Analysis {analysis_id} completed: {len(anomalies)} anomalies found")
            return result
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'analysis_id': analysis_id,
                'anomalies': []
            }
    
    async def _detect_anomalies(self, df, anomaly_types: Optional[List[str]], progress_callback: Optional[callable] = None) -> pd.DataFrame:
        """
        Detect anomalies using SFD.py logic
        
        Implements:
        - AIS beacon off/on detection
        - Speed anomalies (position jumps)
        - COG/Heading inconsistencies
        - Loitering detection
        - Rendezvous detection
        """
        if df.empty or 'BaseDateTime' not in df.columns:
            logger.warning("DataFrame is empty or missing BaseDateTime column")
            return pd.DataFrame()
        
        # Convert to CPU DataFrame if needed
        if hasattr(df, 'to_pandas'):
            df = df.to_pandas()
        
        # Ensure BaseDateTime is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['BaseDateTime']):
            df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'])
        
        # Configuration defaults (matching SFD.py)
        config = {
            'BEACON_TIME_THRESHOLD_HOURS': 6.0,
            'SPEED_THRESHOLD': 102.0,  # knots
            'TIME_DIFF_THRESHOLD_MIN': 240,  # 4 hours
            'COG_HEADING_MAX_DIFF': 45.0,
            'MIN_SPEED_FOR_COG_CHECK': 10.0,
            'LOITERING_RADIUS_NM': 5.0,
            'LOITERING_DURATION_HOURS': 24.0,
            'RENDEZVOUS_PROXIMITY_NM': 0.5,
            'RENDEZVOUS_DURATION_MINUTES': 30,
            'ais_beacon_on': True,
            'ais_beacon_off': True,
            'excessive_travel_distance_fast': True,
            'cog-heading_inconsistency': True,
            'loitering': True,
            'rendezvous': True,
            'USE_GPU': self.gpu_available
        }
        
        # Filter anomaly types if specified
        if anomaly_types:
            for key in ['ais_beacon_on', 'ais_beacon_off', 'excessive_travel_distance_fast', 
                       'cog-heading_inconsistency', 'loitering', 'rendezvous']:
                if key not in anomaly_types:
                    config[key] = False
        
        all_anomalies = []
        
        # Split DataFrame by date for day-by-day processing
        df['Date'] = df['BaseDateTime'].dt.date
        dates = sorted(df['Date'].unique())
        
        logger.info(f"Processing {len(dates)} day(s) for anomaly detection")
        
        # Send progress update
        if progress_callback:
            async def _send_progress(stage, message, data=None):
                if asyncio.iscoroutinefunction(progress_callback):
                    await progress_callback(stage, message, data)
                else:
                    progress_callback(stage, message, data)
        else:
            async def _send_progress(stage, message, data=None):
                pass
        
        await _send_progress("anomaly_detection_start", f"[ANALYZING] Starting anomaly detection for {len(dates)} day(s)...")
        
        df_previous_day = None
        processed_first_day = False
        
        for i, current_date in enumerate(dates):
            df_current_day = df[df['Date'] == current_date].copy()
            
            if df_current_day.empty:
                continue
            
            # Send progress update for each day
            await _send_progress("processing_day", f"[PROCESSING] Analyzing day {i+1}/{len(dates)}: {current_date}", {
                "day": i+1,
                "total_days": len(dates),
                "date": str(current_date)
            })
            
            # Skip first day (no previous day to compare)
            if not processed_first_day:
                df_previous_day = df_current_day
                processed_first_day = True
                logger.info(f"Loaded initial day: {current_date}. No comparisons possible yet.")
                await _send_progress("first_day_loaded", f"[LOADED] Initial day loaded: {current_date}")
                continue
            
            # Group by MMSI
            prev_grouped = df_previous_day.groupby('MMSI')
            current_grouped = df_current_day.groupby('MMSI')
            
            report_date = current_date.strftime('%Y-%m-%d')
            
            # Send progress update for anomaly type detection
            await _send_progress("detecting_beacon", f"[DETECTING] Checking AIS beacon anomalies for {current_date}...")
            
            # 1. AIS Beacon on/off anomalies
            if config.get('ais_beacon_on', True) or config.get('ais_beacon_off', True):
                beacon_time_threshold = config['BEACON_TIME_THRESHOLD_HOURS'] * 60  # Convert to minutes
                
                # Beacon On: Vessels that appeared in current day but not in previous day
                if config.get('ais_beacon_on', True):
                    beacon_on_mmsi = set(df_current_day['MMSI'].unique()) - set(df_previous_day['MMSI'].unique())
                    
                    # Process in chunks to yield control periodically
                    beacon_on_list = list(beacon_on_mmsi)
                    chunk_size = 100
                    for chunk_idx in range(0, len(beacon_on_list), chunk_size):
                        chunk = beacon_on_list[chunk_idx:chunk_idx + chunk_size]
                        await asyncio.sleep(0)  # Yield control
                        
                        for mmsi in chunk:
                            vessel_curr = current_grouped.get_group(mmsi).copy()
                            vessel_curr = vessel_curr.sort_values('BaseDateTime')
                            
                            if len(vessel_curr) > 0:
                                first_appearance = vessel_curr.iloc[0]['BaseDateTime']
                                day_start = pd.Timestamp(current_date).replace(hour=0, minute=0, second=0)
                                time_since_day_start = (first_appearance - day_start).total_seconds() / 60
                                
                                if time_since_day_start >= beacon_time_threshold:
                                    first_pos = vessel_curr.iloc[0].copy()
                                    first_pos['AnomalyType'] = 'AIS_Beacon_On'
                                    first_pos['SpeedAnomaly'] = False
                                    first_pos['PositionAnomaly'] = True
                                    first_pos['CourseAnomaly'] = False
                                    first_pos['BeaconAnomaly'] = True
                                    first_pos['BeaconGapMinutes'] = time_since_day_start
                                    first_pos['ReportDate'] = report_date
                                    all_anomalies.append(first_pos.to_dict())
                
                # Beacon Off: Vessels that disappeared in current day but were in previous day
                if config.get('ais_beacon_off', True):
                    beacon_off_mmsi = set(df_previous_day['MMSI'].unique()) - set(df_current_day['MMSI'].unique())
                    
                    # Process in chunks to yield control periodically
                    beacon_off_list = list(beacon_off_mmsi)
                    chunk_size = 100
                    for chunk_idx in range(0, len(beacon_off_list), chunk_size):
                        chunk = beacon_off_list[chunk_idx:chunk_idx + chunk_size]
                        await asyncio.sleep(0)  # Yield control
                        
                        for mmsi in chunk:
                            vessel_prev = prev_grouped.get_group(mmsi).copy()
                            vessel_prev = vessel_prev.sort_values('BaseDateTime')
                            
                            if len(vessel_prev) > 0:
                                last_appearance = vessel_prev.iloc[-1]['BaseDateTime']
                                day_end = pd.Timestamp(dates[i-1]).replace(hour=23, minute=59, second=59)
                                time_to_day_end = (day_end - last_appearance).total_seconds() / 60
                                
                                if time_to_day_end >= beacon_time_threshold:
                                    last_pos = vessel_prev.iloc[-1].copy()
                                    last_pos['AnomalyType'] = 'AIS_Beacon_Off'
                                    last_pos['SpeedAnomaly'] = False
                                    last_pos['PositionAnomaly'] = True
                                    last_pos['CourseAnomaly'] = False
                                    last_pos['BeaconAnomaly'] = True
                                    last_pos['BeaconGapMinutes'] = time_to_day_end
                                    last_pos['ReportDate'] = report_date
                                    all_anomalies.append(last_pos.to_dict())
            
            # Send progress update
            await _send_progress("detecting_speed", f"[DETECTING] Checking speed anomalies for {current_date}...")
            
            # 2. Speed anomalies (position jumps)
            if config.get('excessive_travel_distance_fast', True):
                common_mmsi = set(df_previous_day['MMSI'].unique()) & set(df_current_day['MMSI'].unique())
                
                # Process in chunks to yield control periodically
                common_mmsi_list = list(common_mmsi)
                chunk_size = 100
                for chunk_idx in range(0, len(common_mmsi_list), chunk_size):
                    chunk = common_mmsi_list[chunk_idx:chunk_idx + chunk_size]
                    await asyncio.sleep(0)  # Yield control
                    
                    for mmsi in chunk:
                        vessel_prev = prev_grouped.get_group(mmsi).copy()
                        vessel_curr = current_grouped.get_group(mmsi).copy()
                        
                        vessel_prev = vessel_prev.sort_values('BaseDateTime')
                        vessel_curr = vessel_curr.sort_values('BaseDateTime')
                        
                        if len(vessel_prev) == 0 or len(vessel_curr) == 0:
                            continue
                        
                        last_pos_prev = vessel_prev.iloc[-1]
                        first_pos_curr = vessel_curr.iloc[0]
                        
                        time_diff = (first_pos_curr['BaseDateTime'] - last_pos_prev['BaseDateTime']).total_seconds() / 60
                        
                        if time_diff > config['TIME_DIFF_THRESHOLD_MIN']:
                            continue
                        
                        # Calculate distance
                        distance_df = pd.DataFrame({
                            'LAT1': [last_pos_prev['LAT']],
                            'LON1': [last_pos_prev['LON']],
                            'LAT2': [first_pos_curr['LAT']],
                            'LON2': [first_pos_curr['LON']]
                        })
                        
                        dist_result = haversine_vectorized(distance_df, use_gpu=config.get('USE_GPU', False))
                        if dist_result.empty or pd.isna(dist_result.iloc[0]):
                            continue
                        
                        dist_nm = dist_result.iloc[0]
                        implied_speed = dist_nm / (time_diff / 60) if time_diff > 0 else 0
                        
                        if implied_speed > config['SPEED_THRESHOLD']:
                            anomaly_record = first_pos_curr.copy()
                            anomaly_record['AnomalyType'] = 'Speed'
                            anomaly_record['SpeedAnomaly'] = True
                            anomaly_record['PositionAnomaly'] = False
                            anomaly_record['CourseAnomaly'] = False
                            anomaly_record['Distance'] = dist_nm
                            anomaly_record['TimeDiff'] = time_diff
                            anomaly_record['ImpliedSpeed'] = implied_speed
                            anomaly_record['ReportDate'] = report_date
                            all_anomalies.append(anomaly_record.to_dict())
            
            # Send progress update
            await _send_progress("detecting_course", f"[DETECTING] Checking COG/Heading inconsistencies for {current_date}...")
            
            # 3. COG/Heading inconsistency
            if config.get('cog-heading_inconsistency', True):
                vessel_count = 0
                for mmsi, vessel_data in current_grouped:
                    vessel_count += 1
                    # Yield control every 50 vessels
                    if vessel_count % 50 == 0:
                        await asyncio.sleep(0)
                    min_speed = config['MIN_SPEED_FOR_COG_CHECK']
                    valid_rows = vessel_data[
                        (vessel_data['SOG'] >= min_speed) &
                        vessel_data['COG'].notna() &
                        vessel_data['Heading'].notna()
                    ]
                    
                    if len(valid_rows) == 0:
                        continue
                    
                    valid_rows = valid_rows.copy()
                    valid_rows['CourseHeadingDiff'] = valid_rows.apply(
                        lambda row: normalize_angle_difference(row['COG'] - row['Heading']),
                        axis=1
                    )
                    
                    max_diff = config['COG_HEADING_MAX_DIFF']
                    anomalous_rows = valid_rows[abs(valid_rows['CourseHeadingDiff']) > max_diff]
                    
                    if not anomalous_rows.empty:
                        for _, row in anomalous_rows.iterrows():
                            anomaly_record = row.copy()
                            anomaly_record['AnomalyType'] = 'Course'
                            anomaly_record['SpeedAnomaly'] = False
                            anomaly_record['PositionAnomaly'] = False
                            anomaly_record['CourseAnomaly'] = True
                            anomaly_record['ReportDate'] = report_date
                            all_anomalies.append(anomaly_record.to_dict())
            
            # Send progress update
            await _send_progress("detecting_loitering", f"[DETECTING] Checking loitering patterns for {current_date}...")
            
            # 4. Loitering detection
            if config.get('loitering', True):
                loitering_radius_nm = config['LOITERING_RADIUS_NM']
                loitering_duration_hours = config['LOITERING_DURATION_HOURS']
                
                vessel_count = 0
                for mmsi, vessel_data in current_grouped:
                    vessel_count += 1
                    # Yield control every 50 vessels
                    if vessel_count % 50 == 0:
                        await asyncio.sleep(0)
                    if len(vessel_data) < 10:
                        continue
                    
                    vessel_data = vessel_data.sort_values('BaseDateTime').copy()
                    time_span = (vessel_data['BaseDateTime'].max() - vessel_data['BaseDateTime'].min()).total_seconds() / 3600
                    
                    if time_span < loitering_duration_hours:
                        continue
                    
                    center_lat = vessel_data['LAT'].mean()
                    center_lon = vessel_data['LON'].mean()
                    
                    max_dist = 0
                    for _, row in vessel_data.iterrows():
                        if pd.notna(row['LAT']) and pd.notna(row['LON']):
                            distance_df = pd.DataFrame({
                                'LAT1': [center_lat],
                                'LON1': [center_lon],
                                'LAT2': [row['LAT']],
                                'LON2': [row['LON']]
                            })
                            dist_result = haversine_vectorized(distance_df, use_gpu=config.get('USE_GPU', False))
                            if not dist_result.empty and pd.notna(dist_result.iloc[0]):
                                dist_nm = dist_result.iloc[0]
                                max_dist = max(max_dist, dist_nm)
                    
                    if max_dist < loitering_radius_nm:
                        anomaly_record = vessel_data.iloc[0].copy()
                        anomaly_record['AnomalyType'] = 'Loitering'
                        anomaly_record['SpeedAnomaly'] = False
                        anomaly_record['PositionAnomaly'] = True
                        anomaly_record['CourseAnomaly'] = False
                        anomaly_record['LoiteringRadiusNM'] = max_dist
                        anomaly_record['LoiteringDurationHours'] = time_span
                        anomaly_record['LoiteringRecordCount'] = len(vessel_data)
                        anomaly_record['ReportDate'] = report_date
                        all_anomalies.append(anomaly_record.to_dict())
            
            # Send progress update
            await _send_progress("detecting_rendezvous", f"[DETECTING] Checking vessel rendezvous for {current_date}...")
            
            # 5. Rendezvous detection
            if config.get('rendezvous', True):
                rendezvous_proximity_nm = config['RENDEZVOUS_PROXIMITY_NM']
                rendezvous_duration_minutes = config['RENDEZVOUS_DURATION_MINUTES']
                
                df_current_day_sorted = df_current_day.sort_values('BaseDateTime').copy()
                df_current_day_sorted['TimeWindow'] = df_current_day_sorted['BaseDateTime'].dt.hour
                
                window_count = 0
                for window_id, window_group in df_current_day_sorted.groupby('TimeWindow'):
                    window_count += 1
                    # Yield control every few windows
                    if window_count % 3 == 0:
                        await asyncio.sleep(0)
                    if len(window_group) < 2:
                        continue
                    
                    window_vessels = window_group.groupby('MMSI')
                    if len(window_vessels) < 2:
                        continue
                    
                    vessel_positions = {}
                    for mmsi, vessel_group in window_vessels:
                        if len(vessel_group) >= 3:
                            avg_lat = vessel_group['LAT'].mean()
                            avg_lon = vessel_group['LON'].mean()
                            if pd.notna(avg_lat) and pd.notna(avg_lon):
                                vessel_positions[mmsi] = (avg_lat, avg_lon, len(vessel_group))
                    
                    vessel_list = list(vessel_positions.keys())
                    for i in range(len(vessel_list)):
                        for j in range(i + 1, len(vessel_list)):
                            mmsi1, mmsi2 = vessel_list[i], vessel_list[j]
                            lat1, lon1, count1 = vessel_positions[mmsi1]
                            lat2, lon2, count2 = vessel_positions[mmsi2]
                            
                            distance_df = pd.DataFrame({
                                'LAT1': [lat1],
                                'LON1': [lon1],
                                'LAT2': [lat2],
                                'LON2': [lon2]
                            })
                            dist_result = haversine_vectorized(distance_df, use_gpu=config.get('USE_GPU', False))
                            if dist_result.empty or pd.isna(dist_result.iloc[0]):
                                continue
                            
                            distance_nm = dist_result.iloc[0]
                            
                            if distance_nm < rendezvous_proximity_nm:
                                # Create anomaly record for first vessel
                                vessel1_data = window_group[window_group['MMSI'] == mmsi1]
                                if not vessel1_data.empty:
                                    anomaly_record = vessel1_data.iloc[0].copy()
                                    anomaly_record['AnomalyType'] = 'Rendezvous'
                                    anomaly_record['SpeedAnomaly'] = False
                                    anomaly_record['PositionAnomaly'] = True
                                    anomaly_record['CourseAnomaly'] = False
                                    anomaly_record['RendezvousDistanceNM'] = distance_nm
                                    anomaly_record['RendezvousVesselMMSI'] = mmsi2
                                    anomaly_record['ReportDate'] = report_date
                                    all_anomalies.append(anomaly_record.to_dict())
            
            # Update previous day for next iteration
            df_previous_day = df_current_day
            
            # Send progress update after each day
            day_anomalies = len([a for a in all_anomalies if a.get('ReportDate') == report_date])
            await _send_progress("day_complete", f"[COMPLETE] Day {i+1}/{len(dates)} complete: Found {day_anomalies} anomalies this day ({len(all_anomalies)} total)", {
                "day": i+1,
                "total_days": len(dates),
                "anomalies_this_day": day_anomalies,
                "anomalies_so_far": len(all_anomalies)
            })
        
        logger.info(f"Detected {len(all_anomalies)} total anomalies")
        
        # Final progress update
        await _send_progress("anomaly_detection_complete", f"[COMPLETE] Anomaly detection finished: Found {len(all_anomalies)} total anomalies", {
            "total_anomalies": len(all_anomalies)
        })
        
        if all_anomalies:
            anomalies_df = pd.DataFrame(all_anomalies)
            return anomalies_df
        else:
            return pd.DataFrame()
    
    def _generate_statistics(self, anomalies_df: pd.DataFrame, data_df) -> Dict[str, Any]:
        """Generate analysis statistics"""
        if anomalies_df.empty:
            return {
                'total_anomalies': 0,
                'unique_vessels': 0,
                'anomaly_types': {},
                'date_range_coverage': {}
            }
        
        stats = {
            'total_anomalies': len(anomalies_df),
            'unique_vessels': anomalies_df['MMSI'].nunique() if 'MMSI' in anomalies_df.columns else 0,
            'anomaly_types': {},
            'severity_distribution': {},
            'temporal_distribution': {}
        }
        
        # Anomaly type distribution
        if 'AnomalyType' in anomalies_df.columns:
            type_counts = anomalies_df['AnomalyType'].value_counts()
            stats['anomaly_types'] = type_counts.to_dict()
        
        # Severity distribution
        if 'Severity' in anomalies_df.columns:
            severity_counts = anomalies_df['Severity'].value_counts()
            stats['severity_distribution'] = severity_counts.to_dict()
        
        return stats
    
    async def get_vessel_tracks(self, mmsi_list: List[str], 
                                start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Get complete tracks for specific vessels
        
        Args:
            mmsi_list: List of MMSI numbers
            start_date: Start date
            end_date: End date
        
        Returns:
            Vessel track data
        """
        logger.info(f"Fetching tracks for {len(mmsi_list)} vessels")
        
        try:
            df = self.data_connector.load_date_range(start_date, end_date)
            
            if df.empty:
                return {'success': False, 'error': 'No data found'}
            
            # Filter to requested vessels
            vessel_data = df[df['MMSI'].isin(mmsi_list)]
            
            if vessel_data.empty:
                return {
                    'success': False,
                    'error': f'No data found for specified vessels'
                }
            
            # Group by vessel
            tracks = {}
            for mmsi in mmsi_list:
                vessel_df = vessel_data[vessel_data['MMSI'] == mmsi]
                if not vessel_df.empty:
                    # Sort by time
                    vessel_df = vessel_df.sort_values('BaseDateTime')
                    tracks[mmsi] = {
                        'mmsi': mmsi,
                        'points': vessel_df[['LAT', 'LON', 'BaseDateTime', 'SOG', 'COG']].to_dict('records'),
                        'total_points': len(vessel_df),
                        'first_seen': vessel_df['BaseDateTime'].min(),
                        'last_seen': vessel_df['BaseDateTime'].max()
                    }
            
            return {
                'success': True,
                'tracks': tracks,
                'vessels_found': len(tracks),
                'vessels_requested': len(mmsi_list)
            }
            
        except Exception as e:
            logger.error(f"Error fetching vessel tracks: {e}")
            return {'success': False, 'error': str(e)}
    
    async def get_top_anomaly_vessels(self, start_date: str, end_date: str,
                                     geographic_zone: Optional[Dict[str, Any]] = None,
                                     min_anomalies: int = 5,
                                     top_n: int = 10) -> Dict[str, Any]:
        """
        Identify vessels with most anomalies
        
        Args:
            start_date: Start date
            end_date: End date
            geographic_zone: Optional zone filter
            min_anomalies: Minimum anomalies to include vessel
            top_n: Number of top vessels to return
        
        Returns:
            List of top vessels with anomaly counts
        """
        logger.info(f"Identifying top {top_n} high-risk vessels")
        
        # Run full analysis
        analysis_result = await self.run_analysis(
            start_date, end_date,
            geographic_zone=geographic_zone
        )
        
        if not analysis_result['success'] or not analysis_result['anomalies']:
            return {
                'success': False,
                'error': 'No anomalies found',
                'vessels': []
            }
        
        # Count anomalies per vessel
        anomalies_df = pd.DataFrame(analysis_result['anomalies'])
        vessel_counts = anomalies_df.groupby('MMSI').size().reset_index(name='anomaly_count')
        vessel_counts = vessel_counts[vessel_counts['anomaly_count'] >= min_anomalies]
        vessel_counts = vessel_counts.sort_values('anomaly_count', ascending=False).head(top_n)
        
        return {
            'success': True,
            'vessels': vessel_counts.to_dict('records'),
            'total_high_risk_vessels': len(vessel_counts)
        }
    
    async def generate_report(self, analysis_id: str, report_type: str = 'summary',
                             include_maps: bool = True, 
                             include_vessel_details: bool = True) -> Dict[str, Any]:
        """
        Generate investigation report
        
        NOTE: Placeholder - implement full report generation
        """
        logger.info(f"Generating {report_type} report for analysis {analysis_id}")
        
        # Placeholder
        return {
            'success': True,
            'report_id': str(uuid.uuid4()),
            'analysis_id': analysis_id,
            'report_type': report_type,
            'generated_at': datetime.now().isoformat(),
            'message': 'Report generation placeholder - implement full functionality'
        }
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get system and GPU information
        """
        return {
            'gpu': get_gpu_info(),
            'processing_backend': get_processing_backend(),
            'data_source': self.data_connector.data_source,
            'date_limits': {
                'min': self.data_connector.MIN_DATE.strftime('%Y-%m-%d'),
                'max': self.data_connector.MAX_DATE.strftime('%Y-%m-%d')
            }
        }

