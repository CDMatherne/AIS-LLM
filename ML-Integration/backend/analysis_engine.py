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
import psutil  # For memory checking
import time  # For heartbeat logging

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


def estimate_memory_required(num_days: int, avg_records_per_day: int = 10_000_000) -> float:
    """
    Estimate memory required for analysis in GB.
    
    Args:
        num_days: Number of days to analyze
        avg_records_per_day: Average records per day (default 10M)
    
    Returns:
        Estimated memory in GB
    """
    # Estimate: 11 columns × 8 bytes per float64 + overhead
    bytes_per_record = 11 * 8 * 1.5  # 1.5x for overhead
    total_records = num_days * avg_records_per_day
    estimated_bytes = total_records * bytes_per_record
    estimated_gb = estimated_bytes / (1024 ** 3)
    return estimated_gb


def check_available_memory() -> Dict[str, float]:
    """
    Check available system memory.
    
    Returns:
        Dict with total, available, and used memory in GB
    """
    mem = psutil.virtual_memory()
    return {
        'total_gb': mem.total / (1024 ** 3),
        'available_gb': mem.available / (1024 ** 3),
        'used_gb': mem.used / (1024 ** 3),
        'percent_used': mem.percent
    }


class AISAnalysisEngine:
    """
    AIS analysis engine with GPU acceleration support
    """
    
    def __init__(self, data_connector, session_manager=None):
        self.data_connector = data_connector
        self.session_manager = session_manager  # NEW: For cache access
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
    
    async def _load_data_with_cache(self, start_date: str, end_date: str,
                                     date_range_dates: List[datetime],
                                     vessel_types: Optional[List[str]],
                                     mmsi_filter: Optional[List[str]],
                                     session_id: Optional[str],
                                     progress_callback) -> tuple:
        """
        Load data with session caching to avoid re-loading same dates.
        
        Returns:
            (DataFrame, metadata_dict)
        """
        # Convert datetime dates to strings for comparison
        requested_date_strings = {d.strftime('%Y-%m-%d') for d in date_range_dates}
        
        # Check if we have session caching available
        logger.info(f"[CACHE] Session ID: {session_id}")
        logger.info(f"[CACHE] Session manager available: {self.session_manager is not None}")
        
        if session_id and self.session_manager:
            cache = self.session_manager.get_data_cache(session_id)
            logger.info(f"[CACHE] Cached data found: {cache is not None}")
            
            if cache:
                logger.info(f"[CACHE] Cached records: {len(cache['dataframe'])}")
                logger.info(f"[CACHE] Cached dates: {cache['loaded_dates']}")
                logger.info(f"[CACHE] Cache metadata: {cache.get('metadata', {})}")
                cached_dates = set(cache['loaded_dates'])
                dates_to_load = requested_date_strings - cached_dates
                
                if dates_to_load:
                    # Partial cache hit - load missing dates only
                    logger.info(f"Cache HIT for {len(cached_dates & requested_date_strings)} dates, "
                               f"loading {len(dates_to_load)} new dates")
                    await progress_callback("loading_data", 
                        f"[CACHE] Using cached data for {len(cached_dates & requested_date_strings)} dates, "
                        f"loading {len(dates_to_load)} new dates...")
                    
                    # Load only missing dates
                    new_data_result = self.data_connector.load_date_range(
                        min(dates_to_load), max(dates_to_load),
                        vessel_types=vessel_types,
                        mmsi_filter=mmsi_filter
                    )
                    
                    if isinstance(new_data_result, tuple):
                        new_df, new_metadata = new_data_result
                    else:
                        new_df, new_metadata = new_data_result, {}
                    
                    # Filter cached DataFrame to requested date range
                    cached_df = cache['dataframe']
                    if 'BaseDateTime' in cached_df.columns:
                        # Convert if needed
                        if not pd.api.types.is_datetime64_any_dtype(cached_df['BaseDateTime']):
                            cached_df['BaseDateTime'] = pd.to_datetime(cached_df['BaseDateTime'])
                        
                        cached_df_filtered = cached_df[
                            cached_df['BaseDateTime'].dt.strftime('%Y-%m-%d').isin(requested_date_strings)
                        ].copy()
                    else:
                        cached_df_filtered = cached_df
                    
                    # Merge cached + new data
                    combined_df = pd.concat([cached_df_filtered, new_df], ignore_index=True)
                    
                    # Drop duplicates
                    if 'MMSI' in combined_df.columns and 'BaseDateTime' in combined_df.columns:
                        combined_df = combined_df.drop_duplicates(subset=['MMSI', 'BaseDateTime'], keep='first')
                    
                    # Update cache with ALL loaded data (old + new)
                    all_loaded_dates = sorted(cached_dates | set(new_metadata.get('loaded_dates', [])))
                    self.session_manager.merge_data_cache(
                        session_id, new_df, list(dates_to_load),
                        metadata={'vessel_types': vessel_types}
                    )
                    
                    metadata = {
                        'loaded_dates': list(requested_date_strings),
                        'missing_dates': new_metadata.get('missing_dates', []),
                        'cache_hit': True,
                        'cache_hit_count': len(cached_dates & requested_date_strings),
                        'new_load_count': len(dates_to_load)
                    }
                    
                    return combined_df, metadata
                    
                else:
                    # Full cache hit!
                    logger.info(f"Cache HIT for ALL {len(requested_date_strings)} requested dates")
                    await progress_callback("loading_data", 
                        f"[CACHE] Using cached data for all {len(requested_date_strings)} dates (no loading needed!)")
                    
                    # Filter cached DataFrame to requested dates
                    cached_df = cache['dataframe']
                    if 'BaseDateTime' in cached_df.columns:
                        if not pd.api.types.is_datetime64_any_dtype(cached_df['BaseDateTime']):
                            cached_df['BaseDateTime'] = pd.to_datetime(cached_df['BaseDateTime'])
                        
                        df = cached_df[
                            cached_df['BaseDateTime'].dt.strftime('%Y-%m-%d').isin(requested_date_strings)
                        ].copy()
                    else:
                        df = cached_df.copy()
                    
                    metadata = {
                        'loaded_dates': list(requested_date_strings),
                        'missing_dates': [],
                        'cache_hit': True,
                        'cache_hit_count': len(requested_date_strings),
                        'new_load_count': 0
                    }
                    
                    return df, metadata
        
        # No cache or cache miss - load from storage
        logger.info(f"Loading {len(requested_date_strings)} dates from storage (no cache)")
        await progress_callback("loading_data", 
            f"[LOADING] Loading data from {len(requested_date_strings)} file(s)...")
        
        # Log connector state before loading
        self.data_connector.log_connector_state()
        
        load_result = self.data_connector.load_date_range(
            start_date, end_date,
            vessel_types=vessel_types,
            mmsi_filter=mmsi_filter
        )
        
        if isinstance(load_result, tuple):
            df, metadata = load_result
        else:
            df, metadata = load_result, {}
        
        # Store in cache if session available
        if session_id and self.session_manager and not df.empty:
            loaded_dates = metadata.get('loaded_dates', list(requested_date_strings))
            logger.info(f"[CACHE] Storing data in session cache...")
            logger.info(f"[CACHE] Records to store: {len(df):,}")
            logger.info(f"[CACHE] Dates to store: {loaded_dates}")
            self.session_manager.set_data_cache(
                session_id, df, loaded_dates,
                metadata={'vessel_types': vessel_types}
            )
            logger.info(f"[CACHE] ✓ Stored {len(df):,} records in session cache for {len(loaded_dates)} dates")
        
        return df, metadata
    
    async def _load_data_chunked(self, start_date: str, end_date: str,
                                  date_range_dates: List[datetime],
                                  vessel_types: Optional[List[str]],
                                  mmsi_filter: Optional[List[str]],
                                  session_id: Optional[str],
                                  progress_callback) -> tuple:
        """
        Load data in chunks (day-by-day) to avoid OOM for large datasets.
        
        FIX #4 & #7: Chunked/Day-by-Day Processing
        
        Returns:
            (DataFrame, metadata_dict)
        """
        logger.info(f"Loading {len(date_range_dates)} dates in chunks (day-by-day)")
        
        all_chunks = []
        loaded_dates = []
        missing_dates = []
        
        # Convert datetime dates to strings for processing
        date_strings = [d.strftime('%Y-%m-%d') for d in sorted(date_range_dates)]
        
        # Process dates in small chunks (5 days at a time)
        CHUNK_SIZE = 5
        for chunk_idx in range(0, len(date_strings), CHUNK_SIZE):
            chunk_dates = date_strings[chunk_idx:chunk_idx + CHUNK_SIZE]
            chunk_start = chunk_dates[0]
            chunk_end = chunk_dates[-1]
            
            logger.info(f"Loading chunk {chunk_idx//CHUNK_SIZE + 1}/{(len(date_strings)-1)//CHUNK_SIZE + 1}: "
                       f"{chunk_start} to {chunk_end}")
            
            await progress_callback("loading_chunk", 
                f"[LOADING] Chunk {chunk_idx//CHUNK_SIZE + 1}/{(len(date_strings)-1)//CHUNK_SIZE + 1}: "
                f"{chunk_start} to {chunk_end}...")
            
            # Load this chunk
            try:
                chunk_result = self.data_connector.load_date_range(
                    chunk_start, chunk_end,
                    vessel_types=vessel_types,
                    mmsi_filter=mmsi_filter
                )
                
                if isinstance(chunk_result, tuple):
                    chunk_df, chunk_metadata = chunk_result
                else:
                    chunk_df, chunk_metadata = chunk_result, {}
                
                if not chunk_df.empty:
                    all_chunks.append(chunk_df)
                    loaded_dates.extend(chunk_metadata.get('loaded_dates', chunk_dates))
                    logger.info(f"Loaded chunk: {len(chunk_df):,} records")
                else:
                    missing_dates.extend(chunk_dates)
                    logger.warning(f"No data in chunk {chunk_start} to {chunk_end}")
                
                # Yield control to allow other tasks to run
                await asyncio.sleep(0)
                
            except Exception as e:
                logger.error(f"Error loading chunk {chunk_start} to {chunk_end}: {e}")
                missing_dates.extend(chunk_dates)
        
        # Combine all chunks
        if all_chunks:
            logger.info(f"Combining {len(all_chunks)} chunks...")
            await progress_callback("combining_chunks", f"[LOADING] Combining {len(all_chunks)} data chunks...")
            df = pd.concat(all_chunks, ignore_index=True)
            logger.info(f"Combined total: {len(df):,} records")
        else:
            df = pd.DataFrame()
        
        metadata = {
            'loaded_dates': loaded_dates,
            'missing_dates': missing_dates,
            'total_dates_requested': len(date_strings),
            'cache_hit': False,
            'chunked_loading': True,
            'num_chunks': len(all_chunks)
        }
        
        # Store in cache if session available
        if session_id and self.session_manager and not df.empty:
            self.session_manager.set_data_cache(
                session_id, df, loaded_dates,
                metadata={'vessel_types': vessel_types, 'chunked': True}
            )
            logger.info(f"Stored {len(df):,} records in session cache (chunked load)")
        
        return df, metadata
    
    async def run_analysis(self, 
                          start_date: str, 
                          end_date: str,
                          geographic_zone: Optional[Dict[str, Any]] = None,
                          anomaly_types: List[str] = None,
                          mmsi_filter: Optional[List[str]] = None,
                          vessel_types: Optional[List[str]] = None,
                          progress_callback: Optional[callable] = None,
                          session_id: Optional[str] = None) -> Dict[str, Any]:
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
        # Normalize vessel_types to always be a list (handle string input)
        if vessel_types is not None:
            if isinstance(vessel_types, str):
                vessel_types = [vessel_types]
            elif not isinstance(vessel_types, (list, tuple)):
                logger.warning(f"vessel_types parameter is not a list, string, or None: {type(vessel_types)}, converting to list")
                vessel_types = [vessel_types] if vessel_types else None
            elif isinstance(vessel_types, tuple):
                vessel_types = list(vessel_types)
        
        analysis_id = str(uuid.uuid4())
        logger.info(f"Starting analysis {analysis_id} from {start_date} to {end_date}")
        
        # FIX #6: Pre-flight memory check
        start_dt = datetime.strptime(start_date, '%Y-%m-%d') if isinstance(start_date, str) else start_date
        end_dt = datetime.strptime(end_date, '%Y-%m-%d') if isinstance(end_date, str) else end_date
        num_days = (end_dt - start_dt).days + 1
        
        estimated_memory_gb = estimate_memory_required(num_days)
        available_memory = check_available_memory()
        
        logger.info(f"Pre-flight check: {num_days} days, estimated {estimated_memory_gb:.2f} GB required, "
                   f"{available_memory['available_gb']:.2f} GB available")
        
        # Warn if estimated memory > 80% of available
        if estimated_memory_gb > (available_memory['available_gb'] * 0.8):
            warning_msg = (f"⚠️ Large dataset warning: Estimated memory requirement ({estimated_memory_gb:.1f} GB) "
                          f"is close to or exceeds available memory ({available_memory['available_gb']:.1f} GB). "
                          f"Analysis may be slow or fail. Consider reducing the date range.")
            logger.warning(warning_msg)
        
        # FIX #5: Start heartbeat logging task
        heartbeat_active = True
        heartbeat_start_time = time.time()
        
        async def heartbeat_logger():
            """Log progress every 60 seconds"""
            while heartbeat_active:
                await asyncio.sleep(60)
                if heartbeat_active:
                    elapsed = time.time() - heartbeat_start_time
                    mem = check_available_memory()
                    logger.info(f"[HEARTBEAT] Analysis running for {elapsed/60:.1f} minutes, "
                               f"Memory: {mem['percent_used']:.1f}% used ({mem['available_gb']:.1f} GB available)")
        
        # Start heartbeat task
        heartbeat_task = asyncio.create_task(heartbeat_logger())
        
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
            # Note: datetime is already imported at module level (line 9)
            start_dt_check = datetime.strptime(start_date, '%Y-%m-%d') if isinstance(start_date, str) else start_date
            end_dt_check = datetime.strptime(end_date, '%Y-%m-%d') if isinstance(end_date, str) else end_date
            
            date_range_dates = [
                d for d in available_dates 
                if start_dt_check <= d <= end_dt_check
            ]
            file_count = len(date_range_dates)
            
            await _progress("files_found", f"[FOUND] Found {file_count} data file(s) for the requested date range", {
                "file_count": file_count,
                "date_range": {"start": start_date, "end": end_date}
            })
            
            # Stage 2: Loading data (uses session cache if available for much faster loading)
            # FIX #4 & #7: For large datasets (>7 days), use chunked loading
            if num_days > 7:
                logger.info(f"Large dataset detected ({num_days} days). Using chunked loading to prevent OOM.")
                await _progress("chunked_loading", f"[LOADING] Large dataset - loading in chunks to prevent memory issues...")
                df, load_metadata = await self._load_data_chunked(
                    start_date, end_date, date_range_dates,
                    vessel_types, mmsi_filter, session_id, _progress
                )
            else:
                df, load_metadata = await self._load_data_with_cache(
                    start_date, end_date, date_range_dates,
                    vessel_types, mmsi_filter, session_id, _progress
                )
            
            # Check for missing dates and provide detailed error message
            missing_dates = load_metadata.get('missing_dates', [])
            loaded_dates = load_metadata.get('loaded_dates', [])
            total_dates = load_metadata.get('total_dates_requested', 0)
            
            if df.empty:
                # Build detailed error message about missing dates
                error_msg = f'No data found for the specified date range ({start_date} to {end_date})'
                
                if missing_dates:
                    if len(missing_dates) == total_dates:
                        # All dates missing
                        error_msg += f'. No data files were found for any of the {total_dates} requested date(s).'
                    else:
                        # Some dates missing
                        error_msg += f'. Missing data for {len(missing_dates)} out of {total_dates} date(s): '
                        if len(missing_dates) <= 10:
                            error_msg += ', '.join(missing_dates)
                        else:
                            error_msg += ', '.join(missing_dates[:10]) + f' and {len(missing_dates) - 10} more'
                    
                    error_msg += ' Please check that data files exist for the requested dates or try a different date range.'
                else:
                    error_msg += '. Please verify that data files exist for the requested date range.'
                
                await _progress("error", f"[ERROR] {error_msg}")
                return {
                    'success': False,
                    'error': error_msg,
                    'analysis_id': analysis_id,
                    'anomalies': [],
                    'missing_dates': missing_dates,
                    'loaded_dates': loaded_dates,
                    'date_range': {'start': start_date, 'end': end_date}
                }
            
            # If some dates are missing but we have some data, warn but continue
            if missing_dates and len(missing_dates) < total_dates:
                warning_msg = f"Note: Data was found for {len(loaded_dates)} date(s) but missing for {len(missing_dates)} date(s)"
                if len(missing_dates) <= 5:
                    warning_msg += f": {', '.join(missing_dates)}"
                else:
                    warning_msg += f": {', '.join(missing_dates[:5])} and {len(missing_dates) - 5} more"
                await _progress("warning", f"[WARNING] {warning_msg}")
                logger.warning(warning_msg)
            
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
            
            # Validate df_gpu is a DataFrame-like object
            if not hasattr(df_gpu, 'columns') or not hasattr(df_gpu, '__getitem__'):
                error_msg = f"df_gpu is not a DataFrame-like object: {type(df_gpu)}"
                logger.error(error_msg)
                raise TypeError(error_msg)
            
            # Apply vessel type filter if provided
            if vessel_types and 'VesselType' in df_gpu.columns:
                from .vessel_types import VESSEL_TYPE_CATEGORIES
                
                # Detailed logging for debugging
                logger.info(f"[DEBUG] Vessel type filtering - Input vessel_types: {vessel_types} (type: {type(vessel_types)})")
                
                # Ensure vessel_types is a list (handle case where it might be a string)
                if isinstance(vessel_types, str):
                    logger.info(f"[DEBUG] Converting vessel_types from string to list: '{vessel_types}'")
                    vessel_types = [vessel_types]
                elif not isinstance(vessel_types, (list, tuple)):
                    logger.warning(f"[DEBUG] vessel_types is not a list or string: {type(vessel_types)}, value: {vessel_types}, converting to list")
                    vessel_types = [vessel_types] if vessel_types else []
                elif isinstance(vessel_types, tuple):
                    logger.info(f"[DEBUG] Converting vessel_types from tuple to list")
                    vessel_types = list(vessel_types)
                
                logger.info(f"[DEBUG] Normalized vessel_types: {vessel_types} (type: {type(vessel_types)})")
                logger.info(f"[DEBUG] Available categories in VESSEL_TYPE_CATEGORIES: {list(VESSEL_TYPE_CATEGORIES.keys())}")
                
                type_codes = []
                for vtype in vessel_types:
                    logger.info(f"[DEBUG] Processing vessel type: '{vtype}' (type: {type(vtype)})")
                    if isinstance(vtype, str):
                        if vtype in VESSEL_TYPE_CATEGORIES:
                            category_codes = VESSEL_TYPE_CATEGORIES[vtype]
                            logger.info(f"[DEBUG] Found category '{vtype}' in VESSEL_TYPE_CATEGORIES, codes: {category_codes} (type: {type(category_codes)})")
                            # Ensure category_codes is a list
                            if isinstance(category_codes, (list, tuple)):
                                type_codes.extend(category_codes)
                                logger.info(f"[DEBUG] Extended type_codes, now has {len(type_codes)} codes")
                            else:
                                logger.error(f"[DEBUG] VESSEL_TYPE_CATEGORIES['{vtype}'] is not a list: {type(category_codes)}, value: {category_codes}")
                                raise TypeError(f"VESSEL_TYPE_CATEGORIES['{vtype}'] returned {type(category_codes).__name__}, expected list")
                        else:
                            logger.warning(f"[DEBUG] Vessel type '{vtype}' not found in VESSEL_TYPE_CATEGORIES. Available: {list(VESSEL_TYPE_CATEGORIES.keys())}")
                    else:
                        logger.warning(f"[DEBUG] Vessel type '{vtype}' is not a string (type: {type(vtype)}), skipping")
                
                logger.info(f"[DEBUG] Final type_codes list: {type_codes} (length: {len(type_codes)})")
                
                if type_codes:
                    try:
                        original_count = len(df_gpu)
                        logger.info(f"[DEBUG] About to filter DataFrame. Original count: {original_count}, type_codes: {type_codes}")
                        logger.info(f"[DEBUG] DataFrame type: {type(df_gpu)}, has VesselType column: {'VesselType' in df_gpu.columns}")
                        
                        # Ensure type_codes are integers for filtering
                        type_codes_int = [int(code) if not isinstance(code, int) else code for code in type_codes]
                        logger.info(f"[DEBUG] Converted type_codes to integers: {type_codes_int}")
                        
                        df_gpu = df_gpu[df_gpu['VesselType'].isin(type_codes_int)]
                        filtered_vessels = df_gpu['MMSI'].nunique() if 'MMSI' in df_gpu.columns else 0
                        vessel_types_str = ', '.join(str(vt) for vt in vessel_types) if vessel_types else 'unknown'
                        logger.info(f"[DEBUG] Filtering complete. Filtered count: {len(df_gpu)}, vessels: {filtered_vessels}")
                        
                        await _progress("vessel_filter", f"[FILTERED] Filtered to {filtered_vessels:,} vessels of selected type(s): {vessel_types_str}", {
                            "vessel_types": vessel_types,
                            "filtered_vessels": filtered_vessels,
                            "filtered_records": len(df_gpu),
                            "original_records": original_count
                        })
                    except Exception as filter_error:
                        logger.error(f"[DEBUG] Error during vessel type filtering: {filter_error}")
                        logger.error(f"[DEBUG] df_gpu type: {type(df_gpu)}")
                        logger.error(f"[DEBUG] type_codes: {type_codes}")
                        import traceback
                        logger.error(f"[DEBUG] Filter error traceback:\n{traceback.format_exc()}")
                        raise
            
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
            import traceback
            error_traceback = traceback.format_exc()
            logger.error(f"Analysis failed: {e}")
            logger.error(f"Full traceback:\n{error_traceback}")
            return {
                'success': False,
                'error': str(e),
                'analysis_id': analysis_id,
                'anomalies': [],
                'traceback': error_traceback  # Include traceback for debugging
            }
        finally:
            # FIX #5: Stop heartbeat logging
            heartbeat_active = False
            try:
                await heartbeat_task
            except:
                pass  # Task may already be cancelled
    
    async def _detect_anomalies(self, df, anomaly_types: Optional[List[str]], progress_callback: Optional[callable] = None) -> pd.DataFrame:
        """
        Detect anomalies using SFD.py logic
        
        Implements:
        - AIS beacon off/on detection
        - Speed anomalies (position jumps)
        - COG/Heading inconsistencies
        - Loitering detection
        # - Rendezvous detection (DISABLED - TOO COMPLEX)
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
            # 'RENDEZVOUS_PROXIMITY_NM': 0.5,  # DISABLED - TOO COMPLEX
            # 'RENDEZVOUS_DURATION_MINUTES': 30,  # DISABLED - TOO COMPLEX
            'ais_beacon_on': True,
            'ais_beacon_off': True,
            'excessive_travel_distance_fast': True,
            'cog-heading_inconsistency': True,
            'loitering': True,
            # 'rendezvous': True,  # DISABLED - TOO COMPLEX
            'USE_GPU': self.gpu_available
        }
        
        # Filter anomaly types if specified
        if anomaly_types:
            for key in ['ais_beacon_on', 'ais_beacon_off', 'excessive_travel_distance_fast', 
                       'cog-heading_inconsistency', 'loitering']:  # 'rendezvous' DISABLED - TOO COMPLEX
                if key not in anomaly_types:
                    config[key] = False
        
        all_anomalies = []
        
        # OPTIMIZED: Pre-sort by BaseDateTime once for faster day filtering
        df = df.sort_values('BaseDateTime').copy()
        
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
            # OPTIMIZED: Use boolean indexing instead of copy when possible
            day_mask = df['Date'] == current_date
            df_current_day = df[day_mask]
            
            if df_current_day.empty:
                continue
            
            # Only copy when we need to modify
            df_current_day = df_current_day.copy()
            
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
            
            # OPTIMIZED: Pre-sort before grouping for faster groupby operations
            if 'BaseDateTime' in df_previous_day.columns:
                df_previous_day = df_previous_day.sort_values('BaseDateTime')
            if 'BaseDateTime' in df_current_day.columns:
                df_current_day = df_current_day.sort_values('BaseDateTime')
            
            # Group by MMSI (cached groupby for reuse, sort=False is faster)
            prev_grouped = df_previous_day.groupby('MMSI', sort=False)
            current_grouped = df_current_day.groupby('MMSI', sort=False)
            
            report_date = current_date.strftime('%Y-%m-%d')
            
            # Send progress update for anomaly type detection
            await _send_progress("detecting_beacon", f"[DETECTING] Checking AIS beacon anomalies for {current_date}...")
            
            # 1. AIS Beacon on/off anomalies
            if config.get('ais_beacon_on', True) or config.get('ais_beacon_off', True):
                beacon_time_threshold = config['BEACON_TIME_THRESHOLD_HOURS'] * 60  # Convert to minutes
                
                # Beacon On: Vessels that appeared in current day but not in previous day
                # OPTIMIZED: Use set operations and vectorized processing
                if config.get('ais_beacon_on', True):
                    prev_mmsi_set = set(df_previous_day['MMSI'].unique())
                    curr_mmsi_set = set(df_current_day['MMSI'].unique())
                    beacon_on_mmsi = curr_mmsi_set - prev_mmsi_set
                    
                    if len(beacon_on_mmsi) > 0:
                        # OPTIMIZED: Batch process first appearances using groupby
                        day_start = pd.Timestamp(current_date).replace(hour=0, minute=0, second=0)
                        
                        # Get first appearance for each vessel in one operation
                        first_appearances = df_current_day[df_current_day['MMSI'].isin(beacon_on_mmsi)].groupby('MMSI', sort=False)['BaseDateTime'].first()
                        
                        for mmsi, first_appearance in first_appearances.items():
                            time_since_day_start = (first_appearance - day_start).total_seconds() / 60
                            
                            if time_since_day_start >= beacon_time_threshold:
                                # Get first position efficiently
                                first_pos = df_current_day[
                                    (df_current_day['MMSI'] == mmsi) & 
                                    (df_current_day['BaseDateTime'] == first_appearance)
                                ].iloc[0].copy()
                                
                                first_pos['AnomalyType'] = 'AIS_Beacon_On'
                                first_pos['SpeedAnomaly'] = False
                                first_pos['PositionAnomaly'] = True
                                first_pos['CourseAnomaly'] = False
                                first_pos['BeaconAnomaly'] = True
                                first_pos['BeaconGapMinutes'] = time_since_day_start
                                first_pos['ReportDate'] = report_date
                                all_anomalies.append(first_pos.to_dict())
                        
                        # Yield control after batch
                        if len(beacon_on_mmsi) > 100:
                            await asyncio.sleep(0)
                
                # Beacon Off: Vessels that disappeared in current day but were in previous day
                # OPTIMIZED: Use set operations and vectorized processing
                if config.get('ais_beacon_off', True):
                    prev_mmsi_set = set(df_previous_day['MMSI'].unique())
                    curr_mmsi_set = set(df_current_day['MMSI'].unique())
                    beacon_off_mmsi = prev_mmsi_set - curr_mmsi_set
                    
                    if len(beacon_off_mmsi) > 0:
                        # OPTIMIZED: Batch process last appearances using groupby
                        day_end = pd.Timestamp(dates[i-1]).replace(hour=23, minute=59, second=59)
                        
                        # Get last appearance for each vessel in one operation
                        last_appearances = df_previous_day[df_previous_day['MMSI'].isin(beacon_off_mmsi)].groupby('MMSI', sort=False)['BaseDateTime'].last()
                        
                        for mmsi, last_appearance in last_appearances.items():
                            time_to_day_end = (day_end - last_appearance).total_seconds() / 60
                            
                            if time_to_day_end >= beacon_time_threshold:
                                # Get last position efficiently
                                last_pos = df_previous_day[
                                    (df_previous_day['MMSI'] == mmsi) & 
                                    (df_previous_day['BaseDateTime'] == last_appearance)
                                ].iloc[-1].copy()
                                
                                last_pos['AnomalyType'] = 'AIS_Beacon_Off'
                                last_pos['SpeedAnomaly'] = False
                                last_pos['PositionAnomaly'] = True
                                last_pos['CourseAnomaly'] = False
                                last_pos['BeaconAnomaly'] = True
                                last_pos['BeaconGapMinutes'] = time_to_day_end
                                last_pos['ReportDate'] = report_date
                                all_anomalies.append(last_pos.to_dict())
                        
                        # Yield control after batch
                        if len(beacon_off_mmsi) > 100:
                            await asyncio.sleep(0)
            
            # Send progress update
            await _send_progress("detecting_speed", f"[DETECTING] Checking speed anomalies for {current_date}...")
            
            # 2. Speed anomalies (position jumps) - OPTIMIZED: Batch vectorized processing
            if config.get('excessive_travel_distance_fast', True):
                common_mmsi = set(df_previous_day['MMSI'].unique()) & set(df_current_day['MMSI'].unique())
                
                if len(common_mmsi) > 0:
                    # Pre-sort grouped dataframes once instead of per-vessel
                    prev_sorted = df_previous_day.sort_values('BaseDateTime').groupby('MMSI')
                    curr_sorted = df_current_day.sort_values('BaseDateTime').groupby('MMSI')
                    
                    # Collect all position pairs for batch processing
                    position_pairs = []
                    mmsi_list = []
                    
                    for mmsi in common_mmsi:
                        try:
                            vessel_prev = prev_sorted.get_group(mmsi)
                            vessel_curr = curr_sorted.get_group(mmsi)
                            
                            if len(vessel_prev) == 0 or len(vessel_curr) == 0:
                                continue
                            
                            last_pos_prev = vessel_prev.iloc[-1]
                            first_pos_curr = vessel_curr.iloc[0]
                            
                            time_diff = (first_pos_curr['BaseDateTime'] - last_pos_prev['BaseDateTime']).total_seconds() / 60
                            
                            if time_diff <= config['TIME_DIFF_THRESHOLD_MIN'] and time_diff > 0:
                                position_pairs.append({
                                    'LAT1': last_pos_prev['LAT'],
                                    'LON1': last_pos_prev['LON'],
                                    'LAT2': first_pos_curr['LAT'],
                                    'LON2': first_pos_curr['LON'],
                                    'time_diff': time_diff,
                                    'mmsi': mmsi,
                                    'first_pos': first_pos_curr
                                })
                                mmsi_list.append(mmsi)
                        except KeyError:
                            continue
                    
                    # Batch calculate all distances at once (much faster)
                    if position_pairs:
                        distance_df = pd.DataFrame(position_pairs)
                        dist_results = haversine_vectorized(distance_df, use_gpu=config.get('USE_GPU', False))
                        
                        # Process results
                        for idx, (pair, dist_nm) in enumerate(zip(position_pairs, dist_results)):
                            if pd.isna(dist_nm) or dist_nm <= 0:
                                continue
                            
                            implied_speed = dist_nm / (pair['time_diff'] / 60) if pair['time_diff'] > 0 else 0
                            
                            if implied_speed > config['SPEED_THRESHOLD']:
                                anomaly_record = pair['first_pos'].copy()
                                anomaly_record['AnomalyType'] = 'Speed'
                                anomaly_record['SpeedAnomaly'] = True
                                anomaly_record['PositionAnomaly'] = False
                                anomaly_record['CourseAnomaly'] = False
                                anomaly_record['Distance'] = dist_nm
                                anomaly_record['TimeDiff'] = pair['time_diff']
                                anomaly_record['ImpliedSpeed'] = implied_speed
                                anomaly_record['ReportDate'] = report_date
                                all_anomalies.append(anomaly_record.to_dict())
                        
                        # Yield control after batch processing
                        await asyncio.sleep(0)
            
            # Send progress update
            await _send_progress("detecting_course", f"[DETECTING] Checking COG/Heading inconsistencies for {current_date}...")
            
            # 3. COG/Heading inconsistency - OPTIMIZED: Vectorized batch processing
            if config.get('cog-heading_inconsistency', True):
                min_speed = config['MIN_SPEED_FOR_COG_CHECK']
                max_diff = config['COG_HEADING_MAX_DIFF']
                
                # OPTIMIZED: Filter and process all vessels at once instead of per-vessel loop
                valid_mask = (
                    (df_current_day['SOG'] >= min_speed) &
                    df_current_day['COG'].notna() &
                    df_current_day['Heading'].notna()
                )
                
                if valid_mask.any():
                    valid_rows = df_current_day[valid_mask].copy()
                    
                    # Vectorized angle difference calculation
                    cog_heading_diff = valid_rows['COG'] - valid_rows['Heading']
                    # Vectorized normalization: ((x + 180) % 360) - 180
                    cog_heading_diff = ((cog_heading_diff + 180) % 360) - 180
                    valid_rows['CourseHeadingDiff'] = cog_heading_diff
                    
                    # Vectorized filtering for anomalies
                    anomalous_mask = abs(valid_rows['CourseHeadingDiff']) > max_diff
                    anomalous_rows = valid_rows[anomalous_mask].copy()  # Explicit copy to avoid SettingWithCopyWarning
                    
                    if not anomalous_rows.empty:
                        # Convert to dict records in batch (faster than iterrows)
                        anomalous_rows['AnomalyType'] = 'Course'
                        anomalous_rows['SpeedAnomaly'] = False
                        anomalous_rows['PositionAnomaly'] = False
                        anomalous_rows['CourseAnomaly'] = True
                        anomalous_rows['ReportDate'] = report_date
                        
                        # Convert to dict records efficiently
                        all_anomalies.extend(anomalous_rows.to_dict('records'))
                    
                    # Yield control periodically
                    if len(valid_rows) > 1000:
                        await asyncio.sleep(0)
            
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
            
            # ===================================================================
            # 5. Rendezvous detection (DISABLED - TOO COMPLEX FOR PROCESSING)
            # ===================================================================
            # if config.get('rendezvous', True):
            #     await _send_progress("detecting_rendezvous", f"[DETECTING] Checking vessel rendezvous for {current_date}...")
            #     rendezvous_proximity_nm = config['RENDEZVOUS_PROXIMITY_NM']
            #     rendezvous_duration_minutes = config['RENDEZVOUS_DURATION_MINUTES']
            #     
            #     df_current_day_sorted = df_current_day.sort_values('BaseDateTime').copy()
            #     df_current_day_sorted['TimeWindow'] = df_current_day_sorted['BaseDateTime'].dt.hour
            #     
            #     window_count = 0
            #     for window_id, window_group in df_current_day_sorted.groupby('TimeWindow'):
            #         window_count += 1
            #         # Yield control every few windows
            #         if window_count % 3 == 0:
            #             await asyncio.sleep(0)
            #         if len(window_group) < 2:
            #             continue
            #         
            #         window_vessels = window_group.groupby('MMSI')
            #         if len(window_vessels) < 2:
            #             continue
            #         
            #         vessel_positions = {}
            #         for mmsi, vessel_group in window_vessels:
            #             if len(vessel_group) >= 3:
            #                 avg_lat = vessel_group['LAT'].mean()
            #                 avg_lon = vessel_group['LON'].mean()
            #                 if pd.notna(avg_lat) and pd.notna(avg_lon):
            #                     vessel_positions[mmsi] = (avg_lat, avg_lon, len(vessel_group))
            #         
            #         vessel_list = list(vessel_positions.keys())
            #         for i in range(len(vessel_list)):
            #             for j in range(i + 1, len(vessel_list)):
            #                 mmsi1, mmsi2 = vessel_list[i], vessel_list[j]
            #                 lat1, lon1, count1 = vessel_positions[mmsi1]
            #                 lat2, lon2, count2 = vessel_positions[mmsi2]
            #                 
            #                 distance_df = pd.DataFrame({
            #                     'LAT1': [lat1],
            #                     'LON1': [lon1],
            #                     'LAT2': [lat2],
            #                     'LON2': [lon2]
            #                 })
            #                 dist_result = haversine_vectorized(distance_df, use_gpu=config.get('USE_GPU', False))
            #                 if dist_result.empty or pd.isna(dist_result.iloc[0]):
            #                     continue
            #                 
            #                 distance_nm = dist_result.iloc[0]
            #                 
            #                 if distance_nm < rendezvous_proximity_nm:
            #                     # Create anomaly record for first vessel
            #                     vessel1_data = window_group[window_group['MMSI'] == mmsi1]
            #                     if not vessel1_data.empty:
            #                         anomaly_record = vessel1_data.iloc[0].copy()
            #                         anomaly_record['AnomalyType'] = 'Rendezvous'
            #                         anomaly_record['SpeedAnomaly'] = False
            #                         anomaly_record['PositionAnomaly'] = True
            #                         anomaly_record['CourseAnomaly'] = False
            #                         anomaly_record['RendezvousDistanceNM'] = distance_nm
            #                         anomaly_record['RendezvousVesselMMSI'] = mmsi2
            #                         anomaly_record['ReportDate'] = report_date
            #                         all_anomalies.append(anomaly_record.to_dict())
            # ===================================================================
            
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
            load_result = self.data_connector.load_date_range(start_date, end_date)
            # Handle new tuple format (DataFrame, metadata) or old format (just DataFrame)
            if isinstance(load_result, tuple):
                df, _ = load_result
            else:
                df = load_result
            
            if df.empty:
                return {'success': False, 'error': 'No data found'}
            
            # Filter to requested vessels - handle both string and int MMSI types
            # Convert mmsi_list to match DataFrame MMSI type
            if 'MMSI' not in df.columns:
                return {'success': False, 'error': 'MMSI column not found in data'}
            
            # Check MMSI type in DataFrame
            df_mmsi_type = df['MMSI'].dtype
            logger.debug(f"DataFrame MMSI dtype: {df_mmsi_type}")
            
            # Convert mmsi_list to match DataFrame type
            try:
                if pd.api.types.is_integer_dtype(df_mmsi_type):
                    # DataFrame has int MMSI, convert list to int
                    mmsi_list_converted = [int(m) for m in mmsi_list]
                else:
                    # DataFrame has string MMSI, keep as string
                    mmsi_list_converted = [str(m) for m in mmsi_list]
            except (ValueError, TypeError) as e:
                logger.warning(f"Error converting MMSI list: {e}")
                mmsi_list_converted = mmsi_list
            
            # Filter to requested vessels
            vessel_data = df[df['MMSI'].isin(mmsi_list_converted)]
            
            if vessel_data.empty:
                logger.warning(f"No data found for {len(mmsi_list)} requested vessels. Sample MMSIs: {mmsi_list[:3]}")
                # Check if any of the requested MMSIs exist in the full dataset
                if 'MMSI' in df.columns:
                    sample_mmsis = df['MMSI'].unique()[:5]
                    logger.debug(f"Sample MMSIs in dataset: {sample_mmsis}")
                return {
                    'success': False,
                    'error': f'No data found for specified vessels (checked {len(mmsi_list)} MMSIs)'
                }
            
            # Group by vessel - use converted MMSI list
            tracks = {}
            for mmsi_str, mmsi_converted in zip(mmsi_list, mmsi_list_converted):
                vessel_df = vessel_data[vessel_data['MMSI'] == mmsi_converted]
                if not vessel_df.empty:
                    # Sort by time to ensure chronological order
                    vessel_df = vessel_df.sort_values('BaseDateTime')
                    
                    # Include all available columns, not just LAT/LON/BaseDateTime/SOG/COG
                    # This ensures we have VesselType, VesselName, etc. for filtering
                    available_cols = ['LAT', 'LON', 'BaseDateTime']
                    optional_cols = ['SOG', 'COG', 'VesselType', 'VesselName', 'MMSI']
                    cols_to_include = available_cols + [col for col in optional_cols if col in vessel_df.columns]
                    
                    # Use original string MMSI as key for consistency
                    tracks[mmsi_str] = {
                        'mmsi': mmsi_str,
                        'points': vessel_df[cols_to_include].to_dict('records'),
                        'total_points': len(vessel_df),
                        'first_seen': vessel_df['BaseDateTime'].min(),
                        'last_seen': vessel_df['BaseDateTime'].max(),
                        'vessel_type': vessel_df['VesselType'].iloc[0] if 'VesselType' in vessel_df.columns else None,
                        'vessel_name': vessel_df['VesselName'].iloc[0] if 'VesselName' in vessel_df.columns else None
                    }
            
            return {
                'success': True,
                'tracks': tracks,
                'vessels_found': len(tracks),
                'vessels_requested': len(mmsi_list),
                'vessels_with_no_data': len(mmsi_list) - len(tracks)
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

