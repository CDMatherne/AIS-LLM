"""
Universal data connector supporting both AWS S3 and local file systems
Handles CSV and Parquet formats for AIS data
"""
import os
import re
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import logging

logger = logging.getLogger(__name__)

class AISDataConnector:
    """
    Unified connector for AIS data supporting:
    - AWS S3 (parquet/csv)
    - Local filesystem (parquet/csv)
    - NOAA (downloads on-demand from NOAA website)
    - Date range: January 1, 2021 onwards
    - Optional data cache for fast access
    """
    
    # Date limits - allow fraud detection from January 1, 2021 onwards
    MIN_DATE = datetime(2021, 1, 1)
    MAX_DATE = datetime(2099, 12, 31)  # Far future date to allow all available data
    
    def __init__(self, config: Dict[str, Any], data_cache=None, progress_callback=None):
        """
        Initialize data connector with configuration
        
        Args:
            config: Dictionary containing either:
                - AWS S3 config: {bucket, prefix, region, credentials}
                - Local config: {path, file_format}
                - NOAA config: {temp_dir (optional)}
            progress_callback: Optional callback for progress updates
        """
        self.config = config
        self.data_source = config.get('data_source', 'local')
        self.s3_client = None
        self.noaa_connector = None
        self.data_cache = data_cache  # Optional cache for fast access
        self.progress_callback = progress_callback
        
        # File discovery cache (Improvement #1)
        self._cached_dates: Optional[List[datetime]] = None
        self._cache_timestamp: Optional[float] = None
        self._cache_ttl: float = 300.0  # 5 minutes TTL
        self._cached_dir_mtime: Optional[float] = None
        
        # Pre-compiled regex patterns for filename matching (Improvement #1.2)
        self._date_pattern = re.compile(r'(\d{4})-(\d{2})-(\d{2})')
        self._ais_prefix_pattern = re.compile(r'AIS_(\d{4})_(\d{2})_(\d{2})')
        
        if self.data_source == 'aws':
            self._initialize_s3()
        elif self.data_source == 'noaa':
            self._initialize_noaa()
        else:
            self._initialize_local()
    
    def _initialize_s3(self):
        """Initialize AWS S3 client"""
        aws_config = self.config.get('aws', {})
        
        auth_method = aws_config.get('auth_method', 'credentials')
        
        try:
            if auth_method == 'credentials':
                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=aws_config.get('access_key'),
                    aws_secret_access_key=aws_config.get('secret_key'),
                    aws_session_token=aws_config.get('session_token'),
                    region_name=aws_config.get('region', 'us-east-1')
                )
            elif auth_method == 'profile':
                session = boto3.Session(profile_name=aws_config.get('profile_name', 'default'))
                self.s3_client = session.client('s3')
            else:  # IAM role
                self.s3_client = boto3.client(
                    's3',
                    region_name=aws_config.get('region', 'us-east-1')
                )
            
            self.bucket = aws_config['bucket']
            self.prefix = aws_config.get('prefix', '').rstrip('/')
            
            logger.info(f"S3 client initialized for bucket: {self.bucket}")
            
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {e}")
            raise
    
    def _initialize_local(self):
        """Initialize local filesystem access"""
        local_config = self.config.get('local', {})
        self.local_path = Path(local_config.get('path', '.'))
        self.file_format = local_config.get('file_format', 'auto')
        
        if not self.local_path.exists():
            raise FileNotFoundError(f"Local data path does not exist: {self.local_path}")
        
        logger.info(f"Local filesystem initialized at: {self.local_path}")
    
    def _initialize_noaa(self):
        """Initialize NOAA data connector"""
        from .noaa_data_connector import NOAADataConnector
        
        noaa_config = self.config.get('noaa', {})
        temp_dir = noaa_config.get('temp_dir', None)
        
        self.noaa_connector = NOAADataConnector(
            temp_dir=temp_dir,
            progress_callback=self.progress_callback
        )
        
        cache_info = self.noaa_connector.get_cache_info()
        logger.info(f"NOAA connector initialized - Cache dir: {cache_info['cache_dir']}")
        logger.info(f"  Available years: {', '.join(cache_info['available_years'])}")
        logger.info(f"  Cached files: {cache_info['cached_files']}")
        logger.info(f"  Cache size: {cache_info['total_size_mb']:.2f} MB")
    
    def validate_date_range(self, start_date: Union[str, datetime], 
                          end_date: Union[str, datetime]) -> tuple:
        """
        Validate and enforce date limits (January 1, 2021 onwards)
        Returns: (start_datetime, end_datetime) within valid range
        """
        # Convert to datetime if strings
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Enforce minimum date (January 1, 2021)
        if start_date < self.MIN_DATE:
            logger.warning(f"Start date {start_date} before minimum {self.MIN_DATE}, adjusting")
            start_date = self.MIN_DATE
        
        # Enforce maximum date (far future, but don't allow dates beyond reasonable range)
        if end_date > self.MAX_DATE:
            logger.warning(f"End date {end_date} after maximum {self.MAX_DATE}, adjusting")
            end_date = self.MAX_DATE
        
        # Ensure end date is not before start date
        if end_date < start_date:
            raise ValueError(
                f"End date {end_date.date()} cannot be before start date {start_date.date()}"
            )
        
        if start_date > self.MAX_DATE or end_date < self.MIN_DATE:
            raise ValueError(
                f"Requested date range is outside available data range "
                f"({self.MIN_DATE.date()} to {self.MAX_DATE.date()})"
            )
        
        return start_date, end_date
    
    def list_available_dates(self) -> List[datetime]:
        """
        List all available dates in the data source
        Returns: List of datetime objects
        """
        if self.data_source == 'aws':
            return self._list_s3_dates()
        else:
            return self._list_local_dates()
    
    def get_available_dates(self) -> List[datetime]:
        """
        Get all available dates in the data source (alias for list_available_dates)
        Returns: List of datetime objects
        """
        return self.list_available_dates()
    
    def _list_s3_dates(self) -> List[datetime]:
        """List available dates in S3 bucket"""
        dates = []
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            prefix = f"{self.prefix}/" if self.prefix else ""
            
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                if 'Contents' not in page:
                    continue
                
                for obj in page['Contents']:
                    key = obj['Key']
                    # Extract date from filename (supports YYYY-MM-DD and ais-YYYY-MM-DD formats)
                    filename = os.path.basename(key)
                    
                    # Try to parse date from filename
                    for ext in ['.parquet', '.csv']:
                        if filename.endswith(ext):
                            date_str = filename.replace(ext, '')
                            
                            # Try standard format (YYYY-MM-DD)
                            try:
                                date = datetime.strptime(date_str, '%Y-%m-%d')
                                if self.MIN_DATE <= date <= self.MAX_DATE:
                                    dates.append(date)
                                    break
                            except ValueError:
                                pass
                            
                            # Try prefixed format (ais-YYYY-MM-DD)
                            if date_str.startswith('ais-'):
                                try:
                                    date = datetime.strptime(date_str[4:], '%Y-%m-%d')
                                    if self.MIN_DATE <= date <= self.MAX_DATE:
                                        dates.append(date)
                                        break
                                except ValueError:
                                    pass
                            
                            # Try AIS_YYYY_MM_DD format
                            if date_str.startswith('AIS_'):
                                try:
                                    date_part = date_str[4:].replace('_', '-')
                                    date = datetime.strptime(date_part, '%Y-%m-%d')
                                    if self.MIN_DATE <= date <= self.MAX_DATE:
                                        dates.append(date)
                                        break
                                except ValueError:
                                    pass
            
            return sorted(list(set(dates)))
            
        except Exception as e:
            logger.error(f"Error listing S3 dates: {e}")
            return []
    
    def _list_local_dates(self) -> List[datetime]:
        """
        List available dates in local filesystem
        
        IMPROVEMENT #1: Added caching and optimized file scanning
        """
        # Guard: Only works for local data source
        if not hasattr(self, 'local_path'):
            logger.warning("_list_local_dates called but local_path not initialized (data source may not be 'local')")
            return []
        
        # Check cache validity (Improvement #1.1)
        if self._cached_dates is not None and self._cache_timestamp is not None:
            cache_age = time.time() - self._cache_timestamp
            if cache_age < self._cache_ttl:
                # Check if directory modification time changed
                try:
                    current_mtime = os.path.getmtime(self.local_path)
                    if self._cached_dir_mtime == current_mtime:
                        logger.debug(f"Using cached dates list ({len(self._cached_dates)} dates)")
                        return self._cached_dates
                    else:
                        logger.debug("Directory modified, invalidating cache")
                except OSError:
                    pass  # If we can't get mtime, proceed with scan
        
        dates = []
        
        # Optimized file scanning using os.scandir() (Improvement #1.2)
        try:
            with os.scandir(self.local_path) as entries:
                for entry in entries:
                    if not entry.is_file():
                        continue
                    
                    name = entry.name
                    ext = os.path.splitext(name)[1].lower()
                    
                    if ext not in ['.parquet', '.csv']:
                        continue
                    
                    stem = os.path.splitext(name)[0]
                    date = None
                    
                    # Try standard format (YYYY-MM-DD) using regex
                    match = self._date_pattern.match(stem)
                    if match:
                        try:
                            date = datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)))
                        except ValueError:
                            pass
                    
                    # Try prefixed format (ais-YYYY-MM-DD)
                    if date is None and stem.startswith('ais-'):
                        match = self._date_pattern.match(stem[4:])
                        if match:
                            try:
                                date = datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)))
                            except ValueError:
                                pass
                    
                    # Try AIS_YYYY_MM_DD format using pre-compiled pattern
                    if date is None:
                        match = self._ais_prefix_pattern.match(stem)
                        if match:
                            try:
                                date = datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)))
                            except ValueError:
                                pass
                    
                    # Validate date is within range
                    if date and self.MIN_DATE <= date <= self.MAX_DATE:
                        dates.append(date)
        
        except Exception as e:
            logger.error(f"Error scanning directory: {e}")
            return sorted(list(set(dates))) if dates else []
        
        # Cache result (Improvement #1.1)
        self._cached_dates = sorted(list(set(dates)))
        self._cache_timestamp = time.time()
        try:
            self._cached_dir_mtime = os.path.getmtime(self.local_path)
        except OSError:
            self._cached_dir_mtime = None
        
        logger.debug(f"Cached {len(self._cached_dates)} dates from directory scan")
        return self._cached_dates
    
    def log_connector_state(self):
        """Log current state of data connector for debugging"""
        logger.info(f"[CONNECTOR] Data source: {self.data_source}")
        logger.info(f"[CONNECTOR] S3 client initialized: {self.s3_client is not None}")
        logger.info(f"[CONNECTOR] NOAA connector initialized: {self.noaa_connector is not None}")
        
        if self.noaa_connector:
            cache_info = self.noaa_connector.get_cache_info()
            logger.info(f"[CONNECTOR] NOAA cache dir: {cache_info['cache_dir']}")
            logger.info(f"[CONNECTOR] NOAA cached files: {cache_info['cached_files']}")
            logger.info(f"[CONNECTOR] NOAA cache size: {cache_info['total_size_mb']:.2f} MB")
            
            # Verify cache integrity
            integrity = self.noaa_connector.verify_cache_integrity()
            logger.info(f"[CONNECTOR] Cache integrity: {integrity}")
    
    def load_date_range(self, start_date: Union[str, datetime], 
                       end_date: Union[str, datetime],
                       columns: Optional[List[str]] = None,
                       vessel_types: Optional[List[int]] = None,
                       mmsi_filter: Optional[List[str]] = None) -> tuple:
        """
        Load AIS data for date range
        
        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            columns: Optional list of columns to load (None = all)
            vessel_types: Optional list of vessel type codes to filter (for cache optimization)
            mmsi_filter: Optional list of MMSI to filter (for cache optimization)
        
        Returns:
            Tuple of (DataFrame, dict) where dict contains:
            - 'loaded_dates': List of dates that had data
            - 'missing_dates': List of dates that had no data
            - 'total_dates_requested': Total number of dates in range
        """
        # Try cache first if available
        if self.data_cache:
            try:
                start_str = start_date.strftime('%Y-%m-%d') if isinstance(start_date, datetime) else start_date
                end_str = end_date.strftime('%Y-%m-%d') if isinstance(end_date, datetime) else end_date
                
                cached_data = self.data_cache.load_date_range(
                    start_str, end_str,
                    vessel_types=vessel_types,
                    mmsi_filter=mmsi_filter,
                    columns=columns
                )
                
                if not cached_data.empty:
                    logger.info(f"Loaded {len(cached_data):,} records from cache (fast path)")
                    # For cache, we don't track individual dates, so return empty missing_dates
                    return cached_data, {
                        'loaded_dates': [],
                        'missing_dates': [],
                        'total_dates_requested': 0,
                        'from_cache': True
                    }
                else:
                    logger.debug("Cache miss - falling back to direct load")
            except Exception as e:
                logger.warning(f"Cache access failed, using direct load: {e}")
        
        # Fall back to direct loading
        # Validate dates
        start_dt, end_dt = self.validate_date_range(start_date, end_date)
        
        # Generate list of dates to load
        dates_to_load = []
        current = start_dt
        while current <= end_dt:
            dates_to_load.append(current)
            current += timedelta(days=1)
        
        logger.info(f"Loading {len(dates_to_load)} days of data from {start_dt.date()} to {end_dt.date()}")
        
        # Load data for each date in parallel for faster loading
        dfs = []
        loaded_dates = []
        missing_dates = []
        
        # Use ThreadPoolExecutor for parallel I/O-bound file loading
        from concurrent.futures import ThreadPoolExecutor, as_completed
        max_workers = min(len(dates_to_load), 8)  # Limit to 8 parallel workers
        
        def load_single_date_wrapper(date):
            """Wrapper for parallel date loading"""
            try:
                df = self._load_single_date(date, columns)
                date_str = date.strftime('%Y-%m-%d')
                if df is not None and len(df) > 0:
                    return ('success', date_str, df)
                else:
                    return ('missing', date_str, None)
            except Exception as e:
                date_str = date.strftime('%Y-%m-%d')
                logger.warning(f"Could not load data for {date.date()}: {e}")
                return ('error', date_str, None)
        
        # Load files in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_date = {executor.submit(load_single_date_wrapper, date): date for date in dates_to_load}
            
            for future in as_completed(future_to_date):
                status, date_str, df = future.result()
                if status == 'success' and df is not None:
                    dfs.append(df)
                    loaded_dates.append(date_str)
                    logger.info(f"Loaded {len(df)} records for {date_str}")
                else:
                    missing_dates.append(date_str)
                    if status == 'missing':
                        logger.warning(f"No data found for {date_str}")
        
        # Combine all dataframes efficiently
        if dfs:
            # Use pd.concat with ignore_index=True for better performance
            combined_df = pd.concat(dfs, ignore_index=True, sort=False)
            logger.info(f"Total records loaded: {len(combined_df):,} from {len(loaded_dates)} date(s)")
        else:
            combined_df = pd.DataFrame()
            logger.warning(f"No data loaded for any date in range {start_dt.date()} to {end_dt.date()}")
        
        return combined_df, {
            'loaded_dates': loaded_dates,
            'missing_dates': missing_dates,
            'total_dates_requested': len(dates_to_load),
            'from_cache': False
        }
    
    def _load_single_date(self, date: datetime, 
                         columns: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
        """Load data for a single date"""
        if self.data_source == 'aws':
            return self._load_s3_date(date, columns)
        elif self.data_source == 'noaa':
            return self._load_noaa_date(date, columns)
        else:
            return self._load_local_date(date, columns)
    
    def _load_noaa_date(self, date: datetime, columns: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
        """
        Load data for a single date from NOAA (downloads if needed)
        
        Args:
            date: Date to load
            columns: Optional list of columns to load
            
        Returns:
            DataFrame or None if not available
        """
        date_str = date.strftime('%Y-%m-%d')
        logger.info(f"[NOAA] Attempting to load data for {date_str}")
        
        if not self.noaa_connector:
            logger.error("[NOAA] NOAA connector not initialized! Check setup configuration.")
            return None
        
        logger.info(f"[NOAA] NOAA connector exists, calling get_data_for_date() for {date_str}")
        
        # Get parquet file (downloads if needed)
        parquet_path = self.noaa_connector.get_data_for_date(date)
        
        if not parquet_path:
            logger.warning(f"[NOAA] get_data_for_date returned None for {date_str} - no data available or download failed")
            return None
        
        logger.info(f"[NOAA] Parquet file obtained: {parquet_path}")
        
        try:
            # Load parquet file
            logger.info(f"[NOAA] Reading parquet file: {parquet_path}")
            if columns:
                df = pd.read_parquet(parquet_path, columns=columns, engine='pyarrow')
            else:
                df = pd.read_parquet(parquet_path, engine='pyarrow')
            
            logger.info(f"[NOAA] Successfully loaded {len(df):,} records for {date_str}")
            
            # Convert BaseDateTime if string
            if 'BaseDateTime' in df.columns and df['BaseDateTime'].dtype == 'object':
                df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'])
            
            logger.info(f"Loaded {len(df):,} records from NOAA for {date.date()}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading NOAA data for {date.date()}: {e}")
            return None
    
    def _load_s3_date(self, date: datetime, 
                     columns: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
        """Load data from S3 for specific date (supports YYYY-MM-DD, ais-YYYY-MM-DD, and AIS_YYYY_MM_DD formats)"""
        date_str = date.strftime('%Y-%m-%d')
        
        # Try multiple filename patterns with parquet and csv
        # Support: YYYY-MM-DD, ais-YYYY-MM-DD, AIS_YYYY_MM_DD formats
        filename_patterns = [
            date_str,  # YYYY-MM-DD
            f"ais-{date_str}",  # ais-YYYY-MM-DD
            f"AIS_{date_str.replace('-', '_')}"  # AIS_YYYY_MM_DD
        ]
        
        for pattern in filename_patterns:
            for ext in ['parquet', 'csv']:
                key = f"{self.prefix}/{pattern}.{ext}" if self.prefix else f"{pattern}.{ext}"
                
                try:
                    # Download to temp file using temp_file_manager
                    from temp_file_manager import create_temp_file
                    temp_path = create_temp_file(suffix=f'.{ext}', prefix='ais_s3_download_')
                    
                    try:
                        # Download file from S3
                        self.s3_client.download_file(self.bucket, key, temp_path)
                        
                        # Read file (file is closed by download_file, so no locking issues)
                        if ext == 'parquet':
                            # OPTIMIZED: Use column projection and optimized parquet reading
                            df = pd.read_parquet(
                                temp_path,
                                columns=columns,
                                engine='pyarrow',
                                use_pandas_metadata=True
                            )
                            # OPTIMIZED: Convert BaseDateTime during load if present
                            if 'BaseDateTime' in df.columns and df['BaseDateTime'].dtype == 'object':
                                df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'], errors='coerce')
                                logger.debug(f"Converted BaseDateTime from string to datetime for S3 file {key}")
                        else:
                            df = pd.read_csv(temp_path, usecols=columns)
                            # Convert BaseDateTime for CSV too
                            if 'BaseDateTime' in df.columns and df['BaseDateTime'].dtype == 'object':
                                df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'], errors='coerce')
                        
                        logger.info(f"Loaded S3 file: {key} ({len(df):,} rows)")
                        return df
                        
                    finally:
                        # Clean up temp file immediately after reading
                        try:
                            if os.path.exists(temp_path):
                                os.unlink(temp_path)
                        except Exception as e:
                            logger.warning(f"Could not delete temp file {temp_path}: {e}")
                        
                except ClientError as e:
                    if e.response['Error']['Code'] == '404':
                        continue  # Try next format/pattern
                    else:
                        raise
        
        logger.warning(f"No data file found in S3 for {date_str} (tried {date_str}.* and ais-{date_str}.*)")
        return None
    
    def _load_local_date(self, date: datetime, 
                        columns: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
        """Load data from local filesystem for specific date (supports YYYY-MM-DD and ais-YYYY-MM-DD formats)"""
        date_str = date.strftime('%Y-%m-%d')
        
        # Try multiple filename patterns with parquet and csv
        # Support: YYYY-MM-DD, ais-YYYY-MM-DD, AIS_YYYY_MM_DD formats
        filename_patterns = [
            date_str,  # YYYY-MM-DD
            f"ais-{date_str}",  # ais-YYYY-MM-DD
            f"AIS_{date_str.replace('-', '_')}"  # AIS_YYYY_MM_DD
        ]
        
        for pattern in filename_patterns:
            for ext in ['parquet', 'csv']:
                file_path = self.local_path / f"{pattern}.{ext}"
                
                if file_path.exists():
                    try:
                        if ext == 'parquet':
                            # OPTIMIZED: Use column projection and parse dates during read
                            df = pd.read_parquet(
                                file_path,
                                columns=columns,
                                engine='pyarrow',
                                use_pandas_metadata=True
                            )
                            # OPTIMIZED: Convert BaseDateTime during load if present
                            if 'BaseDateTime' in df.columns and df['BaseDateTime'].dtype == 'object':
                                df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'], errors='coerce')
                                logger.debug(f"Converted BaseDateTime from string to datetime for {date_str}")
                        else:
                            df = pd.read_csv(file_path, usecols=columns)
                            # Convert BaseDateTime for CSV too
                            if 'BaseDateTime' in df.columns and df['BaseDateTime'].dtype == 'object':
                                df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'], errors='coerce')
                        
                        logger.info(f"Loaded local file: {file_path} ({len(df):,} rows)")
                        return df
                    except Exception as e:
                        logger.error(f"Error reading {file_path}: {e}")
                        continue
        
        logger.warning(f"No data file found locally for {date_str} (tried {len(filename_patterns)} patterns)")
        return None
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Test connection and return status
        Returns: {success: bool, message: str, file_count: int}
        """
        try:
            available_dates = self.list_available_dates()
            
            return {
                'success': True,
                'message': f'Connection successful! Found {len(available_dates)} days of data',
                'file_count': len(available_dates),
                'date_range': {
                    'available_start': min(available_dates).strftime('%Y-%m-%d') if available_dates else None,
                    'available_end': max(available_dates).strftime('%Y-%m-%d') if available_dates else None,
                    'test_limit_start': self.MIN_DATE.strftime('%Y-%m-%d'),
                    'test_limit_end': self.MAX_DATE.strftime('%Y-%m-%d')
                }
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Connection failed: {str(e)}',
                'error': str(e)
            }

