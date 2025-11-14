"""
Universal data connector supporting both AWS S3 and local file systems
Handles CSV and Parquet formats for AIS data
"""
import os
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
    - Date range: 2024-10-15 to 2025-03-30 (TEST IMPLEMENTATION)
    - Optional data cache for fast access
    """
    
    # Hard-coded date limits for test implementation
    MIN_DATE = datetime(2024, 10, 15)
    MAX_DATE = datetime(2025, 3, 30)
    
    def __init__(self, config: Dict[str, Any], data_cache=None):
        """
        Initialize data connector with configuration
        
        Args:
            config: Dictionary containing either:
                - AWS S3 config: {bucket, prefix, region, credentials}
                - Local config: {path, file_format}
        """
        self.config = config
        self.data_source = config.get('data_source', 'local')
        self.s3_client = None
        self.data_cache = data_cache  # Optional cache for fast access
        
        if self.data_source == 'aws':
            self._initialize_s3()
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
    
    def validate_date_range(self, start_date: Union[str, datetime], 
                          end_date: Union[str, datetime]) -> tuple:
        """
        Validate and enforce test implementation date limits
        Returns: (start_datetime, end_datetime) within valid range
        """
        # Convert to datetime if strings
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Enforce limits
        if start_date < self.MIN_DATE:
            logger.warning(f"Start date {start_date} before minimum {self.MIN_DATE}, adjusting")
            start_date = self.MIN_DATE
        
        if end_date > self.MAX_DATE:
            logger.warning(f"End date {end_date} after maximum {self.MAX_DATE}, adjusting")
            end_date = self.MAX_DATE
        
        if start_date > self.MAX_DATE or end_date < self.MIN_DATE:
            raise ValueError(
                f"Requested date range is outside test data availability "
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
            
            return sorted(list(set(dates)))
            
        except Exception as e:
            logger.error(f"Error listing S3 dates: {e}")
            return []
    
    def _list_local_dates(self) -> List[datetime]:
        """List available dates in local filesystem"""
        dates = []
        
        # Look for files matching date pattern (YYYY-MM-DD and ais-YYYY-MM-DD)
        for ext in ['*.parquet', '*.csv']:
            for file_path in self.local_path.glob(ext):
                filename = file_path.stem
                
                # Try standard format (YYYY-MM-DD)
                try:
                    date = datetime.strptime(filename, '%Y-%m-%d')
                    if self.MIN_DATE <= date <= self.MAX_DATE:
                        dates.append(date)
                        continue
                except ValueError:
                    pass
                
                # Try prefixed format (ais-YYYY-MM-DD)
                if filename.startswith('ais-'):
                    try:
                        date = datetime.strptime(filename[4:], '%Y-%m-%d')
                        if self.MIN_DATE <= date <= self.MAX_DATE:
                            dates.append(date)
                    except ValueError:
                        pass
        
        return sorted(list(set(dates)))
    
    def load_date_range(self, start_date: Union[str, datetime], 
                       end_date: Union[str, datetime],
                       columns: Optional[List[str]] = None,
                       vessel_types: Optional[List[int]] = None,
                       mmsi_filter: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load AIS data for date range
        
        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            columns: Optional list of columns to load (None = all)
            vessel_types: Optional list of vessel type codes to filter (for cache optimization)
            mmsi_filter: Optional list of MMSI to filter (for cache optimization)
        
        Returns:
            Combined DataFrame with all data
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
                    return cached_data
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
        
        # Load data for each date
        dfs = []
        for date in dates_to_load:
            try:
                df = self._load_single_date(date, columns)
                if df is not None and len(df) > 0:
                    dfs.append(df)
                    logger.info(f"Loaded {len(df)} records for {date.date()}")
            except Exception as e:
                logger.warning(f"Could not load data for {date.date()}: {e}")
                continue
        
        if not dfs:
            logger.warning("No data loaded for specified date range")
            return pd.DataFrame()
        
        # Combine all dataframes
        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Total records loaded: {len(combined_df)}")
        
        return combined_df
    
    def _load_single_date(self, date: datetime, 
                         columns: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
        """Load data for a single date"""
        if self.data_source == 'aws':
            return self._load_s3_date(date, columns)
        else:
            return self._load_local_date(date, columns)
    
    def _load_s3_date(self, date: datetime, 
                     columns: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
        """Load data from S3 for specific date (supports YYYY-MM-DD and ais-YYYY-MM-DD formats)"""
        date_str = date.strftime('%Y-%m-%d')
        
        # Try both filename patterns with parquet and csv
        filename_patterns = [date_str, f"ais-{date_str}"]
        
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
                            df = pd.read_parquet(temp_path, columns=columns)
                        else:
                            df = pd.read_csv(temp_path, usecols=columns)
                        
                        logger.info(f"Loaded S3 file: {key}")
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
        
        # Try both filename patterns with parquet and csv
        filename_patterns = [date_str, f"ais-{date_str}"]
        
        for pattern in filename_patterns:
            for ext in ['parquet', 'csv']:
                file_path = self.local_path / f"{pattern}.{ext}"
                
                if file_path.exists():
                    try:
                        if ext == 'parquet':
                            df = pd.read_parquet(file_path, columns=columns)
                        else:
                            df = pd.read_csv(file_path, usecols=columns)
                        
                        logger.info(f"Loaded local file: {file_path}")
                        return df
                    except Exception as e:
                        logger.error(f"Error reading {file_path}: {e}")
                        continue
        
        logger.warning(f"No data file found locally for {date_str} (tried {date_str}.* and ais-{date_str}.*)")
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

