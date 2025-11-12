"""
Data Cache and Pre-processing Module

Pre-loads and caches all AIS data in optimized format for fast access.
Automatically syncs with S3 to detect and load new data.
"""
import os
import pickle
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set
import pandas as pd
import numpy as np
import logging
from threading import Lock
import asyncio

logger = logging.getLogger(__name__)

# Try to import pyarrow for faster parquet operations
try:
    import pyarrow.parquet as pq
    import pyarrow as pa
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False
    logger.warning("PyArrow not available - using pandas for parquet (slower)")


class AISDataCache:
    """
    Manages pre-processed and cached AIS data for fast access.
    Automatically syncs with S3 to detect and load new data.
    """
    
    def __init__(self, data_connector, cache_dir: Optional[str] = None):
        """
        Initialize data cache
        
        Args:
            data_connector: AISDataConnector instance
            cache_dir: Directory for cache files (defaults to backend/data_cache)
        """
        self.data_connector = data_connector
        self.cache_lock = Lock()
        
        # Set up cache directory
        if cache_dir is None:
            cache_dir = Path(__file__).parent / "data_cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache metadata file
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        
        # Cache files
        self.data_cache_file = self.cache_dir / "ais_data_cache.parquet"
        self.index_cache_file = self.cache_dir / "ais_index_cache.pkl"
        
        # Load metadata
        self.metadata = self._load_metadata()
        
        # Cache state
        self._data_cache: Optional[pd.DataFrame] = None
        self._date_index: Dict[str, List[int]] = {}  # date -> list of row indices
        self._vessel_type_index: Dict[int, List[int]] = {}  # vessel_type -> list of row indices
        self._mmsi_index: Dict[str, List[int]] = {}  # mmsi -> list of row indices
        
        logger.info(f"Data cache initialized at: {self.cache_dir}")
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache metadata: {e}")
        
        return {
            "cached_dates": [],
            "last_sync": None,
            "total_records": 0,
            "cache_version": "1.0"
        }
    
    def _save_metadata(self):
        """Save cache metadata"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save cache metadata: {e}")
    
    def get_cached_dates(self) -> Set[str]:
        """Get set of dates currently cached"""
        return set(self.metadata.get("cached_dates", []))
    
    def get_available_dates(self) -> List[datetime]:
        """Get all available dates from data source"""
        return self.data_connector.get_available_dates()
    
    async def sync_with_source(self, force_full_reload: bool = False) -> Dict[str, Any]:
        """
        Sync cache with data source (S3 or local).
        Checks for new data and loads it if found.
        
        Args:
            force_full_reload: If True, reload all data even if already cached
            
        Returns:
            Dict with sync results
        """
        logger.info("Starting data sync with source...")
        
        # Get available dates from source
        available_dates = self.get_available_dates()
        available_date_strings = {d.strftime('%Y-%m-%d') for d in available_dates}
        
        # Get cached dates
        cached_dates = self.get_cached_dates()
        
        # Find missing dates
        missing_dates = available_date_strings - cached_dates
        
        if force_full_reload:
            missing_dates = available_date_strings
            logger.info("Force full reload requested - will reload all data")
        
        if not missing_dates:
            logger.info("Cache is up to date - no new data found")
            return {
                "success": True,
                "new_dates": [],
                "total_cached": len(cached_dates),
                "message": "Cache is up to date"
            }
        
        logger.info(f"Found {len(missing_dates)} new date(s) to load: {sorted(missing_dates)}")
        
        # Load missing data
        new_data = []
        failed_dates = []
        
        for date_str in sorted(missing_dates):
            try:
                date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                logger.info(f"Loading data for {date_str}...")
                
                df = self.data_connector.load_date_range(date_str, date_str)
                
                if not df.empty:
                    # Add date column for indexing
                    df['CacheDate'] = date_str
                    new_data.append(df)
                    logger.info(f"Loaded {len(df)} records for {date_str}")
                else:
                    logger.warning(f"No data found for {date_str}")
                    failed_dates.append(date_str)
                    
            except Exception as e:
                logger.error(f"Failed to load data for {date_str}: {e}")
                failed_dates.append(date_str)
        
        if not new_data:
            logger.warning("No new data loaded")
            return {
                "success": False,
                "new_dates": [],
                "failed_dates": failed_dates,
                "message": "No new data could be loaded"
            }
        
        # Combine with existing cache
        with self.cache_lock:
            existing_data = self._load_cache_data()
            
            if existing_data is not None and not existing_data.empty:
                # Remove dates that are being reloaded
                dates_to_remove = missing_dates & cached_dates
                if dates_to_remove:
                    existing_data = existing_data[~existing_data['CacheDate'].isin(dates_to_remove)]
                    logger.info(f"Removed {len(dates_to_remove)} date(s) from cache for reload")
                
                # Combine
                combined_data = pd.concat([existing_data] + new_data, ignore_index=True)
            else:
                combined_data = pd.concat(new_data, ignore_index=True)
            
            # Optimize and save
            self._save_cache_data(combined_data)
            self._build_indexes(combined_data)
            
            # Update metadata
            self.metadata["cached_dates"] = sorted(available_date_strings)
            self.metadata["last_sync"] = datetime.now().isoformat()
            self.metadata["total_records"] = len(combined_data)
            self._save_metadata()
        
        logger.info(f"Sync complete: Added {len(missing_dates)} date(s), {len(combined_data):,} total records")
        
        return {
            "success": True,
            "new_dates": sorted(missing_dates),
            "failed_dates": failed_dates,
            "total_cached": len(combined_data),
            "total_dates": len(available_date_strings),
            "message": f"Successfully synced {len(missing_dates)} new date(s)"
        }
    
    def _load_cache_data(self) -> Optional[pd.DataFrame]:
        """Load cached data from disk"""
        if not self.data_cache_file.exists():
            return None
        
        try:
            logger.info("Loading cached data from disk...")
            if PYARROW_AVAILABLE:
                # Use pyarrow for faster loading
                table = pq.read_table(self.data_cache_file)
                df = table.to_pandas()
            else:
                df = pd.read_parquet(self.data_cache_file)
            
            logger.info(f"Loaded {len(df):,} records from cache")
            return df
        except Exception as e:
            logger.error(f"Failed to load cache data: {e}")
            return None
    
    def _save_cache_data(self, df: pd.DataFrame):
        """Save data to cache with optimization"""
        try:
            logger.info(f"Saving {len(df):,} records to cache...")
            
            # Optimize data types to reduce size
            df_optimized = self._optimize_dataframe(df)
            
            # Save with compression
            if PYARROW_AVAILABLE:
                table = pa.Table.from_pandas(df_optimized)
                pq.write_table(
                    table,
                    self.data_cache_file,
                    compression='snappy',  # Fast compression
                    use_dictionary=True,  # Dictionary encoding for better compression
                    write_statistics=True  # Enable statistics for faster filtering
                )
            else:
                df_optimized.to_parquet(
                    self.data_cache_file,
                    compression='snappy',
                    index=False
                )
            
            logger.info(f"Cache saved successfully ({self._get_file_size_mb(self.data_cache_file):.2f} MB)")
        except Exception as e:
            logger.error(f"Failed to save cache data: {e}")
            raise
    
    def _optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame data types to reduce memory and storage"""
        df = df.copy()
        
        # Optimize numeric columns
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        # Optimize datetime
        if 'BaseDateTime' in df.columns:
            df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'])
        
        # Convert string columns to category if they have few unique values
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() < len(df) * 0.5:  # If less than 50% unique values
                df[col] = df[col].astype('category')
        
        return df
    
    def _build_indexes(self, df: pd.DataFrame):
        """Build indexes for fast lookups"""
        logger.info("Building indexes for fast lookups...")
        
        # Date index
        if 'CacheDate' in df.columns:
            self._date_index = {}
            for date_str in df['CacheDate'].unique():
                self._date_index[date_str] = df[df['CacheDate'] == date_str].index.tolist()
        
        # Vessel type index
        if 'VesselType' in df.columns:
            self._vessel_type_index = {}
            for vessel_type in df['VesselType'].unique():
                if pd.notna(vessel_type):
                    self._vessel_type_index[int(vessel_type)] = df[df['VesselType'] == vessel_type].index.tolist()
        
        # MMSI index
        if 'MMSI' in df.columns:
            self._mmsi_index = {}
            for mmsi in df['MMSI'].unique():
                if pd.notna(mmsi):
                    self._mmsi_index[str(mmsi)] = df[df['MMSI'] == mmsi].index.tolist()
        
        # Save indexes
        self._save_indexes()
        
        logger.info("Indexes built successfully")
    
    def _save_indexes(self):
        """Save indexes to disk"""
        try:
            indexes = {
                'date_index': self._date_index,
                'vessel_type_index': {k: v for k, v in self._vessel_type_index.items()},
                'mmsi_index': self._mmsi_index
            }
            with open(self.index_cache_file, 'wb') as f:
                pickle.dump(indexes, f)
        except Exception as e:
            logger.error(f"Failed to save indexes: {e}")
    
    def _load_indexes(self):
        """Load indexes from disk"""
        if not self.index_cache_file.exists():
            return
        
        try:
            with open(self.index_cache_file, 'rb') as f:
                indexes = pickle.load(f)
                self._date_index = indexes.get('date_index', {})
                self._vessel_type_index = {int(k): v for k, v in indexes.get('vessel_type_index', {}).items()}
                self._mmsi_index = indexes.get('mmsi_index', {})
        except Exception as e:
            logger.error(f"Failed to load indexes: {e}")
    
    def load_date_range(self, start_date: str, end_date: str, 
                       vessel_types: Optional[List[int]] = None,
                       mmsi_filter: Optional[List[str]] = None,
                       columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load data from cache for date range (much faster than loading from source)
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            vessel_types: Optional list of vessel type codes to filter
            mmsi_filter: Optional list of MMSI to filter
            columns: Optional list of columns to load
            
        Returns:
            DataFrame with filtered data
        """
        with self.cache_lock:
            # Load cache if not already loaded
            if self._data_cache is None:
                self._data_cache = self._load_cache_data()
                if self._data_cache is None:
                    logger.warning("Cache not available - returning empty DataFrame")
                    return pd.DataFrame()
                self._load_indexes()
            
            # Get date range
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            # Find matching dates
            matching_indices = set()
            current = start_dt
            while current <= end_dt:
                date_str = current.strftime('%Y-%m-%d')
                if date_str in self._date_index:
                    matching_indices.update(self._date_index[date_str])
                current += timedelta(days=1)
            
            if not matching_indices:
                logger.warning(f"No cached data found for date range {start_date} to {end_date}")
                return pd.DataFrame()
            
            # Apply vessel type filter using index
            if vessel_types:
                vessel_indices = set()
                for vessel_type in vessel_types:
                    if vessel_type in self._vessel_type_index:
                        vessel_indices.update(self._vessel_type_index[vessel_type])
                matching_indices &= vessel_indices
            
            # Apply MMSI filter using index
            if mmsi_filter:
                mmsi_indices = set()
                for mmsi in mmsi_filter:
                    mmsi_str = str(mmsi)
                    if mmsi_str in self._mmsi_index:
                        mmsi_indices.update(self._mmsi_index[mmsi_str])
                matching_indices &= mmsi_indices
            
            # Get data using indices
            if not matching_indices:
                return pd.DataFrame()
            
            # Use iloc for faster access
            result_df = self._data_cache.iloc[list(matching_indices)].copy()
            
            # Select columns if specified
            if columns:
                available_columns = [c for c in columns if c in result_df.columns]
                result_df = result_df[available_columns]
            
            logger.info(f"Loaded {len(result_df):,} records from cache for {start_date} to {end_date}")
            return result_df
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "cached_dates": len(self.metadata.get("cached_dates", [])),
            "total_records": self.metadata.get("total_records", 0),
            "last_sync": self.metadata.get("last_sync"),
            "cache_size_mb": self._get_file_size_mb(self.data_cache_file) if self.data_cache_file.exists() else 0,
            "index_size_mb": self._get_file_size_mb(self.index_cache_file) if self.index_cache_file.exists() else 0
        }
    
    def _get_file_size_mb(self, file_path: Path) -> float:
        """Get file size in MB"""
        if file_path.exists():
            return file_path.stat().st_size / (1024 * 1024)
        return 0.0
    
    def clear_cache(self):
        """Clear all cached data"""
        with self.cache_lock:
            if self.data_cache_file.exists():
                self.data_cache_file.unlink()
            if self.index_cache_file.exists():
                self.index_cache_file.unlink()
            if self.metadata_file.exists():
                self.metadata_file.unlink()
            
            self._data_cache = None
            self._date_index = {}
            self._vessel_type_index = {}
            self._mmsi_index = {}
            self.metadata = {
                "cached_dates": [],
                "last_sync": None,
                "total_records": 0,
                "cache_version": "1.0"
            }
            
            logger.info("Cache cleared successfully")

