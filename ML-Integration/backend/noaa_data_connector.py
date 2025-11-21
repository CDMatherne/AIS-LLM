"""
NOAA AIS Data Connector
Downloads, extracts, and converts NOAA AIS data on-the-fly for analysis
"""
import os
import re
import requests
import zipfile
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class NOAADataConnector:
    """
    Downloads and processes AIS data from NOAA on-demand
    Handles year-based URL construction and caches data in temp folder
    """
    
    # NOAA AIS data base URL pattern
    BASE_URL_PATTERN = "https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{year}/"
    
    # Available years (can be expanded)
    AVAILABLE_YEARS = ["2021", "2022", "2023", "2024"]
    
    def __init__(self, temp_dir: Optional[str] = None, progress_callback: Optional[callable] = None):
        """
        Initialize NOAA data connector
        
        Args:
            temp_dir: Temporary directory for caching downloaded data
            progress_callback: Optional callback function for progress updates
        """
        if temp_dir:
            self.temp_dir = Path(temp_dir)
        else:
            self.temp_dir = Path(tempfile.gettempdir()) / "ais_noaa_cache"
        
        # Create cache directory with enhanced error handling
        try:
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"[NOAA] Cache directory created/verified: {self.temp_dir}")
            logger.info(f"[NOAA] Directory exists: {self.temp_dir.exists()}")
            logger.info(f"[NOAA] Directory is writable: {os.access(self.temp_dir, os.W_OK)}")
            logger.info(f"[NOAA] Absolute path: {self.temp_dir.absolute()}")
        except Exception as e:
            logger.error(f"[NOAA] Failed to create cache directory: {e}")
            raise RuntimeError(f"Cannot create NOAA cache directory at {self.temp_dir}: {e}")
        
        self.progress_callback = progress_callback
        
        # Verify cache integrity on initialization
        cache_status = self.verify_cache_integrity()
        logger.info(f"[NOAA] Cache integrity check: {cache_status}")
        logger.info(f"NOAA Data Connector initialized with cache dir: {self.temp_dir}")
    
    def _send_progress(self, stage: str, message: str, **kwargs):
        """Send progress update via callback"""
        if self.progress_callback:
            try:
                self.progress_callback(stage, message, **kwargs)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")
    
    def _get_year_url(self, year: int) -> str:
        """Construct NOAA URL for a specific year"""
        return self.BASE_URL_PATTERN.format(year=year)
    
    def _get_available_files(self, year: int) -> List[Dict[str, Any]]:
        """
        Get list of available zip files for a year from NOAA
        
        Args:
            year: Year to query
            
        Returns:
            List of dicts with file info (name, url, date)
        """
        url = self._get_year_url(year)
        
        try:
            self._send_progress("listing_files", f"Querying NOAA for {year} data...")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            files = []
            
            for a_tag in soup.find_all('a'):
                href = a_tag.get('href')
                if href and href.endswith('.zip'):
                    # Parse date from filename (e.g., AIS_2024_01_01.zip)
                    match = re.match(r'AIS_(\d{4})_(\d{2})_(\d{2})\.zip', href)
                    if match:
                        y, m, d = match.groups()
                        files.append({
                            'name': href,
                            'url': f"{url}{href}",
                            'date': datetime(int(y), int(m), int(d))
                        })
            
            logger.info(f"Found {len(files)} files for year {year}")
            return files
            
        except Exception as e:
            logger.error(f"Error fetching file list from NOAA for {year}: {e}")
            return []
    
    def _download_file(self, url: str, output_path: Path) -> bool:
        """
        Download a file with progress bar
        
        Args:
            url: URL to download
            output_path: Path to save file
            
        Returns:
            True if successful, False otherwise
        """
        # Skip if already exists and complete
        if output_path.exists():
            try:
                # Quick validation - try to open as zip
                with zipfile.ZipFile(output_path, 'r') as zf:
                    logger.info(f"File already exists and is valid: {output_path.name}")
                    return True
            except:
                logger.warning(f"Existing file is corrupt, re-downloading: {output_path.name}")
                output_path.unlink()
        
        try:
            self._send_progress("downloading", f"Downloading {output_path.name}...")
            
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024 * 1024  # 1 MB
            
            with open(output_path, 'wb') as f:
                if total_size:
                    for data in tqdm(response.iter_content(block_size), 
                                   total=total_size // block_size, 
                                   unit='MB', 
                                   desc=output_path.name,
                                   disable=False):
                        f.write(data)
                else:
                    for data in response.iter_content(block_size):
                        f.write(data)
            
            logger.info(f"Downloaded: {output_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading {url}: {e}")
            if output_path.exists():
                output_path.unlink()
            return False
    
    def _extract_zip(self, zip_path: Path, extract_dir: Path) -> List[Path]:
        """
        Extract CSV files from zip
        
        Args:
            zip_path: Path to zip file
            extract_dir: Directory to extract to
            
        Returns:
            List of extracted CSV paths
        """
        extract_dir.mkdir(parents=True, exist_ok=True)
        extracted_files = []
        
        try:
            self._send_progress("extracting", f"Extracting {zip_path.name}...")
            
            with zipfile.ZipFile(zip_path, 'r') as zf:
                for file_info in zf.filelist:
                    if file_info.filename.lower().endswith('.csv'):
                        zf.extract(file_info, extract_dir)
                        extracted_files.append(extract_dir / file_info.filename)
            
            logger.info(f"Extracted {len(extracted_files)} CSV files from {zip_path.name}")
            return extracted_files
            
        except Exception as e:
            logger.error(f"Error extracting {zip_path}: {e}")
            return []
    
    def _convert_csv_to_parquet(self, csv_path: Path, output_dir: Path) -> Optional[Path]:
        """
        Convert CSV to Parquet format
        
        Args:
            csv_path: Path to CSV file
            output_dir: Directory to save parquet file
            
        Returns:
            Path to parquet file if successful, None otherwise
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = output_dir / f"{csv_path.stem}.parquet"
        
        # Skip if already exists
        if parquet_path.exists():
            logger.info(f"Parquet file already exists: {parquet_path.name}")
            return parquet_path
        
        try:
            self._send_progress("converting", f"Converting {csv_path.name} to Parquet...")
            
            # Read CSV in chunks for large files
            chunk_size = 100000
            chunks = []
            
            for chunk in pd.read_csv(csv_path, chunksize=chunk_size, low_memory=False):
                chunks.append(chunk)
            
            # Combine and write to parquet
            df = pd.concat(chunks, ignore_index=True)
            df.to_parquet(parquet_path, engine='pyarrow', index=False)
            
            logger.info(f"Converted to parquet: {parquet_path.name} ({len(df):,} rows)")
            
            # Clean up CSV
            csv_path.unlink()
            
            return parquet_path
            
        except Exception as e:
            logger.error(f"Error converting {csv_path} to parquet: {e}")
            return None
    
    def get_data_for_date(self, date: datetime) -> Optional[Path]:
        """
        Get parquet file for a specific date, downloading if necessary
        
        Args:
            date: Date to get data for
            
        Returns:
            Path to parquet file, or None if unavailable
        """
        date_str_formatted = date.strftime('%Y-%m-%d')
        logger.info(f"[NOAA] get_data_for_date called for {date_str_formatted}")
        logger.info(f"[NOAA] Cache directory: {self.temp_dir}")
        logger.info(f"[NOAA] Cache directory exists: {self.temp_dir.exists()}")
        
        year = date.year
        
        # Check if year is available
        if str(year) not in self.AVAILABLE_YEARS:
            logger.warning(f"[NOAA] Year {year} not available from NOAA (available: {', '.join(self.AVAILABLE_YEARS)})")
            return None
        
        # Expected filename format
        date_str = date.strftime("%Y_%m_%d")
        expected_filename = f"AIS_{date_str}"
        
        # Check cache first
        parquet_dir = self.temp_dir / str(year)
        parquet_path = parquet_dir / f"{expected_filename}.parquet"
        
        logger.info(f"[NOAA] Looking for cached file: {parquet_path}")
        logger.info(f"[NOAA] Parquet directory exists: {parquet_dir.exists()}")
        
        if parquet_path.exists():
            file_size = parquet_path.stat().st_size / (1024**2)  # MB
            logger.info(f"[NOAA] ✓ Found cached parquet: {parquet_path.name} ({file_size:.2f} MB)")
            logger.info(f"[NOAA] File last modified: {datetime.fromtimestamp(parquet_path.stat().st_mtime)}")
            return parquet_path
        
        # Need to download
        logger.info(f"[NOAA] ✗ Not in cache, downloading from NOAA for {date_str_formatted}")
        
        # Get available files for this year
        available_files = self._get_available_files(year)
        
        # Find matching file
        target_file = None
        for file_info in available_files:
            if file_info['date'] == date:
                target_file = file_info
                break
        
        if not target_file:
            logger.warning(f"No data available from NOAA for {date.strftime('%Y-%m-%d')}")
            return None
        
        # Download zip file
        zip_dir = self.temp_dir / "downloads" / str(year)
        zip_dir.mkdir(parents=True, exist_ok=True)
        zip_path = zip_dir / target_file['name']
        
        logger.info(f"[NOAA] Downloading to: {zip_path}")
        if not self._download_file(target_file['url'], zip_path):
            logger.error(f"[NOAA] ✗ Download failed for {date_str_formatted}")
            return None
        
        logger.info(f"[NOAA] ✓ Download complete: {zip_path.stat().st_size / (1024**2):.2f} MB")
        
        # Extract CSV
        csv_dir = self.temp_dir / "csv" / str(year)
        logger.info(f"[NOAA] Extracting to: {csv_dir}")
        extracted_files = self._extract_zip(zip_path, csv_dir)
        
        if not extracted_files:
            logger.error(f"[NOAA] ✗ Extraction failed for {date_str_formatted}")
            return None
        
        logger.info(f"[NOAA] ✓ Extracted {len(extracted_files)} file(s)")
        
        # Convert to parquet
        parquet_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"[NOAA] Converting to parquet in: {parquet_dir}")
        
        for csv_file in extracted_files:
            logger.info(f"[NOAA] Converting: {csv_file.name}")
            parquet_file = self._convert_csv_to_parquet(csv_file, parquet_dir)
            if parquet_file and parquet_file.stem == expected_filename:
                file_size = parquet_file.stat().st_size / (1024**2)  # MB
                logger.info(f"[NOAA] ✓ Conversion complete: {parquet_file.name} ({file_size:.2f} MB)")
                # Clean up zip file
                zip_path.unlink()
                logger.info(f"[NOAA] Cleaned up zip file: {zip_path.name}")
                return parquet_file
        
        # Clean up zip file even if conversion failed
        if zip_path.exists():
            zip_path.unlink()
            logger.warning(f"[NOAA] Cleaned up zip file after conversion failure")
        
        logger.error(f"[NOAA] ✗ No matching parquet file created for {date_str_formatted}")
        return None
    
    def get_data_for_date_range(self, start_date: datetime, end_date: datetime) -> List[Path]:
        """
        Get parquet files for a date range
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            List of parquet file paths (may be incomplete if some dates unavailable)
        """
        from datetime import timedelta
        
        parquet_files = []
        current_date = start_date
        
        while current_date <= end_date:
            parquet_file = self.get_data_for_date(current_date)
            if parquet_file:
                parquet_files.append(parquet_file)
            current_date += timedelta(days=1)
        
        return parquet_files
    
    def verify_cache_integrity(self) -> Dict[str, Any]:
        """
        Verify cache directory and files are intact
        
        Returns:
            Dictionary with cache status information
        """
        import os
        
        status = {
            'cache_dir_exists': self.temp_dir.exists(),
            'cache_dir_writable': False,
            'cached_parquet_files': 0,
            'total_size_mb': 0.0,
            'years_with_data': []
        }
        
        if self.temp_dir.exists():
            status['cache_dir_writable'] = os.access(self.temp_dir, os.W_OK)
            
            # Count parquet files and calculate size
            parquet_files = list(self.temp_dir.rglob('*.parquet'))
            status['cached_parquet_files'] = len(parquet_files)
            status['total_size_mb'] = sum(f.stat().st_size for f in parquet_files) / (1024**2)
            
            # Find years with data
            year_dirs = [d for d in self.temp_dir.iterdir() if d.is_dir() and d.name.isdigit()]
            status['years_with_data'] = sorted([d.name for d in year_dirs])
        
        return status
    
    def cleanup_cache(self, keep_days: int = 7):
        """
        Clean up old cached files
        
        Args:
            keep_days: Keep files accessed within this many days
        """
        import time
        
        cutoff_time = time.time() - (keep_days * 24 * 3600)
        removed_count = 0
        
        for parquet_file in self.temp_dir.rglob("*.parquet"):
            if parquet_file.stat().st_atime < cutoff_time:
                parquet_file.unlink()
                removed_count += 1
        
        logger.info(f"Cleaned up {removed_count} old cached files")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached data"""
        parquet_files = list(self.temp_dir.rglob("*.parquet"))
        
        total_size = sum(f.stat().st_size for f in parquet_files)
        
        return {
            'cache_dir': str(self.temp_dir),
            'cached_files': len(parquet_files),
            'total_size_mb': total_size / (1024 * 1024),
            'available_years': self.AVAILABLE_YEARS
        }

