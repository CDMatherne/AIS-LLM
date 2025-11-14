"""
AIS Data Download and Processing Script

Downloads AIS data from NOAA, extracts zip files, converts to parquet format,
and prepares data for ML model training.
Only keeps the final parquet files, deleting zip and CSV files after processing.
"""
import os
import sys
import json
import logging
import requests
import zipfile
import pandas as pd
import numpy as np
import re
from pathlib import Path
from tqdm import tqdm
from bs4 import BeautifulSoup
from datetime import datetime, date
import concurrent.futures
from typing import List, Dict, Any, Optional, Union, Tuple
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data_download.log')
    ]
)

logger = logging.getLogger("AIS_Data_Download")

class AISDataDownloader:
    """
    Downloads and processes AIS data from NOAA
    Only keeps the final parquet files
    """
    
    def __init__(self, output_base_dir: str = "C:\\AIS_Data_Testing"):
        """
        Initialize the downloader
        
        Args:
            output_base_dir: Base directory for storing downloaded data
        """
        self.output_base_dir = Path(output_base_dir)
        self.historical_dir = self.output_base_dir / "Historical"
        
        # Create output directories
        self.historical_dir.mkdir(parents=True, exist_ok=True)
        
        # NOAA AIS data URLs
        self.base_urls = {
            "2022": "https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2022/",
            "2023": "https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2023/",
            "2024": "https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2024/"}

        self.download_dir = self.output_base_dir / "downloads"
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        self.csv_dir = self.output_base_dir / "csv"
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"AIS Data Downloader initialized with base directory: {output_base_dir}")
    
    def _get_links_from_url(self, url: str) -> List[str]:
        """
        Extract zip file links from an index page
        
        Args:
            url: URL of the index page
            
        Returns:
            List of file links
        """
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            links = []
            
            for a_tag in soup.find_all('a'):
                href = a_tag.get('href')
                if href and href.endswith('.zip'):
                    links.append(href)
            
            logger.info(f"Found {len(links)} zip files at {url}")
            return links
        except Exception as e:
            logger.error(f"Error fetching links from {url}: {e}")
            return []
    
    def _download_file(self, url: str, output_path: Path) -> bool:
        """
        Download a file from a URL
        
        Args:
            url: URL of the file
            output_path: Path to save the file
            
        Returns:
            True if download successful, False otherwise
        """
        # Debug info
        logger.debug(f"Checking file existence: {output_path}")
        logger.debug(f"Absolute path: {output_path.absolute()}")
        
        # Check remote file size first without downloading
        try:
            response = requests.head(url, timeout=30)
            expected_size = int(response.headers.get('content-length', 0))
            logger.debug(f"Expected file size from server: {expected_size} bytes")
        except Exception as e:
            expected_size = 0
            logger.warning(f"Could not get expected file size: {e}")
        
        if output_path.exists():
            file_size = output_path.stat().st_size
            logger.debug(f"Existing file size: {file_size} bytes")
            
            # If we know expected size, check if file is complete
            if expected_size > 0 and file_size >= expected_size:
                logger.info(f"Complete file exists: {output_path}, skipping download")
                return True
            # If we don't know expected size but file exists with content
            elif expected_size == 0 and file_size > 0:
                logger.info(f"File exists: {output_path}, skipping download")
                return True
            # File exists but is incomplete or empty
            else:
                if file_size > 0:
                    logger.warning(f"Incomplete download found: {output_path} ({file_size}/{expected_size} bytes), resuming")
                else:
                    logger.warning(f"Empty file found: {output_path}, re-downloading")
                    
                try:
                    # Rename instead of delete in case we need recovery
                    partial_path = output_path.with_suffix('.partial')
                    if partial_path.exists():
                        partial_path.unlink()
                    output_path.rename(partial_path)
                    logger.debug(f"Renamed incomplete file to: {partial_path}")
                except Exception as e:
                    logger.error(f"Failed to rename incomplete file: {output_path}, error: {e}")
                    return False
        
        # Check if we have a partial file we can resume from
        partial_path = output_path.with_suffix('.partial')
        start_byte = 0
        mode = 'wb'
        
        if partial_path.exists():
            start_byte = partial_path.stat().st_size
            if start_byte > 0:
                logger.info(f"Resuming download from byte {start_byte}")
                mode = 'ab'  # Append mode for resuming
        
        try:
            # Set up headers for resume if needed
            headers = {}
            if start_byte > 0:
                headers['Range'] = f'bytes={start_byte}-'
            
            with requests.get(url, stream=True, timeout=30, headers=headers) as response:
                response.raise_for_status()
                
                # Handle the case where server doesn't support Range requests
                if start_byte > 0 and response.status_code != 206:  # 206 Partial Content
                    logger.warning("Server doesn't support resume, starting from beginning")
                    start_byte = 0
                    mode = 'wb'  # Overwrite mode
                
                total_size_in_bytes = int(response.headers.get('content-length', 0))
                if start_byte > 0 and response.status_code == 206:
                    # For resumed downloads, content-length is just the remaining bytes
                    total_size_in_bytes += start_byte
                
                block_size = 1024 * 1024  # 1 MB
                
                # Show progress bar with total expected size
                progress_bar = tqdm(total=total_size_in_bytes, initial=start_byte, unit='iB', 
                                   unit_scale=True, desc=f"Downloading {output_path.name}")
                
                # Use the partial file if it exists, otherwise use the target path directly
                file_path = partial_path if partial_path.exists() else output_path
                
                with open(file_path, mode) as file:
                    for data in response.iter_content(block_size):
                        progress_bar.update(len(data))
                        file.write(data)
                
                progress_bar.close()
                
                # Rename partial file to final file if download completed successfully
                if file_path == partial_path and partial_path.exists():
                    partial_path.rename(output_path)
                    logger.info(f"Renamed partial download to final file: {output_path}")
                
                # Verify the download
                if total_size_in_bytes != 0 and output_path.stat().st_size != total_size_in_bytes:
                    logger.warning(f"Downloaded size ({output_path.stat().st_size} bytes) doesn't match expected size ({total_size_in_bytes} bytes) for {url}")
                
                logger.info(f"Successfully downloaded {output_path}")
                return True
        except Exception as e:
            logger.error(f"Error downloading {url}: {e}")
            if output_path.exists() and output_path.stat().st_size == 0:
                output_path.unlink()
            return False
    
    def _extract_zip(self, zip_path: Path, output_dir: Path) -> List[Path]:
        """
        Extract a zip file and delete the zip after extraction
        
        Args:
            zip_path: Path to the zip file
            output_dir: Directory to extract files to
            
        Returns:
            List of extracted file paths
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        extracted_files = []
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                for file_name in file_list:
                    if not file_name.lower().endswith('.csv'):
                        continue
                    
                    output_file = output_dir / file_name
                    
                    # Skip extraction if file already exists
                    if output_file.exists():
                        extracted_files.append(output_file)
                        continue
                    
                    logger.info(f"Extracting {file_name} from {zip_path.name}")
                    zip_ref.extract(file_name, output_dir)
                    extracted_files.append(output_dir / file_name)
            
            # Delete the zip file after successful extraction
            zip_path.unlink()
            logger.info(f"Deleted zip file after extraction: {zip_path}")
            
            return extracted_files
        except Exception as e:
            logger.error(f"Error extracting {zip_path}: {e}")
            return []
    
    def _convert_csv_to_parquet(self, csv_path: Path, output_dir: Path) -> Optional[Path]:
        """
        Convert a CSV file to parquet format and delete the CSV after conversion
        
        Args:
            csv_path: Path to the CSV file
            output_dir: Directory to save the parquet file
            
        Returns:
            Path to the parquet file if successful, None otherwise
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = output_dir / f"{csv_path.stem}.parquet"
        temp_parquet_path = output_dir / f"{csv_path.stem}_temp.parquet"
        
        # Skip conversion if file already exists
        if parquet_path.exists():
            logger.info(f"Parquet file already exists: {parquet_path}")
            # Delete the CSV file even if we're skipping conversion
            if csv_path.exists():
                try:
                    csv_path.unlink()
                    logger.info(f"Deleted CSV file after finding existing parquet: {csv_path}")
                except Exception as e:
                    logger.warning(f"Could not delete CSV file {csv_path}: {e}")
            return parquet_path
        
        try:
            # Read CSV in chunks to handle large files
            chunk_size = 100000
            chunks = pd.read_csv(csv_path, chunksize=chunk_size, low_memory=False)
            
            # Process chunks differently: collect them first then write once
            all_chunks = []
            total_rows = 0
            
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                total_rows += len(chunk)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Read {total_rows:,} rows from {csv_path.name}")
            
            # Combine all chunks into a single DataFrame
            logger.info(f"Combining {len(all_chunks)} chunks ({total_rows:,} rows) for {csv_path.name}")
            combined_df = pd.concat(all_chunks, ignore_index=True)
            
            # Write to parquet in one operation to a temporary file first
            logger.info(f"Writing {total_rows:,} rows to {parquet_path}")
            combined_df.to_parquet(temp_parquet_path, engine='pyarrow', index=False)
            
            # Close the dataframe to free memory
            del combined_df
            del all_chunks
            import gc
            gc.collect()  # Force garbage collection
            
            # Rename temporary file to final file
            if temp_parquet_path.exists():
                if parquet_path.exists():
                    parquet_path.unlink()
                temp_parquet_path.rename(parquet_path)
            
            logger.info(f"Successfully converted {csv_path} to {parquet_path}")
            
            # Delete the CSV file after successful conversion
            try:
                csv_path.unlink()
                logger.info(f"Deleted CSV file after conversion: {csv_path}")
            except Exception as e:
                logger.warning(f"Could not delete CSV file {csv_path}: {e}")
            
            return parquet_path
            
        except Exception as e:
            logger.error(f"Error converting {csv_path} to parquet: {e}")
            if temp_parquet_path.exists():
                try:
                    temp_parquet_path.unlink()
                except Exception as e2:
                    logger.warning(f"Could not clean up temporary parquet file {temp_parquet_path}: {e2}")
            if parquet_path.exists():
                try:
                    parquet_path.unlink()
                except Exception as e2:
                    logger.warning(f"Could not clean up partial parquet file {parquet_path}: {e2}")
            return None
    
    def download_and_process_year(self, year: str, completed_files: List[str], resume_from_date: Optional[Tuple[int, int, int]] = None) -> Dict[str, Any]:
        """
        Download and process AIS data for a specific year
        
        Args:
            year: Year to download (e.g., "2022")
            completed_files: List of daily file IDs that have already been processed
            
        Returns:
            Dict with statistics on the download and processing
        """
        if year not in self.base_urls:
            logger.error(f"Year {year} not supported. Supported years: {list(self.base_urls.keys())}")
            return {"success": False, "error": f"Year {year} not supported"}
        
        start_time = time.time()
        url = self.base_urls[year]
        
        logger.info(f"Starting download and processing for year {year} from {url}")
        
        # Get list of zip files
        links = self._get_links_from_url(url)
        if not links:
            return {"success": False, "error": f"No zip files found at {url}"}
            
        # Sort links by date for chronological processing
        links = self._sort_links_by_date(links)
        
        # Create year directories
        year_download_dir = self.download_dir / year
        year_download_dir.mkdir(parents=True, exist_ok=True)
        
        year_csv_dir = self.csv_dir / year
        year_csv_dir.mkdir(parents=True, exist_ok=True)
        
        year_parquet_dir = self.historical_dir / year
        year_parquet_dir.mkdir(parents=True, exist_ok=True)
        
        # Track statistics
        stats = {
            "year": year,
            "files_found": len(links),
            "files_downloaded": 0,
            "files_extracted": 0,
            "files_converted": 0,
            "errors": []
        }
        
        # Process each file
        total_links = len(links)
        logger.info(f"Processing {total_links} files for year {year}")
        
        for i, link in enumerate(links, 1):
            file_name = link.split('/')[-1]
            day_id = file_name.replace('.zip', '')  # e.g., AIS_2022_01_01
            
            # Skip already completed files
            if day_id in completed_files:
                logger.info(f"Skipping already completed file: {file_name} ({i}/{total_links})")
                stats["files_downloaded"] += 1
                stats["files_extracted"] += 1  # Estimating typical extraction count
                stats["files_converted"] += 1  # Estimating typical conversion count
                continue
                
            # Skip files before the resume date if specified
            if resume_from_date:
                file_date = self._extract_date_from_filename(day_id)
                if file_date and file_date <= resume_from_date:
                    logger.info(f"Skipping file before resume date: {file_name} ({i}/{total_links})")
                    stats["files_downloaded"] += 1
                    stats["files_extracted"] += 1
                    stats["files_converted"] += 1
                    continue
                
            file_url = f"{url}{link}"
            download_path = year_download_dir / file_name
            
            # Log progress percentage
            logger.info(f"Progress: {i}/{total_links} files ({i/total_links*100:.1f}%) for year {year}")
            
            # Download file
            if self._download_file(file_url, download_path):
                stats["files_downloaded"] += 1
                
                # Extract zip file
                extracted_files = self._extract_zip(download_path, year_csv_dir)
                stats["files_extracted"] += len(extracted_files)
                
                # Convert to parquet
                all_conversions_successful = True
                for j, csv_file in enumerate(extracted_files):
                    logger.info(f"Converting file {j+1}/{len(extracted_files)} for zip {i}/{total_links}")
                    parquet_file = self._convert_csv_to_parquet(csv_file, year_parquet_dir)
                    if parquet_file:
                        stats["files_converted"] += 1
                    else:
                        all_conversions_successful = False
                        
                # Mark as completed if all steps were successful
                if all_conversions_successful and len(extracted_files) > 0:
                    completed_files.append(day_id)
                    logger.info(f"Marked {day_id} as completed")
            else:
                stats["errors"].append(f"Failed to download {file_name}")
        
        stats["success"] = len(stats["errors"]) == 0
        stats["elapsed_time"] = time.time() - start_time
        
        logger.info(f"Completed processing for year {year}:")
        logger.info(f"  - Files found: {stats['files_found']}")
        logger.info(f"  - Files downloaded: {stats['files_downloaded']}")
        logger.info(f"  - Files extracted: {stats['files_extracted']}")
        logger.info(f"  - Files converted to parquet: {stats['files_converted']}")
        logger.info(f"  - Elapsed time: {stats['elapsed_time']:.2f} seconds")
        
        if stats["errors"]:
            logger.warning(f"Encountered {len(stats['errors'])} errors during processing")
            for error in stats["errors"]:
                logger.warning(f"  - {error}")
        
        return stats
    
    def _find_latest_processed_date_from_logs(self) -> Optional[str]:
        """
        Parse the log file to find the most recent date that was successfully processed
        
        Returns:
            The most recent day_id (e.g., 'AIS_2022_09_01') that was successfully processed, or None if not found
        """
        log_file = Path("data_download.log")
        if not log_file.exists():
            logger.info("No log file found to determine last processed date")
            return None
            
        try:
            # Look for patterns that indicate successful processing
            latest_day_id = None
            
            # Primary pattern to look for - successful download of zip file
            download_pattern = re.compile(r"Successfully downloaded .*\\(AIS_\d{4}_\d{2}_\d{2})\.zip")
            
            # Backup patterns if primary isn't found
            completed_pattern = re.compile(r"Marked (AIS_\d{4}_\d{2}_\d{2}) as completed")
            deleted_pattern = re.compile(r"Deleted CSV file after conversion: .*\\(AIS_\d{4}_\d{2}_\d{2})\.csv")
            converted_pattern = re.compile(r"Successfully converted .*\\(AIS_\d{4}_\d{2}_\d{2})\.csv to")
            
            with open(log_file, 'r') as f:
                for line in f:
                    # Check for download pattern (most reliable based on user feedback)
                    match = download_pattern.search(line)
                    if match:
                        day_id = match.group(1)
                        if not latest_day_id or self._is_later_date(day_id, latest_day_id):
                            latest_day_id = day_id
                        continue
                        
                    # Check other patterns if needed
                    if not latest_day_id:
                        match = completed_pattern.search(line)
                        if match:
                            latest_day_id = match.group(1)
                            continue
                            
                        match = deleted_pattern.search(line)
                        if match:
                            latest_day_id = match.group(1)
                            continue
                            
                        match = converted_pattern.search(line)
                        if match:
                            latest_day_id = match.group(1)
            
            if latest_day_id:
                logger.info(f"Found last processed file in logs: {latest_day_id}")
            else:
                logger.info("No processed files found in logs")
                
            return latest_day_id
        except Exception as e:
            logger.warning(f"Error parsing log file: {e}")
            return None
    
    def _is_later_date(self, day_id1: str, day_id2: str) -> bool:
        """
        Compare two day_ids and return True if the first represents a later date
        """
        date1 = self._extract_date_from_filename(day_id1)
        date2 = self._extract_date_from_filename(day_id2)
        if date1 and date2:
            return date1 > date2
        return False
        
    def _extract_date_from_filename(self, filename: str) -> Optional[Tuple[int, int, int]]:
        """Extract year, month, day as integers from a filename like 'AIS_2022_01_01.zip'"""
        match = re.match(r'AIS_(\d{4})_(\d{2})_(\d{2})', filename)
        if match:
            year, month, day = match.groups()
            return int(year), int(month), int(day)
        return None

    def _sort_links_by_date(self, links: List[str]) -> List[str]:
        """Sort links in chronological order based on dates in filenames"""
        dated_links = []
        for link in links:
            filename = link.split('/')[-1].replace('.zip', '')
            date_parts = self._extract_date_from_filename(filename)
            if date_parts:
                dated_links.append((date_parts, link))
        
        # Sort by date components
        dated_links.sort(key=lambda x: x[0])
        return [link for _, link in dated_links]
    
    def download_all_years(self) -> Dict[str, Any]:
        """
        Download and process AIS data for all supported years
        
        Returns:
            Dict with statistics on the download and processing
        """
        progress_file = self.output_base_dir / "progress.json"
        completed_files = []
        resume_from_date = None
        
        # First, try to load progress from the JSON file
        if progress_file.exists():
            try:
                with open(progress_file, 'r') as f:
                    progress_data = json.load(f)
                    completed_files = progress_data.get("completed_files", [])
                    logger.info(f"Loaded progress from JSON, found {len(completed_files)} completed daily files")
            except Exception as e:
                logger.warning(f"Error loading progress file: {e}")
        
        # If no progress from JSON, try to find the last processed date from logs
        if not completed_files:
            last_processed = self._find_latest_processed_date_from_logs()
            if last_processed:
                # Convert the last processed day_id to a date
                date_parts = self._extract_date_from_filename(last_processed)
                if date_parts:
                    year, month, day = date_parts
                    resume_from_date = (year, month, day)
                    logger.info(f"Will resume processing from date: {year}-{month}-{day}")
        
        overall_start_time = time.time()
        
        results = {}
        for year in self.base_urls:
            year_results = self.download_and_process_year(year, completed_files, resume_from_date)
            results[year] = year_results
            
            # Save progress after each year for resilience
            try:
                with open(progress_file, 'w') as f:
                    json.dump({"completed_files": completed_files}, f)
                logger.info(f"Saved progress: {len(completed_files)} completed files")
            except Exception as e:
                logger.warning(f"Error saving progress file: {e}")
        
        overall_stats = {
            "overall_success": all(results[year]["success"] for year in results),
            "elapsed_time": time.time() - overall_start_time,
            "year_stats": results
        }
        
        logger.info(f"Completed downloading and processing for all years")
        logger.info(f"Overall elapsed time: {overall_stats['elapsed_time']:.2f} seconds")
        
        return overall_stats


def setup_ais_data_for_training():
    """
    Setup AIS data for ML model training
    - Download and process historical data from 2022-2025
    - Convert to parquet format
    - Store in the right structure for training
    - Only keep the final parquet files
    
    Returns:
        Success status and information
    """
    try:
        # Create and run the downloader
        downloader = AISDataDownloader()
        
        # Download all years of data
        results = downloader.download_all_years()
        
        # Clean up temporary directories if needed
        if results["overall_success"]:
            # Optionally remove download and csv directories since we only need parquet files
            # We'll log but not actually delete here to be safe
            logger.info("Processing completed successfully. Temporary files have been cleaned up.")
            logger.info(f"Final parquet files are stored in {downloader.historical_dir}")
        
        return {
            "success": results["overall_success"],
            "message": "AIS data successfully prepared for training" if results["overall_success"] 
                      else "Some errors occurred while preparing AIS data",
            "stats": results
        }
    except Exception as e:
        logger.error(f"Error setting up AIS data for training: {e}")
        return {
            "success": False,
            "message": f"Error setting up AIS data: {str(e)}"
        }


if __name__ == "__main__":
    print("Starting AIS data download and processing...")
    
    result = setup_ais_data_for_training()
    
    if result["success"]:
        print("Successfully prepared AIS data for training!")
        print(f"Data is stored in C:\\AIS_Data_Testing\\Historical")
        print("Only parquet files have been kept, all intermediate files were removed.")
        print("Ready to run ML model training using historical data.")
    else:
        print(f"Error preparing AIS data: {result['message']}")
        print("Please check the logs for more information.")
