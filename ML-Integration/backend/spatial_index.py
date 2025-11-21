"""
Spatial Indexing Module for AIS Data
Provides spatial indexing capabilities for efficient geographic queries.
"""
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Union
import pandas as pd
from shapely.geometry import Point, Polygon, MultiPolygon
import geopandas as gpd
import h3
import s2sphere
import rtree
from datetime import datetime
import time
import json

# Import GPU support if available
try:
    from gpu_support import (
        GPU_AVAILABLE, convert_to_gpu_dataframe, convert_to_cpu_dataframe
    )
except ImportError:
    GPU_AVAILABLE = False
    def convert_to_gpu_dataframe(df): return df
    def convert_to_cpu_dataframe(df): return df

logger = logging.getLogger(__name__)

# Constants
DEFAULT_H3_RESOLUTION = 7  # ~0.38 km2 hexagons
DEFAULT_S2_LEVEL = 15      # ~0.3 km2 cells
MAX_POINTS_PER_LEAF = 100  # For R-tree index

class SpatialIndexManager:
    """Manages multiple spatial indexing strategies for AIS data"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize spatial index manager"""
        # Set up cache directory
        if cache_dir is None:
            cache_dir = Path(__file__).parent / "spatial_cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache files
        self.h3_index_file = self.cache_dir / "h3_index.pkl"
        self.s2_index_file = self.cache_dir / "s2_index.pkl"
        self.rtree_index_path = str(self.cache_dir / "rtree_index")
        self.metadata_file = self.cache_dir / "spatial_index_metadata.json"
        
        # Initialize indices
        self.h3_index: Dict[str, Set[str]] = {}  # h3_cell -> set of MMSI values
        self.s2_index: Dict[str, Set[str]] = {}  # s2_cellid -> set of MMSI values
        self.rtree_index = None  # Will be initialized when needed
        self.vessel_positions: Dict[str, Dict[str, Any]] = {}  # MMSI -> position data
        
        # Metadata
        self.metadata = self._load_metadata()
        
        # Load existing indices if available
        self._load_indices()
        
        logger.info(f"Spatial index manager initialized at: {self.cache_dir}")
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load spatial index metadata"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load spatial index metadata: {e}")
        
        return {
            "indexed_dates": [],
            "h3_resolution": DEFAULT_H3_RESOLUTION,
            "s2_level": DEFAULT_S2_LEVEL,
            "last_update": None,
            "total_vessels": 0,
            "index_version": "1.0"
        }
    
    def _save_metadata(self):
        """Save spatial index metadata"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save spatial index metadata: {e}")
    
    def _load_indices(self):
        """Load existing indices from disk"""
        # Load H3 index
        if self.h3_index_file.exists():
            try:
                with open(self.h3_index_file, 'rb') as f:
                    self.h3_index = pickle.load(f)
                logger.info(f"Loaded H3 index with {len(self.h3_index)} cells")
            except Exception as e:
                logger.error(f"Failed to load H3 index: {e}")
                self.h3_index = {}
        
        # Load S2 index
        if self.s2_index_file.exists():
            try:
                with open(self.s2_index_file, 'rb') as f:
                    self.s2_index = pickle.load(f)
                logger.info(f"Loaded S2 index with {len(self.s2_index)} cells")
            except Exception as e:
                logger.error(f"Failed to load S2 index: {e}")
                self.s2_index = {}
        
        # Load vessel positions
        vessel_positions_file = self.cache_dir / "vessel_positions.pkl"
        if vessel_positions_file.exists():
            try:
                with open(vessel_positions_file, 'rb') as f:
                    self.vessel_positions = pickle.load(f)
                logger.info(f"Loaded positions for {len(self.vessel_positions)} vessels")
            except Exception as e:
                logger.error(f"Failed to load vessel positions: {e}")
                self.vessel_positions = {}
    
    def _create_rtree_index(self):
        """Create or load R-tree index for spatial queries"""
        # Check if index exists on disk
        if os.path.exists(self.rtree_index_path + ".dat"):
            try:
                # Load existing index
                self.rtree_index = rtree.index.Index(self.rtree_index_path)
                logger.info("Loaded R-tree index from disk")
                return
            except Exception as e:
                logger.error(f"Failed to load R-tree index, creating new one: {e}")
        
        # Create new index
        prop = rtree.index.Property()
        prop.dimension = 2
        prop.leaf_capacity = MAX_POINTS_PER_LEAF
        self.rtree_index = rtree.index.Index(self.rtree_index_path, properties=prop)
        
        # Populate from vessel positions if available
        if self.vessel_positions:
            for idx, (mmsi, pos_data) in enumerate(self.vessel_positions.items()):
                if 'LAT' in pos_data and 'LON' in pos_data:
                    lat, lon = pos_data['LAT'], pos_data['LON']
                    # Insert as point (same coords for min and max)
                    self.rtree_index.insert(idx, (lon, lat, lon, lat), obj=mmsi)
            
            logger.info(f"Created R-tree index with {len(self.vessel_positions)} vessels")
    
    def _save_indices(self):
        """Save indices to disk"""
        # Save H3 index
        try:
            with open(self.h3_index_file, 'wb') as f:
                pickle.dump(self.h3_index, f)
        except Exception as e:
            logger.error(f"Failed to save H3 index: {e}")
        
        # Save S2 index
        try:
            with open(self.s2_index_file, 'wb') as f:
                pickle.dump(self.s2_index, f)
        except Exception as e:
            logger.error(f"Failed to save S2 index: {e}")
        
        # Save vessel positions
        try:
            vessel_positions_file = self.cache_dir / "vessel_positions.pkl"
            with open(vessel_positions_file, 'wb') as f:
                pickle.dump(self.vessel_positions, f)
        except Exception as e:
            logger.error(f"Failed to save vessel positions: {e}")
        
        # Update and save metadata
        self.metadata["last_update"] = datetime.now().isoformat()
        self.metadata["total_vessels"] = len(self.vessel_positions)
        self._save_metadata()
    
    def build_indices(self, df: pd.DataFrame, date_str: str = None) -> Dict[str, Any]:
        """
        Build spatial indices from AIS data
        
        Args:
            df: DataFrame containing AIS data with LAT, LON, MMSI columns
            date_str: Optional date string for this data
            
        Returns:
            Dict with indexing results
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for spatial indexing")
            return {"success": False, "message": "Empty DataFrame provided"}
        
        required_cols = ['MMSI', 'LAT', 'LON']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            logger.error(f"Missing required columns for spatial indexing: {missing}")
            return {"success": False, "message": f"Missing columns: {missing}"}
        
        # Use GPU if available
        use_gpu = GPU_AVAILABLE
        if use_gpu:
            try:
                df_gpu = convert_to_gpu_dataframe(df)
                df = df_gpu
                logger.info("Using GPU acceleration for spatial indexing")
            except Exception as e:
                logger.warning(f"Failed to convert to GPU DataFrame: {e}")
                use_gpu = False
        
        # Convert to GeoDataFrame for spatial operations
        try:
            # If already GPU DataFrame, convert back temporarily for geopandas
            if use_gpu:
                df_cpu = convert_to_cpu_dataframe(df)
                gdf = gpd.GeoDataFrame(
                    df_cpu, 
                    geometry=gpd.points_from_xy(df_cpu['LON'], df_cpu['LAT']),
                    crs="EPSG:4326"
                )
            else:
                gdf = gpd.GeoDataFrame(
                    df, 
                    geometry=gpd.points_from_xy(df['LON'], df['LAT']),
                    crs="EPSG:4326"
                )
        except Exception as e:
            logger.error(f"Failed to create GeoDataFrame: {e}")
            return {"success": False, "message": f"GeoDataFrame creation failed: {e}"}
        
        # Start timing for performance metrics
        start_time = time.time()
        total_points = len(gdf)
        
        # Build H3 index
        h3_resolution = self.metadata.get("h3_resolution", DEFAULT_H3_RESOLUTION)
        h3_start = time.time()
        h3_points = 0
        
        # Process in batches to avoid memory issues
        batch_size = 10000
        for start_idx in range(0, len(gdf), batch_size):
            end_idx = min(start_idx + batch_size, len(gdf))
            batch = gdf.iloc[start_idx:end_idx]
            
            for idx, row in batch.iterrows():
                try:
                    mmsi = str(row['MMSI'])
                    lat, lon = row['LAT'], row['LON']
                    
                    if pd.isna(lat) or pd.isna(lon):
                        continue
                    
                    # Get H3 cell and add to index
                    h3_cell = h3.geo_to_h3(lat, lon, h3_resolution)
                    if h3_cell not in self.h3_index:
                        self.h3_index[h3_cell] = set()
                    self.h3_index[h3_cell].add(mmsi)
                    
                    # Update vessel position
                    self.vessel_positions[mmsi] = {
                        'LAT': lat,
                        'LON': lon,
                        'BaseDateTime': row['BaseDateTime'] if 'BaseDateTime' in row else None,
                        'LastUpdate': datetime.now().isoformat()
                    }
                    
                    h3_points += 1
                except Exception as e:
                    logger.debug(f"Failed to index point {idx}: {e}")
        
        h3_duration = time.time() - h3_start
        
        # Build S2 index
        s2_level = self.metadata.get("s2_level", DEFAULT_S2_LEVEL)
        s2_start = time.time()
        s2_points = 0
        
        for start_idx in range(0, len(gdf), batch_size):
            end_idx = min(start_idx + batch_size, len(gdf))
            batch = gdf.iloc[start_idx:end_idx]
            
            for idx, row in batch.iterrows():
                try:
                    mmsi = str(row['MMSI'])
                    lat, lon = row['LAT'], row['LON']
                    
                    if pd.isna(lat) or pd.isna(lon):
                        continue
                    
                    # Get S2 cell and add to index
                    latlng = s2sphere.LatLng.from_degrees(lat, lon)
                    cell = s2sphere.CellId.from_lat_lng(latlng).parent(s2_level)
                    cell_id = str(cell.id())
                    
                    if cell_id not in self.s2_index:
                        self.s2_index[cell_id] = set()
                    self.s2_index[cell_id].add(mmsi)
                    
                    s2_points += 1
                except Exception as e:
                    logger.debug(f"Failed to S2 index point {idx}: {e}")
        
        s2_duration = time.time() - s2_start
        
        # Initialize or update R-tree index
        rtree_start = time.time()
        
        if self.rtree_index is None:
            self._create_rtree_index()
        
        # Add new points to R-tree
        rtree_points = 0
        
        for start_idx in range(0, len(gdf), batch_size):
            end_idx = min(start_idx + batch_size, len(gdf))
            batch = gdf.iloc[start_idx:end_idx]
            
            for idx, row in batch.iterrows():
                try:
                    mmsi = str(row['MMSI'])
                    lat, lon = row['LAT'], row['LON']
                    
                    if pd.isna(lat) or pd.isna(lon):
                        continue
                    
                    # Generate a stable ID for this MMSI
                    # Using hash to avoid index size explosion for incremental updates
                    stable_id = hash(mmsi) % (10**10)
                    
                    # Insert as point (same coords for min and max)
                    self.rtree_index.insert(stable_id, (lon, lat, lon, lat), obj=mmsi)
                    
                    rtree_points += 1
                except Exception as e:
                    logger.debug(f"Failed to R-tree index point {idx}: {e}")
        
        rtree_duration = time.time() - rtree_start
        
        # Update metadata
        total_duration = time.time() - start_time
        
        if date_str and date_str not in self.metadata.get("indexed_dates", []):
            self.metadata.setdefault("indexed_dates", []).append(date_str)
        
        # Save indices to disk
        self._save_indices()
        
        return {
            "success": True,
            "total_points": total_points,
            "h3_points": h3_points,
            "s2_points": s2_points,
            "rtree_points": rtree_points,
            "h3_duration_seconds": h3_duration,
            "s2_duration_seconds": s2_duration,
            "rtree_duration_seconds": rtree_duration,
            "total_duration_seconds": total_duration,
            "message": f"Indexed {total_points} points successfully"
        }
    
    def query_by_point(self, lat: float, lon: float, radius_km: float = 5.0) -> List[str]:
        """
        Query vessels near a point within specified radius
        
        Args:
            lat: Latitude of query point
            lon: Longitude of query point
            radius_km: Search radius in kilometers
            
        Returns:
            List of MMSI values for vessels in the area
        """
        if self.rtree_index is None:
            self._create_rtree_index()
        
        # Convert radius from km to degrees (approximate)
        radius_degrees = radius_km / 111.0  # ~111km per degree at equator
        
        # Query R-tree
        results = list(self.rtree_index.intersection(
            (lon - radius_degrees, lat - radius_degrees, 
             lon + radius_degrees, lat + radius_degrees),
            objects=True
        ))
        
        # Extract MMSI values from results
        mmsi_list = [result.object for result in results]
        
        return mmsi_list
    
    def query_by_polygon(self, polygon: Union[Polygon, MultiPolygon, List[List[float]]]) -> List[str]:
        """
        Query vessels within a polygon
        
        Args:
            polygon: Shapely Polygon or MultiPolygon, or list of [lon, lat] coordinates
            
        Returns:
            List of MMSI values for vessels in the polygon
        """
        if isinstance(polygon, list):
            # Convert list of [lon, lat] to Polygon
            polygon = Polygon(polygon)
        
        # Get bounding box of polygon
        minx, miny, maxx, maxy = polygon.bounds
        
        # Query R-tree with bounding box first
        candidate_results = list(self.rtree_index.intersection(
            (minx, miny, maxx, maxy),
            objects=True
        ))
        
        # Filter candidates by exact polygon containment
        results = []
        for result in candidate_results:
            mmsi = result.object
            if mmsi in self.vessel_positions:
                vessel = self.vessel_positions[mmsi]
                point = Point(vessel['LON'], vessel['LAT'])
                if polygon.contains(point):
                    results.append(mmsi)
        
        return results
    
    def query_by_h3_cells(self, h3_cells: List[str]) -> List[str]:
        """
        Query vessels by H3 cell IDs
        
        Args:
            h3_cells: List of H3 cell IDs
            
        Returns:
            List of MMSI values for vessels in the cells
        """
        results = set()
        for cell in h3_cells:
            if cell in self.h3_index:
                results.update(self.h3_index[cell])
        
        return list(results)
    
    def query_by_region(self, region_name: str, regions_gdf: gpd.GeoDataFrame) -> List[str]:
        """
        Query vessels by named region in a GeoDataFrame
        
        Args:
            region_name: Name of region to query
            regions_gdf: GeoDataFrame containing region geometries with a 'name' column
            
        Returns:
            List of MMSI values for vessels in the region
        """
        if 'name' not in regions_gdf.columns:
            raise ValueError("GeoDataFrame must have a 'name' column")
        
        # Find the region
        region_row = regions_gdf[regions_gdf['name'] == region_name]
        if len(region_row) == 0:
            return []
        
        # Get the polygon
        polygon = region_row.iloc[0].geometry
        
        # Query by polygon
        return self.query_by_polygon(polygon)
