"""
ML Preprocessing Module for AIS Data
Provides data preparation and feature engineering for machine learning based anomaly detection.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

# Import spatial indexing for geographic features
try:
    from spatial_index import SpatialIndexManager
    SPATIAL_INDEX_AVAILABLE = True
except ImportError:
    SPATIAL_INDEX_AVAILABLE = False

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
DEFAULT_WINDOW_SIZE = 5  # Default window size for rolling statistics
MAX_HISTORY_DAYS = 30   # Maximum history to consider


class MLPreprocessor:
    """Machine Learning Preprocessor for AIS data"""
    
    def __init__(self, model_dir=None, use_gpu=True, spatial_index=None):
        """Initialize ML preprocessor"""
        # Set up model directory
        if model_dir is None:
            model_dir = Path(__file__).parent / "ml_models"
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # GPU support
        self.use_gpu = use_gpu and GPU_AVAILABLE
        
        # Spatial index for geographic features
        self.spatial_index = spatial_index
        self.has_spatial = SPATIAL_INDEX_AVAILABLE and spatial_index is not None
        
        # Initialize scalers and transformers
        self.scalers = {}
        self._load_scalers()
        
        logger.info(f"ML preprocessor initialized with model_dir: {self.model_dir}")
    
    def _load_scalers(self):
        """Load existing scalers from disk"""
        scaler_dir = self.model_dir / "scalers"
        
        if scaler_dir.exists():
            for scaler_file in scaler_dir.glob("*.joblib"):
                feature_set = scaler_file.stem
                try:
                    self.scalers[feature_set] = joblib.load(scaler_file)
                    logger.info(f"Loaded scaler for {feature_set}")
                except Exception as e:
                    logger.error(f"Failed to load scaler for {feature_set}: {e}")
    
    def _extract_trajectory_features(self, vessel_df):
        """Extract features related to vessel trajectory"""
        features = {}
        
        # Calculate speed statistics
        if 'SOG' in vessel_df.columns:
            features['speed_mean'] = vessel_df['SOG'].mean()
            features['speed_std'] = vessel_df['SOG'].std()
            features['speed_max'] = vessel_df['SOG'].max()
        
        # Calculate course statistics
        if 'COG' in vessel_df.columns:
            # Convert to radians for circular statistics
            cog_rad = np.radians(vessel_df['COG'])
            sin_cog = np.sin(cog_rad)
            cos_cog = np.cos(cog_rad)
            features['course_consistency'] = np.sqrt(sin_cog.mean()**2 + cos_cog.mean()**2)
        
        # Calculate acceleration features
        if 'SOG' in vessel_df.columns and len(vessel_df) > 1:
            vessel_df['speed_diff'] = vessel_df['SOG'].diff()
            vessel_df['time_diff'] = vessel_df['BaseDateTime'].diff().dt.total_seconds() / 3600  # in hours
            vessel_df['acceleration'] = vessel_df['speed_diff'] / vessel_df['time_diff'].replace(0, np.nan)
            features['acceleration_mean'] = vessel_df['acceleration'].mean()
            features['acceleration_max'] = vessel_df['acceleration'].abs().max()
        
        return features
    
    def _extract_temporal_features(self, vessel_df):
        """Extract features related to time patterns"""
        features = {}
        
        # Extract time-based features
        if 'BaseDateTime' in vessel_df.columns:
            # Hour of day patterns
            hours = vessel_df['BaseDateTime'].dt.hour
            features['hour_mean'] = hours.mean()
            
            # Day of week patterns
            day_of_week = vessel_df['BaseDateTime'].dt.dayofweek
            features['day_of_week_mean'] = day_of_week.mean()
            
            # Time consistency (standard deviation of hours)
            features['time_consistency'] = hours.std()
        
        return features
    
    def _extract_spatial_features(self, vessel_df):
        """Extract features related to spatial patterns"""
        features = {}
        
        if not self.has_spatial:
            return features
        
        # Last known position
        last_pos = vessel_df.iloc[-1]
        if 'LAT' in last_pos and 'LON' in last_pos:
            features['last_lat'] = last_pos['LAT']
            features['last_lon'] = last_pos['LON']
        
        # Calculate distance from shore/port if available
        # This would use the spatial index to find nearest shore/port
        
        return features
    
    def _extract_vessel_features(self, vessel_df):
        """Extract features related to vessel characteristics"""
        features = {}
        
        # Get vessel type
        if 'VesselType' in vessel_df.columns:
            vessel_type = vessel_df['VesselType'].iloc[0]
            features['vessel_type'] = vessel_type
        
        # Get vessel dimensions if available
        for dim in ['Length', 'Width', 'Draft']:
            if dim in vessel_df.columns:
                features[f'vessel_{dim.lower()}'] = vessel_df[dim].iloc[0]
        
        return features
    
    def preprocess_data(self, df, feature_sets=None, fit_scalers=False):
        """Preprocess AIS data and extract features"""
        if df.empty:
            logger.warning("Empty DataFrame provided for preprocessing")
            return pd.DataFrame()
        
        # Ensure BaseDateTime is datetime
        if 'BaseDateTime' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['BaseDateTime']):
            df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'])
        
        # Use GPU if available and requested
        if self.use_gpu:
            try:
                df_gpu = convert_to_gpu_dataframe(df)
                df = df_gpu
            except Exception as e:
                logger.warning(f"Failed to convert to GPU DataFrame: {e}")
                self.use_gpu = False
        
        # Sort by MMSI and time
        df = df.sort_values(['MMSI', 'BaseDateTime'])
        
        # Default feature sets
        if feature_sets is None:
            feature_sets = ['trajectory', 'temporal', 'vessel']
            if self.has_spatial:
                feature_sets.append('spatial')
        
        # Process by vessel to extract features
        feature_dfs = []
        
        # Group by MMSI to process each vessel separately
        for mmsi, vessel_df in df.groupby('MMSI'):
            try:
                # Process only if we have enough data
                if len(vessel_df) < 3:
                    continue
                
                vessel_features = {'MMSI': mmsi}
                
                # Extract features for requested feature sets
                if 'trajectory' in feature_sets:
                    traj_features = self._extract_trajectory_features(vessel_df)
                    vessel_features.update(traj_features)
                
                if 'temporal' in feature_sets:
                    temporal_features = self._extract_temporal_features(vessel_df)
                    vessel_features.update(temporal_features)
                
                if 'spatial' in feature_sets and self.has_spatial:
                    spatial_features = self._extract_spatial_features(vessel_df)
                    vessel_features.update(spatial_features)
                
                if 'vessel' in feature_sets:
                    vessel_type_features = self._extract_vessel_features(vessel_df)
                    vessel_features.update(vessel_type_features)
                
                # Add to feature dataframes
                feature_dfs.append(vessel_features)
                
            except Exception as e:
                logger.error(f"Error processing vessel {mmsi}: {e}")
                continue
        
        if not feature_dfs:
            logger.warning("No features extracted")
            return pd.DataFrame()
            
        # Combine all vessel features
        result_df = pd.DataFrame(feature_dfs)
        
        # Apply scaling if requested
        if fit_scalers:
            self._fit_and_transform_features(result_df, feature_sets)
        else:
            self._transform_features(result_df, feature_sets)
        
        return result_df
    
    def _fit_and_transform_features(self, df, feature_sets):
        """Fit scalers to data and transform"""
        # Ensure scaler directory exists
        scaler_dir = self.model_dir / "scalers"
        scaler_dir.mkdir(parents=True, exist_ok=True)
        
        # Create scalers for each feature set
        for feature_set in feature_sets:
            # Get columns for this feature set
            cols = [col for col in df.columns if col != 'MMSI']
            if not cols:
                continue
                
            # Create and fit scaler
            scaler = StandardScaler()
            df[cols] = scaler.fit_transform(df[cols].fillna(0))
            
            # Save scaler
            self.scalers[feature_set] = scaler
            joblib.dump(scaler, scaler_dir / f"{feature_set}.joblib")
    
    def _transform_features(self, df, feature_sets):
        """Transform features using pre-fit scalers"""
        for feature_set in feature_sets:
            if feature_set in self.scalers:
                # Get columns for this feature set
                cols = [col for col in df.columns if col != 'MMSI']
                if not cols:
                    continue
                    
                # Apply scaler
                df[cols] = self.scalers[feature_set].transform(df[cols].fillna(0))
        
        return df
