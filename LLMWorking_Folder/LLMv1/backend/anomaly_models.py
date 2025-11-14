"""
Anomaly Detection Models Module
Implements machine learning models for detecting anomalies in AIS vessel data.
"""
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import json
import time
from datetime import datetime

# Import sklearn models
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor

# Try importing TensorFlow
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)

class AnomalyModel:
    """Base class for anomaly detection models"""
    
    def __init__(self, model_name, model_dir=None):
        """Initialize anomaly model"""
        self.model_name = model_name
        
        # Set up model directory
        if model_dir is None:
            model_dir = Path(__file__).parent / "ml_models"
        self.model_dir = Path(model_dir) / model_name
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Model state
        self.model = None
        self.is_fitted = False
        self.metadata = {
            "model_name": model_name,
            "created_at": datetime.now().isoformat(),
            "last_trained": None
        }
        
        # Try to load existing model
        self.load_model()
    
    def save_model(self):
        """Save model to disk"""
        if self.model is None:
            return
        
        try:
            # Save metadata
            self.metadata["last_saved"] = datetime.now().isoformat()
            with open(self.model_dir / "metadata.json", "w") as f:
                json.dump(self.metadata, f, indent=2)
            
            # Save model
            joblib.dump(self.model, self.model_dir / f"{self.model_name}_model.joblib")
            logger.info(f"Model {self.model_name} saved successfully")
        except Exception as e:
            logger.error(f"Failed to save model {self.model_name}: {e}")
    
    def load_model(self):
        """Load model from disk"""
        model_file = self.model_dir / f"{self.model_name}_model.joblib"
        metadata_file = self.model_dir / "metadata.json"
        
        if not model_file.exists():
            return False
        
        try:
            # Load metadata
            if metadata_file.exists():
                with open(metadata_file, "r") as f:
                    self.metadata = json.load(f)
            
            # Load model
            self.model = joblib.load(model_file)
            self.is_fitted = True
            logger.info(f"Model {self.model_name} loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            return False


class IsolationForestModel(AnomalyModel):
    """Isolation Forest model for anomaly detection"""
    
    def __init__(self, model_dir=None, contamination=0.01, n_estimators=100, random_state=42):
        """Initialize Isolation Forest model"""
        super().__init__("isolation_forest", model_dir)
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        
        if self.model is None:
            self.model = IsolationForest(
                contamination=contamination,
                n_estimators=n_estimators,
                random_state=random_state,
                n_jobs=-1
            )
    
    def fit(self, X):
        """Train the Isolation Forest model"""
        logger.info(f"Training Isolation Forest model on {len(X)} samples")
        start_time = time.time()
        
        try:
            # Filter out non-numeric columns
            X_numeric = X.select_dtypes(include=['number'])
            
            # Train the model
            self.model.fit(X_numeric)
            self.is_fitted = True
            
            # Update metadata
            self.metadata["last_trained"] = datetime.now().isoformat()
            self.metadata["training_time"] = time.time() - start_time
            self.metadata["num_features"] = X_numeric.shape[1]
            self.metadata["num_samples"] = X_numeric.shape[0]
            
            # Save model
            self.save_model()
            
            return {
                "success": True,
                "training_time": time.time() - start_time,
                "model": self.model_name
            }
            
        except Exception as e:
            logger.error(f"Error training Isolation Forest: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def predict(self, X):
        """Predict anomalies (1 for inliers, -1 for outliers)"""
        if not self.is_fitted:
            logger.error("Model not fitted")
            return np.zeros(len(X))
        
        try:
            # Filter non-numeric columns
            X_numeric = X.select_dtypes(include=['number'])
            
            # Handle missing columns
            missing_cols = set(self.metadata.get("feature_names", [])) - set(X_numeric.columns)
            if missing_cols:
                logger.warning(f"Missing columns in prediction data: {missing_cols}")
            
            # Make prediction
            return self.model.predict(X_numeric)
        
        except Exception as e:
            logger.error(f"Error predicting with Isolation Forest: {e}")
            return np.zeros(len(X))
    
    def decision_function(self, X):
        """Get anomaly scores (lower = more anomalous)"""
        if not self.is_fitted:
            logger.error("Model not fitted")
            return np.zeros(len(X))
        
        try:
            # Filter non-numeric columns
            X_numeric = X.select_dtypes(include=['number'])
            
            # Get anomaly scores
            return self.model.decision_function(X_numeric)
        
        except Exception as e:
            logger.error(f"Error getting anomaly scores: {e}")
            return np.zeros(len(X))


class DBSCANModel(AnomalyModel):
    """DBSCAN model for anomaly detection"""
    
    def __init__(self, model_dir=None, eps=0.5, min_samples=5):
        """Initialize DBSCAN model"""
        super().__init__("dbscan", model_dir)
        self.eps = eps
        self.min_samples = min_samples
        
        if self.model is None:
            self.model = DBSCAN(
                eps=eps,
                min_samples=min_samples,
                n_jobs=-1
            )
    
    def fit(self, X):
        """Train the DBSCAN model"""
        logger.info(f"Training DBSCAN model on {len(X)} samples")
        start_time = time.time()
        
        try:
            # Filter non-numeric columns
            X_numeric = X.select_dtypes(include=['number'])
            
            # Train model
            self.model.fit(X_numeric)
            self.is_fitted = True
            
            # Save cluster labels
            self.labels_ = self.model.labels_
            
            # Update metadata
            self.metadata["last_trained"] = datetime.now().isoformat()
            self.metadata["training_time"] = time.time() - start_time
            self.metadata["num_features"] = X_numeric.shape[1]
            self.metadata["num_samples"] = X_numeric.shape[0]
            self.metadata["num_clusters"] = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
            self.metadata["outlier_ratio"] = np.sum(self.labels_ == -1) / len(self.labels_)
            
            # Save model
            self.save_model()
            
            return {
                "success": True,
                "training_time": time.time() - start_time,
                "model": self.model_name,
                "num_clusters": self.metadata["num_clusters"],
                "outlier_ratio": self.metadata["outlier_ratio"]
            }
            
        except Exception as e:
            logger.error(f"Error training DBSCAN: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def predict(self, X):
        """Predict clusters/anomalies (-1 for outliers)"""
        if not self.is_fitted:
            logger.error("Model not fitted")
            return np.zeros(len(X))
        
        try:
            # Filter non-numeric columns
            X_numeric = X.select_dtypes(include=['number'])
            
            # Use nearest neighbors to assign to clusters
            # This is a simplified approach for DBSCAN prediction
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=1).fit(X_numeric)
            indices = nbrs.kneighbors(X_numeric, return_distance=False)
            
            return np.array([self.labels_[i[0]] for i in indices])
            
        except Exception as e:
            logger.error(f"Error predicting with DBSCAN: {e}")
            return np.zeros(len(X))


class LOFModel(AnomalyModel):
    """Local Outlier Factor model for anomaly detection"""
    
    def __init__(self, model_dir=None, n_neighbors=20, contamination=0.01):
        """Initialize LOF model"""
        super().__init__("lof", model_dir)
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        
        if self.model is None:
            self.model = LocalOutlierFactor(
                n_neighbors=n_neighbors,
                contamination=contamination,
                n_jobs=-1
            )
    
    def fit(self, X):
        """Train the LOF model"""
        logger.info(f"Training LOF model on {len(X)} samples")
        start_time = time.time()
        
        try:
            # Filter non-numeric columns
            X_numeric = X.select_dtypes(include=['number'])
            
            # LOF doesn't support true "fit" - it's fit_predict in one step
            # We'll store the data for later use in predict
            self.X_train = X_numeric
            self.y_pred = self.model.fit_predict(X_numeric)
            self.negative_outlier_factor_ = self.model.negative_outlier_factor_
            self.is_fitted = True
            
            # Update metadata
            self.metadata["last_trained"] = datetime.now().isoformat()
            self.metadata["training_time"] = time.time() - start_time
            self.metadata["num_features"] = X_numeric.shape[1]
            self.metadata["num_samples"] = X_numeric.shape[0]
            self.metadata["outlier_ratio"] = np.sum(self.y_pred == -1) / len(self.y_pred)
            
            # Save model
            self.save_model()
            
            return {
                "success": True,
                "training_time": time.time() - start_time,
                "model": self.model_name,
                "outlier_ratio": self.metadata["outlier_ratio"]
            }
            
        except Exception as e:
            logger.error(f"Error training LOF: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def predict(self, X):
        """Predict anomalies (1 for inliers, -1 for outliers)"""
        if not self.is_fitted:
            logger.error("Model not fitted")
            return np.zeros(len(X))
        
        try:
            # Filter non-numeric columns
            X_numeric = X.select_dtypes(include=['number'])
            
            # Create a new LOF model with fixed neighbors
            lof = LocalOutlierFactor(
                n_neighbors=self.n_neighbors, 
                contamination=self.contamination,
                novelty=True,
                n_jobs=-1
            )
            
            # Fit on the training data
            lof.fit(self.X_train)
            
            # Predict on the new data
            return lof.predict(X_numeric)
            
        except Exception as e:
            logger.error(f"Error predicting with LOF: {e}")
            return np.zeros(len(X))
    
    def decision_function(self, X):
        """Get anomaly scores (lower = more anomalous)"""
        if not self.is_fitted:
            logger.error("Model not fitted")
            return np.zeros(len(X))
        
        try:
            # Filter non-numeric columns
            X_numeric = X.select_dtypes(include=['number'])
            
            # Create a new LOF model with fixed neighbors
            lof = LocalOutlierFactor(
                n_neighbors=self.n_neighbors, 
                contamination=self.contamination,
                novelty=True,
                n_jobs=-1
            )
            
            # Fit on the training data
            lof.fit(self.X_train)
            
            # Get scores
            return lof.decision_function(X_numeric)
            
        except Exception as e:
            logger.error(f"Error getting LOF scores: {e}")
            return np.zeros(len(X))


# Factory function to create models
def create_model(model_type, **kwargs):
    """Create a new anomaly detection model of the specified type"""
    model_map = {
        "isolation_forest": IsolationForestModel,
        "dbscan": DBSCANModel,
        "lof": LOFModel
    }
    
    if model_type not in model_map:
        logger.error(f"Unknown model type: {model_type}")
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model_map[model_type](**kwargs)
