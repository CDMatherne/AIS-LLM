"""
ML Model Training and Testing Script
Train, test, and validate ML-based anomaly detection models using AIS data.
"""
import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import time
import json
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ml_training.log')
    ]
)

logger = logging.getLogger("ML_Training")

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from backend.data_connector import AISDataConnector
from backend.data_cache import AISDataCache
from backend.ml_anomaly_detection import MLAnomalyDetector
from backend.analysis_engine import AISAnalysisEngine

def setup_data_connector(data_path: str) -> AISDataConnector:
    """
    Set up data connector to load AIS data from specified path
    
    Args:
        data_path: Path to AIS data
        
    Returns:
        Data connector instance
    """
    # Configuration for local data source
    config = {
        "data_source": "local",
        "local": {
            "path": data_path,
            "file_format": "auto"
        }
    }
    
    try:
        # Create data connector
        connector = AISDataConnector(config)
        
        # Test connection
        test_result = connector.test_connection()
        
        if test_result["success"]:
            logger.info(f"Data connector successfully initialized: {test_result['message']}")
            logger.info(f"Data date range: {test_result['date_range']['available_start']} to {test_result['date_range']['available_end']}")
            return connector
        else:
            logger.error(f"Failed to initialize data connector: {test_result['message']}")
            raise RuntimeError(f"Data connector initialization failed: {test_result['message']}")
            
    except Exception as e:
        logger.error(f"Error setting up data connector: {e}")
        raise

def setup_data_cache(connector: AISDataConnector, cache_dir: str) -> AISDataCache:
    """
    Set up data cache for faster data access
    
    Args:
        connector: Data connector instance
        cache_dir: Directory for cache files
        
    Returns:
        Data cache instance
    """
    try:
        # Create data cache
        cache = AISDataCache(connector, cache_dir=cache_dir)
        
        # Get cache stats before sync
        stats_before = cache.get_cache_stats()
        logger.info(f"Initial cache stats: {stats_before}")
        
        return cache
    except Exception as e:
        logger.error(f"Error setting up data cache: {e}")
        raise

async def train_models(engine: AISAnalysisEngine, training_start_date: str, training_end_date: str) -> Dict[str, Any]:
    """
    Train ML models on specified date range
    
    Args:
        engine: Analysis engine with ML detector
        training_start_date: Start date for training data
        training_end_date: End date for training data
        
    Returns:
        Training results
    """
    logger.info(f"Starting model training on data from {training_start_date} to {training_end_date}")
    
    try:
        # Define progress callback
        async def progress_callback(stage: str, message: str, data: Optional[Dict] = None):
            logger.info(f"{stage}: {message}")
        
        # Train models
        results = await engine.train_ml_models(
            training_start_date, 
            training_end_date,
            progress_callback=progress_callback
        )
        
        if results['success']:
            logger.info(f"Model training completed successfully in {results['training_details']['training_time']:.2f} seconds")
            logger.info(f"Trained on {results['total_records']:,} records")
            logger.info(f"Models trained: {results['training_details']['models']}")
            return results
        else:
            logger.error(f"Model training failed: {results.get('error', 'Unknown error')}")
            return results
    
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        return {"success": False, "error": str(e)}

async def test_detection(engine: AISAnalysisEngine, test_start_date: str, test_end_date: str) -> Dict[str, Any]:
    """
    Test ML-based anomaly detection on specified date range
    
    Args:
        engine: Analysis engine with ML detector
        test_start_date: Start date for test data
        test_end_date: End date for test data
        
    Returns:
        Test results
    """
    logger.info(f"Testing anomaly detection on data from {test_start_date} to {test_end_date}")
    
    try:
        # Define progress callback
        async def progress_callback(stage: str, message: str, data: Optional[Dict] = None):
            logger.info(f"{stage}: {message}")
        
        # Run rule-based analysis
        rule_results = await engine.run_analysis(
            test_start_date, 
            test_end_date, 
            use_ml_detection=False,
            progress_callback=progress_callback
        )
        
        # Run combined rule-based and ML analysis
        combined_results = await engine.run_analysis(
            test_start_date, 
            test_end_date, 
            use_ml_detection=True,
            progress_callback=progress_callback
        )
        
        # Calculate statistics and compare results
        rule_anomaly_count = len(rule_results.get('anomalies', [])) if rule_results.get('success') else 0
        combined_anomaly_count = len(combined_results.get('anomalies', [])) if combined_results.get('success') else 0
        
        logger.info(f"Rule-based detection found {rule_anomaly_count} anomalies")
        logger.info(f"Combined detection found {combined_anomaly_count} anomalies")
        
        # Calculate overlap if both methods found anomalies
        overlap = {}
        if rule_anomaly_count > 0 and combined_anomaly_count > 0:
            rule_mmsis = set([a['MMSI'] for a in rule_results.get('anomalies', [])])
            combined_mmsis = set([a['MMSI'] for a in combined_results.get('anomalies', [])])
            
            common_mmsis = rule_mmsis & combined_mmsis
            only_rule = rule_mmsis - combined_mmsis
            only_ml = combined_mmsis - rule_mmsis
            
            overlap = {
                'both_methods': len(common_mmsis),
                'rule_only': len(only_rule),
                'ml_only': len(only_ml),
                'agreement_rate': len(common_mmsis) / len(rule_mmsis) if rule_mmsis else 0
            }
            
            logger.info(f"Agreement statistics:")
            logger.info(f"  - Vessels found by both methods: {overlap['both_methods']}")
            logger.info(f"  - Vessels found only by rules: {overlap['rule_only']}")
            logger.info(f"  - Vessels found only by ML: {overlap['ml_only']}")
            logger.info(f"  - Agreement rate: {overlap['agreement_rate']:.2f}")
        
        return {
            'success': True,
            'rule_results': {
                'anomaly_count': rule_anomaly_count,
                'success': rule_results.get('success', False)
            },
            'combined_results': {
                'anomaly_count': combined_anomaly_count,
                'success': combined_results.get('success', False)
            },
            'overlap': overlap
        }
    
    except Exception as e:
        logger.error(f"Error during detection testing: {e}")
        return {"success": False, "error": str(e)}

async def main():
    """Main function to train and test ML models"""
    try:
        logger.info("Starting ML training and testing")
        
        # Setup data paths
        data_path = "C:\\AIS_Data"
        cache_dir = "./data_cache"
        
        # Dates for training and testing
        training_start_date = "2024-10-01"
        training_end_date = "2024-11-30"
        testing_start_date = "2024-12-01"
        testing_end_date = "2025-01-31"
        
        logger.info(f"Data path: {data_path}")
        logger.info(f"Training period: {training_start_date} to {training_end_date}")
        logger.info(f"Testing period: {testing_start_date} to {testing_end_date}")
        
        # Set up data connector and cache
        connector = setup_data_connector(data_path)
        cache = setup_data_cache(connector, cache_dir)
        
        # Sync cache with data source
        logger.info("Syncing data cache with source...")
        sync_result = await cache.sync_with_source()
        logger.info(f"Cache sync results: {sync_result}")
        
        # Set up analysis engine
        logger.info("Initializing analysis engine...")
        engine = AISAnalysisEngine(cache)
        
        # Train ML models
        logger.info("Starting model training...")
        training_results = await train_models(engine, training_start_date, training_end_date)
        
        if not training_results.get('success', False):
            logger.error("Model training failed, cannot proceed to testing")
            return
        
        # Test detection
        logger.info("Starting detection testing...")
        testing_results = await test_detection(engine, testing_start_date, testing_end_date)
        
        # Save results
        results_dir = Path("./results")
        results_dir.mkdir(exist_ok=True)
        
        # Save testing results
        with open(results_dir / "testing_results.json", "w") as f:
            json.dump(testing_results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_dir}")
        logger.info("ML training and testing completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main process: {e}")
        raise

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
