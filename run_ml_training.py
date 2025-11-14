"""
Runner script for ML training and testing
Executes the ML training and testing process
"""
import os
import sys
import asyncio
import logging
from datetime import datetime

# Add backend directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend'))

# Import the training script
from backend.train_test_ml import main

if __name__ == "__main__":
    # Configure logging
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"ml_training_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
    
    logger = logging.getLogger("ML_Training_Runner")
    logger.info("Starting ML training and testing process")
    
    try:
        # Run the main training and testing process
        asyncio.run(main())
        logger.info("Training and testing completed successfully")
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error running training and testing: {e}")
        sys.exit(1)
