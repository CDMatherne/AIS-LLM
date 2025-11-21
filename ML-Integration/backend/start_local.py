#!/usr/bin/env python3
"""
Simplified starter script for local testing
Uses local data sources and minimal dependencies
"""
import logging
import sys
import os
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main entrypoint for local testing"""
    # Ensure we're in the backend directory
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(script_dir)
    
    # Add both the backend directory and its parent to sys.path
    # This allows both direct imports and backend.* style imports
    parent_dir = os.path.dirname(script_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, str(parent_dir))
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    
    logger.info(f"Python path updated: {sys.path[:2]}...")
    
    # Load local environment
    from dotenv import load_dotenv
    load_dotenv()
    
    # Set data source to local
    os.environ['data_Source'] = 'local'
    
    # Import our fixed direct_bedrock_patch
    try:
        import direct_bedrock_patch
        logger.info("✓ Bedrock patch imported successfully")
    except ImportError as e:
        logger.warning(f"Failed to import direct_bedrock_patch: {e}")
    
    # Try to import key modules
    try:
        import optimized_claude_agent
        logger.info("✓ Claude agent imported successfully")
        
        import geographic_tools
        logger.info("✓ Geographic tools imported successfully")
        
        import data_connector
        logger.info("✓ Data connector imported successfully")
        
        logger.info("All core modules imported successfully!")
        logger.info("Local setup verification successful")
        return True
    except ImportError as e:
        logger.error(f"Failed to import core modules: {e}")
        logger.error("Local setup verification failed")
        return False
        
if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
