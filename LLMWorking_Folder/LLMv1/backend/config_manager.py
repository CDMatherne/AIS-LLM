"""
Configuration Manager

Centralized configuration management for the AIS Law Enforcement LLM application.
Handles loading environment variables and providing configuration settings to other modules.
"""
import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Default configuration values
DEFAULT_CONFIG = {
    'data_source': 'local',
    'aws_region': 'us-east-1',
    'aws_prefix': '',
    'local_file_format': 'auto',
    'output_directory': str(Path.home() / 'Downloads' / 'AISDS_Output'),
    'date_range': {
        'start': '2024-10-15',
        'end': '2025-03-30'
    }
}

def load_config() -> Dict[str, Any]:
    """
    Load configuration from environment variables with fallbacks to defaults
    """
    # Load environment variables from .env file if it exists
    try:
        dotenv_path = Path(__file__).parent.parent / '.env'
        if dotenv_path.exists():
            load_dotenv(dotenv_path)
            logger.info(f"Loaded environment variables from {dotenv_path}")
        else:
            # Try the backend directory's .env
            dotenv_path = Path(__file__).parent / '.env'
            if dotenv_path.exists():
                load_dotenv(dotenv_path)
                logger.info(f"Loaded environment variables from {dotenv_path}")
            else:
                logger.warning("No .env file found. Using environment variables and defaults.")
    except Exception as e:
        logger.warning(f"Error loading .env file: {e}")
    
    # Create base configuration
    config = DEFAULT_CONFIG.copy()
    
    # Load Claude API key (required)
    claude_api_key = os.getenv('CLAUDE_API_KEY')
    if claude_api_key:
        config['claude_api_key'] = claude_api_key
    else:
        logger.warning("No Claude API key found in environment variables")
    
    # Data source configuration
    data_source = os.getenv('DATA_SOURCE', 'local').lower()
    config['data_source'] = data_source
    
    # Source-specific configuration
    if data_source == 'aws':
        aws_config = {
            'access_key': os.getenv('AWS_ACCESS_KEY_ID'),
            'secret_key': os.getenv('AWS_SECRET_ACCESS_KEY'),
            'session_token': os.getenv('AWS_SESSION_TOKEN'),
            'region': os.getenv('AWS_REGION', config['aws_region']),
            'bucket': os.getenv('AWS_BUCKET'),
            'prefix': os.getenv('AWS_PREFIX', config['aws_prefix'])
        }
        
        if not aws_config['access_key'] or not aws_config['secret_key'] or not aws_config['bucket']:
            logger.warning("AWS data source selected but credentials are incomplete")
            
        config['aws'] = aws_config
        config['local'] = None
    else:
        # Local data source configuration
        local_config = {
            'path': os.getenv('LOCAL_DATA_PATH'),
            'file_format': os.getenv('LOCAL_FILE_FORMAT', config['local_file_format'])
        }
        
        if not local_config['path']:
            logger.warning("Local data source selected but no path provided")
            
        config['aws'] = None
        config['local'] = local_config
    
    # Output directory
    output_dir = os.getenv('OUTPUT_DIRECTORY')
    if output_dir:
        config['output_directory'] = output_dir
    
    return config

def get_config() -> Dict[str, Any]:
    """
    Get the current application configuration
    
    This is a convenience method that can be extended to add caching
    or other features in the future.
    """
    return load_config()

# Initialize configuration on module load
try:
    config = load_config()
    logger.info(f"Configuration loaded successfully. Data source: {config['data_source']}")
except Exception as e:
    logger.error(f"Failed to load configuration: {e}")
    config = DEFAULT_CONFIG
