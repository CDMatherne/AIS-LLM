"""
Main FastAPI application for AIS Fraud Detection LLM Assistant
"""
# Initialize path first to make imports work in all contexts
import sys
import os
from pathlib import Path

# Add both the backend directory and its parent to sys.path
# This allows both direct imports and backend.* style imports
script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, str(parent_dir))
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from fastapi import FastAPI, WebSocket, HTTPException, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime
import uuid
import logging
from dotenv import load_dotenv
import pandas as pd  # Import at module level for consistency

# Load environment variables early
load_dotenv()

# Set up logging with console output for debugging
log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
enable_console = os.getenv('ENABLE_CONSOLE_LOGGING', 'true').lower() == 'true'
log_format = os.getenv('LOG_FORMAT', 'detailed')

# Configure logging format
if log_format == 'detailed':
    log_format_str = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
else:
    log_format_str = '%(asctime)s - %(levelname)s - %(message)s'

# Set up logging with console handler for debugging
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format=log_format_str,
    handlers=[
        logging.StreamHandler() if enable_console else logging.NullHandler()
    ]
)

logger = logging.getLogger(__name__)

# Log startup information
if enable_console:
    logger.info(f"Logging configured: Level={log_level}, Console={'enabled' if enable_console else 'disabled'}")

# Environment variables loaded by config_manager
from config_manager import get_config

# Helper function for importing modules in both contexts
def import_local_module(module_name):
    """Import a module that could be in 'backend' or in the current directory"""
    try:
        # Try importing assuming we're in the project root
        return __import__(f'backend.{module_name}', fromlist=['*'])
    except ImportError:
        # If that fails, we're likely in the backend directory
        return __import__(module_name, fromlist=['*'])

# Import our bedrock resilience patch (must be imported before any boto3 usage)
try:
    # Try relative import first (when running from parent directory)
    try:
        from backend import direct_bedrock_patch
    except ImportError:
        # If that fails, we're likely running from backend directory, try direct import
        import direct_bedrock_patch
except Exception as e:
    logger.warning(f"Failed to import direct_bedrock_patch: {e}")
    logger.info("Application will continue without AWS Bedrock resilience")

# Import our custom modules using the helper function
optimized_claude_agent = import_local_module('optimized_claude_agent')
OptimizedAISFraudDetectionAgent = optimized_claude_agent.OptimizedAISFraudDetectionAgent

# Import the consolidated bedrock resilience implementation
try:
    bedrock_resilience = import_local_module('bedrock_resilience')
    get_resilient_client = bedrock_resilience.get_resilient_client
    BEDROCK_RESILIENCE_AVAILABLE = True
except ImportError:
    logger.warning("bedrock_resilience module not found, continuing without AWS Bedrock resilience")
    BEDROCK_RESILIENCE_AVAILABLE = False

import boto3
import anthropic

# Import more modules using our helper function
analysis_engine = import_local_module('analysis_engine')
AISAnalysisEngine = analysis_engine.AISAnalysisEngine

geographic_tools = import_local_module('geographic_tools')
GeographicZoneManager = geographic_tools.GeographicZoneManager

data_connector = import_local_module('data_connector')
AISDataConnector = data_connector.AISDataConnector

data_cache = import_local_module('data_cache')
AISDataCache = data_cache.AISDataCache

session_manager = import_local_module('session_manager')
SessionManager = session_manager.SessionManager

gpu_support = import_local_module('gpu_support')
get_gpu_info = gpu_support.get_gpu_info
get_installation_instructions = gpu_support.get_installation_instructions
# Continue importing more modules using our helper function
export_utils = import_local_module('export_utils')
AISExporter = export_utils.AISExporter

visualization_engine = import_local_module('visualization_engine')
viz_engine = visualization_engine.viz_engine

map_creator = import_local_module('map_creator')
map_creator = map_creator.map_creator

chart_creator = import_local_module('chart_creator')
chart_creator = chart_creator.chart_creator

temp_file_manager = import_local_module('temp_file_manager')
temp_manager = temp_file_manager.temp_manager
cleanup_session = temp_file_manager.cleanup_session

interaction_logger = import_local_module('interaction_logger')
interaction_logger = interaction_logger.interaction_logger

# Import tool handlers (new registry-based system)
# First try the backend.tool_handlers path
try:
    from backend.tool_handlers.registry import get_handler, is_tool_registered, list_registered_tools
    import backend.tool_handlers.analysis_handlers
    import backend.tool_handlers.geographic_handlers
    TOOL_HANDLERS_PREFIX = 'backend.tool_handlers'
except ImportError:
    # If that fails, try with just tool_handlers
    try:
        from tool_handlers.registry import get_handler, is_tool_registered, list_registered_tools
        import tool_handlers.analysis_handlers
        import tool_handlers.geographic_handlers
        TOOL_HANDLERS_PREFIX = 'tool_handlers'
    except ImportError:
        logger.error("Failed to import tool handlers. Some functionality may be limited.")
        TOOL_HANDLERS_PREFIX = None
# Import the remaining handlers using the detected prefix
if TOOL_HANDLERS_PREFIX:
    try:
        __import__(f'{TOOL_HANDLERS_PREFIX}.chart_handlers')
        __import__(f'{TOOL_HANDLERS_PREFIX}.map_handlers')
        __import__(f'{TOOL_HANDLERS_PREFIX}.export_handlers')
        __import__(f'{TOOL_HANDLERS_PREFIX}.visualization_handlers')
        __import__(f'{TOOL_HANDLERS_PREFIX}.ui_handlers')
    except ImportError as e:
        logger.warning(f"Failed to import some tool handlers: {e}")
        logger.info("Some functionality may be limited.")


# Logging already configured

# Configuration now handled by config_manager module

# =============================================================================
# Default Output Directory Configuration
# =============================================================================

def get_default_output_directory():
    """
    Get the default output directory path.
    Creates AISDS_Output folder in user's Downloads directory.
    
    Returns:
        Path: Path to the default output directory
    """
    from pathlib import Path
    
    # Get user's home directory
    home_dir = Path.home()
    
    # Construct Downloads path (works on Windows, Mac, Linux)
    downloads_dir = home_dir / "Downloads"
    
    # Create AISDS_Output folder
    output_dir = downloads_dir / "AISDS_Output"
    
    # Create directory if it doesn't exist
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using output directory: {output_dir}")
    except Exception as e:
        logger.error(f"Failed to create output directory {output_dir}: {e}")
        # Fallback to current directory if Downloads is not accessible
        output_dir = Path("AISDS_Output")
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.warning(f"Using fallback output directory: {output_dir}")
    
    return str(output_dir)

# Initialize FastAPI
app = FastAPI(
    title="AIS Law Enforcement Assistant",
    description="Claude-powered AIS fraud detection system for law enforcement",
    version="1.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get default output directory
DEFAULT_OUTPUT_DIR = get_default_output_directory()

# Global instances
session_manager = SessionManager(session_timeout_minutes=120)  # 2 hour timeout
geo_manager = GeographicZoneManager()
exporter = AISExporter(output_directory=DEFAULT_OUTPUT_DIR)

# Create resilient Bedrock client using the factory function
resilient_bedrock_client = None

# Initialize Bedrock client if the module is available
if BEDROCK_RESILIENCE_AVAILABLE:
    try:
        # Get configuration from config_manager
        config = get_config()
        
        # Create AWS Bedrock client
        bedrock_client = None
        anthropic_client = None
        
        if os.getenv('CLAUDE_API_KEY'):
            anthropic_client = anthropic.Anthropic(api_key=os.getenv('CLAUDE_API_KEY'))
        
        # Initialize resilient client using factory function
        resilient_bedrock_client = get_resilient_client(
            bedrock_client=bedrock_client,
            anthropic_client=anthropic_client,
            failure_threshold=2,
            recovery_timeout=60
        )
        logger.info("Bedrock resilience layer initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Bedrock resilience layer: {e}")
else:
    logger.info("Skipping AWS Bedrock resilience layer initialization - module not available")

# Global data cache (initialized on startup)
data_cache: Optional[AISDataCache] = None
cache_sync_task: Optional[asyncio.Task] = None

# Register session_manager with tool handlers (must be done AFTER creation)
if TOOL_HANDLERS_PREFIX:
    try:
        if TOOL_HANDLERS_PREFIX == 'backend.tool_handlers':
            from backend.tool_handlers.dependencies import register_session_manager
        else:
            from tool_handlers.dependencies import register_session_manager
        register_session_manager(session_manager)
        logger.info("✅ SessionManager registered with tool handlers")
    except ImportError as e:
        logger.warning(f"Could not register SessionManager with tool handlers: {e}")
        logger.info("Some functionality may be limited.")

# Pydantic models
class SetupRequest(BaseModel):
    data_source: str  # 'aws' or 'local'
    claude_api_key: str
    aws: Optional[Dict[str, Any]] = None
    local: Optional[Dict[str, Any]] = None
    date_range: Dict[str, str]
    output_folder: Optional[str] = None  # Optional custom output directory
    user_name: Optional[str] = None  # Optional user name for personalization

class ChatMessage(BaseModel):
    session_id: str
    message: str
    map_context: Optional[Dict[str, Any]] = None

class AnalysisRequest(BaseModel):
    session_id: str
    start_date: str
    end_date: str
    geographic_zone: Optional[Dict[str, Any]] = None
    anomaly_types: Optional[List[str]] = None
    mmsi_filter: Optional[List[str]] = None


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "AIS Law Enforcement Assistant",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "setup": "/api/setup",
            "chat": "/api/chat",
            "websocket": "/ws/chat/{session_id}",
            "gpu_status": "/api/gpu-status",
            "health": "/api/health"
        }
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    gpu_info = get_gpu_info()
    
    # Check Bedrock service status
    bedrock_status = "unavailable"
    if resilient_bedrock_client:
        try:
            bedrock_status = "open" if resilient_bedrock_client.circuit_breaker.state == "OPEN" else "available"
        except AttributeError:
            # The dummy BedrockResilientClient class won't have a circuit_breaker attribute
            bedrock_status = "unavailable"
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "gpu_available": gpu_info['available'],
        "gpu_type": gpu_info['type'],
        "active_sessions": session_manager.get_session_count(),
        "bedrock_status": bedrock_status,
        "optimizer_enabled": True
    }


@app.get("/api/gpu-status")
async def get_gpu_status():
    """
    Get GPU status and information
    """
    gpu_info = get_gpu_info()
    instructions = get_installation_instructions()
    
    return {
        'gpu_info': gpu_info,
        'installation_instructions': instructions,
        'recommendations': _get_gpu_recommendations(gpu_info)
    }


def _get_gpu_recommendations(gpu_info):
    """Provide simplified recommendations based on GPU status"""
    if not gpu_info['available']:
        return {
            'status': 'no_gpu',
            'message': 'No GPU detected. System will use CPU processing.'
        }
    elif gpu_info['type'] == 'NVIDIA':
        return {
            'status': 'optimal',
            'message': f"NVIDIA GPU detected with CUDA. Optimal performance available."
        }
    elif gpu_info['type'] == 'AMD':
        return {
            'status': 'good',
            'message': f"AMD GPU detected with {gpu_info['backend']}."
        }


@app.post("/api/setup-from-env")
async def setup_from_env():
    """
    Initialize system using environment variables.
    """
    try:
        config = get_config()
        
        if not config:
            return {
                'success': False,
                'error': 'Environment variables not configured. Missing required variables.'
            }
        
        # Create data connector (with cache if available)
        connector_config = {
            'data_source': config['data_source'],
            'aws': config.get('aws'),
            'local': config.get('local'),
            'date_range': config.get('date_range', {})
        }
        
        data_connector = AISDataConnector(connector_config, data_cache=data_cache)
        
        # Test connection
        connection_test = data_connector.test_connection()
        if not connection_test['success']:
            return {
                'success': False,
                'error': connection_test.get('message', 'Connection test failed')
            }
        
        # Create analysis engine
        analysis_engine = AISAnalysisEngine(data_connector)
        
        # Create Claude agent
        agent = OptimizedAISFraudDetectionAgent(config['claude_api_key'])
        
        # Create session
        session_id = str(uuid.uuid4())
        session_manager.create_session(
            session_id, 
            agent, 
            data_connector=data_connector,
            config=connector_config
        )
        
        # Store analysis engine in session
        session = session_manager.get_session(session_id)
        session['analysis_engine'] = analysis_engine
        
        # Set output directory if specified in env
        if 'output_directory' in config:
            session['output_directory'] = config['output_directory']
        
        # Send initial welcome message from Claude
        data_source_info = "AWS S3" if config['data_source'] == "aws" else "local storage"
        welcome_response = await agent.chat(
            f"Hello! Please introduce yourself and explain that the system is NOW CONFIGURED and READY with {data_source_info} access. Explain your capabilities and the available date range (October 15, 2024 to March 30, 2025). Make it clear you can analyze the data immediately - no uploads needed.",
            session_context={}
        )
        
        logger.info(f"Session {session_id} created successfully from environment variables")
        
        return {
            'success': True,
            'session_id': session_id,
            'welcome_message': welcome_response['message'],
            'date_range': connection_test.get('date_range', {}),
            'system_info': analysis_engine.get_system_info(),
            'source': 'environment_variables',
            'config': {
                'data_source': config['data_source'],
                'aws': config.get('aws'),
                'local': config.get('local')
            }
        }
        
    except Exception as e:
        logger.error(f"Setup from env failed: {e}")
        return {
            'success': False,
            'error': str(e)
        }


@app.post("/api/setup")
async def setup_system(config: SetupRequest):
    """
    Initialize system with user configuration
    """
    try:
        # Validate Claude API key
        claude_api_key = config.claude_api_key
        if not claude_api_key:
            raise HTTPException(status_code=400, detail="Claude API key required")
        
        # Create data connector
        connector_config = {
            'data_source': config.data_source,
            'aws': config.aws,
            'local': config.local,
            'date_range': config.date_range
        }
        
        data_connector = AISDataConnector(connector_config, data_cache=data_cache)
        
        # Test connection
        connection_test = data_connector.test_connection()
        if not connection_test['success']:
            return {
                'success': False,
                'error': connection_test['message']
            }
        
        # Create analysis engine
        analysis_engine = AISAnalysisEngine(data_connector)
        
        # Create Claude agent
        agent = OptimizedAISFraudDetectionAgent(claude_api_key)
        
        # Create session
        session_id = str(uuid.uuid4())
        session_manager.create_session(
            session_id, 
            agent, 
            data_connector=data_connector,
            config=connector_config
        )
        
        # Store analysis engine in session
        session = session_manager.get_session(session_id)
        session['analysis_engine'] = analysis_engine
        
        # Store output folder if provided
        if config.output_folder:
            session['output_directory'] = config.output_folder
            # Update exporter with custom output directory
            global exporter
            exporter = AISExporter(output_directory=config.output_folder)
        
        # Store user name if provided
        if config.user_name:
            session['user_name'] = config.user_name
        
        # Send initial welcome message from Claude
        data_source_info = "AWS S3" if config.data_source == "aws" else "local storage"
        welcome_response = await agent.chat(
            f"Hello! Please introduce yourself and explain that the system is NOW CONFIGURED and READY with {data_source_info} access. Explain your capabilities and the available date range (October 15, 2024 to March 30, 2025). Make it clear you can analyze the data immediately - no uploads needed.",
            session_context={}
        )
        
        logger.info(f"Session {session_id} created successfully")
        
        return {
            'success': True,
            'session_id': session_id,
            'welcome_message': welcome_response['message'],
            'date_range': connection_test['date_range'],
            'system_info': analysis_engine.get_system_info()
        }
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        return {
            'success': False,
            'error': str(e)
        }


@app.get("/api/check-env-config")
async def check_env_config():
    """
    Check if environment variables are configured and test connection.
    Returns config status and connection test result.
    """
    config = get_config()
    
    if not config:
        return {
            'success': False,
            'env_configured': False,
            'message': 'Environment variables not configured. Missing required variables.',
            'missing': []
        }
    
    # Test connection
    try:
        connector_config = {
            'data_source': config['data_source'],
            'aws': config.get('aws'),
            'local': config.get('local'),
            'date_range': config.get('date_range', {})
        }
        
        data_connector = AISDataConnector(connector_config)
        connection_test = data_connector.test_connection()
        
        return {
            'success': connection_test['success'],
            'env_configured': True,
            'data_source': config['data_source'],
            'connection_test': connection_test,
            'message': 'Environment variables configured and connection tested'
        }
    except Exception as e:
        logger.error(f"Connection test failed: {e}")
        return {
            'success': False,
            'env_configured': True,
            'connection_test': {
                'success': False,
                'error': str(e)
            },
            'message': 'Environment variables configured but connection test failed'
        }


@app.post("/api/test-connection")
async def test_data_connection(config: dict):
    """
    Test data source connection without creating session
    """
    try:
        data_connector = AISDataConnector(config, data_cache=data_cache)
        result = data_connector.test_connection()
        return result
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


@app.post("/api/test-local-path")
async def test_local_path(request: dict):
    """
    Test if local directory is accessible and contains data files.
    Reads first 5 lines of first available file (ais-YYYY-MM-DD.parquet/csv or YYYY-MM-DD.parquet/csv)
    """
    try:
        import pandas as pd
        from pathlib import Path
        
        local_dir = request.get('local_directory')
        
        if not local_dir:
            return {"success": False, "error": "No directory specified"}
        
        path = Path(local_dir)
        
        if not path.exists():
            return {"success": False, "error": f"Directory does not exist: {local_dir}"}
        
        if not path.is_dir():
            return {"success": False, "error": f"Path is not a directory: {local_dir}"}
        
        # Look for files in preferred order: ais-YYYY-MM-DD.parquet, YYYY-MM-DD.parquet, then csv
        file_patterns = [
            "ais-*.parquet",  # Preferred format
            "*.parquet",       # Alternative format
            "ais-*.csv",       # CSV preferred format
            "*.csv"            # CSV alternative format
        ]
        
        first_file = None
        file_type = None
        
        for pattern in file_patterns:
            files = sorted(list(path.glob(pattern)))
            if files:
                first_file = files[0]
                if first_file.suffix == '.parquet':
                    file_type = 'parquet'
                else:
                    file_type = 'csv'
                break
        
        if not first_file:
            return {
                "success": False,
                "error": f"No data files found in {local_dir}. Expected format: ais-YYYY-MM-DD.parquet or YYYY-MM-DD.parquet (or .csv)"
            }
        
        # Read first 5 lines
        try:
            if file_type == 'parquet':
                # Parquet doesn't support nrows, so read full file and take first 5 rows
                df = pd.read_parquet(first_file)
                df = df.head(5)
            else:
                df = pd.read_csv(first_file, nrows=5)
            
            # Convert to dict for JSON serialization (df already has first 5 rows)
            sample_data = df.to_dict('records')
            columns = list(df.columns)
            
            return {
                "success": True,
                "file_found": first_file.name,
                "file_path": str(first_file),
                "file_type": file_type,
                "total_files": sum(len(list(path.glob(p))) for p in file_patterns),
                "columns": columns,
                "column_count": len(columns),
                "sample_data": sample_data,
                "row_count_sample": len(sample_data),
                "message": f"Successfully read first 5 rows from {first_file.name}. File format: {file_type.upper()}"
            }
            
        except Exception as e:
            logger.error(f"Error reading file {first_file}: {e}")
            return {
                "success": False,
                "error": f"Found file {first_file.name} but could not read it: {str(e)}"
            }
        
    except Exception as e:
        logger.error(f"Local path test error: {e}")
        return {"success": False, "error": str(e)}


@app.post("/api/test-aws")
async def test_aws_connection(config: dict):
    """
    Test AWS S3 connection and read first 5 lines of first available file.
    Looks for files in format: ais-YYYY-MM-DD.parquet/csv or YYYY-MM-DD.parquet/csv
    """
    try:
        import boto3
        import pandas as pd
        from botocore.exceptions import ClientError, NoCredentialsError
        from temp_file_manager import create_temp_file
        import os
        
        # Create S3 client with provided credentials
        session_config = {}
        if config.get('access_key') and config.get('secret_key'):
            session_config['aws_access_key_id'] = config['access_key']
            session_config['aws_secret_access_key'] = config['secret_key']
            if config.get('session_token'):
                session_config['aws_session_token'] = config['session_token']
        
        session_config['region_name'] = config.get('region', 'us-east-1')
        
        s3_client = boto3.client('s3', **session_config)
        
        # Test listing objects in bucket
        bucket = config.get('bucket')
        prefix = config.get('prefix', '')
        
        response = s3_client.list_objects_v2(
            Bucket=bucket,
            Prefix=prefix,
            MaxKeys=100  # Get more files to find first matching one
        )
        
        object_count = response.get('KeyCount', 0)
        
        if object_count == 0:
            return {
                "success": False,
                "error": f"No objects found in bucket '{bucket}' with prefix '{prefix}'. Expected format: ais-YYYY-MM-DD.parquet or YYYY-MM-DD.parquet (or .csv)"
            }
        
        # Find first file matching expected patterns (prefer ais-YYYY-MM-DD.parquet)
        first_file_key = None
        file_type = None
        
        objects = response.get('Contents', [])
        
        # Sort by key to get first file
        objects.sort(key=lambda x: x['Key'])
        
        # Look for files in preferred order
        for obj in objects:
            key = obj['Key']
            filename = os.path.basename(key)
            
            # Check if it matches our patterns
            if filename.startswith('ais-') and filename.endswith('.parquet'):
                first_file_key = key
                file_type = 'parquet'
                break
            elif filename.endswith('.parquet') and not filename.startswith('ais-'):
                # Check if it's YYYY-MM-DD format
                try:
                    from datetime import datetime
                    date_part = filename.replace('.parquet', '')
                    datetime.strptime(date_part, '%Y-%m-%d')
                    if not first_file_key:  # Only use if we haven't found ais-* yet
                        first_file_key = key
                        file_type = 'parquet'
                except ValueError:
                    pass
            elif filename.startswith('ais-') and filename.endswith('.csv'):
                if not first_file_key:
                    first_file_key = key
                    file_type = 'csv'
            elif filename.endswith('.csv') and not filename.startswith('ais-'):
                try:
                    from datetime import datetime
                    date_part = filename.replace('.csv', '')
                    datetime.strptime(date_part, '%Y-%m-%d')
                    if not first_file_key:
                        first_file_key = key
                        file_type = 'csv'
                except ValueError:
                    pass
        
        if not first_file_key:
            sample_files = [os.path.basename(obj['Key']) for obj in objects[:5]]
            return {
                "success": False,
                "error": f"No files found matching expected format (ais-YYYY-MM-DD.parquet/csv or YYYY-MM-DD.parquet/csv). Found {object_count} objects. Sample files: {', '.join(sample_files)}"
            }
        
        # Download and read first 5 lines
        temp_path = None
        try:
            # Create temp file
            temp_path = create_temp_file(suffix=f'.{file_type}', prefix='s3_test_')
            
            # Download file from S3
            s3_client.download_file(bucket, first_file_key, temp_path)
            
            # Read first 5 lines
            if file_type == 'parquet':
                # Parquet doesn't support nrows, so read full file and take first 5 rows
                df = pd.read_parquet(temp_path)
                df = df.head(5)
            else:
                df = pd.read_csv(temp_path, nrows=5)
            
            # Convert to dict for JSON serialization (df already has first 5 rows)
            sample_data = df.to_dict('records')
            columns = list(df.columns)
            
            return {
                "success": True,
                "bucket": bucket,
                "prefix": prefix,
                "file_found": os.path.basename(first_file_key),
                "file_key": first_file_key,
                "file_type": file_type,
                "object_count": object_count,
                "columns": columns,
                "column_count": len(columns),
                "sample_data": sample_data,
                "row_count_sample": len(sample_data),
                "message": f"Successfully read first 5 rows from {os.path.basename(first_file_key)}. File format: {file_type.upper()}"
            }
            
        except Exception as e:
            logger.error(f"Error reading S3 file {first_file_key}: {e}")
            return {
                "success": False,
                "error": f"Found file {os.path.basename(first_file_key)} but could not read it: {str(e)}"
            }
        finally:
            # Clean up temp file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass
        
    except NoCredentialsError:
        return {"success": False, "error": "AWS credentials not found or invalid"}
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_msg = e.response['Error']['Message']
        return {"success": False, "error": f"{error_code}: {error_msg}"}
    except Exception as e:
        logger.error(f"AWS test error: {e}")
        return {"success": False, "error": str(e)}


@app.post("/api/chat")
async def chat(message: ChatMessage):
    """
    Handle chat messages with Claude
    """
    # Start logging this interaction
    interaction_id = interaction_logger.start_interaction(message.session_id, message.message)
    
    agent = session_manager.get_agent(message.session_id)
    if not agent:
        interaction_logger.end_interaction(success=False)
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    try:
        # Get Claude's response
        response = await agent.chat(
            message.message,
            session_context={"map_context": message.map_context}
        )
        
        # Log LLM response
        interaction_logger.log_llm_response(response)
        
        # If Claude wants to use tools, execute them
        tool_calls = response.get("tool_calls", [])
        if tool_calls and isinstance(tool_calls, list):
            tool_results = []
            for tool_call in tool_calls:
                # Validate tool_call structure
                if not isinstance(tool_call, dict) or "name" not in tool_call or "input" not in tool_call or "id" not in tool_call:
                    logger.warning(f"Invalid tool_call structure: {tool_call}")
                    continue
                
                try:
                    result = await execute_tool(
                        tool_call["name"],
                        tool_call["input"],
                        message.session_id
                    )
                    tool_results.append({
                        "tool_call_id": tool_call["id"],
                        "result": result
                    })
                except Exception as tool_error:
                    logger.error(f"Error executing tool {tool_call.get('name', 'unknown')}: {tool_error}")
                    tool_results.append({
                        "tool_call_id": tool_call.get("id", "unknown"),
                        "result": {"error": str(tool_error)}
                    })
            
            # Send results back to Claude
            if tool_results:
                try:
                    final_response = await agent.process_tool_results(tool_results)
                    if not isinstance(final_response, dict) or "message" not in final_response:
                        logger.warning(f"Invalid final_response structure: {final_response}")
                        final_response = {"message": "Tool execution completed, but received invalid response format."}
                    
                    interaction_logger.end_interaction(success=True)
                    
                    return {
                        "message": final_response["message"],
                        "tool_results": tool_results,
                        "has_more_tools": len(final_response.get("tool_calls", [])) > 0
                    }
                except Exception as process_error:
                    logger.error(f"Error processing tool results: {process_error}")
                    interaction_logger.end_interaction(success=False)
                    raise HTTPException(status_code=500, detail=f"Error processing tool results: {str(process_error)}")
        
        interaction_logger.end_interaction(success=True)
        response_message = response.get("message", "")
        return {"message": response_message}
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        interaction_logger.end_interaction(success=False)
        raise HTTPException(status_code=500, detail=str(e))


async def execute_tool(tool_name: str, tool_input: Dict[str, Any], 
                       session_id: str, websocket: Optional[WebSocket] = None) -> Any:
    """
    Execute a tool called by Claude.
    
    Uses the new registry-based handler system for cleaner, more maintainable code.
    Falls back to legacy handlers if a tool is not registered.
    
    Args:
        tool_name: Name of the tool to execute
        tool_input: Tool parameters
        session_id: User session ID
        websocket: Optional WebSocket for progress updates
    
    Returns:
        Tool execution result
    """
    import time
    
    # Log tool execution start
    interaction_logger.log_tool_execution_start(tool_name, tool_input, session_id)
    start_time = time.time()
    
    # Set current session for tool handlers
    session_manager.current_session_id = session_id
    
    session = session_manager.get_session(session_id)
    if not session:
        return {"error": "Session not found"}
    
    analysis_engine = session.get('analysis_engine')
    
    # Create progress callback if websocket is available
    async def progress_callback(stage: str, message: str, data: Optional[Dict] = None):
        """Send progress updates via WebSocket and store for LLM access"""
        # Send to frontend via WebSocket
        if websocket:
            try:
                await websocket.send_json({
                    "type": "progress",
                    "stage": stage,
                    "message": message,
                    "data": data or {}
                })
            except Exception as e:
                logger.warning(f"Failed to send progress update: {e}")
        
        # Store progress in session for LLM to access
        if 'progress_updates' not in session:
            session['progress_updates'] = []
        session['progress_updates'].append({
            "stage": stage,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "data": data or {}
        })
        # Keep only last 50 progress updates to avoid memory issues
        if len(session['progress_updates']) > 50:
            session['progress_updates'] = session['progress_updates'][-50:]
    
    # Store progress callback in session for tool handlers to access
    session['progress_callback'] = progress_callback
    
    try:
        # NEW: Check registry-based handlers first
        if TOOL_HANDLERS_PREFIX and 'is_tool_registered' in globals() and is_tool_registered(tool_name):
            logger.debug(f"Executing tool via registry: {tool_name}")
            handler = get_handler(tool_name)
            result = await handler(tool_input, session_id)
            
            # Log successful execution
            duration_ms = (time.time() - start_time) * 1000
            interaction_logger.log_tool_execution_end(tool_name, result, duration_ms)
            
            return result
        
        # LEGACY: Fall back to old if/elif chain for any unregistered tools
        logger.warning(f"Tool '{tool_name}' not in registry, using legacy handler")
        if tool_name == "run_anomaly_analysis":
            result = await analysis_engine.run_analysis(
                start_date=tool_input["start_date"],
                end_date=tool_input["end_date"],
                geographic_zone=tool_input.get("geographic_zone"),
                anomaly_types=tool_input.get("anomaly_types", []),
                mmsi_filter=tool_input.get("mmsi_filter")
            )
            # Store result in session
            if result['success']:
                session_manager.store_analysis_result(
                    session_id,
                    result['analysis_id'],
                    result
                )
            return result
        
        elif tool_name == "get_vessel_history":
            result = await analysis_engine.get_vessel_tracks(
                mmsi_list=tool_input["mmsi"],
                start_date=tool_input["start_date"],
                end_date=tool_input["end_date"]
            )
            return result
        
        elif tool_name == "create_geographic_zone":
            zone_type = tool_input["zone_type"]
            
            if zone_type == "polygon":
                zone = geo_manager.create_polygon_zone(
                    tool_input["coordinates"],
                    tool_input["name"]
                )
            elif zone_type == "circle":
                center = tool_input["coordinates"]
                zone = geo_manager.create_circle_zone(
                    center[0], center[1],
                    tool_input["radius_nm"],
                    tool_input["name"]
                )
            elif zone_type == "oval":
                center = tool_input["coordinates"]
                zone = geo_manager.create_oval_zone(
                    center[0], center[1],
                    tool_input["major_axis_nm"],
                    tool_input["minor_axis_nm"],
                    tool_input.get("rotation", 0),
                    tool_input["name"]
                )
            elif zone_type == "rectangle":
                coords = tool_input["coordinates"]
                zone = geo_manager.create_rectangle_zone(
                    coords[0], coords[1], coords[2], coords[3],
                    tool_input["name"]
                )
            
            # Save zone to session
            session_manager.add_zone(session_id, zone)
            
            # Return serializable version
            return {
                "success": True,
                "zone": {
                    "name": zone["name"],
                    "type": zone["type"],
                    "bounds": zone["bounds"],
                    "area_km2": zone["area_km2"]
                }
            }
        
        elif tool_name == "identify_high_risk_vessels":
            result = await analysis_engine.get_top_anomaly_vessels(
                start_date=tool_input["start_date"],
                end_date=tool_input["end_date"],
                geographic_zone=tool_input.get("geographic_zone"),
                min_anomalies=tool_input.get("min_anomalies", 5),
                top_n=tool_input.get("top_n", 10)
            )
            return result
        
        elif tool_name == "list_all_water_bodies":
            # Get all water bodies from geographic tools
            water_bodies_stats = geo_manager.get_water_body_stats()
            return {
                "success": True,
                "total_water_bodies": water_bodies_stats["total"],
                "by_type": water_bodies_stats["by_type"],
                "message": "The system recognizes 24 major water bodies worldwide. Use 'lookup_geographic_region' to get detailed information about any specific water body."
            }
        
        elif tool_name == "identify_vessel_location":
            # Identify which water body a location is in
            # If MMSI provided but no coordinates, look up vessel position from session timespan
            lon = tool_input.get("longitude")
            lat = tool_input.get("latitude")
            
            if "mmsi" in tool_input and tool_input["mmsi"] and (lon is None or lat is None):
                # Get latest vessel position from session's analysis timespan
                timespan = session.get('analysis_timespan')
                if not timespan:
                    return {
                        "success": False,
                        "error": "No analysis timespan set. Please set a date range first using set_analysis_timespan, or provide longitude and latitude directly."
                    }
                
                # Get vessel tracks for the timespan
                try:
                    vessel_result = await analysis_engine.get_vessel_tracks(
                        mmsi_list=[tool_input["mmsi"]],
                        start_date=timespan["start_date"],
                        end_date=timespan["end_date"]
                    )
                    
                    if vessel_result.get("success") and tool_input["mmsi"] in vessel_result.get("tracks", {}):
                        points = vessel_result["tracks"][tool_input["mmsi"]].get("points", [])
                        if points:
                            latest_point = points[-1]  # Last point is latest (sorted by time)
                            lon = latest_point.get("LON")
                            lat = latest_point.get("LAT")
                            logger.info(f"Retrieved position for MMSI {tool_input['mmsi']}: ({lon}, {lat})")
                        else:
                            return {
                                "success": False,
                                "error": f"No position data found for vessel MMSI {tool_input['mmsi']} in the specified date range"
                            }
                    else:
                        return {
                            "success": False,
                            "error": vessel_result.get("error", f"No data found for vessel MMSI {tool_input['mmsi']}")
                        }
                except Exception as e:
                    logger.error(f"Error getting vessel position: {e}")
                    return {
                        "success": False,
                        "error": f"Error retrieving vessel position: {str(e)}"
                    }
            
            if lon is None or lat is None:
                return {
                    "success": False,
                    "error": "Missing coordinates. Provide longitude and latitude, or MMSI with a set analysis timespan."
                }
            location_info = geo_manager.identify_location(lon, lat)
            
            if location_info:
                return {
                    "success": True,
                    "location": {
                        "longitude": lon,
                        "latitude": lat,
                        "water_body": location_info["name"],
                        "type": location_info["type"],
                        "parent": location_info.get("parent", "")
                    },
                    "message": f"Location ({lon}, {lat}) is in the {location_info['name']}"
                }
            else:
                return {
                    "success": False,
                    "location": {
                        "longitude": lon,
                        "latitude": lat
                    },
                    "message": f"Location ({lon}, {lat}) is not in any defined water body (may be on land or in an undefined region)"
                }
        
        elif tool_name == "lookup_geographic_region":
            # Try new water bodies system first
            try:
                zone = geo_manager.get_water_body_zone(tool_input["region_name"])
                return {
                    "success": True,
                    "region": tool_input["region_name"],
                    "zone": {
                        "name": zone["name"],
                        "type": zone["type"],
                        "parent": zone.get("parent", ""),
                        "bounds": zone["bounds"],
                        "area_km2": zone["area_km2"],
                        "coordinates_count": len(zone["coordinates"])
                    },
                    "message": f"Water body '{zone['name']}' is a {zone['type']} covering approximately {zone['area_km2']:.0f} km²"
                }
            except ValueError:
                # Fall back to old named regions system
                try:
                    zone = geo_manager.named_region_to_zone(tool_input["region_name"])
                    return {
                        "success": True,
                        "region": tool_input["region_name"],
                        "zone": {
                            "name": zone["name"],
                            "type": zone["type"],
                            "bounds": zone["bounds"],
                            "area_km2": zone["area_km2"]
                        }
                    }
                except ValueError as e:
                    return {
                        "success": False,
                        "error": f"Unknown region: {tool_input['region_name']}. Use 'list_all_water_bodies' to see available regions."
                    }
        
        elif tool_name == "list_vessel_types":
            # List all vessel types, optionally filtered by category
            category = tool_input.get("category", "all")
            
            if category == "all":
                stats = geo_manager.get_vessel_type_stats()
                return {
                    "success": True,
                    "total_vessel_types": stats["total"],
                    "categories": stats["by_category"],
                    "message": f"There are {stats['total']} vessel types across 8 categories. Key types: Cargo (70-79), Tanker (80-89), Fishing (30), Law Enforcement (55)"
                }
            else:
                codes = geo_manager.get_vessel_types_by_category(category)
                vessel_list = []
                for code in codes:
                    info = geo_manager.get_vessel_type_info(code)
                    if info:
                        vessel_list.append(f"{code}: {info['name']}")
                
                return {
                    "success": True,
                    "category": category,
                    "vessel_types": vessel_list,
                    "count": len(codes),
                    "message": f"Category '{category}' contains {len(codes)} vessel types"
                }
        
        elif tool_name == "get_vessel_type_info":
            # Get detailed information about a vessel type
            vessel_type_code = tool_input["vessel_type_code"]
            info = geo_manager.get_vessel_type_info(vessel_type_code)
            
            if info:
                is_hazmat = geo_manager.is_hazardous_vessel(vessel_type_code)
                is_commercial = geo_manager.is_commercial(vessel_type_code)
                is_special = geo_manager.is_special_purpose(vessel_type_code)
                
                return {
                    "success": True,
                    "code": vessel_type_code,
                    "name": info["name"],
                    "category": info["category"],
                    "description": info["description"],
                    "is_hazardous_cargo": is_hazmat,
                    "is_commercial": is_commercial,
                    "is_special_purpose": is_special,
                    "message": f"Vessel type {vessel_type_code}: {info['name']} ({info['category']})"
                }
            else:
                return {
                    "success": False,
                    "error": f"Unknown vessel type code: {vessel_type_code}. Valid codes are 20-99."
                }
        
        elif tool_name == "set_analysis_timespan":
            # Set or update the date range for this session
            start_date = tool_input["start_date"]
            end_date = tool_input["end_date"]
            
            # Store in session data
            session_id = getattr(session_manager, 'current_session_id', None)
            if session_id and session_id in session_manager.sessions:
                session_manager.sessions[session_id]['analysis_timespan'] = {
                    "start_date": start_date,
                    "end_date": end_date,
                    "set_at": datetime.now().isoformat()
                }
                logger.info(f"Session {session_id} timespan set: {start_date} to {end_date}")
                
                return {
                    "success": True,
                    "start_date": start_date,
                    "end_date": end_date,
                    "message": f"Analysis timespan set to {start_date} through {end_date}. You can now run anomaly analysis for this period."
                }
            else:
                return {
                    "success": False,
                    "error": "Unable to set timespan - no active session"
                }
        
        elif tool_name == "get_current_timespan":
            # Get the current date range for this session
            session_id = getattr(session_manager, 'current_session_id', None)
            if session_id and session_id in session_manager.sessions:
                timespan = session_manager.sessions[session_id].get('analysis_timespan')
                if timespan:
                    return {
                        "success": True,
                        "timespan_set": True,
                        "start_date": timespan["start_date"],
                        "end_date": timespan["end_date"],
                        "set_at": timespan.get("set_at", "unknown"),
                        "message": f"Current timespan: {timespan['start_date']} to {timespan['end_date']}"
                    }
                else:
                    return {
                        "success": True,
                        "timespan_set": False,
                        "message": "No timespan has been set yet. Please use set_analysis_timespan to choose dates before running analysis."
                    }
            else:
                return {
                    "success": False,
                    "error": "Unable to get timespan - no active session"
                }
        
        elif tool_name == "generate_investigation_report":
            result = await analysis_engine.generate_report(
                analysis_id=tool_input["analysis_id"],
                report_type=tool_input.get("report_type", "summary"),
                include_maps=tool_input.get("include_maps", True),
                include_vessel_details=tool_input.get("include_vessel_details", True)
            )
            return result
        
        elif tool_name == "export_to_csv":
            # Get analysis result
            analysis_result = session_manager.get_analysis_result(
                session_id,
                tool_input["analysis_id"]
            )
            
            if not analysis_result:
                return {"error": "Analysis not found"}
            
            export_type = tool_input.get("export_type", "anomalies")
            
            if export_type == "anomalies":
                if not analysis_result.get('anomalies'):
                    return {'success': False, 'error': 'No anomalies to export'}
                
                import pandas as pd
                anomalies_df = pd.DataFrame(analysis_result['anomalies'])
                result = exporter.export_anomalies_csv(anomalies_df)
                
            else:  # statistics
                statistics = analysis_result.get('statistics', {})
                result = exporter.export_statistics_csv(statistics)
            
            if result['success']:
                result['download_url'] = f"/api/exports/download/{os.path.basename(result['file_path'])}"
            
            return result
        
        elif tool_name == "export_to_excel":
            # Get analysis result
            analysis_result = session_manager.get_analysis_result(
                session_id,
                tool_input["analysis_id"]
            )
            
            if not analysis_result:
                return {"error": "Analysis not found"}
            
            import pandas as pd
            statistics = analysis_result.get('statistics', {})
            anomalies_df = pd.DataFrame(analysis_result.get('anomalies', []))
            
            result = exporter.export_statistics_excel(statistics, anomalies_df)
            
            if result['success']:
                result['download_url'] = f"/api/exports/download/{os.path.basename(result['file_path'])}"
            
            return result
        
        elif tool_name == "list_available_exports":
            exports = exporter.list_exports()
            
            # Add download URLs
            for export in exports:
                export['download_url'] = f"/api/exports/download/{export['filename']}"
            
            return {
                "success": True,
                "exports": exports,
                "count": len(exports)
            }
        
        elif tool_name == "create_custom_visualization":
            # Get analysis result
            analysis_result = session_manager.get_analysis_result(
                session_id,
                tool_input["analysis_id"]
            )
            
            if not analysis_result:
                return {"error": "Analysis not found"}
            
            # Execute custom visualization
            import pandas as pd
            anomalies_df = pd.DataFrame(analysis_result.get('anomalies', []))
            
            result = viz_engine.create_and_execute(
                code=tool_input["code"],
                data=anomalies_df,
                parameters=tool_input.get("parameters", {}),
                name=tool_input.get("name"),
                description=tool_input.get("description"),
                save_to_registry=tool_input.get("save_to_registry", False)
            )
            
            # Add download URL if successful
            if result.get('success') and 'file_path' in result:
                filename = os.path.basename(result['file_path'])
                result['download_url'] = f"/api/exports/download/{filename}"
            
            return result
        
        elif tool_name == "execute_saved_visualization":
            # Get analysis result
            analysis_result = session_manager.get_analysis_result(
                session_id,
                tool_input["analysis_id"]
            )
            
            if not analysis_result:
                return {"error": "Analysis not found"}
            
            # Execute saved visualization
            import pandas as pd
            anomalies_df = pd.DataFrame(analysis_result.get('anomalies', []))
            
            result = viz_engine.execute_visualization(
                viz_id=tool_input["viz_id"],
                data=anomalies_df,
                parameters=tool_input.get("parameters", {})
            )
            
            # Add download URL if successful
            if result.get('success') and 'file_path' in result.get('result', {}):
                filename = os.path.basename(result['result']['file_path'])
                result['download_url'] = f"/api/exports/download/{filename}"
            
            return result
        
        elif tool_name == "list_custom_visualizations":
            viz_type = tool_input.get("viz_type")
            vizs = viz_engine.registry.list_visualizations(viz_type=viz_type)
            
            return {
                "success": True,
                "visualizations": vizs,
                "count": len(vizs),
                "message": f"Found {len(vizs)} custom visualizations. Use execute_saved_visualization to run them."
            }
        
        elif tool_name == "get_visualization_templates":
            templates = viz_engine.get_visualization_templates()
            
            return {
                "success": True,
                "templates": templates,
                "count": len(templates),
                "message": "Use these templates as a starting point for create_custom_visualization"
            }
        
        elif tool_name == "create_all_anomalies_map":
            # Get analysis result
            analysis_result = session_manager.get_analysis_result(
                session_id,
                tool_input["analysis_id"]
            )
            
            if not analysis_result:
                return {"error": "Analysis not found"}
            
            # Create map
            import pandas as pd
            anomalies_df = pd.DataFrame(analysis_result.get('anomalies', []))
            
            result = map_creator.create_all_anomalies_map(
                anomalies_df=anomalies_df,
                show_clustering=tool_input.get("show_clustering", True),
                show_heatmap=tool_input.get("show_heatmap", False),
                show_grid=tool_input.get("show_grid", False)
            )
            
            # Add download URL if successful
            if result.get('success') and 'file_path' in result:
                filename = os.path.basename(result['file_path'])
                result['download_url'] = f"/api/exports/download/{filename}"
            
            return result
        
        elif tool_name == "create_vessel_track_map":
            # Get analysis result
            analysis_result = session_manager.get_analysis_result(
                session_id,
                tool_input["analysis_id"]
            )
            
            if not analysis_result:
                return {"error": "Analysis not found"}
            
            # Create vessel track map
            import pandas as pd
            anomalies_df = pd.DataFrame(analysis_result.get('anomalies', []))
            mmsi = tool_input["mmsi"]
            
            # Filter data for specific vessel
            vessel_data = anomalies_df[anomalies_df['MMSI'] == int(mmsi)] if 'MMSI' in anomalies_df.columns else pd.DataFrame()
            
            result = map_creator.create_vessel_track_map(
                vessel_data=vessel_data,
                mmsi=mmsi,
                anomalies_df=vessel_data
            )
            
            # Add download URL if successful
            if result.get('success') and 'file_path' in result:
                filename = os.path.basename(result['file_path'])
                result['download_url'] = f"/api/exports/download/{filename}"
            
            return result
        
        elif tool_name == "create_anomaly_heatmap":
            # Get analysis result
            analysis_result = session_manager.get_analysis_result(
                session_id,
                tool_input["analysis_id"]
            )
            
            if not analysis_result:
                return {"error": "Analysis not found"}
            
            # Create heatmap
            import pandas as pd
            anomalies_df = pd.DataFrame(analysis_result.get('anomalies', []))
            
            result = map_creator.create_all_anomalies_map(
                anomalies_df=anomalies_df,
                output_filename="Anomaly_Heatmap.html",
                show_clustering=False,
                show_heatmap=True,
                show_grid=False,
                title="AIS Anomaly Density Heatmap"
            )
            
            # Add download URL if successful
            if result.get('success') and 'file_path' in result:
                filename = os.path.basename(result['file_path'])
                result['download_url'] = f"/api/exports/download/{filename}"
            
            return result
        
        elif tool_name == "create_anomaly_types_chart":
            # Get analysis result
            analysis_result = session_manager.get_analysis_result(
                session_id,
                tool_input["analysis_id"]
            )
            
            if not analysis_result:
                return {"error": "Analysis not found"}
            
            # Create chart
            import pandas as pd
            anomalies_df = pd.DataFrame(analysis_result.get('anomalies', []))
            
            result = chart_creator.create_anomaly_types_distribution(
                anomalies_df=anomalies_df,
                chart_type=tool_input.get("chart_type", "bar")
            )
            
            # Add download URL if successful
            if result.get('success') and 'file_path' in result:
                filename = os.path.basename(result['file_path'])
                result['download_url'] = f"/api/exports/download/{filename}"
            
            return result
        
        elif tool_name == "create_top_vessels_chart":
            # Get analysis result
            analysis_result = session_manager.get_analysis_result(
                session_id,
                tool_input["analysis_id"]
            )
            
            if not analysis_result:
                return {"error": "Analysis not found"}
            
            # Create chart
            import pandas as pd
            anomalies_df = pd.DataFrame(analysis_result.get('anomalies', []))
            
            result = chart_creator.create_top_vessels_chart(
                anomalies_df=anomalies_df,
                top_n=tool_input.get("top_n", 10)
            )
            
            # Add download URL if successful
            if result.get('success') and 'file_path' in result:
                filename = os.path.basename(result['file_path'])
                result['download_url'] = f"/api/exports/download/{filename}"
            
            return result
        
        elif tool_name == "create_anomalies_by_date_chart":
            # Get analysis result
            analysis_result = session_manager.get_analysis_result(
                session_id,
                tool_input["analysis_id"]
            )
            
            if not analysis_result:
                return {"error": "Analysis not found"}
            
            # Create chart
            import pandas as pd
            anomalies_df = pd.DataFrame(analysis_result.get('anomalies', []))
            
            result = chart_creator.create_anomalies_by_date_chart(
                anomalies_df=anomalies_df
            )
            
            # Add download URL if successful
            if result.get('success') and 'file_path' in result:
                filename = os.path.basename(result['file_path'])
                result['download_url'] = f"/api/exports/download/{filename}"
            
            return result
        
        elif tool_name == "create_3d_bar_chart":
            # Get analysis result
            analysis_result = session_manager.get_analysis_result(
                session_id,
                tool_input["analysis_id"]
            )
            
            if not analysis_result:
                return {"error": "Analysis not found"}
            
            # Create chart
            import pandas as pd
            anomalies_df = pd.DataFrame(analysis_result.get('anomalies', []))
            
            result = chart_creator.create_3d_bar_chart(
                anomalies_df=anomalies_df
            )
            
            # Add download URL if successful
            if result.get('success') and 'file_path' in result:
                filename = os.path.basename(result['file_path'])
                result['download_url'] = f"/api/exports/download/{filename}"
            
            return result
        
        elif tool_name == "create_scatterplot":
            # Get analysis result
            analysis_result = session_manager.get_analysis_result(
                session_id,
                tool_input["analysis_id"]
            )
            
            if not analysis_result:
                return {"error": "Analysis not found"}
            
            # Create chart
            import pandas as pd
            anomalies_df = pd.DataFrame(analysis_result.get('anomalies', []))
            
            result = chart_creator.create_scatterplot_interactive(
                anomalies_df=anomalies_df
            )
            
            # Add download URL if successful
            if result.get('success') and 'file_path' in result:
                filename = os.path.basename(result['file_path'])
                result['download_url'] = f"/api/exports/download/{filename}"
            
            return result
        
        else:
            result = {"error": f"Unknown tool: {tool_name}"}
            duration_ms = (time.time() - start_time) * 1000
            interaction_logger.log_tool_execution_end(tool_name, result, duration_ms)
            return result
            
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        interaction_logger.log_tool_execution_error(tool_name, e, duration_ms)
        logger.error(f"Tool execution error ({tool_name}): {e}")
        return {"error": str(e), "tool": tool_name}


@app.websocket("/ws/chat/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time chat"""
    await websocket.accept()
    
    agent = session_manager.get_agent(session_id)
    if not agent:
        await websocket.close(code=1008, reason="Session not found")
        return
    
    try:
        while True:
            # Receive message
            try:
                data = await websocket.receive_json()
            except ValueError as json_error:
                logger.error(f"Invalid JSON received: {json_error}")
                await websocket.send_json({
                    "type": "error",
                    "content": "Invalid message format. Please send valid JSON."
                })
                continue
            
            # Validate data structure
            if not isinstance(data, dict):
                logger.error(f"Invalid data type received: {type(data)}")
                await websocket.send_json({
                    "type": "error",
                    "content": "Invalid message format. Expected JSON object."
                })
                continue
            
            # Handle user selection responses from popups
            if data.get("type") == "user_selection":
                selection_type = data.get("selection_type")
                session = session_manager.get_session(session_id)
                
                # Store selection in session for LLM to access
                if not session.get("user_selections"):
                    session["user_selections"] = {}
                
                if selection_type == "date_range":
                    session["user_selections"]["date_range"] = {
                        "start_date": data.get("start_date"),
                        "end_date": data.get("end_date")
                    }
                    message = f"User selected date range: {data.get('start_date')} to {data.get('end_date')}"
                elif selection_type == "vessel_types":
                    session["user_selections"]["vessel_types"] = data.get("vessel_types", [])
                    message = f"User selected vessel types: {', '.join(data.get('vessel_types', []))}"
                elif selection_type == "anomaly_types":
                    session["user_selections"]["anomaly_types"] = data.get("anomaly_types", [])
                    message = f"User selected anomaly types: {', '.join(data.get('anomaly_types', []))}"
                elif selection_type == "output_formats":
                    session["user_selections"]["output_formats"] = data.get("output_formats", [])
                    message = f"User selected output formats: {', '.join(data.get('output_formats', []))}"
                else:
                    message = "User made a selection"
            else:
                message = data.get("message", "")
            
            # Validate message is not empty
            if not message or not message.strip():
                await websocket.send_json({
                    "type": "error",
                    "content": "Message cannot be empty. Please provide a valid message."
                })
                continue
            
            # Validate conversation history before processing new message
            # This ensures any orphaned tool_use blocks are cleaned up
            try:
                if hasattr(agent, '_validate_conversation_history'):
                    agent._validate_conversation_history()
            except Exception as e:
                logger.warning(f"Conversation history validation failed: {e}")
            
            # Process with Claude
            response = await agent.chat(message, session_context={})
            
            # Validate response structure
            if not isinstance(response, dict):
                raise ValueError(f"Invalid response type from agent: {type(response)}")
            
            # Send response (use get() with default to handle missing message key)
            response_message = response.get("message", "")
            await websocket.send_json({
                "type": "message",
                "content": response_message
            })
            
            # Execute any tool calls
            tool_calls = response.get("tool_calls", [])
            if tool_calls and isinstance(tool_calls, list):
                # Validate tool_calls structure
                valid_tool_calls = []
                tool_names = []
                for tc in tool_calls:
                    if isinstance(tc, dict) and "name" in tc and "input" in tc and "id" in tc:
                        valid_tool_calls.append(tc)
                        tool_names.append(tc["name"])
                    else:
                        logger.warning(f"Invalid tool_call structure: {tc}")
                
                if valid_tool_calls:
                    await websocket.send_json({
                        "type": "tools_executing",
                        "tools": tool_names
                    })
                    
                    tool_results = []
                    for tool_call in valid_tool_calls:
                        try:
                            result = await execute_tool(
                                tool_call["name"],
                                tool_call["input"],
                                session_id,
                                websocket=websocket  # Pass websocket for progress updates
                            )
                            tool_results.append({
                                "tool_call_id": tool_call["id"],
                                "tool_name": tool_call["name"],  # Include tool name for progress tracking
                                "result": result
                            })
                            
                            # Stream tool result
                            await websocket.send_json({
                                "type": "tool_result",
                                "tool_name": tool_call["name"],
                                "result": result
                            })
                        except Exception as tool_error:
                            logger.error(f"Error executing tool {tool_call.get('name', 'unknown')}: {tool_error}")
                            tool_results.append({
                                "tool_call_id": tool_call.get("id", "unknown"),
                                "tool_name": tool_call.get("name", "unknown"),
                                "result": {"error": str(tool_error)}
                            })
                    
                    # Get Claude's interpretation
                    if tool_results:
                        try:
                            final_response = await agent.process_tool_results(tool_results)
                            if isinstance(final_response, dict) and "message" in final_response:
                                await websocket.send_json({
                                    "type": "message",
                                    "content": final_response["message"]
                                })
                            else:
                                logger.warning(f"Invalid final_response structure: {final_response}")
                                await websocket.send_json({
                                    "type": "message",
                                    "content": "Tool execution completed, but received invalid response format."
                                })
                        except Exception as process_error:
                            logger.error(f"Error processing tool results: {process_error}")
                            await websocket.send_json({
                                "type": "message",
                                "content": f"Error processing tool results: {str(process_error)}"
                            })
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()


@app.get("/api/sessions/{session_id}/info")
async def get_session_info(session_id: str):
    """Get information about a session"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session_id,
        "created_at": session['created_at'].isoformat(),
        "last_activity": session['last_activity'].isoformat(),
        "zones_count": len(session['zones']),
        "analyses_count": len(session['analyses']),
        "conversation_length": session['agent'].get_conversation_length()
    }


async def background_cache_sync():
    """Background task to periodically sync cache with data source"""
    global data_cache
    
    # Wait 1 hour before first sync
    await asyncio.sleep(3600)
    
    while True:
        try:
            if data_cache:
                logger.info("Running background cache sync...")
                result = await data_cache.sync_with_source()
                if result.get("success"):
                    logger.info(f"Background sync complete: {result.get('message')}")
                else:
                    logger.warning(f"Background sync had issues: {result.get('message')}")
            
            # Wait 6 hours before next sync
            await asyncio.sleep(21600)  # 6 hours
        except asyncio.CancelledError:
            logger.info("Background cache sync task cancelled")
            break
        except Exception as e:
            logger.error(f"Error in background cache sync: {e}")
            await asyncio.sleep(3600)  # Wait 1 hour before retrying on error


@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    global data_cache, cache_sync_task
    
    logger.info("=== AIS Law Enforcement Assistant Starting ===")
    logger.info(f"GPU Support: {get_gpu_info()}")
    logger.info("Claude Model: Auto-detection enabled (will select best available model per session)")
    
    # Log registered tools
    try:
        if TOOL_HANDLERS_PREFIX and 'list_registered_tools' in globals():
            registered_tools = list_registered_tools()
            logger.info(f"Tool Handler Registry: {len(registered_tools)} tools registered")
            logger.debug(f"Registered tools: {', '.join(sorted(registered_tools))}")
        else:
            logger.warning("Tool handlers not available - list_registered_tools not imported")
    except Exception as e:
        logger.warning(f"Could not list registered tools: {e}")
    
    logger.info("Output Capabilities: Maps, Charts, Exports, Dynamic Visualizations")
    
    # Initialize data cache if data source is configured
    try:
        config = get_config()  # Use get_config() instead of non-existent load_config_from_env()
        if config and config.get('data_source') == 'aws':
            # Create connector config dict (AISDataConnector expects specific format)
            connector_config = {
                'data_source': config.get('data_source'),
                'aws': config.get('aws'),
                'local': config.get('local'),
                'date_range': config.get('date_range', {})
            }
            # Create a temporary connector for cache initialization
            temp_connector = AISDataConnector(connector_config)
            data_cache = AISDataCache(temp_connector)
            
            # Initial sync - load all available data
            logger.info("Initializing data cache - loading all available data...")
            sync_result = await data_cache.sync_with_source(force_full_reload=True)
            
            if sync_result.get("success"):
                stats = data_cache.get_cache_stats()
                logger.info(f"✅ Data cache initialized: {stats['cached_dates']} dates, {stats['total_records']:,} records, {stats['cache_size_mb']:.2f} MB")
            else:
                logger.warning(f"⚠️ Cache initialization had issues: {sync_result.get('message')}")
            
            # Start background sync task
            cache_sync_task = asyncio.create_task(background_cache_sync())
            logger.info("Background cache sync task started (checks for new data every 6 hours)")
        else:
            logger.info("Data cache disabled (local data source or no config)")
    except Exception as e:
        logger.error(f"Failed to initialize data cache: {e}")
        logger.warning("Continuing without cache - data will be loaded on-demand")
    
    # Clean up old temporary files (7+ days old)
    try:
        cleaned = temp_manager.cleanup_old_system_temps(days_old=7)
        logger.info(f"Cleaned {cleaned} old temporary files from previous sessions")
    except Exception as e:
        logger.warning(f"Failed to clean old temporary files: {e}")
    
    logger.info("=== Server Ready ===")


# Cache management endpoints have been consolidated into the new config_manager module
# These endpoints were rarely used in the main workflow


@app.post("/api/export/csv")
async def export_to_csv(request: dict):
    """
    Export analysis results to CSV
    """
    session_id = request.get('session_id')
    analysis_id = request.get('analysis_id')
    export_type = request.get('export_type', 'anomalies')  # 'anomalies' or 'statistics'
    
    if not session_id or not analysis_id:
        raise HTTPException(status_code=400, detail="session_id and analysis_id required")
    
    # Get analysis result from session
    result = session_manager.get_analysis_result(session_id, analysis_id)
    if not result:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    try:
        if export_type == 'anomalies':
            # Export anomalies CSV
            if not result.get('anomalies'):
                return {'success': False, 'error': 'No anomalies to export'}
            
            anomalies_df = pd.DataFrame(result['anomalies'])
            export_result = exporter.export_anomalies_csv(anomalies_df)
            
        elif export_type == 'statistics':
            # Export statistics CSV
            statistics = result.get('statistics', {})
            export_result = exporter.export_statistics_csv(statistics)
        
        else:
            raise HTTPException(status_code=400, detail="Invalid export_type")
        
        return export_result
        
    except Exception as e:
        logger.error(f"Export error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/export/excel")
async def export_to_excel(request: dict):
    """
    Export analysis results to Excel
    """
    session_id = request.get('session_id')
    analysis_id = request.get('analysis_id')
    
    if not session_id or not analysis_id:
        raise HTTPException(status_code=400, detail="session_id and analysis_id required")
    
    # Get analysis result from session
    result = session_manager.get_analysis_result(session_id, analysis_id)
    if not result:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    try:
        statistics = result.get('statistics', {})
        anomalies_df = pd.DataFrame(result.get('anomalies', []))
        
        export_result = exporter.export_statistics_excel(statistics, anomalies_df)
        return export_result
        
    except Exception as e:
        logger.error(f"Excel export error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/exports/list")
async def list_exports():
    """
    List all available export files
    """
    try:
        exports = exporter.list_exports()
        return {
            'success': True,
            'exports': exports,
            'count': len(exports)
        }
    except Exception as e:
        logger.error(f"Error listing exports: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/exports/download/{filename}")
async def download_export(filename: str):
    """
    Download an export file
    """
    from fastapi.responses import FileResponse
    
    file_info = exporter.get_export_file_info(filename)
    if not file_info:
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_info['path'],
        filename=filename,
        media_type='application/octet-stream'
    )


@app.post("/api/visualizations/create")
async def create_custom_visualization(request: dict):
    """
    Create and execute a custom visualization
    Claude can generate code on-the-fly to create new visualizations
    """
    session_id = request.get('session_id')
    analysis_id = request.get('analysis_id')
    code = request.get('code')
    parameters = request.get('parameters', {})
    name = request.get('name')
    description = request.get('description')
    save_to_registry = request.get('save_to_registry', False)
    
    if not all([session_id, analysis_id, code]):
        raise HTTPException(status_code=400, detail="session_id, analysis_id, and code required")
    
    # Get analysis result
    result = session_manager.get_analysis_result(session_id, analysis_id)
    if not result:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    try:
        # Get anomalies data
        anomalies_df = pd.DataFrame(result.get('anomalies', []))
        
        # Execute visualization
        viz_result = viz_engine.create_and_execute(
            code=code,
            data=anomalies_df,
            parameters=parameters,
            name=name,
            description=description,
            save_to_registry=save_to_registry
        )
        
        # Add download URL if successful
        if viz_result['success'] and 'file_path' in viz_result:
            filename = os.path.basename(viz_result['file_path'])
            viz_result['download_url'] = f"/api/exports/download/{filename}"
        
        return viz_result
        
    except Exception as e:
        logger.error(f"Error creating visualization: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/visualizations/execute/{viz_id}")
async def execute_saved_visualization(viz_id: str, request: dict):
    """
    Execute a saved visualization from the registry
    """
    session_id = request.get('session_id')
    analysis_id = request.get('analysis_id')
    parameters = request.get('parameters', {})
    
    if not all([session_id, analysis_id]):
        raise HTTPException(status_code=400, detail="session_id and analysis_id required")
    
    # Get analysis result
    result = session_manager.get_analysis_result(session_id, analysis_id)
    if not result:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    try:
        anomalies_df = pd.DataFrame(result.get('anomalies', []))
        
        # Execute from registry
        viz_result = viz_engine.execute_visualization(
            viz_id=viz_id,
            data=anomalies_df,
            parameters=parameters
        )
        
        # Add download URL if successful
        if viz_result['success'] and 'file_path' in viz_result.get('result', {}):
            filename = os.path.basename(viz_result['result']['file_path'])
            viz_result['download_url'] = f"/api/exports/download/{filename}"
        
        return viz_result
        
    except Exception as e:
        logger.error(f"Error executing visualization: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/visualizations/list")
async def list_visualizations(viz_type: Optional[str] = None):
    """
    List all available custom visualizations
    """
    try:
        vizs = viz_engine.registry.list_visualizations(viz_type=viz_type)
        return {
            'success': True,
            'visualizations': vizs,
            'count': len(vizs)
        }
    except Exception as e:
        logger.error(f"Error listing visualizations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/visualizations/templates")
async def get_visualization_templates():
    """
    Get code templates for creating custom visualizations
    """
    templates = viz_engine.get_visualization_templates()
    return {
        'success': True,
        'templates': templates,
        'count': len(templates)
    }


@app.post("/api/visualizations/{viz_id}/rate")
async def rate_visualization(viz_id: str, request: dict):
    """
    Rate a visualization (1-5 stars)
    """
    rating = request.get('rating')
    
    if not rating or rating < 1 or rating > 5:
        raise HTTPException(status_code=400, detail="Rating must be between 1 and 5")
    
    try:
        viz_engine.registry.add_rating(viz_id, rating)
        return {
            'success': True,
            'viz_id': viz_id,
            'rating': rating
        }
    except Exception as e:
        logger.error(f"Error rating visualization: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/visualizations/{viz_id}")
async def delete_visualization(viz_id: str):
    """
    Delete a visualization from the registry
    """
    try:
        success = viz_engine.registry.delete_visualization(viz_id)
        if success:
            return {'success': True, 'message': f'Visualization {viz_id} deleted'}
        else:
            raise HTTPException(status_code=404, detail="Visualization not found")
    except Exception as e:
        logger.error(f"Error deleting visualization: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ===== Map Generation Endpoints =====

@app.post("/api/maps/all-anomalies")
async def create_all_anomalies_map(request: dict):
    """
    Create an interactive map showing all anomalies
    """
    session_id = request.get('session_id')
    analysis_id = request.get('analysis_id')
    show_clustering = request.get('show_clustering', True)
    show_heatmap = request.get('show_heatmap', False)
    show_grid = request.get('show_grid', False)
    
    if not session_id or not analysis_id:
        raise HTTPException(status_code=400, detail="session_id and analysis_id required")
    
    # Get analysis result
    result = session_manager.get_analysis_result(session_id, analysis_id)
    if not result:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    try:
        import pandas as pd
        anomalies_df = pd.DataFrame(result.get('anomalies', []))
        
        if anomalies_df.empty:
            return {
                'success': False,
                'error': 'No anomalies to map'
            }
        
        # Create map
        map_result = map_creator.create_all_anomalies_map(
            anomalies_df=anomalies_df,
            show_clustering=show_clustering,
            show_heatmap=show_heatmap,
            show_grid=show_grid
        )
        
        return map_result
        
    except Exception as e:
        logger.error(f"Error creating anomalies map: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/maps/vessel-track")
async def create_vessel_track_map(request: dict):
    """
    Create a map showing a vessel's complete track
    """
    session_id = request.get('session_id')
    analysis_id = request.get('analysis_id')
    mmsi = request.get('mmsi')
    
    if not all([session_id, analysis_id, mmsi]):
        raise HTTPException(status_code=400, detail="session_id, analysis_id, and mmsi required")
    
    # Get analysis result
    result = session_manager.get_analysis_result(session_id, analysis_id)
    if not result:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    try:
        import pandas as pd
        
        # Get all vessel data from session (would need to be stored during analysis)
        # For now, just use anomalies for that vessel
        anomalies_df = pd.DataFrame(result.get('anomalies', []))
        vessel_data = anomalies_df[anomalies_df['MMSI'] == int(mmsi)] if 'MMSI' in anomalies_df.columns else pd.DataFrame()
        vessel_anomalies = vessel_data.copy()
        
        if vessel_data.empty:
            return {
                'success': False,
                'error': f'No data found for vessel {mmsi}'
            }
        
        # Create track map
        map_result = map_creator.create_vessel_track_map(
            vessel_data=vessel_data,
            mmsi=mmsi,
            anomalies_df=vessel_anomalies
        )
        
        return map_result
        
    except Exception as e:
        logger.error(f"Error creating vessel track map: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/maps/heatmap")
async def create_anomaly_heatmap(request: dict):
    """
    Create a heatmap showing anomaly density
    """
    session_id = request.get('session_id')
    analysis_id = request.get('analysis_id')
    
    if not session_id or not analysis_id:
        raise HTTPException(status_code=400, detail="session_id and analysis_id required")
    
    # Get analysis result
    result = session_manager.get_analysis_result(session_id, analysis_id)
    if not result:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    try:
        import pandas as pd
        anomalies_df = pd.DataFrame(result.get('anomalies', []))
        
        if anomalies_df.empty:
            return {
                'success': False,
                'error': 'No anomalies to map'
            }
        
        # Create heatmap (uses the all_anomalies_map with heatmap enabled and no markers)
        map_result = map_creator.create_all_anomalies_map(
            anomalies_df=anomalies_df,
            output_filename="Anomaly_Heatmap.html",
            show_clustering=False,
            show_heatmap=True,
            show_grid=False,
            title="AIS Anomaly Density Heatmap"
        )
        
        return map_result
        
    except Exception as e:
        logger.error(f"Error creating heatmap: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ===== Chart Generation Endpoints =====

@app.post("/api/charts/anomaly-types")
async def create_anomaly_types_chart(request: dict):
    """
    Create a chart showing distribution of anomaly types
    """
    session_id = request.get('session_id')
    analysis_id = request.get('analysis_id')
    chart_type = request.get('chart_type', 'bar')  # 'bar', 'pie', or 'both'
    
    if not session_id or not analysis_id:
        raise HTTPException(status_code=400, detail="session_id and analysis_id required")
    
    result = session_manager.get_analysis_result(session_id, analysis_id)
    if not result:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    try:
        import pandas as pd
        anomalies_df = pd.DataFrame(result.get('anomalies', []))
        
        chart_result = chart_creator.create_anomaly_types_distribution(
            anomalies_df=anomalies_df,
            chart_type=chart_type
        )
        
        return chart_result
        
    except Exception as e:
        logger.error(f"Error creating anomaly types chart: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/charts/top-vessels")
async def create_top_vessels_chart(request: dict):
    """
    Create a chart showing vessels with most anomalies
    """
    session_id = request.get('session_id')
    analysis_id = request.get('analysis_id')
    top_n = request.get('top_n', 10)
    
    if not session_id or not analysis_id:
        raise HTTPException(status_code=400, detail="session_id and analysis_id required")
    
    result = session_manager.get_analysis_result(session_id, analysis_id)
    if not result:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    try:
        import pandas as pd
        anomalies_df = pd.DataFrame(result.get('anomalies', []))
        
        chart_result = chart_creator.create_top_vessels_chart(
            anomalies_df=anomalies_df,
            top_n=top_n
        )
        
        return chart_result
        
    except Exception as e:
        logger.error(f"Error creating top vessels chart: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/charts/by-date")
async def create_by_date_chart(request: dict):
    """
    Create a time series chart showing anomalies over time
    """
    session_id = request.get('session_id')
    analysis_id = request.get('analysis_id')
    
    if not session_id or not analysis_id:
        raise HTTPException(status_code=400, detail="session_id and analysis_id required")
    
    result = session_manager.get_analysis_result(session_id, analysis_id)
    if not result:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    try:
        import pandas as pd
        anomalies_df = pd.DataFrame(result.get('anomalies', []))
        
        chart_result = chart_creator.create_anomalies_by_date_chart(
            anomalies_df=anomalies_df
        )
        
        return chart_result
        
    except Exception as e:
        logger.error(f"Error creating by-date chart: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/charts/3d-bar")
async def create_3d_bar_chart(request: dict):
    """
    Create a 3D bar chart showing anomalies by type and date
    """
    session_id = request.get('session_id')
    analysis_id = request.get('analysis_id')
    
    if not session_id or not analysis_id:
        raise HTTPException(status_code=400, detail="session_id and analysis_id required")
    
    result = session_manager.get_analysis_result(session_id, analysis_id)
    if not result:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    try:
        import pandas as pd
        anomalies_df = pd.DataFrame(result.get('anomalies', []))
        
        chart_result = chart_creator.create_3d_bar_chart(
            anomalies_df=anomalies_df
        )
        
        return chart_result
        
    except Exception as e:
        logger.error(f"Error creating 3D bar chart: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/charts/scatterplot")
async def create_scatterplot(request: dict):
    """
    Create an interactive scatterplot showing geographic distribution
    """
    session_id = request.get('session_id')
    analysis_id = request.get('analysis_id')
    
    if not session_id or not analysis_id:
        raise HTTPException(status_code=400, detail="session_id and analysis_id required")
    
    result = session_manager.get_analysis_result(session_id, analysis_id)
    if not result:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    try:
        import pandas as pd
        anomalies_df = pd.DataFrame(result.get('anomalies', []))
        
        chart_result = chart_creator.create_scatterplot_interactive(
            anomalies_df=anomalies_df
        )
        
        return chart_result
        
    except Exception as e:
        logger.error(f"Error creating scatterplot: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# File Management Endpoints
# =============================================================================

@app.get("/api/files/list")
async def list_generated_files():
    """List all generated files from output folder"""
    try:
        import os
        from pathlib import Path
        
        # Get output folder from session or use default
        output_folder = Path(DEFAULT_OUTPUT_DIR)
        
        # Check if folder exists
        if not output_folder.exists():
            return {"success": True, "files": []}
        
        files = []
        file_id = 0
        
        # Scan Maps folder
        maps_folder = output_folder / "Path_Maps"
        if maps_folder.exists():
            for file_path in maps_folder.glob("*.html"):
                stat = file_path.stat()
                files.append({
                    "id": f"maps_{file_id}",
                    "name": file_path.name,
                    "path": str(file_path),
                    "size": stat.st_size,
                    "modified": stat.st_mtime,
                    "category": "maps",
                    "type": "html",
                    "viewable": True
                })
                file_id += 1
        
        # Scan Charts folder
        charts_folder = output_folder / "Charts"
        if charts_folder.exists():
            for file_path in charts_folder.iterdir():
                if file_path.is_file():
                    stat = file_path.stat()
                    ext = file_path.suffix.lower()
                    files.append({
                        "id": f"charts_{file_id}",
                        "name": file_path.name,
                        "path": str(file_path),
                        "size": stat.st_size,
                        "modified": stat.st_mtime,
                        "category": "charts",
                        "type": ext[1:],  # Remove the dot
                        "viewable": ext in ['.html', '.png', '.jpg', '.svg']
                    })
                    file_id += 1
        
        # Scan root Output folder for CSV/Excel
        for file_path in output_folder.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in ['.csv', '.xlsx', '.xls']:
                stat = file_path.stat()
                files.append({
                    "id": f"exports_{file_id}",
                    "name": file_path.name,
                    "path": str(file_path),
                    "size": stat.st_size,
                    "modified": stat.st_mtime,
                    "category": "exports",
                    "type": file_path.suffix[1:].lower(),
                    "viewable": False
                })
                file_id += 1
        
        # Sort by modified time (newest first)
        files.sort(key=lambda x: x['modified'], reverse=True)
        
        return {"success": True, "files": files}
        
    except Exception as e:
        logger.error(f"Error listing files: {e}")
        return {"success": False, "error": str(e)}


@app.get("/api/files/view/{file_id}")
async def view_file(file_id: str):
    """View a file (for HTML files, charts, etc.)"""
    try:
        from fastapi.responses import FileResponse
        import os
        from pathlib import Path
        
        # Get all files to find the matching ID
        response = await list_generated_files()
        if not response["success"]:
            raise HTTPException(status_code=404, detail="Could not list files")
        
        # Find file with matching ID
        file_info = next((f for f in response["files"] if f["id"] == file_id), None)
        if not file_info:
            raise HTTPException(status_code=404, detail="File not found")
        
        file_path = Path(file_info["path"])
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File no longer exists")
        
        # Determine media type
        media_types = {
            'html': 'text/html',
            'png': 'image/png',
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'svg': 'image/svg+xml',
            'csv': 'text/csv',
            'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        }
        
        media_type = media_types.get(file_info["type"], 'application/octet-stream')
        
        return FileResponse(
            path=str(file_path),
            media_type=media_type,
            filename=file_info["name"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error viewing file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/files/download/{file_id}")
async def download_file(file_id: str):
    """Download a file"""
    try:
        from fastapi.responses import FileResponse
        from pathlib import Path
        
        # Get all files to find the matching ID
        response = await list_generated_files()
        if not response["success"]:
            raise HTTPException(status_code=404, detail="Could not list files")
        
        # Find file with matching ID
        file_info = next((f for f in response["files"] if f["id"] == file_id), None)
        if not file_info:
            raise HTTPException(status_code=404, detail="File not found")
        
        file_path = Path(file_info["path"])
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File no longer exists")
        
        return FileResponse(
            path=str(file_path),
            media_type='application/octet-stream',
            filename=file_info["name"],
            headers={"Content-Disposition": f"attachment; filename={file_info['name']}"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown"""
    global cache_sync_task
    
    # Cancel background sync task
    if cache_sync_task:
        cache_sync_task.cancel()
        try:
            await cache_sync_task
        except asyncio.CancelledError:
            pass
        logger.info("Background cache sync task stopped")
    logger.info("=== AIS Law Enforcement Assistant Shutting Down ===")
    
    # Clean up all temporary files
    try:
        files, dirs = temp_manager.cleanup_all()
        logger.info(f"Cleaned up {files} temporary files and {dirs} temporary directories")
    except Exception as e:
        logger.error(f"Error during temporary file cleanup: {e}")
    
    logger.info("Shutdown complete")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

