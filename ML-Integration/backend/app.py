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
from datetime import datetime, date
import uuid
import logging
from dotenv import load_dotenv
import pandas as pd  # Import at module level for consistency
import numpy as np
import json

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
# Use the standard Claude agent (optimized version removed due to reliability issues)
claude_agent = import_local_module('claude_agent')
AISFraudDetectionAgent = claude_agent.AISFraudDetectionAgent
logger.info("Using standard Claude agent")

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
# Allow all origins including 'null' (file:// protocol) for local development
# Note: When allow_origins=["*"], allow_credentials must be False
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=False,  # Must be False when allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],  # Expose all headers
)


# ============================================================
# JSON SERIALIZATION HELPERS
# ============================================================

def make_json_serializable(obj):
    """
    Convert pandas/numpy types to JSON-serializable Python types.
    
    Handles:
    - pd.Timestamp -> ISO string
    - datetime.date -> ISO string
    - datetime.datetime -> ISO string
    - np.integer -> int
    - np.floating -> float (with NaN handling)
    - np.ndarray -> list
    - float NaN/Inf -> None
    - dict/list recursion
    """
    import math
    
    # Check for NaN/Inf values first (before type checking)
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        float_val = float(obj)
        if math.isnan(float_val) or math.isinf(float_val):
            return None
    
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, date):  # datetime.date - explicit check
        return obj.isoformat()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        float_val = float(obj)
        # Double-check for NaN/Inf after conversion
        if math.isnan(float_val) or math.isinf(float_val):
            return None
        return float_val
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(make_json_serializable(item) for item in obj)
    elif pd.isna(obj):  # Handle NaN, NaT (pandas-specific)
        return None
    else:
        return obj


def truncate_result_for_llm(result: Dict[str, Any], max_anomalies: int = 50, max_json_size: int = 50000) -> Dict[str, Any]:
    """
    Truncate large result sets to prevent 413 errors when sending to Claude API.
    
    For analysis results with many anomalies, we:
    1. Keep essential summary statistics (truncate large nested structures)
    2. Truncate the anomaly list to max_anomalies
    3. Limit total JSON size to prevent 413 errors
    4. Add a note that full results are stored in session
    
    Args:
        result: Tool execution result
        max_anomalies: Maximum number of anomalies to include (default: 50, reduced from 100)
        max_json_size: Maximum size of serialized JSON in bytes (default: 50KB)
        
    Returns:
        Truncated result safe for LLM API
    """
    if not isinstance(result, dict):
        return result
    
    # Create a truncated copy
    truncated_result = result.copy()
    
    # Check if this is an analysis result with many anomalies
    if 'anomalies' in result and isinstance(result.get('anomalies'), list):
        anomaly_count = len(result['anomalies'])
        
        if anomaly_count > max_anomalies:
            # Truncate anomalies
            truncated_result['anomalies'] = result['anomalies'][:max_anomalies]
            truncated_result['anomalies_truncated'] = True
            truncated_result['total_anomalies_in_full_result'] = anomaly_count
            truncated_result['note'] = f"Showing first {max_anomalies} of {anomaly_count:,} anomalies. Full results stored in session."
            
            logger.info(f"Truncated anomaly list from {anomaly_count:,} to {max_anomalies} for LLM")
    
    # Truncate other potentially large fields
    # Limit statistics to essential fields only
    if 'statistics' in truncated_result and isinstance(truncated_result['statistics'], dict):
        stats = truncated_result['statistics']
        # Keep only essential statistics, remove large nested structures
        essential_stats = {
            'total_anomalies': stats.get('total_anomalies'),
            'unique_vessels': stats.get('unique_vessels'),
            'date_range': stats.get('date_range'),
            'anomaly_types': stats.get('anomaly_types'),
            'vessel_types': stats.get('vessel_types')
        }
        # Remove None values
        essential_stats = {k: v for k, v in essential_stats.items() if v is not None}
        truncated_result['statistics'] = essential_stats
    
    # Remove large metadata fields that aren't needed for LLM
    fields_to_remove = ['raw_data', 'data_metadata', 'processing_metadata', 'cache_info']
    for field in fields_to_remove:
        truncated_result.pop(field, None)
    
    # Check total size and further truncate if needed
    import json
    try:
        test_json = json.dumps(truncated_result, default=str)
        if len(test_json) > max_json_size:
            # If still too large, reduce anomalies further
            if 'anomalies' in truncated_result and isinstance(truncated_result['anomalies'], list):
                current_count = len(truncated_result['anomalies'])
                # Reduce to 25% of current or 10, whichever is larger
                new_max = max(10, current_count // 4)
                truncated_result['anomalies'] = truncated_result['anomalies'][:new_max]
                truncated_result['note'] = f"Showing first {new_max} of {truncated_result.get('total_anomalies_in_full_result', current_count):,} anomalies. Full results stored in session."
                logger.warning(f"Result still too large after truncation, reduced anomalies to {new_max}")
    except Exception as e:
        logger.warning(f"Error checking JSON size: {e}")
    
    return truncated_result

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
    data_source: str  # 'aws', 'local', or 'noaa'
    claude_api_key: str
    aws: Optional[Dict[str, Any]] = None
    local: Optional[Dict[str, Any]] = None
    noaa: Optional[Dict[str, Any]] = None
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


@app.get("/api/frontend-config")
async def get_frontend_config():
    """
    Get frontend configuration from environment variables.
    This allows the frontend to use .env values without hardcoding URLs.
    """
    # Get server host and port from environment or use defaults
    server_host = os.getenv('SERVER_HOST', '0.0.0.0')
    server_port = os.getenv('SERVER_PORT', '8000')
    
    # Determine the API base URL
    # If running in production, use the actual host
    # For development, use localhost
    if server_host == '0.0.0.0' or server_host == '127.0.0.1':
        # Development mode - use localhost
        api_base_url = f"http://localhost:{server_port}"
        ws_base_url = f"ws://localhost:{server_port}"
    else:
        # Production mode - use the configured host
        # Check if HTTPS is enabled
        use_https = os.getenv('USE_HTTPS', 'false').lower() == 'true'
        protocol = 'https' if use_https else 'http'
        ws_protocol = 'wss' if use_https else 'ws'
        api_base_url = f"{protocol}://{server_host}:{server_port}"
        ws_base_url = f"{ws_protocol}://{server_host}:{server_port}"
    
    # Allow override via environment variable
    api_base_url = os.getenv('API_BASE_URL', api_base_url)
    ws_base_url = os.getenv('WS_BASE_URL', ws_base_url)
    
    return {
        "api_base_url": api_base_url,
        "ws_base_url": ws_base_url,
        "server_host": server_host,
        "server_port": server_port,
        "version": "1.0.0"
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


@app.post("/api/pre-initialize-session")
async def pre_initialize_session():
    """
    Pre-initialize session components in the background to speed up apparent start time.
    Creates Claude agent, data connector, and analysis engine without full setup.
    Returns session_id that can be used later when user clicks "Continue".
    """
    try:
        config = get_config()
        
        if not config:
            return {
                'success': False,
                'error': 'Environment variables not configured. Missing required variables.'
            }
        
        # Create data connector (with cache if available)
        data_source = config.get('data_source', 'noaa')  # Default to NOAA if not specified
        connector_config = {
            'data_source': data_source,
            'aws': config.get('aws'),
            'local': config.get('local'),
            'noaa': config.get('noaa'),  # Include NOAA config
            'date_range': config.get('date_range', {})
        }
        
        data_connector = AISDataConnector(connector_config, data_cache=data_cache)
        
        # Create analysis engine with session_manager for data caching
        analysis_engine = AISAnalysisEngine(data_connector, session_manager=session_manager)
        
        # Create Claude agent (this may take a moment for model detection)
        claude_api_key = config.get('claude_api_key')
        if not claude_api_key:
            logger.error("Claude API key not found in config")
            return {
                'success': False,
                'error': 'Claude API key not configured'
            }
        
        try:
            agent = AISFraudDetectionAgent(claude_api_key)
        except Exception as e:
            logger.error(f"Failed to create Claude agent during pre-initialization: {e}")
            return {
                'success': False,
                'error': f'Failed to initialize Claude agent: {str(e)}'
            }
        
        # Create session with pre-initialized components
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
        
        # Mark as pre-initialized (not fully activated yet)
        session['pre_initialized'] = True
        session['initialized_at'] = datetime.now()
        
        # Set output directory if specified in env
        if 'output_directory' in config:
            session['output_directory'] = config['output_directory']
        
        logger.info(f"Pre-initialized session {session_id} (ready for activation)")
        
        return {
            'success': True,
            'session_id': session_id,
            'status': 'pre_initialized',
            'message': 'Session components initialized and ready'
        }
        
    except Exception as e:
        logger.error(f"Pre-initialization failed: {e}")
        return {
            'success': False,
            'error': str(e)
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
        
        # Check if there's a pre-initialized session we can reuse
        pre_initialized_session_id = None
        for sid, session_data in session_manager.sessions.items():
            if session_data.get('pre_initialized') and not session_data.get('activated'):
                # Found a pre-initialized session, reuse it
                pre_initialized_session_id = sid
                logger.info(f"Reusing pre-initialized session {pre_initialized_session_id}")
                break
        
        if pre_initialized_session_id:
            # Reuse pre-initialized session
            session_id = pre_initialized_session_id
            session = session_manager.get_session(session_id)
            if not session:
                logger.error(f"Pre-initialized session {session_id} not found")
                # Fall through to create new session
                pre_initialized_session_id = None
            else:
                agent = session.get('agent')
                data_connector = session.get('data_connector')
                analysis_engine = session.get('analysis_engine')
                
                if not agent or not data_connector or not analysis_engine:
                    logger.error(f"Pre-initialized session {session_id} missing required components")
                    # Fall through to create new session
                    pre_initialized_session_id = None
                else:
                    # Mark as activated
                    session['pre_initialized'] = False
                    session['activated'] = True
                    session['activated_at'] = datetime.now()
                    
                    # Test connection (quick check)
                    connection_test = data_connector.test_connection()
                    if not connection_test or not connection_test.get('success'):
                        return {
                            'success': False,
                            'error': connection_test.get('message', 'Connection test failed') if connection_test else 'Connection test returned None'
                        }
        
        if not pre_initialized_session_id:
            # Create new session (fallback if pre-initialization didn't happen)
            data_source = config.get('data_source', 'noaa')  # Default to NOAA if not specified
            connector_config = {
                'data_source': data_source,
                'aws': config.get('aws'),
                'local': config.get('local'),
                'noaa': config.get('noaa'),
                'date_range': config.get('date_range', {})
            }
            
            data_connector = AISDataConnector(connector_config, data_cache=data_cache)
            
            # Test connection
            connection_test = data_connector.test_connection()
            if not connection_test or not connection_test.get('success'):
                return {
                    'success': False,
                    'error': connection_test.get('message', 'Connection test failed') if connection_test else 'Connection test returned None'
                }
            
            # Create analysis engine with session_manager for data caching
            analysis_engine = AISAnalysisEngine(data_connector, session_manager=session_manager)
            
            # Create Claude agent
            claude_api_key = config.get('claude_api_key')
            if not claude_api_key:
                return {
                    'success': False,
                    'error': 'Claude API key not configured'
                }
            agent = AISFraudDetectionAgent(claude_api_key)
            
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
            if not session:
                logger.error(f"Failed to retrieve session {session_id} after creation")
                return {
                    'success': False,
                    'error': 'Failed to create session'
                }
            session['analysis_engine'] = analysis_engine
        
        # Ensure session exists (should be set in either branch above)
        if 'session' not in locals() or not session:
            logger.error("Session not properly initialized")
            return {
                'success': False,
                'error': 'Session initialization failed'
            }
        
        # Set output directory if specified in env
        if 'output_directory' in config:
            session['output_directory'] = config['output_directory']
        
        # Send initial welcome message from Claude (only if not already sent)
        if not session.get('welcome_sent'):
            data_source = config.get('data_source', 'noaa')
            data_source_info = "AWS S3" if data_source == "aws" else ("NOAA AIS database" if data_source == "noaa" else "local storage")
            welcome_response = await agent.chat(
                f"Hello! Please introduce yourself and explain that the system is NOW CONFIGURED and READY with {data_source_info} access. Explain your capabilities and the available date range (January 1, 2021 onwards). Make it clear you can analyze the data immediately - no uploads needed.",
                session_context={}
            )
            session['welcome_sent'] = True
            welcome_message = welcome_response['message']
        else:
            # Welcome already sent during pre-initialization or previous activation
            welcome_message = "Welcome back! The system is ready for analysis."
        
        logger.info(f"Session {session_id} created successfully from environment variables")
        
        # Ensure connection_test exists (it should from the if/else blocks above)
        date_range = {}
        if 'connection_test' in locals() and connection_test:
            date_range = connection_test.get('date_range', {})
        
        return {
            'success': True,
            'session_id': session_id,
            'welcome_message': welcome_message,
            'date_range': date_range,
            'system_info': analysis_engine.get_system_info(),
            'source': 'environment_variables',
            'pre_initialized': pre_initialized_session_id is not None,
            'config': {
                'data_source': config.get('data_source', 'noaa'),
                'aws': config.get('aws'),
                'local': config.get('local'),
                'noaa': config.get('noaa')
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
        
        # Validate NOAA config if applicable
        if config.data_source == 'noaa' and config.noaa and config.noaa.get('temp_dir'):
            temp_dir = Path(config.noaa['temp_dir'])
            if not temp_dir.exists():
                try:
                    temp_dir.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Created NOAA cache directory: {temp_dir}")
                except Exception as e:
                    logger.error(f"Cannot create NOAA cache directory: {e}")
                    return{
                        'success': False,
                        'error': f"Cannot create cache directory: {str(e)}"
                    }
        
        # Create data connector
        connector_config = {
            'data_source': config.data_source,
            'aws': config.aws,
            'local': config.local,
            'noaa': config.noaa,
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
        try:
            agent = AISFraudDetectionAgent(claude_api_key)
        except RuntimeError as model_error:
            # Model detection failed - provide more helpful error
            logger.error(f"Failed to initialize Claude agent: {model_error}")
            error_message = str(model_error)
            
            # Check if it's an authentication issue
            if "authentication" in error_message.lower() or "invalid" in error_message.lower():
                error_message += "\n\nPossible causes:\n"
                error_message += "1. API key is incorrect or expired\n"
                error_message += "2. API key doesn't have access to Claude models\n"
                error_message += "3. Account may need to be activated at https://console.anthropic.com/\n"
                error_message += "4. Check your API key format (should start with 'sk-ant-')"
            
            return {
                'success': False,
                'error': error_message
            }
        except Exception as agent_error:
            # Other errors during agent creation
            logger.error(f"Failed to create Claude agent: {agent_error}", exc_info=True)
            return {
                'success': False,
                'error': f"Failed to initialize Claude agent: {str(agent_error)}"
            }
        
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
        if config.data_source == "aws":
            data_source_info = "AWS S3"
        elif config.data_source == "noaa":
            data_source_info = "NOAA (on-demand download)"
        else:
            data_source_info = "local storage"
        
        welcome_response = await agent.chat(
            f"Hello! Please introduce yourself and explain that the system is NOW CONFIGURED and READY with {data_source_info} access. Explain your capabilities and the available date range (January 1, 2021 onwards). Make it clear you can analyze the data immediately - no uploads needed.",
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
    Check if environment variables are configured and return config for auto-population.
    Returns config status WITHOUT requiring a full connection test.
    """
    config = get_config()
    
    if not config:
        return {
            'success': False,
            'env_configured': False,
            'message': 'Environment variables not configured. Missing required variables.',
            'missing': []
        }
    
    # Return configuration for auto-population (without full connection test)
    response = {
        'success': True,
        'env_configured': True,
        'data_source': config.get('data_source'),
        'message': 'Environment variables configured'
    }
    
    # Include AWS config if available
    if config.get('aws'):
        response['aws_config'] = {
            'bucket': config['aws'].get('bucket'),
            'prefix': config['aws'].get('prefix'),
            'region': config['aws'].get('region')
        }
    
    # Include Local config if available
    if config.get('local'):
        response['local_config'] = {
            'path': config['local'].get('path')
        }
    
    # Include NOAA config if available
    if config.get('noaa'):
        response['noaa_config'] = {
            'temp_dir': config['noaa'].get('temp_dir'),
            'cache_days': config['noaa'].get('cache_days', 7)
        }
    
    # Include Claude API key (masked for security)
    if config.get('claude_api_key'):
        # Only include if it exists, frontend will auto-populate
        response['claude_api_key'] = config['claude_api_key']
    
    # Include output folder if available
    if config.get('output_folder'):
        response['output_folder'] = config['output_folder']
    
    return response


@app.get("/api/default-settings")
async def get_default_settings():
    """
    Get default settings for data source and output folder.
    Used by the welcome modal to display current/default configuration.
    """
    config = get_config()
    default_output_dir = str(DEFAULT_OUTPUT_DIR)
    
    # Determine data source type and details
    # Default to NOAA if no config or no data_source specified
    data_source_type = None
    data_source_details = None
    
    if config:
        data_source_type = config.get('data_source')
    
    # If no data source configured, default to NOAA
    if not data_source_type:
        data_source_type = 'noaa'
        data_source_details = {
            'type': 'noaa',
            'temp_dir': 'Default location',
            'cache_days': 7,
            'is_default': True
        }
    elif data_source_type == 'aws' and config.get('aws'):
        data_source_details = {
            'type': 'aws',
            'bucket': config['aws'].get('bucket', 'Not configured'),
            'prefix': config['aws'].get('prefix', ''),
            'region': config['aws'].get('region', 'Not configured')
        }
    elif data_source_type == 'local' and config.get('local'):
        data_source_details = {
            'type': 'local',
            'path': config['local'].get('path', 'Not configured')
        }
    elif data_source_type == 'noaa' and config.get('noaa'):
        data_source_details = {
            'type': 'noaa',
            'temp_dir': config['noaa'].get('temp_dir', 'Default location'),
            'cache_days': config['noaa'].get('cache_days', 7)
        }
    elif data_source_type == 'noaa':
        # NOAA selected but no specific config - use defaults
        data_source_details = {
            'type': 'noaa',
            'temp_dir': 'Default location',
            'cache_days': 7,
            'is_default': True
        }
    
    # Check if defaults are acceptable (has valid data source)
    defaults_acceptable = False
    if data_source_type:
        if data_source_type == 'aws' and config and config.get('aws') and config['aws'].get('bucket'):
            defaults_acceptable = True
        elif data_source_type == 'local' and config and config.get('local') and config['local'].get('path'):
            defaults_acceptable = True
        elif data_source_type == 'noaa':
            defaults_acceptable = True  # NOAA is always acceptable (default or configured)
    
    return {
        'success': True,
        'data_source': {
            'type': data_source_type or 'Not configured',
            'details': data_source_details or {'type': 'none', 'message': 'No data source configured'}
        },
        'output_folder': default_output_dir,
        'defaults_acceptable': defaults_acceptable,
        'has_claude_key': bool(config and config.get('claude_api_key')) if config else False
    }


@app.get("/api/test-env-connection")
async def test_env_connection():
    """
    Test connection using environment variables.
    Separate endpoint for actual connection testing.
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
            'noaa': config.get('noaa'),
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
                    logger.error(f"Error executing tool {tool_call.get('name', 'unknown')}: {tool_error}", exc_info=True)
                    # Return graceful error that won't disrupt conversation
                    tool_results.append({
                        "tool_call_id": tool_call.get("id", "unknown"),
                        "result": {
                            "success": False,
                            "error": f"An error occurred: {str(tool_error)}. Please try again or select different parameters.",
                            "tool_name": tool_call.get('name', 'unknown')
                        }
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
        # Send to frontend via WebSocket (check connection state first)
        if websocket:
            try:
                # Check if WebSocket is still connected before sending
                try:
                    # Check WebSocket state (0=CONNECTING, 1=OPEN, 2=CLOSING, 3=CLOSED)
                    if websocket.client_state.value != 1:  # Not OPEN
                        logger.debug(f"WebSocket not open (state: {websocket.client_state.value}), skipping progress update")
                        # Still store in session for LLM access
                    else:
                        await websocket.send_json({
                            "type": "progress",
                            "stage": stage,
                            "message": message,
                            "data": data or {}
                        })
                except AttributeError:
                    # Fallback if client_state is not available - try sending anyway
                    try:
                        await websocket.send_json({
                            "type": "progress",
                            "stage": stage,
                            "message": message,
                            "data": data or {}
                        })
                    except Exception as send_error:
                        # Connection may have closed - log but don't fail
                        logger.debug(f"Progress update send failed (connection may be closed): {send_error}")
            except Exception as e:
                # Connection may have closed - log but don't fail
                logger.debug(f"Progress update send failed (connection may be closed): {e}")
        
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
    logger.info(f"WebSocket connection attempt for session: {session_id}")
    
    try:
        await websocket.accept()
        logger.info(f"WebSocket accepted for session: {session_id}")
    except Exception as accept_error:
        logger.error(f"Failed to accept WebSocket connection: {accept_error}")
        return
    
    agent = session_manager.get_agent(session_id)
    if not agent:
        logger.error(f"Agent not found for session: {session_id}")
        try:
            await websocket.close(code=1008, reason="Session not found")
        except:
            pass
        return
    
    logger.info(f"WebSocket connection established for session: {session_id}")
    
    # Track connection state
    connection_active = True
    
    # Helper function to check if WebSocket is still connected
    def is_websocket_connected():
        """Check if WebSocket connection is still active"""
        try:
            # Check WebSocket state (0=CONNECTING, 1=OPEN, 2=CLOSING, 3=CLOSED)
            return websocket.client_state.value == 1  # OPEN state
        except:
            return False
    
    # Helper function to safely send WebSocket messages
    async def safe_send_json(message: dict):
        """Safely send JSON message, handling connection errors"""
        nonlocal connection_active
        if not connection_active:
            return False
        try:
            # Check connection state before sending
            if not is_websocket_connected():
                connection_active = False
                logger.warning(f"WebSocket connection lost for session {session_id}")
                return False
            await websocket.send_json(message)
            return True
        except Exception as send_error:
            connection_active = False
            logger.warning(f"Failed to send WebSocket message: {send_error}")
            return False
    
    # Keepalive task to prevent connection timeout during long operations
    async def keepalive_task():
        """Send periodic ping messages to keep WebSocket connection alive"""
        nonlocal connection_active
        try:
            while connection_active:
                await asyncio.sleep(30)  # Send ping every 30 seconds
                if connection_active and is_websocket_connected():
                    try:
                        # Send a lightweight keepalive message
                        await websocket.send_json({
                            "type": "keepalive",
                            "timestamp": datetime.now().isoformat()
                        })
                    except Exception as e:
                        logger.debug(f"Keepalive ping failed: {e}")
                        connection_active = False
                        break
                else:
                    connection_active = False
                    break
        except asyncio.CancelledError:
            logger.debug(f"Keepalive task cancelled for session {session_id}")
        except Exception as e:
            logger.warning(f"Keepalive task error: {e}")
    
    # Start keepalive task
    keepalive_task_handle = asyncio.create_task(keepalive_task())
    
    try:
        while True:
            # Receive message
            try:
                data = await websocket.receive_json()
                logger.debug(f"Received message from session {session_id}: {data.get('type', 'unknown')}")
            except ValueError as json_error:
                logger.error(f"Invalid JSON received from session {session_id}: {json_error}")
                if not await safe_send_json({
                    "type": "error",
                    "content": "Invalid message format. Please send valid JSON."
                }):
                    break  # Connection lost, exit loop
                continue
            
            # Validate data structure
            if not isinstance(data, dict):
                logger.error(f"Invalid data type received: {type(data)}")
                if not await safe_send_json({
                    "type": "error",
                    "content": "Invalid message format. Expected JSON object."
                }):
                    break  # Connection lost, exit loop
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
                if not await safe_send_json({
                    "type": "error",
                    "content": "Message cannot be empty. Please provide a valid message."
                }):
                    break  # Connection lost, exit loop
                continue
            
            # Validate conversation history before processing new message
            # This ensures any orphaned tool_use blocks are cleaned up
            try:
                if hasattr(agent, '_validate_conversation_history'):
                    agent._validate_conversation_history()
            except Exception as e:
                logger.warning(f"Conversation history validation failed: {e}")
            
            # Process with Claude - wrap in try/except to handle errors gracefully
            try:
                response = await agent.chat(message, session_context={})
                
                # Validate response structure
                if not isinstance(response, dict):
                    logger.error(f"Invalid response type from agent: {type(response)}")
                    if not await safe_send_json({
                        "type": "error",
                        "content": "Received invalid response format from AI agent. Please try again."
                    }):
                        break  # Connection lost, exit loop
                    continue
                
                # Check if response contains an error
                if response.get("error"):
                    logger.error(f"Agent returned error: {response.get('error')}")
                    if not await safe_send_json({
                        "type": "error",
                        "content": f"AI agent error: {response.get('error')}. Please try again."
                    }):
                        break  # Connection lost, exit loop
                    continue
                
                # Send response (use get() with default to handle missing message key)
                response_message = response.get("message", "")
                if response_message:
                    if not await safe_send_json({
                        "type": "message",
                        "content": response_message
                    }):
                        break  # Connection lost, exit loop
                    
            except Exception as chat_error:
                # Handle errors in agent.chat() gracefully without closing connection
                logger.error(f"Error in agent.chat(): {chat_error}", exc_info=True)
                error_message = f"I encountered an error processing your message: {str(chat_error)}. Please try again or rephrase your request."
                if not await safe_send_json({
                    "type": "error",
                    "content": error_message
                }):
                    break  # Connection lost, exit loop
                # Also send as a message so user sees it in chat
                if not await safe_send_json({
                    "type": "message",
                    "content": error_message
                }):
                    break  # Connection lost, exit loop
                continue
            
            # Execute any tool calls (response is guaranteed to be defined here)
            tool_calls = response.get("tool_calls", []) if response else []
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
                    if not await safe_send_json({
                        "type": "tools_executing",
                        "tools": tool_names
                    }):
                        break  # Connection lost, exit loop
                    
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
                            
                            # Store full result server-side (already done in execute_tool for analysis)
                            # Send minimal notification to frontend for UI updates (not full data to Claude)
                            if tool_call["name"] == "run_anomaly_analysis" and isinstance(result, dict) and result.get("success"):
                                # Send notification with analysis_id so frontend can fetch full results from server
                                if not await safe_send_json({
                                    "type": "analysis_complete",
                                    "analysis_id": result.get("analysis_id"),
                                    "anomalies_found": result.get("anomalies_found", 0),
                                    "files_generated": result.get("files_generated", []),
                                    "message": f"Analysis complete. Found {result.get('anomalies_found', 0):,} anomalies. Results stored server-side."
                                }):
                                    break  # Connection lost, exit loop
                            else:
                                # For other tools, send result for UI updates
                                # IMPORTANT: Include all fields needed for popup requests (action, etc.)
                                if isinstance(result, dict):
                                    # For UI interaction tools (popups), include all popup-related fields
                                    if result.get("action"):
                                        # This is a popup request - send full result structure
                                        minimal_result = result
                                    else:
                                        # Regular tool - send minimal result
                                        minimal_result = {
                                            "success": result.get("success", True),
                                            "file_path": result.get("file_path"),
                                            "message": result.get("message", "Tool execution completed")
                                        }
                                else:
                                    minimal_result = {
                                        "success": True,
                                        "message": "Tool execution completed"
                                    }
                                if not await safe_send_json({
                                    "type": "tool_result",
                                    "tool_name": tool_call["name"],
                                    "result": minimal_result
                                }):
                                    break  # Connection lost, exit loop
                        except Exception as tool_error:
                            logger.error(f"Error executing tool {tool_call.get('name', 'unknown')}: {tool_error}")
                            tool_results.append({
                                "tool_call_id": tool_call.get("id", "unknown"),
                                "tool_name": tool_call.get("name", "unknown"),
                                "result": {"error": str(tool_error)}
                            })
                    
                    # Send summary to Claude (not full results - results stored server-side)
                    if tool_results:
                        try:
                            # Create minimal summaries for Claude - full results stored server-side
                            summary_tool_results = []
                            for tool_result in tool_results:
                                tool_name = tool_result.get("tool_name", "unknown")
                                result_data = tool_result.get("result", {})
                                
                                # Create minimal summary based on tool type
                                if tool_name == "run_anomaly_analysis":
                                    # For analysis: send only success status and summary stats
                                    if isinstance(result_data, dict):
                                        summary = {
                                            "success": result_data.get("success", False),
                                            "analysis_id": result_data.get("analysis_id"),
                                            "anomalies_found": result_data.get("anomalies_found", 0),
                                            "total_records": result_data.get("total_records", 0),
                                            "unique_vessels": result_data.get("statistics", {}).get("unique_vessels", 0) if isinstance(result_data.get("statistics"), dict) else 0,
                                            "date_range": result_data.get("date_range", {}),
                                            "files_generated_count": len(result_data.get("files_generated", [])),
                                            "message": f"Analysis completed. Found {result_data.get('anomalies_found', 0):,} anomalies from {result_data.get('statistics', {}).get('unique_vessels', 0) if isinstance(result_data.get('statistics'), dict) else 0:,} vessels. Full results stored server-side and available via files tab."
                                        }
                                        # Add error if present
                                        if result_data.get("error"):
                                            summary["error"] = result_data.get("error")
                                    else:
                                        summary = {"success": False, "message": "Analysis completed with unknown result format"}
                                    
                                    summary_tool_results.append({
                                        "tool_call_id": tool_result.get("tool_call_id"),
                                        "tool_name": tool_name,
                                        "result": summary
                                    })
                                else:
                                    # For other tools: send minimal summary
                                    if isinstance(result_data, dict):
                                        summary = {
                                            "success": result_data.get("success", False),
                                            "message": result_data.get("message", f"{tool_name} completed"),
                                            "file_path": result_data.get("file_path") if "file_path" in result_data else None
                                        }
                                        if result_data.get("error"):
                                            summary["error"] = result_data.get("error")
                                    else:
                                        summary = {"success": True, "message": f"{tool_name} completed"}
                                    
                                    summary_tool_results.append({
                                        "tool_call_id": tool_result.get("tool_call_id"),
                                        "tool_name": tool_name,
                                        "result": summary
                                    })
                            
                            # Send minimal summaries to Claude
                            final_response = await agent.process_tool_results(summary_tool_results)
                            if isinstance(final_response, dict) and "message" in final_response:
                                if not await safe_send_json({
                                    "type": "message",
                                    "content": final_response["message"]
                                }):
                                    break  # Connection lost, exit loop
                            else:
                                logger.warning(f"Invalid final_response structure: {final_response}")
                                if not await safe_send_json({
                                    "type": "message",
                                    "content": "Tool execution completed, but received invalid response format."
                                }):
                                    break  # Connection lost, exit loop
                        except Exception as process_error:
                            logger.error(f"Error processing tool results: {process_error}")
                            if not await safe_send_json({
                                "type": "message",
                                "content": f"Error processing tool results: {str(process_error)}"
                            }):
                                break  # Connection lost, exit loop
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
        connection_active = False
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        # Try to send error message before closing
        try:
            await websocket.send_json({
                "type": "error",
                "content": f"A critical error occurred: {str(e)}. The connection will be closed."
            })
        except:
            pass  # Connection may already be closed
        # Only close if connection is still open
        try:
            await websocket.close(code=1011, reason="Internal server error")
        except:
            pass  # Connection may already be closed
    finally:
        # Cancel keepalive task when connection closes
        if 'keepalive_task_handle' in locals():
            keepalive_task_handle.cancel()
            try:
                await keepalive_task_handle
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.debug(f"Error cancelling keepalive task: {e}")
        connection_active = False


@app.post("/api/manual-analysis")
async def manual_analysis(request: dict):
    """
    Run analysis directly without LLM conversation.
    Accepts all analysis parameters and returns results.
    """
    try:
        # Get or create session
        session_id = request.get('session_id')
        if not session_id:
            session_id = str(uuid.uuid4())
        
        session = session_manager.get_session(session_id)
        if not session:
            # Create session if it doesn't exist
            config = get_config()
            if not config:
                return {"success": False, "error": "System not configured"}
            
            # Create data connector
            connector_config = {
                'data_source': config.get('data_source', 'noaa'),
                'aws': config.get('aws'),
                'local': config.get('local'),
                'noaa': config.get('noaa'),
                'date_range': config.get('date_range', {})
            }
            
            data_connector = AISDataConnector(connector_config, data_cache=data_cache)
            analysis_engine = AISAnalysisEngine(data_connector, session_manager=session_manager)
            
            claude_api_key = config.get('claude_api_key')
            if claude_api_key:
                agent = AISFraudDetectionAgent(claude_api_key)
            else:
                agent = None
            
            session_manager.create_session(
                session_id,
                agent,
                data_connector=data_connector,
                config=connector_config
            )
            session = session_manager.get_session(session_id)
            session['analysis_engine'] = analysis_engine
        
        analysis_engine = session.get('analysis_engine')
        if not analysis_engine:
            return {"success": False, "error": "Analysis engine not available"}
        
        # ============================================================
        # DEBUG: Log exactly what is received from frontend
        # ============================================================
        logger.info("=" * 80)
        logger.info("📥 MANUAL ANALYSIS REQUEST - EXACT PAYLOAD RECEIVED:")
        logger.info("=" * 80)
        logger.info(f"Full request dict: {json.dumps(request, indent=2, default=str)}")
        logger.info(f"\n📋 Request Details:")
        logger.info(f"  - session_id: {request.get('session_id')} (type: {type(request.get('session_id')).__name__})")
        logger.info(f"  - start_date: {request.get('start_date')} (type: {type(request.get('start_date')).__name__})")
        logger.info(f"  - end_date: {request.get('end_date')} (type: {type(request.get('end_date')).__name__})")
        
        vessel_types_raw = request.get('vessel_types')
        logger.info(f"  - vessel_types (raw): {vessel_types_raw} (type: {type(vessel_types_raw).__name__})")
        if vessel_types_raw is not None:
            logger.info(f"    - Is list: {isinstance(vessel_types_raw, list)}")
            logger.info(f"    - Is tuple: {isinstance(vessel_types_raw, tuple)}")
            logger.info(f"    - Is string: {isinstance(vessel_types_raw, str)}")
            if isinstance(vessel_types_raw, (list, tuple)):
                logger.info(f"    - Length: {len(vessel_types_raw)}")
                logger.info(f"    - Contents: {vessel_types_raw}")
        
        anomaly_types_raw = request.get('anomaly_types')
        logger.info(f"  - anomaly_types (raw): {anomaly_types_raw} (type: {type(anomaly_types_raw).__name__})")
        if anomaly_types_raw is not None:
            logger.info(f"    - Is list: {isinstance(anomaly_types_raw, list)}")
            logger.info(f"    - Length: {len(anomaly_types_raw) if isinstance(anomaly_types_raw, (list, tuple)) else 'N/A'}")
            if isinstance(anomaly_types_raw, (list, tuple)):
                logger.info(f"    - Contents: {anomaly_types_raw}")
        
        logger.info(f"  - mmsi_filter: {request.get('mmsi_filter')} (type: {type(request.get('mmsi_filter')).__name__})")
        logger.info(f"  - geographic_zone: {request.get('geographic_zone')} (type: {type(request.get('geographic_zone')).__name__})")
        logger.info(f"  - output_formats: {request.get('output_formats')} (type: {type(request.get('output_formats')).__name__})")
        logger.info("=" * 80)
        
        # Extract parameters from request
        start_date = request.get('start_date')
        end_date = request.get('end_date')
        vessel_types = request.get('vessel_types')
        anomaly_types = request.get('anomaly_types', [])
        mmsi_filter = request.get('mmsi_filter')
        geographic_zone = request.get('geographic_zone')
        
        # Normalize vessel_types: convert None or empty list to None, ensure it's a list if provided
        if vessel_types is None:
            vessel_types = None
        elif isinstance(vessel_types, list) and len(vessel_types) == 0:
            vessel_types = None
        elif isinstance(vessel_types, str):
            vessel_types = [vessel_types]
        elif not isinstance(vessel_types, list):
            logger.warning(f"vessel_types is not None, list, or string: {type(vessel_types)}, converting to list")
            vessel_types = [vessel_types] if vessel_types else None
        
        # Ensure anomaly_types is always a list (empty list if None)
        if anomaly_types is None:
            anomaly_types = []
        elif not isinstance(anomaly_types, list):
            if isinstance(anomaly_types, str):
                anomaly_types = [anomaly_types]
            else:
                logger.warning(f"anomaly_types is not None or list: {type(anomaly_types)}, converting to list")
                anomaly_types = [anomaly_types] if anomaly_types else []
        
        logger.info(f"\n📤 NORMALIZED PARAMETERS FOR ANALYSIS ENGINE:")
        logger.info(f"  - start_date: {start_date} (type: {type(start_date).__name__})")
        logger.info(f"  - end_date: {end_date} (type: {type(end_date).__name__})")
        logger.info(f"  - vessel_types: {vessel_types} (type: {type(vessel_types).__name__})")
        logger.info(f"  - anomaly_types: {anomaly_types} (type: {type(anomaly_types).__name__}, length: {len(anomaly_types)})")
        logger.info(f"  - mmsi_filter: {mmsi_filter} (type: {type(mmsi_filter).__name__})")
        logger.info(f"  - geographic_zone: {geographic_zone} (type: {type(geographic_zone).__name__})")
        logger.info("=" * 80)
        
        if not start_date or not end_date:
            return {"success": False, "error": "start_date and end_date are required"}
        
        # Create a simple progress callback that stores updates
        progress_updates = []
        
        async def progress_callback(stage: str, message: str, data: dict = None):
            progress_updates.append({
                "stage": stage,
                "message": message,
                "timestamp": datetime.now().isoformat(),
                "data": data or {}
            })
        
        # ============================================================
        # DEBUG: Log exactly what is being passed to analysis engine
        # ============================================================
        logger.info(f"\n🔧 CALLING ANALYSIS ENGINE WITH:")
        logger.info(f"  - start_date: {start_date} (type: {type(start_date).__name__})")
        logger.info(f"  - end_date: {end_date} (type: {type(end_date).__name__})")
        logger.info(f"  - vessel_types: {vessel_types} (type: {type(vessel_types).__name__})")
        logger.info(f"  - anomaly_types: {anomaly_types} (type: {type(anomaly_types).__name__}, length: {len(anomaly_types)})")
        logger.info(f"  - mmsi_filter: {mmsi_filter} (type: {type(mmsi_filter).__name__})")
        logger.info(f"  - geographic_zone: {geographic_zone} (type: {type(geographic_zone).__name__})")
        logger.info(f"  - session_id: {session_id} (type: {type(session_id).__name__})")
        logger.info("=" * 80)
        
        # Run analysis
        result = await analysis_engine.run_analysis(
            start_date=start_date,
            end_date=end_date,
            geographic_zone=geographic_zone,
            anomaly_types=anomaly_types,
            mmsi_filter=mmsi_filter,
            vessel_types=vessel_types,
            progress_callback=progress_callback,
            session_id=session_id
        )
        
        logger.info(f"\n✅ ANALYSIS ENGINE RETURNED:")
        logger.info(f"  - success: {result.get('success')}")
        logger.info(f"  - analysis_id: {result.get('analysis_id')}")
        logger.info(f"  - anomalies_found: {result.get('anomalies_found', result.get('total_anomalies', 'N/A'))}")
        logger.info("=" * 80)
        
        # Store result in session
        if result.get('success'):
            analysis_id = result.get('analysis_id')
            session_manager.store_analysis_result(session_id, analysis_id, result)
            result['progress_updates'] = progress_updates
            
            # Auto-generate outputs if requested (similar to analysis_handlers)
            output_formats = request.get('output_formats', [])
            if output_formats:
                try:
                    from tool_handlers.analysis_handlers import _generate_all_outputs
                    generated_files = await _generate_all_outputs(
                        result, 
                        session_id, 
                        {
                            'start_date': start_date,
                            'end_date': end_date
                        },
                        output_formats=output_formats  # Pass requested formats
                    )
                    result['files_generated'] = generated_files
                    logger.info(f"Generated {len(generated_files)} output files from {len(output_formats)} requested formats")
                except Exception as e:
                    logger.error(f"Failed to generate outputs: {e}", exc_info=True)
                    result['output_generation_error'] = str(e)
        
        # Make result JSON-serializable (handle NaN, Inf, and other non-serializable types)
        logger.info("Making result JSON-serializable (handling NaN/Inf values)...")
        result = make_json_serializable(result)
        
        # Return with CORS headers to handle file:// protocol
        from fastapi.responses import JSONResponse
        return JSONResponse(
            content=result,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "*",
                "Access-Control-Allow-Headers": "*",
            }
        )
        
    except Exception as e:
        logger.error(f"Manual analysis error: {e}", exc_info=True)
        error_result = {
            "success": False,
            "error": str(e)
        }
        # Return error with CORS headers
        from fastapi.responses import JSONResponse
        return JSONResponse(
            content=error_result,
            status_code=500,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "*",
                "Access-Control-Allow-Headers": "*",
            }
        )


@app.get("/api/sessions/{session_id}/anomalies-for-map")
async def get_anomalies_for_map(session_id: str):
    """
    Get anomaly data formatted for map display.
    Returns minimal data (lat, lon, mmsi, type) for efficient map rendering.
    """
    try:
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get latest analysis from session
        if 'analyses' not in session or not session['analyses']:
            return {
                "success": False,
                "error": "No analysis results found"
            }
        
        # Get the most recent analysis
        latest_analysis_id = max(session['analyses'].keys(), 
                                key=lambda x: session['analyses'][x].get('timestamp', datetime.min))
        latest_analysis = session_manager.get_analysis_result(session_id, latest_analysis_id)
        
        if not latest_analysis:
            return {
                "success": False,
                "error": "No analysis results found"
            }
        
        anomalies = latest_analysis.get('anomalies', [])
        
        # Format for map (minimal data)
        map_data = []
        for anomaly in anomalies:
            if 'LAT' in anomaly and 'LON' in anomaly:
                map_data.append({
                    'lat': float(anomaly.get('LAT', 0)),
                    'lon': float(anomaly.get('LON', 0)),
                    'mmsi': anomaly.get('MMSI'),
                    'type': anomaly.get('AnomalyType', 'Unknown'),
                    'datetime': anomaly.get('BaseDateTime'),
                    'vessel_type': anomaly.get('VesselType')
                })
        
        return {
            "success": True,
            "anomalies": map_data,
            "count": len(map_data)
        }
    except Exception as e:
        logger.error(f"Error getting anomalies for map: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }


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

@app.get("/api/sessions/{session_id}/latest-analysis")
async def get_latest_analysis(session_id: str):
    """
    Get the latest analysis result for a session (server-side storage).
    Returns full analysis data including anomalies, statistics, and generated files.
    """
    try:
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get all analyses for this session
        analyses = session.get('analyses', {})
        if not analyses:
            return {
                "success": False,
                "error": "No analysis results found for this session"
            }
        
        # Get the most recent analysis
        latest_analysis_id = None
        latest_timestamp = None
        for analysis_id, analysis_data in analyses.items():
            timestamp = analysis_data.get('timestamp')
            if timestamp and (latest_timestamp is None or timestamp > latest_timestamp):
                latest_timestamp = timestamp
                latest_analysis_id = analysis_id
        
        if not latest_analysis_id:
            return {
                "success": False,
                "error": "No valid analysis results found"
            }
        
        analysis_result = analyses[latest_analysis_id]['result']
        
        # Return full result (stored server-side)
        return {
            "success": True,
            "analysis_id": latest_analysis_id,
            "result": analysis_result,
            "timestamp": latest_timestamp.isoformat() if latest_timestamp else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting latest analysis: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@app.get("/api/sessions/{session_id}/analysis/{analysis_id}")
async def get_analysis_by_id(session_id: str, analysis_id: str):
    """
    Get a specific analysis result by ID (server-side storage).
    """
    try:
        analysis_result = session_manager.get_analysis_result(session_id, analysis_id)
        if not analysis_result:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        return {
            "success": True,
            "analysis_id": analysis_id,
            "result": analysis_result
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting analysis: {e}")
        return {
            "success": False,
            "error": str(e)
        }


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

