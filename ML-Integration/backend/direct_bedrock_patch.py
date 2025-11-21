"""
Direct AWS Bedrock Patch 

This module monkey patches the boto3 bedrock-runtime client to automatically
use our resilience layer for all invoke_model calls.

Import this module at the beginning of app.py to ensure all Bedrock calls
are resilient to 503 Service Unavailable errors.
"""
import logging
import boto3
import types
from functools import wraps
import os
from dotenv import load_dotenv

# Set up logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Import our resilience tools using factory function approach
try:
    # First try with dot (relative import)
    try:
        from .bedrock_resilience import get_resilient_client
        import anthropic
        RESILIENCE_IMPORTS_AVAILABLE = True
    except ImportError:
        # Try direct import (when running from backend directory)
        from bedrock_resilience import get_resilient_client
        import anthropic
        RESILIENCE_IMPORTS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Could not import resilience modules: {e}")
    RESILIENCE_IMPORTS_AVAILABLE = False

# Create our global resilient client
_resilient_bedrock_client = None

def initialize_resilient_client():
    """Initialize the resilient Bedrock client if not already done"""
    global _resilient_bedrock_client
    
    if _resilient_bedrock_client is not None:
        return _resilient_bedrock_client
    
    # Check if resilience imports are available
    if not RESILIENCE_IMPORTS_AVAILABLE:
        logger.warning("Skipping resilient client initialization - required modules not available")
        return None
    
    try:
        # Create AWS Bedrock client
        aws_credentials = {}
        if os.getenv('AWS_ACCESS_KEY_ID') and os.getenv('AWS_SECRET_ACCESS_KEY'):
            aws_credentials['aws_access_key_id'] = os.getenv('AWS_ACCESS_KEY_ID')
            aws_credentials['aws_secret_access_key'] = os.getenv('AWS_SECRET_ACCESS_KEY')
            if os.getenv('AWS_SESSION_TOKEN'):
                aws_credentials['aws_session_token'] = os.getenv('AWS_SESSION_TOKEN')
        
        region = os.getenv('AWS_REGION', 'us-east-1')
        
        try:
            # Try to create Bedrock client using the original (non-patched) boto3.client
            bedrock_client = original_client('bedrock-runtime', region_name=region, **aws_credentials)
        except Exception as e:
            logger.warning(f"Failed to create Bedrock client, proceeding without AWS Bedrock: {e}")
            return None
        
        # Create Anthropic client (direct API fallback)
        api_key = os.getenv('CLAUDE_API_KEY')
        if api_key:
            try:
                anthropic_client = anthropic.Anthropic(api_key=api_key)
                
                # Initialize resilient client using factory function
                _resilient_bedrock_client = get_resilient_client(
                    bedrock_client=bedrock_client,
                    anthropic_client=anthropic_client,
                    failure_threshold=2,
                    recovery_timeout=60
                )
                logger.info("Bedrock resilience layer initialized successfully")
                return _resilient_bedrock_client
            except Exception as e:
                logger.warning(f"Could not initialize Anthropic client: {e}")
                return None
        else:
            logger.warning("Could not initialize Bedrock resilience layer - missing Claude API key")
            return None
            
    except Exception as e:
        logger.error(f"Failed to initialize Bedrock resilience layer: {e}")
        return None

# Patch the boto3 client factory to inject our resilient wrapper
original_client = boto3.client

@wraps(original_client)
def patched_client(*args, **kwargs):
    """
    Patched boto3 client factory that wraps bedrock-runtime clients 
    with our resilience layer
    """
    client = original_client(*args, **kwargs)
    
    # Only patch bedrock-runtime clients
    if args and args[0] == 'bedrock-runtime':
        # Check if we should try to patch with resilience layer
        if not RESILIENCE_IMPORTS_AVAILABLE:
            logger.info("Skipping bedrock-runtime patching - required modules not available")
            return client
            
        # Initialize our resilient client if not done yet
        resilient_client = initialize_resilient_client()
        
        if resilient_client:
            # Replace the invoke_model method with our resilient version
            original_invoke = client.invoke_model
            
            @wraps(original_invoke)
            def resilient_invoke_model(**invoke_kwargs):
                """
                Resilient version of invoke_model that handles 503 errors
                by falling back to direct Anthropic API
                """
                try:
                    # Try to use our resilient client first
                    return resilient_client.resilient_invoke_model(**invoke_kwargs)
                except Exception as e:
                    logger.error(f"Resilient client failed, trying original: {e}")
                    # If that fails for any reason, fall back to the original
                    return original_invoke(**invoke_kwargs)
                
            # Replace the method
            client.invoke_model = resilient_invoke_model
            logger.info("Patched bedrock-runtime client with resilience layer")
    
    return client

# Apply the patch
boto3.client = patched_client

# Initialize the resilient client right away, but don't fail if it doesn't work
try:
    initialize_resilient_client()
    logger.info("AWS Bedrock resilience patch applied")
except Exception as e:
    logger.warning(f"Failed to initialize AWS Bedrock resilience layer: {e}")
    logger.info("Application will continue without AWS Bedrock resilience")
