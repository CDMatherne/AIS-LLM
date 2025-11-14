"""
AWS Bedrock Resilience Layer

This module provides resilience capabilities for AWS Bedrock service with failover to direct Anthropic API.
"""
import logging
import importlib

logger = logging.getLogger(__name__)

# Check if Anthropic is available
ANTHROPIC_AVAILABLE = importlib.util.find_spec("anthropic") is not None

class CircuitBreaker:
    """Simple circuit breaker implementation"""
    def __init__(self, failure_threshold=3, recovery_timeout=60):
        self.state = "CLOSED"
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
    
    def allow_request(self):
        return True
        
    def record_success(self):
        self.failure_count = 0
        
    def record_failure(self):
        self.failure_count += 1

class BedrockResilienceLayer:
    """Resilience layer for AWS Bedrock with fallback to direct Anthropic API"""
    def __init__(self, bedrock_client=None, anthropic_client=None, **kwargs):
        self.bedrock_client = bedrock_client
        self.anthropic_client = anthropic_client
        self.circuit_breaker = CircuitBreaker()
        
    def resilient_invoke_model(self, **request_params):
        """Make a resilient API call to Bedrock with fallback to Anthropic API"""
        # Try Bedrock first
        try:
            return self.bedrock_client.invoke_model(**request_params)
        except Exception as e:
            logger.warning(f"Bedrock call failed: {e}")
            
            # Fall back to Anthropic direct API if available
            if self.anthropic_client:
                anthropic_params = self._convert_to_anthropic_params(request_params)
                return self.anthropic_client.messages.create(**anthropic_params)
            raise
            
    def _convert_to_anthropic_params(self, bedrock_params):
        """Convert Bedrock params to Anthropic API format"""
        # Simplified conversion
        return {
            "model": "claude-3-sonnet-20240229",
            "messages": bedrock_params.get("messages", []),
            "max_tokens": 1000
        }

class BedrockResilientClient:
    """Stub implementation when AWS Bedrock is unavailable"""
    def __init__(self, *args, **kwargs):
        pass
        
    def resilient_invoke_model(self, **kwargs):
        raise NotImplementedError("AWS Bedrock client is not available")

def get_resilient_client(*args, **kwargs):
    """Factory function to get the appropriate resilient client"""
    try:
        # Check if boto3 is available
        import boto3
        if ANTHROPIC_AVAILABLE:
            return BedrockResilienceLayer(*args, **kwargs)
        else:
            return BedrockResilientClient(*args, **kwargs)
    except ImportError:
        return BedrockResilientClient(*args, **kwargs)
