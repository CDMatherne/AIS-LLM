"""
AWS Bedrock Resilience Layer

This module provides resilience capabilities for AWS Bedrock service,
including circuit breaker pattern, automatic failover to direct Anthropic API,
and exponential backoff strategies.

It also includes a stub implementation for when AWS Bedrock is unavailable.
"""
import time
import random
import logging
import threading
import importlib
from datetime import datetime, timedelta
from typing import Dict, Any, Callable, Optional, Union, List

logger = logging.getLogger(__name__)

# Check if Anthropic is available
ANTHROPIC_AVAILABLE = importlib.util.find_spec("anthropic") is not None

class CircuitBreaker:
    """
    Circuit breaker implementation for API calls.
    
    Prevents repeated failures by temporarily "opening the circuit" after 
    a specified number of consecutive failures. This stops further calls
    until a cooldown period has elapsed.
    """
    
    # States
    CLOSED = "closed"     # Normal operation, allowing calls
    OPEN = "open"         # Not allowing calls due to failures
    HALF_OPEN = "half_open"  # Testing if service is back
    
    def __init__(self, failure_threshold: int = 3, recovery_timeout: int = 60):
        """
        Initialize the circuit breaker
        
        Args:
            failure_threshold: Number of consecutive failures before opening circuit
            recovery_timeout: Seconds to wait before allowing retry in half-open state
        """
        self.state = self.CLOSED
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.last_failure_time = None
        self.lock = threading.RLock()
        
    def record_success(self):
        """Record a successful call, closing the circuit if in half-open state"""
        with self.lock:
            if self.state == self.HALF_OPEN:
                logger.info("Circuit breaker: success in half-open state, closing circuit")
                self.state = self.CLOSED
            self.failure_count = 0
    
    def record_failure(self):
        """Record a failed call, possibly opening the circuit"""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            if self.state == self.CLOSED and self.failure_count >= self.failure_threshold:
                logger.warning(f"Circuit breaker: {self.failure_count} consecutive failures, opening circuit")
                self.state = self.OPEN
                
    def allow_request(self) -> bool:
        """
        Check if a request is allowed based on circuit state
        
        Returns:
            True if request is allowed, False if circuit is open
        """
        with self.lock:
            if self.state == self.CLOSED:
                return True
                
            elif self.state == self.OPEN:
                # Check if recovery timeout has elapsed
                if self.last_failure_time is None:
                    return True
                    
                elapsed = (datetime.now() - self.last_failure_time).total_seconds()
                if elapsed >= self.recovery_timeout:
                    logger.info(f"Circuit breaker: recovery timeout elapsed ({elapsed:.1f}s), entering half-open state")
                    self.state = self.HALF_OPEN
                    return True
                return False
                
            elif self.state == self.HALF_OPEN:
                return True
                
            return True  # Unknown state, allow by default

class BedrockResilienceLayer:
    """
    Resilience layer for AWS Bedrock service with failover to direct Anthropic API
    """
    
    def __init__(self, 
                 bedrock_client=None,
                 anthropic_client=None,
                 failure_threshold: int = 3,
                 recovery_timeout: int = 60):
        """
        Initialize the resilience layer
        
        Args:
            bedrock_client: AWS Bedrock client
            anthropic_client: Direct Anthropic client
            failure_threshold: Number of failures before circuit opens
            recovery_timeout: Seconds to wait before testing service again
        """
        self.bedrock_client = bedrock_client
        self.anthropic_client = anthropic_client
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout
        )
        
        # Model mapping between AWS Bedrock and Anthropic API
        self.model_mapping = {
            "anthropic.claude-3-7-sonnet-20250219-v1:0": "claude-3-sonnet-20240229",
            "anthropic.claude-3-5-sonnet-20240620-v1:0": "claude-3-5-sonnet-20240620",
            "anthropic.claude-3-haiku-20240307-v1:0": "claude-3-haiku-20240307",
            "anthropic.claude-instant-v1": "claude-2.0",
        }
        
        # Keep track of service status
        self.service_status = {
            "bedrock_available": True,
            "anthropic_available": True,
            "last_bedrock_failure": None,
            "last_anthropic_failure": None,
            "failover_count": 0
        }
        
    def _translate_bedrock_to_anthropic_model(self, bedrock_model: str) -> str:
        """
        Translate AWS Bedrock model name to Anthropic API model name
        
        Args:
            bedrock_model: AWS Bedrock model name
            
        Returns:
            Equivalent Anthropic API model name
        """
        return self.model_mapping.get(bedrock_model, "claude-3-sonnet-20240229")  # Default fallback
        
    def _retry_with_backoff(self, func: Callable, max_retries: int = 5, base_delay: float = 1.0, 
                           max_delay: float = 30.0, jitter: float = 0.1) -> Any:
        """
        Execute a function with exponential backoff retry logic
        
        Args:
            func: Function to execute
            max_retries: Maximum number of retry attempts
            base_delay: Base delay in seconds
            max_delay: Maximum delay in seconds
            jitter: Random jitter factor (0-1)
            
        Returns:
            Function result
            
        Raises:
            Last exception if all retries fail
        """
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return func()
            except Exception as e:
                last_exception = e
                
                if attempt == max_retries:
                    break
                    
                # Calculate delay with exponential backoff and jitter
                delay = min(base_delay * (2 ** attempt), max_delay)
                jitter_amount = delay * jitter
                actual_delay = delay + random.uniform(-jitter_amount, jitter_amount)
                
                logger.warning(f"Request failed (attempt {attempt+1}/{max_retries+1}): {str(e)}")
                logger.info(f"Retrying in {actual_delay:.2f}s")
                time.sleep(max(0.1, actual_delay))  # Ensure positive delay
                
        # If we've exhausted retries, raise the last exception
        if last_exception:
            logger.error(f"All {max_retries+1} attempts failed")
            raise last_exception
            
    def invoke_with_fallback(self, invoke_func: Callable, fallback_func: Optional[Callable] = None) -> Any:
        """
        Invoke a function with circuit breaker and optional fallback
        
        Args:
            invoke_func: Primary function to invoke
            fallback_func: Optional fallback function
            
        Returns:
            Function result
            
        Raises:
            Exception if both primary and fallback functions fail
        """
        # Check if circuit breaker allows the request
        if not self.circuit_breaker.allow_request():
            logger.warning("Circuit breaker open, bypassing primary service")
            if fallback_func:
                try:
                    result = fallback_func()
                    return result
                except Exception as fallback_e:
                    logger.error(f"Fallback function failed: {str(fallback_e)}")
                    raise fallback_e
            else:
                raise RuntimeError("Service unavailable and no fallback provided")
                
        # Try the primary function
        try:
            result = self._retry_with_backoff(invoke_func)
            self.circuit_breaker.record_success()
            return result
        except Exception as e:
            logger.error(f"Primary function failed: {str(e)}")
            self.circuit_breaker.record_failure()
            
            # Try the fallback if provided
            if fallback_func:
                try:
                    logger.info("Attempting fallback")
                    result = fallback_func()
                    self.service_status["failover_count"] += 1
                    return result
                except Exception as fallback_e:
                    logger.error(f"Fallback function failed: {str(fallback_e)}")
                    raise fallback_e
            else:
                raise e
                
    def bedrock_to_anthropic_request(self, bedrock_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert an AWS Bedrock request to an Anthropic API request
        
        Args:
            bedrock_request: AWS Bedrock request parameters
            
        Returns:
            Equivalent Anthropic API request parameters
        """
        # Extract the model from the Bedrock request
        bedrock_model = bedrock_request.get("modelId", "")
        anthropic_model = self._translate_bedrock_to_anthropic_model(bedrock_model)
        
        # Extract the messages from the Bedrock request
        request_body = bedrock_request.get("body", {})
        if isinstance(request_body, str):
            import json
            try:
                request_body = json.loads(request_body)
            except:
                request_body = {}
                
        # Convert parameters
        anthropic_request = {
            "model": anthropic_model,
            "max_tokens": request_body.get("max_tokens", 1024),
            "messages": request_body.get("messages", []),
            "temperature": request_body.get("temperature", 0.7),
        }
        
        # Handle system prompt if present
        if "system" in request_body:
            anthropic_request["system"] = request_body["system"]
            
        return anthropic_request
        
    def resilient_invoke_model(self, **request_params) -> Dict[str, Any]:
        """
        Make a resilient API call to Bedrock with fallback to Anthropic API
        
        Args:
            request_params: Request parameters
            
        Returns:
            API response
        """
        if not self.bedrock_client or not self.anthropic_client:
            raise ValueError("Both Bedrock and Anthropic clients must be provided for resilient_invoke")
            
        # Define the primary function (Bedrock)
        def bedrock_call():
            return self.bedrock_client.invoke_model(**request_params)
            
        # Define the fallback function (Anthropic API)
        def anthropic_call():
            anthropic_params = self.bedrock_to_anthropic_request(request_params)
            return self.anthropic_client.messages.create(**anthropic_params)
            
        # Invoke with fallback
        return self.invoke_with_fallback(bedrock_call, anthropic_call)


# Stub implementation for when AWS Bedrock is not available
class BedrockResilientClient:
    """
    Stub implementation of the BedrockResilientClient class for use when AWS Bedrock is unavailable.
    Avoids import errors but doesn't provide actual functionality.
    """
    
    def __init__(self, bedrock_client=None, anthropic_client=None, failure_threshold=3, recovery_timeout=60):
        self.bedrock_client = bedrock_client
        self.anthropic_client = anthropic_client
        self.circuit_breaker = CircuitBreaker()
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout

    def resilient_invoke_model(self, **kwargs):
        """Stub method that raises an exception"""
        raise NotImplementedError("AWS Bedrock client is not available")


# Export the appropriate class based on AWS availability
def get_resilient_client(*args, **kwargs):
    """
    Factory function to get the appropriate resilient client implementation
    based on AWS availability.
    """
    try:
        # Check if boto3 is available and AWS credentials are set
        import boto3
        if ANTHROPIC_AVAILABLE:
            return BedrockResilienceLayer(*args, **kwargs)
        else:
            logger.warning("Anthropic SDK not available, using stub implementation")
            return BedrockResilientClient(*args, **kwargs)
    except ImportError:
        logger.warning("AWS boto3 not available, using stub implementation")
        return BedrockResilientClient(*args, **kwargs)
