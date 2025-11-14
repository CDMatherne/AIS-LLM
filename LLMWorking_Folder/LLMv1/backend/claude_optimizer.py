"""
Claude API Performance Optimizer

This module implements the Claude optimization techniques from the Claude Optimization Kit
to improve response speed and reduce costs for the AIS Law Enforcement LLM project.

Key optimizations:
- Prompt caching (90% cost reduction for repetitive patterns)
- Smart model selection (Haiku for simple queries, Sonnet for complex ones)
- Connection resilience with exponential backoff
- Performance monitoring
"""
import time
import random
import logging
import hashlib
import anthropic
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__)

class ClaudeOptimizer:
    """
    Optimizes Claude API calls for better performance, cost efficiency, and reliability.
    Implements techniques from the Claude Optimization Kit including prompt caching,
    smart model selection, and connection resilience.
    """
    
    def __init__(self, api_key: str):
        """
        Initialize the Claude optimizer with API key and optimization settings
        
        Args:
            api_key: Anthropic API key
        """
        self.api_key = api_key
        self.client = anthropic.Anthropic(api_key=api_key)
        
        # Models configuration
        self.haiku_models = [
            "claude-3-haiku-20240307",
            "claude-3-5-haiku-20240620",
            "claude-haiku-4"
        ]
        self.sonnet_models = [
            "claude-3-sonnet-20240229",
            "claude-3-5-sonnet-20240620",
            "claude-sonnet-4"
        ]
        
        # Performance monitoring
        self.performance_stats = {
            "calls": 0,
            "total_time": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "model_usage": {},
            "total_input_tokens": 0,
            "total_output_tokens": 0
        }
        
    def _select_model_for_query(self, query: str, haiku_model: str, sonnet_model: str) -> str:
        """
        Select the appropriate model based on query complexity
        
        Args:
            query: User query text
            haiku_model: Preferred Haiku model to use
            sonnet_model: Preferred Sonnet model to use
            
        Returns:
            Selected model name
        """
        query_lower = query.lower()
        
        # Patterns indicating complex queries needing Sonnet
        complex_indicators = [
            'analyze', 'compare', 'explain why', 'step by step', 'detailed',
            'write code', 'implement', 'debug', 'create', 'design', 'improve',
            'optimize', 'architecture'
        ]
        
        # Check for complexity indicators
        if any(indicator in query_lower for indicator in complex_indicators):
            logger.debug(f"Using {sonnet_model} for complex query: {query[:50]}...")
            return sonnet_model
        
        # Long queries are likely complex
        if len(query.split()) > 50:
            logger.debug(f"Using {sonnet_model} for long query ({len(query.split())} words)")
            return sonnet_model
        
        # Multiple questions indicate complexity
        if query.count('?') > 1:
            logger.debug(f"Using {sonnet_model} for multi-question query ({query.count('?')} questions)")
            return sonnet_model
        
        # Default to Haiku for speed
        logger.debug(f"Using {haiku_model} for simple query")
        return haiku_model
        
    def _add_cache_control(self, system_prompt: Union[str, List]) -> List:
        """
        Add cache control headers to system prompt
        
        Args:
            system_prompt: Original system prompt as string or list
            
        Returns:
            System prompt with cache control headers
        """
        if isinstance(system_prompt, str):
            return [{
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"}
            }]
        elif isinstance(system_prompt, list):
            return system_prompt  # Assume it's already properly formatted
        else:
            raise ValueError("System prompt must be a string or list")
        
    def _resilient_api_call(self, call_func, max_retries=3, base_delay=1.0):
        """
        Make API calls with exponential backoff retry logic
        
        Args:
            call_func: Function to call
            max_retries: Maximum number of retry attempts
            base_delay: Base delay for exponential backoff
            
        Returns:
            API call result
        
        Raises:
            Last exception if all retries fail
        """
        retries = 0
        last_exception = None
        
        while retries < max_retries:
            try:
                return call_func()
            except anthropic.RateLimitError as e:
                delay = base_delay * (2 ** retries) + random.uniform(0, 1)
                logger.warning(f"Rate limit hit, retrying in {delay:.2f}s (retry {retries+1}/{max_retries})")
                time.sleep(delay)
                retries += 1
                last_exception = e
            except anthropic.APIStatusError as e:
                if e.status_code >= 500:  # Server-side errors
                    delay = base_delay * (2 ** retries) + random.uniform(0, 1)
                    logger.warning(f"Server error: {e}. Retrying in {delay:.2f}s (retry {retries+1}/{max_retries})")
                    time.sleep(delay)
                    retries += 1
                    last_exception = e
                else:  # Client errors - don't retry
                    raise
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise
        
        # If we've exhausted retries, raise the last exception
        if last_exception:
            raise last_exception
            
    def _update_performance_stats(self, model: str, start_time: float, response: Any):
        """
        Update performance statistics based on API response
        
        Args:
            model: Model used for the request
            start_time: Start time of the request
            response: API response object
        """
        elapsed = time.time() - start_time
        
        self.performance_stats["calls"] += 1
        self.performance_stats["total_time"] += elapsed
        
        # Track model usage
        model_count = self.performance_stats["model_usage"].get(model, 0)
        self.performance_stats["model_usage"][model] = model_count + 1
        
        # Try to extract token usage
        if hasattr(response, "usage"):
            usage = response.usage
            self.performance_stats["total_input_tokens"] += usage.input_tokens
            self.performance_stats["total_output_tokens"] += usage.output_tokens
            
            # Record cache hits
            cache_read = getattr(usage, "cache_read_input_tokens", 0)
            if cache_read > 0:
                self.performance_stats["cache_hits"] += 1
            else:
                self.performance_stats["cache_misses"] += 1
                
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get current performance statistics
        
        Returns:
            Dictionary with performance statistics
        """
        total_calls = max(1, self.performance_stats["calls"])  # Avoid division by zero
        return {
            "total_calls": self.performance_stats["calls"],
            "avg_response_time": self.performance_stats["total_time"] / total_calls,
            "cache_hit_rate": self.performance_stats["cache_hits"] / total_calls,
            "model_usage": self.performance_stats["model_usage"],
            "total_input_tokens": self.performance_stats["total_input_tokens"],
            "total_output_tokens": self.performance_stats["total_output_tokens"]
        }
        
    def optimized_request(self, 
                          messages: List[Dict[str, Any]],
                          system_prompt: Union[str, List],
                          tools: Optional[List] = None,
                          max_tokens: Optional[int] = None,
                          temperature: float = 0.7,
                          auto_select_model: bool = True,
                          force_model: Optional[str] = None) -> anthropic.types.Message:
        """
        Make an optimized Claude API request with all performance enhancements
        
        Args:
            messages: List of conversation messages
            system_prompt: System prompt (string or list with cache control)
            tools: List of tools for function calling
            max_tokens: Maximum tokens to generate (auto-selected if None)
            temperature: Sampling temperature
            auto_select_model: Whether to automatically select model based on query
            force_model: Force specific model (overrides auto-select)
            
        Returns:
            Claude API response
        """
        # Extract the user query from the last user message
        user_query = ""
        for msg in reversed(messages):
            if msg["role"] == "user":
                if isinstance(msg["content"], str):
                    user_query = msg["content"]
                elif isinstance(msg["content"], list):
                    for item in msg["content"]:
                        if isinstance(item, dict) and item.get("type") == "text":
                            user_query = item.get("text", "")
                            break
                break
                
        # Determine the models to use
        if force_model:
            selected_model = force_model
        elif auto_select_model:
            # Use the best available models
            sonnet_model = self.sonnet_models[0]  # First one is highest preference
            haiku_model = self.haiku_models[0]    # First one is highest preference
            selected_model = self._select_model_for_query(user_query, haiku_model, sonnet_model)
        else:
            # Default to Haiku for speed
            selected_model = self.haiku_models[0]
            
        # Determine appropriate max_tokens based on model and query
        if max_tokens is None:
            # More tokens for Sonnet (complex queries), fewer for Haiku (simple queries)
            max_tokens = 2048 if "sonnet" in selected_model.lower() else 500
            
        # Add cache control headers for system prompt
        cached_system_prompt = self._add_cache_control(system_prompt)
        
        # Add cache control headers to API request
        extra_headers = {"anthropic-beta": "prompt-caching-2024-07-31"}
        
        # Make the API call with retries
        start_time = time.time()
        
        response = self._resilient_api_call(
            lambda: self.client.messages.create(
                model=selected_model,
                max_tokens=max_tokens,
                system=cached_system_prompt,
                messages=messages,
                tools=tools,
                temperature=temperature,
                extra_headers=extra_headers
            )
        )
        
        # Update performance statistics
        self._update_performance_stats(selected_model, start_time, response)
        
        return response
