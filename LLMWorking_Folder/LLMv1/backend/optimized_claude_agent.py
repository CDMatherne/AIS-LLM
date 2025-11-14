"""
Optimized Claude-powered conversational agent for AIS fraud detection

This module implements an optimized version of the AIS Fraud Detection Agent
using techniques from the Claude Optimization Kit to improve response speed and reliability.
"""
import anthropic
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime

# Set up logging
logger = logging.getLogger(__name__)

# Import the original agent implementation
try:
    # Try as a relative import (when running from backend dir)
    from .claude_agent import AISFraudDetectionAgent, detect_available_claude_model
    from .claude_optimizer import ClaudeOptimizer
except ImportError:
    # Try as an absolute import (when running from project root)
    from backend.claude_agent import AISFraudDetectionAgent, detect_available_claude_model
    from backend.claude_optimizer import ClaudeOptimizer
except ImportError as e:
    # Log error but keep imports for IDE and type checking
    logger.error(f"Failed to import required modules: {e}")
    # Keep the imports for IDE and type checking
    from backend.claude_agent import AISFraudDetectionAgent, detect_available_claude_model
    from backend.claude_optimizer import ClaudeOptimizer

class OptimizedAISFraudDetectionAgent(AISFraudDetectionAgent):
    """
    High-performance version of the AIS Fraud Detection Agent
    using Claude optimization techniques for faster responses and lower costs
    """
    
    def __init__(self, api_key: str, auto_detect_model: bool = True):
        """
        Initialize the optimized agent
        
        Args:
            api_key: Anthropic API key
            auto_detect_model: Whether to auto-detect the best available model
        """
        # Initialize the parent class
        super().__init__(api_key, auto_detect_model)
        
        # Create the optimizer
        self.optimizer = ClaudeOptimizer(api_key)
        
        # Cache the haiku and sonnet model variants
        self.haiku_model = self._find_best_model_variant("haiku")
        self.sonnet_model = self._find_best_model_variant("sonnet")
        logger.info(f"Using Haiku model: {self.haiku_model}")
        logger.info(f"Using Sonnet model: {self.sonnet_model}")
        
        # Convert system prompt to cacheable format
        self.cached_system_prompt = self._prepare_cached_system_prompt()
        
    def _find_best_model_variant(self, model_type: str) -> str:
        """
        Find the best available model variant of a given type
        
        Args:
            model_type: "haiku" or "sonnet"
            
        Returns:
            Best available model name
        """
        # Model preference in order (newest to oldest)
        if model_type.lower() == "haiku":
            variants = [
                "claude-haiku-4",
                "claude-3-5-haiku-20241022",
                "claude-3-5-haiku-20240620",
                "claude-3-haiku-20240307"
            ]
        elif model_type.lower() == "sonnet":
            variants = [
                "claude-sonnet-4-5-20250929",
                "claude-sonnet-4",
                "claude-3-5-sonnet-20241022",
                "claude-3-5-sonnet-20240620",
                "claude-3-sonnet-20240229"
            ]
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Try each variant in order
        for variant in variants:
            try:
                client = anthropic.Anthropic(api_key=self.client.api_key)
                response = client.messages.create(
                    model=variant,
                    max_tokens=10,
                    messages=[{"role": "user", "content": "test"}]
                )
                logger.info(f"✓ Successfully connected to {variant}")
                return variant
            except Exception as e:
                logger.debug(f"Model {variant} not available: {str(e)}")
                continue
                
        # If none work, return the first one (will trigger appropriate errors later)
        logger.warning(f"No {model_type} models available, using default")
        return variants[0]
        
    def _prepare_cached_system_prompt(self) -> List[Dict[str, Any]]:
        """
        Convert the system prompt to a cacheable format
        
        Returns:
            System prompt with cache control headers
        """
        return [{
            "type": "text",
            "text": self.system_prompt,
            "cache_control": {"type": "ephemeral"}
        }]
        
    async def chat(self, user_message: str, session_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process user message and return Claude's response with any tool calls
        
        Overrides the parent class method to use optimized request handling
        
        Args:
            user_message: User's message
            session_context: Context including map state, zones, etc.
        
        Returns:
            Dict with message, tool_calls, and metadata
        """
        # Validate history before adding new message
        self._validate_conversation_history()
        
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        try:
            # Limit conversation history to last 50 messages to reduce token usage and improve speed
            max_history_messages = 50
            if len(self.conversation_history) > max_history_messages:
                # Keep first message (usually important context) and last N messages
                trimmed_history = [self.conversation_history[0]] + self.conversation_history[-max_history_messages+1:]
                logger.debug(f"Trimmed conversation history from {len(self.conversation_history)} to {len(trimmed_history)} messages")
                messages_to_send = trimmed_history
            else:
                messages_to_send = self.conversation_history
            
            # Use the optimizer to make the request
            response = self.optimizer.optimized_request(
                messages=messages_to_send,
                system_prompt=self.cached_system_prompt,
                tools=self.tools,
                auto_select_model=True,
                temperature=0.5  # Reduced from 0.7 for faster, more deterministic responses
            )
            
            # Process response and tool calls
            result = {
                "message": "",
                "tool_calls": [],
                "stop_reason": response.stop_reason
            }
            
            for content_block in response.content:
                if content_block.type == "text":
                    result["message"] += content_block.text
                elif content_block.type == "tool_use":
                    result["tool_calls"].append({
                        "id": content_block.id,
                        "name": content_block.name,
                        "input": content_block.input
                    })
            
            # Add the response to conversation history
            self.conversation_history.append({
                "role": "assistant",
                "content": response.content
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error in optimized chat: {str(e)}")
            # Fall back to the parent implementation if optimization fails
            logger.warning("Falling back to standard implementation")
            return await super().chat(user_message, session_context)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for the optimized agent
        
        Returns:
            Dictionary with performance statistics
        """
        return self.optimizer.get_performance_stats()
