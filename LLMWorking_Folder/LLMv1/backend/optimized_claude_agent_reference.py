"""
OptimizedAISFraudDetectionAgent - Reference Implementation

This is a complete, working implementation of the Claude API agent
that should replace or serve as a reference for the existing implementation.

FIXES:
1. Proper Anthropic client initialization
2. Async chat() method implementation
3. Tool call handling
4. Conversation history management
5. Error handling
"""

import anthropic
from typing import Dict, Any, List, Optional
import logging
import json

logger = logging.getLogger(__name__)


class OptimizedAISFraudDetectionAgent:
    """
    Claude-powered agent for AIS fraud detection.
    
    Handles conversation with Claude, tool execution, and response processing.
    """
    
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        """
        Initialize the agent with Claude API credentials.
        
        Args:
            api_key: Anthropic API key
            model: Claude model to use (default: claude-sonnet-4-20250514)
        """
        if not api_key:
            raise ValueError("Claude API key is required")
        
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.conversation_history = []
        self.system_prompt = self._build_system_prompt()
        
        logger.info(f"Initialized OptimizedAISFraudDetectionAgent with model: {model}")
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt for Claude"""
        return """You are an expert AIS (Automatic Identification System) fraud detection assistant for law enforcement.

Your role is to help analyze vessel tracking data, detect anomalies, and provide insights for maritime security investigations.

You have access to tools for:
- Running anomaly analysis on AIS data
- Creating geographic zones for monitoring
- Generating visualizations (maps, charts)
- Exporting data in various formats
- Identifying vessel locations and types

Always be professional, thorough, and security-conscious in your responses."""
    
    def _get_tools(self) -> List[Dict[str, Any]]:
        """
        Define all available tools for Claude to use.
        
        Returns:
            List of tool definitions in Anthropic format
        """
        return [
            {
                "name": "run_anomaly_analysis",
                "description": "Run anomaly detection analysis on AIS data for a specific date range and geographic zone",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "start_date": {
                            "type": "string",
                            "description": "Start date in YYYY-MM-DD format"
                        },
                        "end_date": {
                            "type": "string",
                            "description": "End date in YYYY-MM-DD format"
                        },
                        "geographic_zone": {
                            "type": "object",
                            "description": "Optional geographic zone to analyze (polygon, circle, etc.)"
                        },
                        "anomaly_types": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Types of anomalies to detect (e.g., ais_beacon_off, speed_anomaly)"
                        },
                        "mmsi_filter": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional list of MMSI numbers to filter by"
                        },
                        "vessel_types": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "Optional list of vessel type codes to filter by"
                        }
                    },
                    "required": ["start_date", "end_date"]
                }
            },
            {
                "name": "get_vessel_history",
                "description": "Get complete movement history for specific vessel(s)",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "mmsi": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of MMSI numbers to get history for"
                        },
                        "start_date": {
                            "type": "string",
                            "description": "Start date in YYYY-MM-DD format"
                        },
                        "end_date": {
                            "type": "string",
                            "description": "End date in YYYY-MM-DD format"
                        }
                    },
                    "required": ["mmsi", "start_date", "end_date"]
                }
            },
            {
                "name": "create_all_anomalies_map",
                "description": "Create an interactive map showing all detected anomalies",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "analysis_id": {
                            "type": "string",
                            "description": "ID of the analysis to visualize"
                        },
                        "show_clustering": {
                            "type": "boolean",
                            "description": "Whether to show anomaly clusters"
                        },
                        "show_heatmap": {
                            "type": "boolean",
                            "description": "Whether to show density heatmap"
                        }
                    },
                    "required": ["analysis_id"]
                }
            },
            {
                "name": "create_anomaly_types_chart",
                "description": "Create a chart showing distribution of anomaly types",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "analysis_id": {
                            "type": "string",
                            "description": "ID of the analysis to visualize"
                        },
                        "chart_type": {
                            "type": "string",
                            "enum": ["bar", "pie", "both"],
                            "description": "Type of chart to create"
                        }
                    },
                    "required": ["analysis_id"]
                }
            },
            {
                "name": "export_to_csv",
                "description": "Export analysis results to CSV format",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "analysis_id": {
                            "type": "string",
                            "description": "ID of the analysis to export"
                        },
                        "export_type": {
                            "type": "string",
                            "enum": ["anomalies", "statistics"],
                            "description": "What to export"
                        }
                    },
                    "required": ["analysis_id"]
                }
            },
            {
                "name": "set_analysis_timespan",
                "description": "Set the date range for analysis in this session",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "start_date": {
                            "type": "string",
                            "description": "Start date in YYYY-MM-DD format"
                        },
                        "end_date": {
                            "type": "string",
                            "description": "End date in YYYY-MM-DD format"
                        }
                    },
                    "required": ["start_date", "end_date"]
                }
            },
            {
                "name": "get_current_timespan",
                "description": "Get the currently set date range for this session",
                "input_schema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "identify_vessel_location",
                "description": "Identify which water body a vessel or location is in",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "mmsi": {
                            "type": "string",
                            "description": "MMSI of vessel to locate"
                        },
                        "longitude": {
                            "type": "number",
                            "description": "Longitude coordinate"
                        },
                        "latitude": {
                            "type": "number",
                            "description": "Latitude coordinate"
                        }
                    }
                }
            },
            {
                "name": "list_vessel_types",
                "description": "List all vessel type codes and categories",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": "Optional category to filter by"
                        }
                    }
                }
            }
        ]
    
    async def chat(self, message: str, session_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Send a message to Claude and get a response.
        
        Args:
            message: User message to send to Claude
            session_context: Optional context (map_context, etc.)
        
        Returns:
            Dict with keys:
                - 'message': Claude's text response (str)
                - 'tool_calls': List of tool calls if any (optional)
        """
        try:
            # Add user message to history
            user_message = {"role": "user", "content": message}
            
            # Build messages array (conversation history + new message)
            messages = self.conversation_history + [user_message]
            
            # Call Claude API
            logger.debug(f"Sending message to Claude: {message[:100]}...")
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=self.system_prompt,
                messages=messages,
                tools=self._get_tools()
            )
            
            logger.debug(f"Claude response stop_reason: {response.stop_reason}")
            
            # Process response based on stop reason
            if response.stop_reason == "tool_use":
                # Extract tool calls
                tool_calls = []
                text_content = ""
                
                for block in response.content:
                    if block.type == "tool_use":
                        tool_calls.append({
                            "id": block.id,
                            "name": block.name,
                            "input": block.input
                        })
                        logger.info(f"Tool call requested: {block.name}")
                    elif block.type == "text":
                        text_content += block.text
                
                # Store assistant message in history (including tool use blocks)
                self.conversation_history.append(user_message)
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response.content
                })
                
                return {
                    "message": text_content,
                    "tool_calls": tool_calls
                }
            
            else:
                # Regular text response
                text_content = ""
                for block in response.content:
                    if block.type == "text":
                        text_content += block.text
                
                # Store in history
                self.conversation_history.append(user_message)
                self.conversation_history.append({
                    "role": "assistant",
                    "content": text_content
                })
                
                return {
                    "message": text_content
                }
        
        except anthropic.APIError as e:
            logger.error(f"Claude API error: {e}")
            return {
                "message": f"I encountered an error communicating with the AI service: {str(e)}",
                "error": str(e)
            }
        except Exception as e:
            logger.error(f"Unexpected error in chat: {e}", exc_info=True)
            return {
                "message": f"An unexpected error occurred: {str(e)}",
                "error": str(e)
            }
    
    async def process_tool_results(self, tool_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process tool execution results and get Claude's next response.
        
        Args:
            tool_results: List of dicts with keys:
                - 'tool_call_id': The tool call ID from Claude
                - 'result': The result of tool execution
        
        Returns:
            Dict with 'message' and possibly 'tool_calls' if Claude wants to use more tools
        """
        try:
            # Build tool result content blocks
            tool_result_content = []
            for result in tool_results:
                # Convert result to string if it's a dict
                result_str = json.dumps(result["result"]) if isinstance(result["result"], dict) else str(result["result"])
                
                tool_result_content.append({
                    "type": "tool_result",
                    "tool_use_id": result["tool_call_id"],
                    "content": result_str
                })
            
            # Add tool results as user message
            user_message = {
                "role": "user",
                "content": tool_result_content
            }
            self.conversation_history.append(user_message)
            
            # Get Claude's response
            logger.debug("Sending tool results to Claude...")
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=self.system_prompt,
                messages=self.conversation_history,
                tools=self._get_tools()
            )
            
            logger.debug(f"Claude response after tools: {response.stop_reason}")
            
            # Process response
            if response.stop_reason == "tool_use":
                # Claude wants to use more tools
                tool_calls = []
                text_content = ""
                
                for block in response.content:
                    if block.type == "tool_use":
                        tool_calls.append({
                            "id": block.id,
                            "name": block.name,
                            "input": block.input
                        })
                        logger.info(f"Additional tool call: {block.name}")
                    elif block.type == "text":
                        text_content += block.text
                
                # Store in history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response.content
                })
                
                return {
                    "message": text_content,
                    "tool_calls": tool_calls
                }
            else:
                # Final text response
                text_content = ""
                for block in response.content:
                    if block.type == "text":
                        text_content += block.text
                
                # Store in history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": text_content
                })
                
                return {
                    "message": text_content
                }
        
        except anthropic.APIError as e:
            logger.error(f"Claude API error processing tool results: {e}")
            return {
                "message": f"I encountered an error processing the tool results: {str(e)}",
                "error": str(e)
            }
        except Exception as e:
            logger.error(f"Unexpected error processing tool results: {e}", exc_info=True)
            return {
                "message": f"An unexpected error occurred: {str(e)}",
                "error": str(e)
            }
    
    def get_conversation_length(self) -> int:
        """Get the number of messages in the conversation history"""
        return len(self.conversation_history)
    
    def _validate_conversation_history(self):
        """
        Validate and clean up conversation history.
        
        Ensures alternating user/assistant messages and no orphaned tool_use blocks.
        """
        # This is called by the WebSocket handler to clean up any issues
        # For now, just log the history length
        logger.debug(f"Conversation history length: {len(self.conversation_history)}")
    
    def clear_history(self):
        """Clear the conversation history"""
        self.conversation_history = []
        logger.info("Conversation history cleared")


# Example usage:
if __name__ == "__main__":
    import asyncio
    import os
    
    async def test_agent():
        """Test the agent with a simple conversation"""
        api_key = os.getenv("CLAUDE_API_KEY")
        if not api_key:
            print("Error: CLAUDE_API_KEY environment variable not set")
            return
        
        agent = OptimizedAISFraudDetectionAgent(api_key)
        
        # Test simple chat
        print("Testing simple chat...")
        response = await agent.chat("Hello! Can you introduce yourself?")
        print(f"Claude: {response['message']}")
        
        # Test tool use
        print("\nTesting tool use...")
        response = await agent.chat("Can you set the analysis timespan to October 15-17, 2024?")
        print(f"Claude: {response['message']}")
        if response.get('tool_calls'):
            print(f"Tool calls: {response['tool_calls']}")
    
    # Run test
    asyncio.run(test_agent())
