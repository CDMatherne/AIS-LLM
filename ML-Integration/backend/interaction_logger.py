"""
Interaction Logger

Captures detailed logs of LLM tool interactions, data gathering, and processing.
Creates structured logs for debugging and monitoring.
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import traceback

# Configure detailed logging
logger = logging.getLogger(__name__)


class InteractionLogger:
    """
    Structured logger for LLM interactions and tool execution.
    Captures detailed information about data gathering and processing.
    """
    
    def __init__(self, log_dir: str = "logs"):
        """
        Initialize interaction logger.
        
        Args:
            log_dir: Directory to store log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create separate log files for different types
        self.setup_file_handlers()
        
        # Current interaction context
        self.current_interaction_id = None
        self.interaction_start_time = None
        
    def setup_file_handlers(self):
        """Set up file handlers for different log types"""
        
        # Main interaction log (all interactions)
        interaction_handler = logging.FileHandler(
            self.log_dir / "interactions.log"
        )
        interaction_handler.setLevel(logging.INFO)
        interaction_handler.setFormatter(logging.Formatter(
            '%(asctime)s - [%(levelname)s] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        
        # Tool execution log (detailed tool calls)
        tool_handler = logging.FileHandler(
            self.log_dir / "tool_execution.log"
        )
        tool_handler.setLevel(logging.DEBUG)
        tool_handler.setFormatter(logging.Formatter(
            '%(asctime)s - [TOOL] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        
        # Data processing log (data loading and analysis)
        data_handler = logging.FileHandler(
            self.log_dir / "data_processing.log"
        )
        data_handler.setLevel(logging.DEBUG)
        data_handler.setFormatter(logging.Formatter(
            '%(asctime)s - [DATA] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        
        # Error log (all errors)
        error_handler = logging.FileHandler(
            self.log_dir / "errors.log"
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(logging.Formatter(
            '%(asctime)s - [ERROR] - %(message)s\n%(exc_info)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        
        # Add handlers to logger
        logger.addHandler(interaction_handler)
        logger.addHandler(tool_handler)
        logger.addHandler(data_handler)
        logger.addHandler(error_handler)
        
    def start_interaction(self, session_id: str, message: str) -> str:
        """
        Log the start of a new user interaction.
        
        Args:
            session_id: User session ID
            message: User's message
            
        Returns:
            Interaction ID
        """
        interaction_id = f"{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        self.current_interaction_id = interaction_id
        self.interaction_start_time = datetime.now()
        
        logger.info("="*80)
        logger.info(f"🔵 NEW INTERACTION: {interaction_id}")
        logger.info(f"Session: {session_id}")
        logger.info(f"User Message: {message}")
        logger.info(f"Timestamp: {self.interaction_start_time.isoformat()}")
        logger.info("="*80)
        
        return interaction_id
    
    def log_llm_response(self, response: Dict[str, Any]):
        """
        Log LLM's response.
        
        Args:
            response: LLM response dictionary
        """
        logger.info("📤 LLM RESPONSE:")
        
        if 'message' in response:
            logger.info(f"Message: {response['message'][:200]}{'...' if len(response['message']) > 200 else ''}")
        
        if 'tool_calls' in response and response['tool_calls']:
            logger.info(f"Tool Calls: {len(response['tool_calls'])} tool(s) requested")
            for i, tool_call in enumerate(response['tool_calls'], 1):
                logger.info(f"  Tool {i}: {tool_call.get('name', 'unknown')}")
    
    def log_tool_execution_start(self, tool_name: str, tool_input: Dict[str, Any], session_id: str):
        """
        Log the start of tool execution.
        
        Args:
            tool_name: Name of the tool being executed
            tool_input: Tool input parameters
            session_id: Session ID
        """
        logger.info("-"*80)
        logger.info(f"[TOOL START] {tool_name}")
        logger.info(f"Session: {session_id}")
        logger.info(f"Input Parameters:")
        
        # Log parameters with sensitive data masked
        safe_input = self._mask_sensitive_data(tool_input)
        for key, value in safe_input.items():
            if isinstance(value, (str, int, float, bool)):
                logger.info(f"  {key}: {value}")
            elif isinstance(value, (list, dict)):
                logger.info(f"  {key}: {json.dumps(value)[:100]}{'...' if len(json.dumps(value)) > 100 else ''}")
            else:
                logger.info(f"  {key}: {type(value).__name__}")
        
        logger.info(f"Execution Time: {datetime.now().isoformat()}")
    
    def log_tool_execution_end(self, tool_name: str, result: Any, duration_ms: float):
        """
        Log the end of tool execution.
        
        Args:
            tool_name: Name of the tool
            result: Tool execution result
            duration_ms: Execution duration in milliseconds
        """
        logger.info(f"[TOOL COMPLETE] {tool_name}")
        logger.info(f"Duration: {duration_ms:.2f}ms")
        
        # Log result summary
        if isinstance(result, dict):
            if 'success' in result:
                logger.info(f"Success: {result['success']}")
            if 'error' in result:
                logger.info(f"Error: {result['error']}")
            if 'analysis_id' in result:
                logger.info(f"Analysis ID: {result['analysis_id']}")
            if 'anomaly_count' in result or 'total_anomalies' in result:
                count = result.get('anomaly_count', result.get('total_anomalies', 0))
                logger.info(f"Anomalies Found: {count}")
        
        logger.info("-"*80)
    
    def log_tool_execution_error(self, tool_name: str, error: Exception, duration_ms: float):
        """
        Log tool execution error.
        
        Args:
            tool_name: Name of the tool
            error: Exception that occurred
            duration_ms: Execution duration in milliseconds
        """
        logger.error(f"[TOOL FAILED] {tool_name}")
        logger.error(f"Duration: {duration_ms:.2f}ms")
        logger.error(f"Error Type: {type(error).__name__}")
        logger.error(f"Error Message: {str(error)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        logger.error("-"*80)
    
    def log_data_loading_start(self, source: str, date_range: str, filters: Optional[Dict] = None):
        """
        Log the start of data loading.
        
        Args:
            source: Data source (S3, local, etc.)
            date_range: Date range being loaded
            filters: Optional filters applied
        """
        logger.info("[DATA LOADING START]")
        logger.info(f"Source: {source}")
        logger.info(f"Date Range: {date_range}")
        if filters:
            logger.info(f"Filters: {json.dumps(filters)}")
    
    def log_data_loading_progress(self, current_date: str, total_dates: int, current_index: int):
        """
        Log data loading progress.
        
        Args:
            current_date: Current date being processed
            total_dates: Total number of dates
            current_index: Current index (0-based)
        """
        percentage = ((current_index + 1) / total_dates) * 100
        logger.info(f"[LOADING] Data for {current_date} ({current_index + 1}/{total_dates} - {percentage:.1f}%)")
    
    def log_data_loading_complete(self, total_records: int, duration_ms: float):
        """
        Log data loading completion.
        
        Args:
            total_records: Total records loaded
            duration_ms: Loading duration in milliseconds
        """
        logger.info("[DATA LOADING COMPLETE]")
        logger.info(f"Total Records: {total_records:,}")
        logger.info(f"Duration: {duration_ms:.2f}ms")
        logger.info(f"Records/second: {(total_records / (duration_ms / 1000)):.2f}")
    
    def log_analysis_start(self, analysis_id: str, config: Dict[str, Any]):
        """
        Log the start of analysis.
        
        Args:
            analysis_id: Unique analysis ID
            config: Analysis configuration
        """
        logger.info("[ANALYSIS START]")
        logger.info(f"Analysis ID: {analysis_id}")
        logger.info(f"Configuration:")
        for key, value in config.items():
            logger.info(f"  {key}: {value}")
    
    def log_analysis_progress(self, stage: str, details: str):
        """
        Log analysis progress.
        
        Args:
            stage: Current analysis stage
            details: Stage details
        """
        logger.info(f"⚙️ Analysis Stage: {stage} - {details}")
    
    def log_analysis_complete(self, analysis_id: str, results: Dict[str, Any], duration_ms: float):
        """
        Log analysis completion.
        
        Args:
            analysis_id: Analysis ID
            results: Analysis results summary
            duration_ms: Analysis duration
        """
        logger.info(f"[ANALYSIS COMPLETE] {analysis_id}")
        logger.info(f"Duration: {duration_ms:.2f}ms")
        
        if 'statistics' in results:
            stats = results['statistics']
            logger.info("Results Summary:")
            for key, value in stats.items():
                logger.info(f"  {key}: {value}")
    
    def end_interaction(self, success: bool = True):
        """
        Log the end of an interaction.
        
        Args:
            success: Whether interaction completed successfully
        """
        if self.interaction_start_time:
            duration = (datetime.now() - self.interaction_start_time).total_seconds()
            
            status = "[SUCCESS]" if success else "[FAILED]"
            logger.info("="*80)
            logger.info(f"[INTERACTION END] {self.current_interaction_id}")
            logger.info(f"Status: {status}")
            logger.info(f"Total Duration: {duration:.2f}s")
            logger.info("="*80)
            logger.info("")  # Blank line for readability
        
        self.current_interaction_id = None
        self.interaction_start_time = None
    
    def _mask_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mask sensitive data in logs.
        
        Args:
            data: Data dictionary
            
        Returns:
            Dictionary with sensitive fields masked
        """
        sensitive_keys = ['api_key', 'password', 'secret', 'token', 'credential']
        
        masked = {}
        for key, value in data.items():
            key_lower = key.lower()
            if any(sensitive in key_lower for sensitive in sensitive_keys):
                masked[key] = "***MASKED***"
            else:
                masked[key] = value
        
        return masked
    
    def log_custom(self, message: str, level: str = "info"):
        """
        Log a custom message.
        
        Args:
            message: Message to log
            level: Log level (debug, info, warning, error)
        """
        log_func = getattr(logger, level.lower(), logger.info)
        log_func(message)


# Global interaction logger instance
interaction_logger = InteractionLogger()

