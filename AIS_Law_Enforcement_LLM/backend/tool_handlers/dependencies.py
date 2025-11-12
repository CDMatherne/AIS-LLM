"""
Dependency Injection for Tool Handlers

Provides access to singleton instances from app.py without circular imports.
Uses a global registry pattern - app.py must call register_session_manager()
during startup to make the session_manager available to tool handlers.
"""

import sys
import logging

logger = logging.getLogger(__name__)

# Global registry for singleton instances
# These will be set by app.py after initialization
_session_manager = None


def register_session_manager(session_manager):
    """
    Register the session_manager singleton from app.py.
    This MUST be called once during app startup.
    
    Args:
        session_manager: The SessionManager instance from app.py
    """
    global _session_manager
    _session_manager = session_manager
    logger.info("✅ SessionManager registered with tool handlers")


def get_session_manager():
    """
    Get the session_manager singleton instance.
    
    Returns:
        SessionManager instance
    
    Raises:
        RuntimeError: If session_manager hasn't been registered yet
    """
    if _session_manager is None:
        # Critical error - this means app.py didn't call register_session_manager()
        logger.error("⚠️ CRITICAL: SessionManager not registered!")
        logger.error("app.py must call register_session_manager() during startup")
        
        # Try emergency fallback
        try:
            import app
            if hasattr(app, 'session_manager'):
                logger.warning("Emergency fallback: Using app.session_manager directly")
                return app.session_manager
        except:
            pass
        
        raise RuntimeError(
            "SessionManager not initialized. This is a configuration error - "
            "app.py must call register_session_manager() during startup."
        )
    
    return _session_manager


def get_analysis_engine(session_id: str):
    """
    Get the analysis engine for a specific session.
    
    Args:
        session_id: User session ID
    
    Returns:
        Analysis engine instance or None
    """
    session_manager = get_session_manager()
    session = session_manager.get_session(session_id)
    if session:
        return session.get('analysis_engine')
    return None
