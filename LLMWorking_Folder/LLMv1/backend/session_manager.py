"""
Session management for user interactions with the AIS LLM Assistant
"""
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import uuid
import logging
from .temp_file_manager import cleanup_session

logger = logging.getLogger(__name__)


class SessionManager:
    """
    Manages user sessions including Claude agents, data connectors, and zones
    """
    
    def __init__(self, session_timeout_minutes: int = 60):
        """
        Initialize session manager
        
        Args:
            session_timeout_minutes: Minutes before session expires
        """
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
    
    def create_session(self, session_id: str, agent, 
                      data_connector=None, config: Dict[str, Any] = None) -> str:
        """
        Create a new session
        
        Args:
            session_id: Unique session identifier
            agent: Claude agent instance
            data_connector: AIS data connector instance
            config: User configuration
        
        Returns:
            Session ID
        """
        self.sessions[session_id] = {
            'id': session_id,
            'agent': agent,
            'data_connector': data_connector,
            'config': config or {},
            'zones': [],
            'analyses': {},
            'created_at': datetime.now(),
            'last_activity': datetime.now(),
            'metadata': {}
        }
        
        logger.info(f"Created session: {session_id}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session by ID
        
        Args:
            session_id: Session identifier
        
        Returns:
            Session data or None if not found/expired
        """
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        
        # Check if session expired
        if datetime.now() - session['last_activity'] > self.session_timeout:
            logger.info(f"Session expired: {session_id}")
            self.delete_session(session_id)
            return None
        
        # Update last activity
        session['last_activity'] = datetime.now()
        
        return session
    
    def get_agent(self, session_id: str):
        """Get Claude agent for session"""
        session = self.get_session(session_id)
        return session['agent'] if session else None
    
    def get_data_connector(self, session_id: str):
        """Get data connector for session"""
        session = self.get_session(session_id)
        return session['data_connector'] if session else None
    
    def add_zone(self, session_id: str, zone: Dict[str, Any]):
        """Add a geographic zone to session"""
        session = self.get_session(session_id)
        if session:
            # Convert shapely geometry to serializable format
            zone_copy = zone.copy()
            if 'geometry' in zone_copy:
                # Store WKT representation instead of geometry object
                zone_copy['geometry_wkt'] = zone_copy['geometry'].wkt
                del zone_copy['geometry']
            
            session['zones'].append(zone_copy)
            logger.info(f"Added zone '{zone['name']}' to session {session_id}")
    
    def get_zones(self, session_id: str) -> list:
        """Get all zones for session"""
        session = self.get_session(session_id)
        return session['zones'] if session else []
    
    def store_analysis_result(self, session_id: str, analysis_id: str, 
                              result: Dict[str, Any]):
        """Store analysis result in session"""
        session = self.get_session(session_id)
        if session:
            session['analyses'][analysis_id] = {
                'result': result,
                'timestamp': datetime.now(),
                'analysis_id': analysis_id
            }
            logger.info(f"Stored analysis result {analysis_id} for session {session_id}")
    
    def get_analysis_result(self, session_id: str, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve stored analysis result"""
        session = self.get_session(session_id)
        if session and analysis_id in session['analyses']:
            return session['analyses'][analysis_id]['result']
        return None
    
    def delete_session(self, session_id: str):
        """Delete a session and cleanup temporary files"""
        if session_id in self.sessions:
            # Clean up temporary files associated with this session
            try:
                cleaned = cleanup_session(session_id)
                logger.info(f"Cleaned up {cleaned} temporary files for session {session_id}")
            except Exception as e:
                logger.warning(f"Error cleaning temp files for session {session_id}: {e}")
            
            del self.sessions[session_id]
            logger.info(f"Deleted session: {session_id}")
    
    def cleanup_expired_sessions(self):
        """Remove all expired sessions"""
        current_time = datetime.now()
        expired = []
        
        for session_id, session in self.sessions.items():
            if current_time - session['last_activity'] > self.session_timeout:
                expired.append(session_id)
        
        for session_id in expired:
            self.delete_session(session_id)
        
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired sessions")
    
    def get_session_count(self) -> int:
        """Get count of active sessions"""
        return len(self.sessions)
    
    def get_all_sessions_info(self) -> list:
        """Get info about all active sessions (for admin/monitoring)"""
        return [
            {
                'id': sid,
                'created_at': session['created_at'].isoformat(),
                'last_activity': session['last_activity'].isoformat(),
                'zones_count': len(session['zones']),
                'analyses_count': len(session['analyses'])
            }
            for sid, session in self.sessions.items()
        ]

