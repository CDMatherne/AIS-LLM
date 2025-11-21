"""
Temporary File Manager
Handles creation, tracking, and cleanup of temporary files throughout the application
"""
import os
import shutil
import tempfile
import atexit
import logging
from pathlib import Path
from typing import List, Set, Optional
from datetime import datetime, timedelta
import threading
import weakref

logger = logging.getLogger(__name__)


class TempFileManager:
    """
    Global temporary file manager with automatic cleanup
    Ensures no temporary files are left behind when application closes
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern to ensure only one manager exists"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the temp file manager"""
        if self._initialized:
            return
            
        self._initialized = True
        self.temp_files: Set[str] = set()
        self.temp_dirs: Set[str] = set()
        self.session_temps: dict = {}  # session_id -> set of temp files
        self._cleanup_lock = threading.Lock()
        
        # Register cleanup on program exit
        atexit.register(self.cleanup_all)
        
        logger.info("TempFileManager initialized")
    
    def create_temp_file(self, suffix: str = '', prefix: str = 'ais_', 
                        session_id: Optional[str] = None, 
                        auto_cleanup: bool = True) -> str:
        """
        Create a temporary file and register it for cleanup
        
        Args:
            suffix: File suffix (e.g., '.csv', '.html')
            prefix: File prefix
            session_id: Optional session ID to associate file with
            auto_cleanup: Whether to auto-cleanup this file
            
        Returns:
            Path to temporary file
        """
        try:
            # Create temporary file
            fd, path = tempfile.mkstemp(suffix=suffix, prefix=prefix)
            os.close(fd)  # Close file descriptor immediately
            
            if auto_cleanup:
                with self._cleanup_lock:
                    self.temp_files.add(path)
                    
                    # Associate with session if provided
                    if session_id:
                        if session_id not in self.session_temps:
                            self.session_temps[session_id] = set()
                        self.session_temps[session_id].add(path)
            
            logger.debug(f"Created temp file: {path}")
            return path
            
        except Exception as e:
            logger.error(f"Failed to create temp file: {e}")
            raise
    
    def create_temp_dir(self, suffix: str = '', prefix: str = 'ais_',
                       session_id: Optional[str] = None,
                       auto_cleanup: bool = True) -> str:
        """
        Create a temporary directory and register it for cleanup
        
        Args:
            suffix: Directory suffix
            prefix: Directory prefix
            session_id: Optional session ID to associate directory with
            auto_cleanup: Whether to auto-cleanup this directory
            
        Returns:
            Path to temporary directory
        """
        try:
            # Create temporary directory
            path = tempfile.mkdtemp(suffix=suffix, prefix=prefix)
            
            if auto_cleanup:
                with self._cleanup_lock:
                    self.temp_dirs.add(path)
                    
                    # Associate with session if provided
                    if session_id:
                        if session_id not in self.session_temps:
                            self.session_temps[session_id] = set()
                        self.session_temps[session_id].add(path)
            
            logger.debug(f"Created temp directory: {path}")
            return path
            
        except Exception as e:
            logger.error(f"Failed to create temp directory: {e}")
            raise
    
    def register_temp_file(self, path: str, session_id: Optional[str] = None):
        """
        Register an existing file as temporary for cleanup
        
        Args:
            path: Path to file to register
            session_id: Optional session ID to associate with
        """
        with self._cleanup_lock:
            self.temp_files.add(path)
            
            if session_id:
                if session_id not in self.session_temps:
                    self.session_temps[session_id] = set()
                self.session_temps[session_id].add(path)
        
        logger.debug(f"Registered temp file: {path}")
    
    def register_temp_dir(self, path: str, session_id: Optional[str] = None):
        """
        Register an existing directory as temporary for cleanup
        
        Args:
            path: Path to directory to register
            session_id: Optional session ID to associate with
        """
        with self._cleanup_lock:
            self.temp_dirs.add(path)
            
            if session_id:
                if session_id not in self.session_temps:
                    self.session_temps[session_id] = set()
                self.session_temps[session_id].add(path)
        
        logger.debug(f"Registered temp directory: {path}")
    
    def cleanup_file(self, path: str) -> bool:
        """
        Clean up a specific temporary file
        
        Args:
            path: Path to file to clean up
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if os.path.exists(path) and os.path.isfile(path):
                os.remove(path)
                logger.debug(f"Cleaned up temp file: {path}")
                
            with self._cleanup_lock:
                self.temp_files.discard(path)
                # Remove from all session temps
                for session_temps in self.session_temps.values():
                    session_temps.discard(path)
            
            return True
            
        except Exception as e:
            logger.warning(f"Failed to cleanup file {path}: {e}")
            return False
    
    def cleanup_dir(self, path: str) -> bool:
        """
        Clean up a specific temporary directory
        
        Args:
            path: Path to directory to clean up
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if os.path.exists(path) and os.path.isdir(path):
                shutil.rmtree(path)
                logger.debug(f"Cleaned up temp directory: {path}")
            
            with self._cleanup_lock:
                self.temp_dirs.discard(path)
                # Remove from all session temps
                for session_temps in self.session_temps.values():
                    session_temps.discard(path)
            
            return True
            
        except Exception as e:
            logger.warning(f"Failed to cleanup directory {path}: {e}")
            return False
    
    def cleanup_session(self, session_id: str) -> int:
        """
        Clean up all temporary files for a specific session
        
        Args:
            session_id: Session ID to clean up
            
        Returns:
            Number of items cleaned up
        """
        cleaned = 0
        
        with self._cleanup_lock:
            session_temps = self.session_temps.get(session_id, set()).copy()
        
        for path in session_temps:
            if os.path.isfile(path):
                if self.cleanup_file(path):
                    cleaned += 1
            elif os.path.isdir(path):
                if self.cleanup_dir(path):
                    cleaned += 1
        
        with self._cleanup_lock:
            if session_id in self.session_temps:
                del self.session_temps[session_id]
        
        logger.info(f"Cleaned up {cleaned} items for session {session_id}")
        return cleaned
    
    def cleanup_all(self) -> tuple:
        """
        Clean up all registered temporary files and directories
        Called automatically on program exit
        
        Returns:
            Tuple of (files_cleaned, dirs_cleaned)
        """
        files_cleaned = 0
        dirs_cleaned = 0
        
        logger.info("Starting cleanup of all temporary files...")
        
        # Clean up files
        with self._cleanup_lock:
            temp_files = self.temp_files.copy()
        
        for path in temp_files:
            if self.cleanup_file(path):
                files_cleaned += 1
        
        # Clean up directories
        with self._cleanup_lock:
            temp_dirs = self.temp_dirs.copy()
        
        for path in temp_dirs:
            if self.cleanup_dir(path):
                dirs_cleaned += 1
        
        logger.info(f"Cleanup complete: {files_cleaned} files, {dirs_cleaned} directories")
        return (files_cleaned, dirs_cleaned)
    
    def cleanup_old_system_temps(self, days_old: int = 7) -> int:
        """
        Clean up old temporary files from system temp directory
        Looks for files with our prefix that are older than specified days
        
        Args:
            days_old: Delete files older than this many days
            
        Returns:
            Number of files cleaned up
        """
        cleaned = 0
        temp_dir = Path(tempfile.gettempdir())
        cutoff_time = datetime.now() - timedelta(days=days_old)
        
        logger.info(f"Scanning for old temp files in {temp_dir} (older than {days_old} days)")
        
        try:
            for item in temp_dir.glob('ais_*'):
                try:
                    # Get modification time
                    mtime = datetime.fromtimestamp(item.stat().st_mtime)
                    
                    if mtime < cutoff_time:
                        if item.is_file():
                            item.unlink()
                            cleaned += 1
                            logger.debug(f"Cleaned old temp file: {item}")
                        elif item.is_dir():
                            shutil.rmtree(item)
                            cleaned += 1
                            logger.debug(f"Cleaned old temp directory: {item}")
                except Exception as e:
                    logger.warning(f"Failed to clean {item}: {e}")
            
            logger.info(f"Cleaned {cleaned} old temporary items")
            
        except Exception as e:
            logger.error(f"Error scanning temp directory: {e}")
        
        return cleaned
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get statistics about tracked temporary files
        
        Returns:
            Dictionary with statistics
        """
        with self._cleanup_lock:
            return {
                'temp_files': len(self.temp_files),
                'temp_dirs': len(self.temp_dirs),
                'sessions': len(self.session_temps),
                'total_session_temps': sum(len(temps) for temps in self.session_temps.values())
            }


# Global instance
temp_manager = TempFileManager()


# Convenience functions
def create_temp_file(suffix: str = '', prefix: str = 'ais_', 
                    session_id: Optional[str] = None) -> str:
    """Create a temporary file with automatic cleanup"""
    return temp_manager.create_temp_file(suffix, prefix, session_id)


def create_temp_dir(suffix: str = '', prefix: str = 'ais_',
                   session_id: Optional[str] = None) -> str:
    """Create a temporary directory with automatic cleanup"""
    return temp_manager.create_temp_dir(suffix, prefix, session_id)


def register_temp_file(path: str, session_id: Optional[str] = None):
    """Register an existing file for automatic cleanup"""
    temp_manager.register_temp_file(path, session_id)


def register_temp_dir(path: str, session_id: Optional[str] = None):
    """Register an existing directory for automatic cleanup"""
    temp_manager.register_temp_dir(path, session_id)


def cleanup_session(session_id: str) -> int:
    """Clean up all temporary files for a session"""
    return temp_manager.cleanup_session(session_id)


def cleanup_all() -> tuple:
    """Clean up all temporary files"""
    return temp_manager.cleanup_all()


# Context manager for temporary files
class TempFile:
    """Context manager for temporary files with automatic cleanup"""
    
    def __init__(self, suffix: str = '', prefix: str = 'ais_', 
                session_id: Optional[str] = None):
        self.suffix = suffix
        self.prefix = prefix
        self.session_id = session_id
        self.path = None
    
    def __enter__(self) -> str:
        self.path = temp_manager.create_temp_file(
            self.suffix, self.prefix, self.session_id, auto_cleanup=False
        )
        return self.path
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.path:
            temp_manager.cleanup_file(self.path)
        return False


class TempDir:
    """Context manager for temporary directories with automatic cleanup"""
    
    def __init__(self, suffix: str = '', prefix: str = 'ais_',
                session_id: Optional[str] = None):
        self.suffix = suffix
        self.prefix = prefix
        self.session_id = session_id
        self.path = None
    
    def __enter__(self) -> str:
        self.path = temp_manager.create_temp_dir(
            self.suffix, self.prefix, self.session_id, auto_cleanup=False
        )
        return self.path
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.path:
            temp_manager.cleanup_dir(self.path)
        return False


if __name__ == "__main__":
    # Test the temp file manager
    logging.basicConfig(level=logging.DEBUG)
    
    print("Testing TempFileManager...")
    
    # Test file creation
    temp_file = create_temp_file(suffix='.txt')
    print(f"Created temp file: {temp_file}")
    assert os.path.exists(temp_file)
    
    # Test directory creation
    temp_dir = create_temp_dir()
    print(f"Created temp directory: {temp_dir}")
    assert os.path.exists(temp_dir)
    
    # Test context managers
    with TempFile(suffix='.csv') as tf:
        print(f"Temp file in context: {tf}")
        assert os.path.exists(tf)
    assert not os.path.exists(tf), "Context manager should have cleaned up file"
    
    with TempDir() as td:
        print(f"Temp dir in context: {td}")
        assert os.path.exists(td)
    assert not os.path.exists(td), "Context manager should have cleaned up directory"
    
    # Test stats
    stats = temp_manager.get_stats()
    print(f"Stats: {stats}")
    
    # Test cleanup
    files, dirs = cleanup_all()
    print(f"Cleaned up: {files} files, {dirs} directories")
    
    # Verify cleanup
    assert not os.path.exists(temp_file)
    assert not os.path.exists(temp_dir)
    
    print("All tests passed!")

