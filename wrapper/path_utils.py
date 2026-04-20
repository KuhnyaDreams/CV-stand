"""
Utilities for path management and file operations.
Centralizes all path discovery and temporary file handling logic.
"""

import os
import tempfile
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class PathManager:
    """Centralized path management for the wrapper module."""
    
    DATA_SUBDIRS = ["data", "../data", "../../data"]
    
    @staticmethod
    def find_image(filename: str) -> str:
        """
        Find image path by searching multiple possible locations.
        
        Args:
            filename: Image filename to search for
            
        Returns:
            Path to the image file
            
        Raises:
            FileNotFoundError: If image not found in any expected location
        """
        possible_paths = [
            filename,
            *[os.path.join(subdir, filename) for subdir in PathManager.DATA_SUBDIRS]
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                logger.debug(f"Found image at: {path}")
                return path
        
        raise FileNotFoundError(
            f"Image '{filename}' not found in: {possible_paths}"
        )
    
    @staticmethod
    def find_data_dir(create: bool = False) -> Path:
        """
        Find or create the data directory.
        
        Args:
            create: If True, create the directory if it doesn't exist
            
        Returns:
            Path object pointing to data directory
            
        Raises:
            FileNotFoundError: If directory not found and create=False
        """
        for subdir in PathManager.DATA_SUBDIRS:
            data_path = Path(subdir)
            if data_path.exists():
                logger.debug(f"Found data directory at: {data_path}")
                return data_path
        
        if create:
            data_path = Path("data")
            data_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created data directory at: {data_path}")
            return data_path
        
        raise FileNotFoundError(
            f"Data directory not found in: {PathManager.DATA_SUBDIRS}"
        )
    
    @staticmethod
    def get_temp_image_path(
        filename: str,
        prefix: str = "adv_",
        suffix: str = ".png"
    ) -> str:
        """
        Get a safe temporary file path in the data directory.
        
        Args:
            filename: Base filename
            prefix: Prefix for temporary file
            suffix: File extension
            
        Returns:
            Path to temporary file in data directory
        """
        try:
            data_dir = PathManager.find_data_dir(create=True)
            temp_name = f"{prefix}{Path(filename).stem}{suffix}"
            temp_path = data_dir / temp_name
            return str(temp_path)
        except Exception as e:
            logger.error(f"Error getting temp path: {e}")
            # Fallback to current directory
            temp_name = f"{prefix}{Path(filename).stem}{suffix}"
            return temp_name
    
    @staticmethod
    def cleanup_file(filepath: str) -> bool:
        """
        Safely remove a file with proper error handling.
        
        Args:
            filepath: Path to file to remove
            
        Returns:
            True if file was removed, False otherwise
        """
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                logger.debug(f"Cleaned up temporary file: {filepath}")
                return True
        except OSError as e:
            logger.warning(f"Failed to clean up {filepath}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error cleaning up {filepath}: {e}")
        
        return False
    
    @staticmethod
    def ensure_directory(path: str) -> Path:
        """
        Ensure directory exists, create if necessary.
        
        Args:
            path: Directory path
            
        Returns:
            Path object
        """
        dir_path = Path(path)
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path
