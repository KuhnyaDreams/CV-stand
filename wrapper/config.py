"""
Configuration management and initialization.
Handles loading and providing access to configuration parameters.
"""

from pathlib import Path
from typing import Dict, Optional
import yaml
import logging

logger = logging.getLogger(__name__)

# Global configuration instance
_CONFIG: Optional[Dict] = None


def load_config(config_path: str = "config.yaml") -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    global _CONFIG
    
    try:
        config_file = Path(config_path)
        if not config_file.exists():
            logger.warning(f"Config file not found: {config_path}")
            logger.info("Using default parameters...")
            _CONFIG = {}
            return _CONFIG
        
        with open(config_file, 'r') as f:
            _CONFIG = yaml.safe_load(f)
            logger.info(f"Loaded config from: {config_path}")
            return _CONFIG
    
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error in {config_path}: {e}")
        _CONFIG = {}
        return _CONFIG
    
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        _CONFIG = {}
        return _CONFIG


def get_config() -> Dict:
    """
    Get current configuration (load if not already loaded).
    
    Returns:
        Configuration dictionary
    """
    global _CONFIG
    
    if _CONFIG is None:
        load_config()
    
    return _CONFIG or {}


def find_image_path(filename: str = "test.jpg") -> str:
    """
    Find image path by searching common locations.
    
    Args:
        filename: Image filename to search for
        
    Returns:
        Path to image file
        
    Raises:
        FileNotFoundError: If image not found
    """
    from path_utils import PathManager
    
    try:
        return PathManager.find_image(filename)
    except FileNotFoundError:
        logger.error(f"Image not found: {filename}")
        # Fallback to default
        return "data/test.jpg"


def configure_logging(level: str = "INFO") -> None:
    """
    Configure logging for the module.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )