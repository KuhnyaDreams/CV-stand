"""
API wrapper for core CV services.
Provides unified interface to detect, segment, classify, and estimate tasks.
"""

import time
import requests
from pathlib import Path
import os
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

# API configuration
CORE_URL = os.getenv("CORE_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 30  # seconds


def _call_core(
    task: str,
    input_path: str,
    class_names: Optional[str] = None,
    save_images: bool = True,
    show_boxes: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Universal API call to core service.
    
    Args:
        task: Task type ('detect', 'segment', 'classify', 'estimate')
        input_path: Input image path
        class_names: Optional class names filter
        save_images: Whether to save result images
        show_boxes: Whether to show bounding boxes
        
    Returns:
        API response dictionary or None on error
        
    Raises:
        ValueError: If task type is unknown
    """
    # Validate task
    valid_tasks = {'detect', 'estimate', 'segment', 'classify'}
    if task not in valid_tasks:
        raise ValueError(f"Unknown task: {task}. Must be one of: {valid_tasks}")
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = Path(input_path)
    name = path.stem
    
    # Map task to output subdirectory
    output_subdir = {
        'detect': 'detection',
        'estimate': 'estimation',
        'segment': 'segmentation',
        'classify': 'classification',
    }.get(task, 'unknown')
    
    # Build request parameters
    params = {
        "input_path": f"/data/{input_path}",
        "output_path": f"/results/{output_subdir}/{timestamp}-{name}",
        "task": task,
        "save_images": save_images,
        "show_boxes": show_boxes,
    }
    
    # Add class names for detection and segmentation only
    if task in ['detect', 'segment']:
        params["class_names"] = class_names
    
    logger.info(f"Calling {task} API: {CORE_URL}/{task}")
    
    try:
        response = requests.post(
            f"{CORE_URL}/{task}",
            json=params,
            timeout=REQUEST_TIMEOUT
        )
        
        if response.status_code == 200:
            logger.info(f"✓ {task} completed successfully")
            return response.json()
        else:
            logger.error(
                f"✗ API error for {task}: "
                f"status {response.status_code}, "
                f"body: {response.text}"
            )
            return None
    
    except requests.Timeout:
        logger.error(f"✗ Request timeout for {task} (>{REQUEST_TIMEOUT}s)")
        return None
    
    except requests.ConnectionError:
        logger.error(f"✗ Connection error: Cannot reach {CORE_URL}")
        return None
    
    except Exception as e:
        logger.error(f"✗ Unexpected error in {task}: {e}")
        return None


def detect(
    input_path: str,
    class_names: Optional[str] = None,
    save_images: bool = True,
    show_boxes: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Run object detection on image.
    
    Args:
        input_path: Path to input image
        class_names: Optional class filter
        save_images: Save result images
        show_boxes: Show bounding boxes
        
    Returns:
        Detection results or None
    """
    return _call_core('detect', input_path, class_names, save_images, show_boxes)


def estimate(
    input_path: str,
    save_images: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Run pose estimation on image.
    
    Args:
        input_path: Path to input image
        save_images: Save result images
        
    Returns:
        Estimation results or None
    """
    return _call_core('estimate', input_path, None, save_images, False)


def segment(
    input_path: str,
    class_names: Optional[str] = None,
    save_images: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Run segmentation on image.
    
    Args:
        input_path: Path to input image
        class_names: Optional class filter
        save_images: Save result images
        
    Returns:
        Segmentation results or None
    """
    return _call_core('segment', input_path, class_names, save_images, False)


def classify(
    input_path: str,
    save_images: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Run image classification.
    
    Args:
        input_path: Path to input image
        save_images: Save result images
        
    Returns:
        Classification results or None
    """
    return _call_core('classify', input_path, None, save_images, False)

def classify(input_path, save_images = True):
    """Классификация изображения/папки."""
    return _call_core('classify', input_path, None, save_images)