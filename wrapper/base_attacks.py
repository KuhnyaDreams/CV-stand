"""
Base class for adversarial attacks.
Provides common functionality for white-box and black-box attacks.
"""

import numpy as np
from typing import Any, Optional
from abc import ABC
import logging

from config_validator import ConfigValidator
from path_utils import PathManager

logger = logging.getLogger(__name__)


class AttackBase(ABC):
    """Base class providing common attack functionality."""
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize attack base.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.validator = ConfigValidator()
    
    def get_config_param(
        self,
        section: str,
        subsection: str,
        param: str,
        default: Any
    ) -> Any:
        """
        Get parameter from configuration with validation.
        
        Args:
            section: Configuration section
            subsection: Configuration subsection
            param: Parameter name
            default: Default value if not found
            
        Returns:
            Parameter value or default
        """
        return ConfigValidator.get_param(
            self.config,
            section,
            subsection,
            param,
            default
        )
    
    @staticmethod
    def validate_image(image: np.ndarray) -> None:
        """
        Validate image array.
        
        Args:
            image: Image array to validate
            
        Raises:
            ValueError: If image is invalid
        """
        if not isinstance(image, np.ndarray):
            raise ValueError(f"Expected ndarray, got {type(image)}")
        
        if len(image.shape) not in (2, 3):
            raise ValueError(
                f"Expected 2D or 3D array, got shape {image.shape}"
            )
        
        if image.size == 0:
            raise ValueError("Image array is empty")
    
    @staticmethod
    def normalize_to_unit(
        image: np.ndarray,
        dtype: np.dtype = np.float32
    ) -> np.ndarray:
        """
        Normalize image to [0, 1] range.
        
        Args:
            image: Image array [0, 255]
            dtype: Output dtype
            
        Returns:
            Normalized image [0, 1]
        """
        AttackBase.validate_image(image)
        return image.astype(dtype) / 255.0
    
    @staticmethod
    def denormalize_to_uint8(image: np.ndarray) -> np.ndarray:
        """
        Convert normalized image [0, 1] to uint8 [0, 255].
        
        Args:
            image: Normalized image [0, 1]
            
        Returns:
            uint8 image [0, 255]
        """
        clipped = np.clip(image, 0, 1)
        return (clipped * 255).astype(np.uint8)
    
    @staticmethod
    def clip_image(
        image: np.ndarray,
        min_val: float = 0.0,
        max_val: float = 255.0
    ) -> np.ndarray:
        """
        Clip image values to valid range.
        
        Args:
            image: Image array
            min_val: Minimum value
            max_val: Maximum value
            
        Returns:
            Clipped image
        """
        if image.max() <= 1.0:
            return np.clip(image, min_val / 255.0, max_val / 255.0)
        return np.clip(image, min_val, max_val)
    
    @staticmethod
    def add_noise(
        image: np.ndarray,
        noise_scale: float,
        noise_type: str = 'gaussian'
    ) -> np.ndarray:
        """
        Add noise to image.
        
        Args:
            image: Image array
            noise_scale: Noise magnitude
            noise_type: Type of noise ('gaussian', 'uniform')
            
        Returns:
            Image with added noise
        """
        AttackBase.validate_image(image)
        
        if noise_type == 'gaussian':
            noise = np.random.normal(0, noise_scale, image.shape)
        elif noise_type == 'uniform':
            noise = np.random.uniform(-noise_scale, noise_scale, image.shape)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
        
        return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    def log_attack(self, attack_name: str, **kwargs) -> None:
        """
        Log attack execution with parameters.
        
        Args:
            attack_name: Name of attack
            **kwargs: Additional parameters to log
        """
        params_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        logger.info(f"Running {attack_name}: {params_str}")
