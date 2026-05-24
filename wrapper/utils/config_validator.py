"""
Configuration validation and management.
Ensures configuration has correct structure before use.
"""

from typing import Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)


class ConfigValidator:
    """Validates and manages configuration structure."""
    
    # Define required configuration schema
    REQUIRED_SECTIONS = {
        'white_box_attacks': [
            'fgsm',
            'pgd',
            'deepfool',
            'jsma'
        ],
        'black_box_attacks': [
            'single_pixel',
            'random_noise',
            'gaussian_blur',
            'patch',
            'blackout'
        ]
    }
    
    DEFAULT_CONFIG = {
        'white_box_attacks': {
            'fgsm': {'epsilon': 0.03},
            'pgd': {'epsilon': 0.03, 'num_steps': 7},
            'deepfool': {'num_classes': 80},
            'jsma': {'theta': 1.0, 'gamma': 0.1}
        },
        'black_box_attacks': {
            'single_pixel': {'num_modifications': 1},
            'random_noise': {'noise_level': 0.1},
            'gaussian_blur': {'kernel_size': 5},
            'patch': {'patch_size': 32, 'patch_color': [255, 0, 0]},
            'blackout': {}
        },
        'defenses': {
            'gaussian_blur': {'kernel_size': 5},
            'jpeg_compression': {'quality': 60},
            'denoise': {},
            'random_resize': {},
            'normalize_lighting': {},
            'combined': {'quality': 70, 'blur_kernel': 3}
        }
    }
    
    @staticmethod
    def validate(config: Dict[str, Any]) -> None:
        """
        Validate configuration structure.
        
        Args:
            config: Configuration dictionary to validate
            
        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(config, dict):
            raise ValueError(f"Config must be a dictionary, got {type(config)}")
        
        for section, required_keys in ConfigValidator.REQUIRED_SECTIONS.items():
            if section not in config:
                raise ValueError(f"Config missing required section: '{section}'")
            
            if not isinstance(config[section], dict):
                raise ValueError(
                    f"Config['{section}'] must be a dictionary, "
                    f"got {type(config[section])}"
                )
            
            logger.debug(f"✓ Config section '{section}' is valid")
    
    @staticmethod
    def get_param(
        config: Dict[str, Any],
        section: str,
        subsection: str,
        param: str,
        default: Any
    ) -> Any:
        """
        Safely get a parameter from nested config structure.
        
        Args:
            config: Configuration dictionary
            section: Top-level section (e.g., 'white_box_attacks')
            subsection: Subsection (e.g., 'fgsm')
            param: Parameter name (e.g., 'epsilon')
            default: Default value if parameter not found
            
        Returns:
            Parameter value or default
        """
        try:
            return config.get(section, {}).get(subsection, {}).get(param, default)
        except (AttributeError, TypeError) as e:
            logger.warning(
                f"Error retrieving {section}.{subsection}.{param}: {e}, "
                f"using default: {default}"
            )
            return default
    
    @staticmethod
    def merge_with_defaults(config: Optional[Dict] = None) -> Dict:
        """
        Merge provided config with defaults.
        
        Args:
            config: User-provided configuration or None
            
        Returns:
            Merged configuration with defaults
        """
        result = ConfigValidator.DEFAULT_CONFIG.copy()
        
        if config:
            try:
                ConfigValidator.validate(config)
                # Deep merge
                for section, values in config.items():
                    if section in result:
                        result[section].update(values)
                    else:
                        result[section] = values
            except ValueError as e:
                logger.warning(f"Invalid config provided: {e}, using defaults")
        
        return result
