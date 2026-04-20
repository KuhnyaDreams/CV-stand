import numpy as np
from typing import Tuple, Optional
import cv2
import logging

from base_attacks import AttackBase
from config_validator import ConfigValidator

logger = logging.getLogger(__name__)


class BlackBoxAttacks(AttackBase):
    """Black-box adversarial attack implementations."""
    
    def single_pixel_attack(
        self,
        image: np.ndarray,
        num_modifications: Optional[int] = None
    ) -> np.ndarray:
        """
        Modify random pixels to create adversarial example.
        
        Args:
            image: Input image array
            num_modifications: Number of pixels to modify
            
        Returns:
            Adversarial image
        """
        self.validate_image(image)
        
        if num_modifications is None:
            num_modifications = self.get_config_param(
                'black_box_attacks',
                'single_pixel',
                'num_modifications',
                1
            )
        
        self.log_attack('single_pixel_attack', num_modifications=num_modifications)
        adversarial = image.copy()
        h, w = image.shape[:2]
        
        for _ in range(num_modifications):
            y = np.random.randint(0, h)
            x = np.random.randint(0, w)
            adversarial[y, x] = np.random.randint(0, 256, 3)
        
        return adversarial
    
    def blackout_attack(self, image: np.ndarray) -> np.ndarray:
        """
        Fill entire image with black color.
        
        Returns:
            Completely black adversarial image
        """
        self.validate_image(image)
        self.log_attack('blackout_attack')
        return np.zeros_like(image, dtype=np.uint8)
    
    def random_noise_attack(
        self,
        image: np.ndarray,
        noise_level: Optional[float] = None
    ) -> np.ndarray:
        """
        Add random Gaussian noise to image.
        
        Args:
            image: Input image array
            noise_level: Standard deviation of noise
            
        Returns:
            Adversarial image with noise
        """
        self.validate_image(image)
        
        if noise_level is None:
            noise_level = self.get_config_param(
                'black_box_attacks',
                'random_noise',
                'noise_level',
                0.1
            )
        
        self.log_attack('random_noise_attack', noise_level=noise_level)
        noise = np.random.normal(0, noise_level * 255, image.shape)
        adversarial = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return adversarial
    
    def gaussian_blur_attack(
        self,
        image: np.ndarray,
        kernel_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Apply Gaussian blur to image.
        
        Args:
            image: Input image array
            kernel_size: Size of blur kernel
            
        Returns:
            Blurred adversarial image
        """
        self.validate_image(image)
        
        if kernel_size is None:
            kernel_size = self.get_config_param(
                'black_box_attacks',
                'gaussian_blur',
                'kernel_size',
                5
            )
        
        # Ensure odd kernel size
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        self.log_attack('gaussian_blur_attack', kernel_size=kernel_size)
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    def patch_attack(
        self,
        image: np.ndarray,
        patch_size: Optional[int] = None,
        patch_color: Optional[Tuple] = None
    ) -> np.ndarray:
        """
        Apply colored patch to random location.
        
        Args:
            image: Input image array
            patch_size: Size of patch
            patch_color: RGB color tuple
            
        Returns:
            Image with adversarial patch
        """
        self.validate_image(image)
        
        if patch_size is None or patch_color is None:
            patch_config = self.get_config_param(
                'black_box_attacks',
                'patch',
                None,
                {}
            ) if isinstance(self.config.get('black_box_attacks', {}).get('patch', {}), dict) else {}
            
            patch_size = (
                patch_size or
                self.get_config_param('black_box_attacks', 'patch', 'patch_size', 32)
            )
            patch_color_list = self.get_config_param(
                'black_box_attacks',
                'patch',
                'patch_color',
                [255, 0, 0]
            )
            patch_color = tuple(patch_color_list) if patch_color is None else patch_color
        
        self.log_attack('patch_attack', patch_size=patch_size, patch_color=patch_color)
        adversarial = image.copy()
        h, w = image.shape[:2]
        
        y = np.random.randint(0, max(1, h - patch_size))
        x = np.random.randint(0, max(1, w - patch_size))
        y_end = min(y + patch_size, h)
        x_end = min(x + patch_size, w)
        
        adversarial[y:y_end, x:x_end] = patch_color
        return adversarial
    
    def brightness_attack(
        self,
        image: np.ndarray,
        factor: Optional[float] = None
    ) -> np.ndarray:
        """
        Modify image brightness.
        
        Args:
            image: Input image array
            factor: Brightness multiplier
            
        Returns:
            Adversarial image with modified brightness
        """
        self.validate_image(image)
        
        if factor is None:
            factor = self.get_config_param(
                'black_box_attacks',
                'brightness',
                'factor',
                0.5
            )
        
        self.log_attack('brightness_attack', factor=factor)
        adversarial = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        return adversarial
    
    def contrast_attack(
        self,
        image: np.ndarray,
        factor: Optional[float] = None
    ) -> np.ndarray:
        """
        Modify image contrast.
        
        Args:
            image: Input image array
            factor: Contrast multiplier
            
        Returns:
            Adversarial image with modified contrast
        """
        self.validate_image(image)
        
        if factor is None:
            factor = self.get_config_param(
                'black_box_attacks',
                'contrast',
                'factor',
                0.5
            )
        
        self.log_attack('contrast_attack', factor=factor)
        mean = np.mean(image, axis=(0, 1))
        adversarial = np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)
        return adversarial
    
    def rotation_attack(
        self,
        image: np.ndarray,
        angle: Optional[float] = None
    ) -> np.ndarray:
        """
        Rotate image by specified angle.
        
        Args:
            image: Input image array
            angle: Rotation angle in degrees
            
        Returns:
            Rotated adversarial image
        """
        self.validate_image(image)
        
        if angle is None:
            angle = self.get_config_param(
                'black_box_attacks',
                'rotation',
                'angle',
                15.0
            )
        
        self.log_attack('rotation_attack', angle=angle)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        adversarial = cv2.warpAffine(image, matrix, (w, h))
        return adversarial
    
    def perspective_transform_attack(
        self,
        image: np.ndarray,
        strength: Optional[float] = None
    ) -> np.ndarray:
        """
        Apply perspective transformation to image.
        
        Args:
            image: Input image array
            strength: Strength of transformation
            
        Returns:
            Perspective-transformed adversarial image
        """
        self.validate_image(image)
        
        if strength is None:
            strength = self.get_config_param(
                'black_box_attacks',
                'perspective',
                'strength',
                20
            )
        
        self.log_attack('perspective_transform_attack', strength=strength)
        h, w = image.shape[:2]
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        pts2 = np.float32([
            [
                np.random.randint(0, int(strength)),
                np.random.randint(0, int(strength))
            ],
            [
                w - np.random.randint(0, int(strength)),
                np.random.randint(0, int(strength))
            ],
            [
                np.random.randint(0, int(strength)),
                h - np.random.randint(0, int(strength))
            ],
            [
                w - np.random.randint(0, int(strength)),
                h - np.random.randint(0, int(strength))
            ]
        ])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        adversarial = cv2.warpPerspective(image, matrix, (w, h))
        return adversarial