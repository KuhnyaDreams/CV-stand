"""
Defense mechanisms against adversarial attacks.
Provides various image preprocessing and transformation methods for robustness.
"""

import cv2
import numpy as np
from typing import List, Callable, Optional
import logging

logger = logging.getLogger(__name__)


class Defenses:
    """Collection of defense methods against adversarial attacks."""
    
    # Defense hyperparameters
    DEFAULT_PARAMS = {
        'gaussian_blur': {'kernel_size': 5},
        'denoise': {'h': 10, 'template_window_size': 10, 'search_window_size': 21},
        'jpeg_compression': {'quality': 60},
        'random_resize': {'scale_range': (0.8, 1.2)},
        'normalize_lighting': {},
    }
    
    @staticmethod
    def gaussian_blur(
        image: np.ndarray,
        kernel_size: int = 5
    ) -> np.ndarray:
        """
        Gaussian blur smoothing.
        Reduces high-frequency noise (e.g., FGSM, random noise).
        
        Args:
            image: Input image
            kernel_size: Size of blur kernel
            
        Returns:
            Blurred image
        """
        if kernel_size % 2 == 0:
            kernel_size += 1
        logger.debug(f"Applying gaussian blur: kernel_size={kernel_size}")
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    @staticmethod
    def denoise(
        image: np.ndarray,
        h: int = 10,
        template_window_size: int = 10,
        search_window_size: int = 21
    ) -> np.ndarray:
        """
        Non-local means denoising with structure preservation.
        Effective against random noise and some white-box attacks.
        
        Args:
            image: Input image
            h: Filter strength
            template_window_size: Size of template window
            search_window_size: Size of search area
            
        Returns:
            Denoised image
        """
        logger.debug(
            f"Applying denoise: h={h}, "
            f"template_size={template_window_size}, "
            f"search_size={search_window_size}"
        )
        return cv2.fastNlMeansDenoisingColored(
            image,
            None,
            h,
            h,
            template_window_size,
            search_window_size
        )
    
    @staticmethod
    def jpeg_compression(
        image: np.ndarray,
        quality: int = 60
    ) -> np.ndarray:
        """
        JPEG compression with quality loss.
        Removes fine perturbations characteristic of adversarial attacks.
        
        Args:
            image: Input image
            quality: JPEG quality (0-100)
            
        Returns:
            Compressed image
        """
        if not 0 <= quality <= 100:
            logger.warning(f"Quality {quality} out of range [0, 100], clamping")
            quality = max(0, min(100, quality))
        
        logger.debug(f"Applying JPEG compression: quality={quality}")
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encimg = cv2.imencode('.jpg', image, encode_param)
        decimg = cv2.imdecode(encimg, 1)
        return decimg
    
    @staticmethod
    def random_resize(
        image: np.ndarray,
        scale_range: tuple = (0.8, 1.2)
    ) -> np.ndarray:
        """
        Random resizing to break perturbation structure.
        Reduces transferability of attacks by disrupting noise patterns.
        
        Args:
            image: Input image
            scale_range: (min_scale, max_scale) for random scaling
            
        Returns:
            Resized image restored to original size
        """
        h, w = image.shape[:2]
        scale = np.random.uniform(scale_range[0], scale_range[1])
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        logger.debug(f"Applying random resize: scale={scale:.2f}")
        resized = cv2.resize(image, (new_w, new_h))
        return cv2.resize(resized, (w, h))
    
    @staticmethod
    def normalize_lighting(image: np.ndarray) -> np.ndarray:
        """
        Brightness normalization.
        Partially compensates for illumination-changing attacks.
        
        Args:
            image: Input image
            
        Returns:
            Normalized image
        """
        logger.debug("Applying lighting normalization")
        return cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    
    @staticmethod
    def combined(
        image: np.ndarray,
        jpeg_quality: int = 70,
        blur_kernel: int = 3
    ) -> np.ndarray:
        """
        Combined defense pipeline with multiple techniques.
        Applies transformations sequentially for enhanced robustness.
        
        Args:
            image: Input image
            jpeg_quality: JPEG quality parameter
            blur_kernel: Gaussian blur kernel size
            
        Returns:
            Defended image after all transformations
        """
        logger.info("Applying combined defense pipeline")
        image = Defenses.jpeg_compression(image, quality=jpeg_quality)
        image = Defenses.gaussian_blur(image, blur_kernel)
        image = Defenses.denoise(image)
        return image


class DefensePipeline:
    """
    Pipeline for sequential application of defense methods.
    Allows flexible composition of multiple defense techniques.
    """
    
    def __init__(self, defenses: Optional[List[Callable]] = None):
        """
        Initialize defense pipeline.
        
        Args:
            defenses: List of callable defense functions or Defenses instance
        """
        self.defenses = defenses
        logger.debug(f"Initialized DefensePipeline with {type(defenses).__name__}")
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply defense pipeline to image.
        
        Args:
            image: Input image
            
        Returns:
            Defended image
            
        Raises:
            TypeError: If defenses parameter is invalid type
        """
        if self.defenses is None:
            logger.warning("No defenses specified, returning image unchanged")
            return image
        
        # Support Defenses instance with combined method
        if hasattr(self.defenses, 'combined') and callable(
            getattr(self.defenses, 'combined')
        ):
            logger.info("Applying Defenses.combined()")
            return self.defenses.combined(image)
        
        # Support single callable
        if callable(self.defenses):
            logger.info(f"Applying single defense: {self.defenses.__name__}")
            return self.defenses(image)
        
        # Support list/iterable of callables
        try:
            iterator = iter(self.defenses)
            logger.info(f"Applying {len(list(self.defenses))} defenses sequentially")
        except TypeError:
            raise TypeError(
                "defenses must be a callable, an iterable of callables, "
                "or a Defenses instance"
            )
        
        # Re-create iterator and apply sequentially
        iterator = iter(self.defenses)
        for i, defense in enumerate(iterator):
            if not callable(defense):
                raise TypeError(
                    f"Element {i} in defenses is not callable: {type(defense)}"
                )
            logger.debug(f"Applying defense {i}: {defense.__name__}")
            image = defense(image)
        
        return image