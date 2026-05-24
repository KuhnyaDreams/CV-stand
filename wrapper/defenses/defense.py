import cv2
import numpy as np
from typing import List, Callable, Optional
import logging

logger = logging.getLogger(__name__)


class Defenses:
    
    DEFAULT_PARAMS = {
        'gaussian_blur': {'kernel_size': 5},
        'denoise': {'h': 10, 'template_window_size': 10, 'search_window_size': 21},
        'jpeg_compression': {'quality': 60},
        'random_resize': {'scale_range': (0.8, 1.2)},
        'normalize_lighting': {},
    }

    @staticmethod
    def _ensure_image(image: np.ndarray, method_name: str = "defense") -> None:
       
        if image is None:
            logger.error(f"Empty image passed to {method_name}()")
            raise ValueError(
                f"Empty image passed to {method_name}(). Check file path and cv2.imread result."
            )

        size = getattr(image, 'size', None)
        if size is None or size == 0:
            logger.error(f"Image with zero size passed to {method_name}()")
            raise ValueError(
                f"Image appears empty in {method_name}(). Check file path and cv2.imread result."
            )
    
    @staticmethod
    def gaussian_blur(
        image: np.ndarray,
        kernel_size: int = 5
    ) -> np.ndarray:
        
        Defenses._ensure_image(image, 'gaussian_blur')

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
       
        Defenses._ensure_image(image, 'denoise')

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
        
        Defenses._ensure_image(image, 'jpeg_compression')

        if not 0 <= quality <= 100:
            logger.warning(f"Quality {quality} out of range [0, 100], clamping")
            quality = max(0, min(100, quality))
        
        if quality == 60: 
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                img_area = gray.size
                if 0.05 < area / img_area < 0.4:
                    peri = cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                    if len(approx) == 4:
                        quality = 40  
                        logger.info(f"Patch detected, using stronger compression: quality={quality}")
                        break
        
        logger.debug(f"Applying JPEG compression: quality={quality}")
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, encimg = cv2.imencode('.jpg', image, encode_param)
        if not success or encimg is None or encimg.size == 0:
            logger.error("JPEG encoding failed - image may be invalid")
            raise ValueError("JPEG compression failed: invalid image or OpenCV error")

        decimg = cv2.imdecode(encimg, 1)
        if decimg is None or getattr(decimg, 'size', 0) == 0:
            logger.error("JPEG decoding failed - result is empty")
            raise ValueError("JPEG compression failed: decoded image is empty")

        return decimg
    
    @staticmethod
    def random_resize(
        image: np.ndarray,
        scale_range: tuple = (0.8, 1.2)
    ) -> np.ndarray:
        
        Defenses._ensure_image(image, 'random_resize')

        h, w = image.shape[:2]
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        is_patch = False
        for contour in contours:
            area = cv2.contourArea(contour)
            img_area = gray.size
            if 0.05 < area / img_area < 0.4:
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                if len(approx) == 4:
                    is_patch = True
                    break
        
        if is_patch:
            scale = np.random.uniform(0.7, 1.3)  # более сильное изменение
            logger.info(f"Patch detected, using stronger resize: scale={scale:.2f}")
        else:
            scale = np.random.uniform(scale_range[0], scale_range[1])
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        logger.debug(f"Applying random resize: scale={scale:.2f}")
        resized = cv2.resize(image, (new_w, new_h))
        return cv2.resize(resized, (w, h))
    
    @staticmethod
    def normalize_lighting(image: np.ndarray) -> np.ndarray:
       
        Defenses._ensure_image(image, 'normalize_lighting')

        logger.debug("Applying lighting normalization")
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            img_area = gray.size
            if 0.05 < area / img_area < 0.4:
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                if len(approx) == 4:
                    # Применяем CLAHE для выравнивания гистограммы
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    if len(image.shape) == 3:
                        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                        lab[:,:,0] = clahe.apply(lab[:,:,0])
                        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                    else:
                        result = clahe.apply(image)
                    logger.info("Patch detected, applying CLAHE normalization")
                    return result
        
        return cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    
    @staticmethod
    def combined(
        image: np.ndarray,
        jpeg_quality: int = 70,
        blur_kernel: int = 3
    ) -> np.ndarray:
       
        Defenses._ensure_image(image, 'combined')

        logger.info("Applying combined defense pipeline")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        is_patch = False
        for contour in contours:
            area = cv2.contourArea(contour)
            img_area = gray.size
            if 0.05 < area / img_area < 0.4:
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                if len(approx) == 4:
                    is_patch = True
                    break
        
        if is_patch:
            jpeg_quality = max(40, jpeg_quality - 20)
            blur_kernel = max(5, blur_kernel + 2)
            logger.info(f"Patch detected, using stronger defense: quality={jpeg_quality}, blur={blur_kernel}")
        
        image = Defenses.jpeg_compression(image, quality=jpeg_quality)
        image = Defenses.gaussian_blur(image, blur_kernel)
        image = Defenses.denoise(image)
        return image