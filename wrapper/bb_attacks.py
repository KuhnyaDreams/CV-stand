import numpy as np
from typing import Any, Dict, Tuple, Optional, List, Union
import cv2
import logging
from coords_extractor import extract_attack_coordinates
from base_attacks import AttackBase
from config_validator import ConfigValidator

logger = logging.getLogger(__name__)


class BlackBoxAttacks(AttackBase):
    """Black-box adversarial attack implementations."""
    
    def single_pixel_attack(
        self,
        image: np.ndarray,
        num_modifications: Optional[int] = None,
        pixel_coordinates: Optional[List[Tuple[int, int]]] = None
    ) -> np.ndarray:
        """
        Modify random pixels to create adversarial example.
        
        Args:
            image: Input image array
            num_modifications: Number of pixels to modify
            pixel_coordinates: Optional list of specific pixel coordinates to modify
            
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
        for i in range(num_modifications):
            # If specific pixel coordinates were provided, use them one-by-one.
            if pixel_coordinates is not None and i < len(pixel_coordinates):
                x, y = pixel_coordinates[i]
                if 0 <= x < w and 0 <= y < h:
                    adversarial[y, x] = np.random.randint(0, 256, 3)
            else:
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
        patch_size: Optional[Union[int, str]] = None,      # int, "auto", или None
        patch_color: Optional[Tuple[int, int, int]] = None,
        patch_coordinates: Optional[Union[Tuple[int, int], List[Dict[str, Any]]]] = None,
        detection_result: Optional[Dict[str, Any]] = None,  # 🔍 YOLO-результаты
        target_class: Optional[Union[str, List[str]]] = None, # 🔍 Класс для атаки
        patch_strategy: str = "center",                    # стратегия выбора координат
        patch_size_ratio: float = 0.15,                    # для patch_size="auto"
        return_metadata: bool = False                      # вернуть инфо о применённом патче
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Apply adversarial patch with optional object targeting via YOLO detection.
        
        Args:
            image: Input image [H, W, C]
            patch_size: 
                - int: фиксированный размер в пикселях
                - "auto": размер вычисляется от bbox (если передан detection_result)
                - None: берётся из конфига
            patch_color: RGB цвет, напр. (255, 0, 0)
            patch_coordinates:
                - Tuple[int, int]: явные координаты (x, y)
                - List[Dict]: результат extract_attack_coordinates(return_patch_info=True)
            detection_result: JSON с детекцией YOLO (опционально)
            target_class: Фильтр по классу для атаки (напр. "cell phone")
            patch_strategy: "center", "random", "corners", "grid" — как выбирать точку внутри bbox
            patch_size_ratio: Доля от bbox для auto-размера
            return_metadata: Если True — возвращает (image, metadata)
            
        Returns:
            np.ndarray или Tuple[np.ndarray, Dict] с метаданными патча
        """
        self.validate_image(image)
        h, w = image.shape[:2]
        
        # 🎨 Загрузка параметров по умолчанию
        if patch_size is None or patch_color is None:
            patch_config = self.config.get('black_box_attacks', {}).get('patch', {})
            if not isinstance(patch_config, dict):
                patch_config = {}
            
            patch_size = patch_size or patch_config.get('patch_size', 32)
            patch_color_list = patch_config.get('patch_color', [255, 0, 0])
            patch_color = tuple(patch_color_list) if patch_color is None else patch_color
        
        # 🎯 Обработка координат и размеров
        patch_placements = []  # список {x, y, size}
        
        if patch_coordinates is not None and isinstance(patch_coordinates, list) and len(patch_coordinates) > 0:
            # 📥 Получили список из extract_attack_coordinates(return_patch_info=True)
            for item in patch_coordinates:
                if isinstance(item, dict) and "x" in item and "y" in item:
                    size = item.get("size", patch_size if isinstance(patch_size, int) else 32)
                    patch_placements.append({
                        "x": item["x"],
                        "y": item["y"],
                        "size": size
                    })
        
        elif patch_coordinates is not None and isinstance(patch_coordinates, tuple):
            # 📍 Явные координаты
            x, y = patch_coordinates
            size = patch_size if isinstance(patch_size, int) else 32
            patch_placements.append({"x": x, "y": y, "size": size})
        
        elif detection_result is not None and target_class is not None:
            # 🔍 Авто-извлечение из детекции
            coords_info = extract_attack_coordinates(
                detection_result=detection_result,
                strategy=patch_strategy,
                target_class=target_class,
                return_patch_info=True,
                patch_size_mode="ratio" if patch_size == "auto" else "fixed",
                patch_size_value=patch_size_ratio if patch_size == "auto" else (patch_size if isinstance(patch_size, int) else 32)
            )
            for info in coords_info:
                patch_placements.append({
                    "x": info["x"],
                    "y": info["y"],
                    "size": info["size"]
                })
        
        else:
            # 🎲 Случайное размещение (старое поведение)
            size = patch_size if isinstance(patch_size, int) else 32
            y_start = np.random.randint(0, max(1, h - size))
            x_start = np.random.randint(0, max(1, w - size))
            patch_placements.append({"x": x_start, "y": y_start, "size": size})
        
        # 🖌️ Применение патчей
        adversarial = image.copy()
        applied_patches = []
        
        for placement in patch_placements:
            x, y, size = placement["x"], placement["y"], placement["size"]
            
            # Корректировка границ
            x_start = max(0, min(x, w - 1))
            y_start = max(0, min(y, h - 1))
            x_end = min(x_start + size, w)
            y_end = min(y_start + size, h)
            
            # Применение цвета
            adversarial[y_start:y_end, x_start:x_end] = patch_color
            
            applied_patches.append({
                "top_left": (x_start, y_start),
                "bottom_right": (x_end, y_end),
                "size": size,
                "color": patch_color
            })
        
        self.log_attack('patch_attack', 
                    patches_applied=len(applied_patches),
                    patch_color=patch_color,
                    target_class=target_class)
        
        if return_metadata:
            return adversarial, {"patches": applied_patches, "image_shape": (h, w)}
        
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