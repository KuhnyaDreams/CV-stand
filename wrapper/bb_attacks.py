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
    
    def _resolve_regions(
        self,
        image_shape: Tuple[int, int],
        attack_coordinates: Optional[Union[List[Tuple[int, int]], List[Dict[str, Any]]]] = None,
        detection_result: Optional[Dict[str, Any]] = None,
        target_class: Optional[Union[str, List[str]]] = None,
        strategy: str = "center",
        points_per_bbox: int = 1,
        region_size: Optional[int] = None,
        default_size: int = 32
    ) -> List[Dict[str, Any]]:
        """
        Normalize various coordinate inputs into a list of region dicts.

        Each region dict will contain either 'bbox':[x1,y1,x2,y2] or 'x','y','size' and optional 'from_center'.
        """
        h, w = image_shape
        regions: List[Dict[str, Any]] = []

        if attack_coordinates:
            for item in attack_coordinates:
                if isinstance(item, tuple) and len(item) >= 2:
                    x, y = item[0], item[1]
                    size = int(region_size or default_size)
                    regions.append({"x": int(x), "y": int(y), "size": size, "from_center": False})
                elif isinstance(item, dict):
                    if "bbox" in item and item["bbox"]:
                        x1, y1, x2, y2 = item["bbox"]
                        regions.append({"bbox": [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]})
                    else:
                        x = item.get("x", item.get("left", 0))
                        y = item.get("y", item.get("top", 0))
                        size = int(item.get("size", region_size or default_size))
                        if "bbox" in item or "class" in item or "confidence" in item:
                            regions.append({"x": int(x), "y": int(y), "size": size, "from_center": True})
                        else:
                            regions.append({"x": int(x), "y": int(y), "size": size, "from_center": False})

        elif detection_result is not None and target_class is not None:
            patch_size_mode = "fixed" if region_size is not None else "ratio"
            patch_size_value = region_size if region_size is not None else 0.15
            coords_info = extract_attack_coordinates(
                detection_result=detection_result,
                strategy=strategy,
                points_per_bbox=points_per_bbox,
                target_class=target_class,
                return_patch_info=True,
                patch_size_mode=patch_size_mode,
                patch_size_value=patch_size_value
            )
            for info in coords_info:
                if "bbox" in info and info.get("bbox"):
                    regions.append({"bbox": [int(round(v)) for v in info["bbox"]]})
                else:
                    size = int(info.get("size", region_size or default_size))
                    regions.append({"x": int(info["x"]), "y": int(info["y"]), "size": size, "from_center": True})

        return regions

    def _apply_op_to_regions(
        self,
        image: np.ndarray,
        regions: List[Dict[str, Any]],
        op_func,
        default_whole_func
    ) -> np.ndarray:
        """
        Apply operation either to whole image (if no regions) or to each region.

        - op_func: function(crop: np.ndarray) -> processed_crop
        - default_whole_func: function(image) -> processed_image
        """
        if not regions:
            return default_whole_func(image)

        adversarial = image.copy()
        h, w = image.shape[:2]

        for reg in regions:
            if "bbox" in reg:
                x1, y1, x2, y2 = reg["bbox"]
                x_start = max(0, min(int(x1), w - 1))
                y_start = max(0, min(int(y1), h - 1))
                x_end = min(int(x2), w)
                y_end = min(int(y2), h)
            else:
                x = int(reg.get("x", 0))
                y = int(reg.get("y", 0))
                size = max(1, int(reg.get("size", 32)))
                if reg.get("from_center", False):
                    x_start = int(round(x - size // 2))
                    y_start = int(round(y - size // 2))
                else:
                    x_start = x
                    y_start = y
                x_start = max(0, min(x_start, w - 1))
                y_start = max(0, min(y_start, h - 1))
                x_end = min(x_start + size, w)
                y_end = min(y_start + size, h)

            if x_end <= x_start or y_end <= y_start:
                continue

            crop = adversarial[y_start:y_end, x_start:x_end].copy()
            processed = op_func(crop)

            if processed is None:
                continue

            if processed.shape[:2] != crop.shape[:2]:
                processed = cv2.resize(processed, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_LINEAR)

            adversarial[y_start:y_end, x_start:x_end] = processed

        return adversarial

    def single_pixel_attack(
        self,
        image: np.ndarray,
        num_modifications: Optional[int] = None,
        pixel_coordinates: Optional[List[Tuple[int, int]]] = None,
        detection_result: Optional[Dict[str, Any]] = None,
        target_class: Optional[Union[str, List[str]]] = None,
        strategy: str = "center"
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
                "black_box_attacks",
                "single_pixel",
                "num_modifications",
                1,
            )

        # If detection_result + target_class provided, extract pixel coords
        if pixel_coordinates is None and detection_result is not None and target_class is not None:
            coords = extract_attack_coordinates(
                detection_result=detection_result,
                strategy=strategy,
                points_per_bbox=num_modifications,
                target_class=target_class,
                return_patch_info=False,
            )
            if coords:
                pixel_coordinates = coords[:num_modifications]

        self.log_attack("single_pixel_attack", num_modifications=num_modifications)
        adversarial = image.copy()
        h, w = image.shape[:2]

        for i in range(num_modifications):
            if pixel_coordinates is not None and i < len(pixel_coordinates):
                x, y = pixel_coordinates[i]
                if 0 <= x < w and 0 <= y < h:
                    adversarial[y, x] = np.random.randint(0, 256, 3, dtype=np.uint8)
            else:
                y = np.random.randint(0, h)
                x = np.random.randint(0, w)
                adversarial[y, x] = np.random.randint(0, 256, 3, dtype=np.uint8)

        return adversarial
    
    def blackout_attack(
        self,
        image: np.ndarray,
        attack_coordinates: Optional[Union[List[Tuple[int, int]], List[Dict[str, Any]]]] = None,
        detection_result: Optional[Dict[str, Any]] = None,
        target_class: Optional[Union[str, List[str]]] = None,
        strategy: str = "center",
        points_per_bbox: int = 1,
        region_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Fill entire image with black color.
        
        Returns:
            Completely black adversarial image
        """
        self.validate_image(image)
        regions = self._resolve_regions(image.shape[:2], attack_coordinates, detection_result, target_class, strategy, points_per_bbox, region_size)
        self.log_attack("blackout_attack", regions=len(regions) if regions else "all")

        return self._apply_op_to_regions(
            image,
            regions,
            lambda crop: np.zeros_like(crop, dtype=np.uint8),
            lambda img: np.zeros_like(img, dtype=np.uint8),
        )
    
    def random_noise_attack(
        self,
        image: np.ndarray,
        noise_level: Optional[float] = None,
        attack_coordinates: Optional[Union[List[Tuple[int, int]], List[Dict[str, Any]]]] = None,
        detection_result: Optional[Dict[str, Any]] = None,
        target_class: Optional[Union[str, List[str]]] = None,
        strategy: str = "center",
        points_per_bbox: int = 1,
        region_size: Optional[int] = None,
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
            noise_level = self.get_config_param("black_box_attacks", "random_noise", "noise_level", 0.1)

        regions = self._resolve_regions(image.shape[:2], attack_coordinates, detection_result, target_class, strategy, points_per_bbox, region_size)
        self.log_attack("random_noise_attack", noise_level=noise_level, regions=len(regions) if regions else "all")

        def op(crop: np.ndarray) -> np.ndarray:
            noise = np.random.normal(0, noise_level * 255, crop.shape)
            return np.clip(crop.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return self._apply_op_to_regions(image, regions, op, lambda img: op(img))
    
    def gaussian_blur_attack(
        self,
        image: np.ndarray,
        kernel_size: Optional[int] = None,
        attack_coordinates: Optional[Union[List[Tuple[int, int]], List[Dict[str, Any]]]] = None,
        detection_result: Optional[Dict[str, Any]] = None,
        target_class: Optional[Union[str, List[str]]] = None,
        strategy: str = "center",
        points_per_bbox: int = 1,
        region_size: Optional[int] = None,
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
            kernel_size = self.get_config_param("black_box_attacks", "gaussian_blur", "kernel_size", 5)

        # Ensure odd kernel size
        if kernel_size % 2 == 0:
            kernel_size += 1

        regions = self._resolve_regions(image.shape[:2], attack_coordinates, detection_result, target_class, strategy, points_per_bbox, region_size)
        self.log_attack("gaussian_blur_attack", kernel_size=kernel_size, regions=len(regions) if regions else "all")

        def op(crop: np.ndarray) -> np.ndarray:
            k = kernel_size
            if k <= 1:
                return crop
            kx = min(k, max(1, crop.shape[0] // 2 * 2 + 1))
            ky = min(k, max(1, crop.shape[1] // 2 * 2 + 1))
            if kx % 2 == 0:
                kx += 1
            if ky % 2 == 0:
                ky += 1
            kx = max(1, kx)
            ky = max(1, ky)
            return cv2.GaussianBlur(crop, (ky, kx), 0)

        return self._apply_op_to_regions(image, regions, op, lambda img: op(img))
    
    def patch_attack(
        self,
        image: np.ndarray,
        patch_size: Optional[Union[int, str]] = None,
        patch_color: Optional[Tuple[int, int, int]] = None,
        patch_coordinates: Optional[Union[Tuple[int, int], List[Dict[str, Any]]]] = None,
        detection_result: Optional[Dict[str, Any]] = None,
        target_class: Optional[Union[str, List[str]]] = None,
        patch_strategy: str = "center",
        patch_size_ratio: float = 0.15,
        return_metadata: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Apply adversarial patch with optional object targeting via YOLO detection.
        Patch is now properly CENTERED on the target object.
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
        
        patch_placements = []
        
        # 🎯 Обработка координат и размеров
        if patch_coordinates is not None and isinstance(patch_coordinates, list) and len(patch_coordinates) > 0:
            for item in patch_coordinates:
                if isinstance(item, dict) and "x" in item and "y" in item:
                    size = item.get("size", patch_size if isinstance(patch_size, int) else 32)
                    patch_placements.append({
                        "x": item["x"], "y": item["y"], "size": size,
                        "is_center": item.get("is_center", False)  # 🔥 новый флаг
                    })
        
        elif patch_coordinates is not None and isinstance(patch_coordinates, tuple):
            x, y = patch_coordinates
            size = patch_size if isinstance(patch_size, int) else 32
            patch_placements.append({"x": x, "y": y, "size": size, "is_center": False})
        
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
                # 🔥 Гарантируем, что координаты интерпретируются как ЦЕНТР объекта
                patch_placements.append({
                    "x": info["x"],
                    "y": info["y"], 
                    "size": info["size"],
                    "is_center": True  # 🔥 Ключевое исправление
                })
        
        else:
            # 🎲 Случайное размещение
            size = patch_size if isinstance(patch_size, int) else 32
            y_start = np.random.randint(0, max(1, h - size))
            x_start = np.random.randint(0, max(1, w - size))
            patch_placements.append({"x": x_start, "y": y_start, "size": size, "is_center": False})
        
        # 🖌️ Применение патчей
        adversarial = image.copy()
        applied_patches = []
        
        for placement in patch_placements:
            x, y, size = placement["x"], placement["y"], placement["size"]
            is_center = placement.get("is_center", False)
            
            # 🔥 Центрируем патч, если координаты указывают на центр объекта
            if is_center:
                x_start = int(round(x - size / 2))
                y_start = int(round(y - size / 2))
            else:
                x_start = int(x)
                y_start = int(y)

            # Корректировка границ с учётом возможного отрицательного смещения
            x_start = max(0, min(x_start, w - 1))
            y_start = max(0, min(y_start, h - 1))
            x_end = min(x_start + size, w)
            y_end = min(y_start + size, h)
            
            # Применение цвета
            adversarial[y_start:y_end, x_start:x_end] = patch_color
            
            applied_patches.append({
                "top_left": (x_start, y_start),
                "bottom_right": (x_end, y_end),
                "size": size,
                "color": patch_color,
                "centered": is_center
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
        factor: Optional[float] = None,
        attack_coordinates: Optional[Union[List[Tuple[int, int]], List[Dict[str, Any]]]] = None,
        detection_result: Optional[Dict[str, Any]] = None,
        target_class: Optional[Union[str, List[str]]] = None,
        strategy: str = "center",
        points_per_bbox: int = 1,
        region_size: Optional[int] = None,
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
            factor = self.get_config_param("black_box_attacks", "brightness", "factor", 0.5)

        regions = self._resolve_regions(image.shape[:2], attack_coordinates, detection_result, target_class, strategy, points_per_bbox, region_size)
        self.log_attack("brightness_attack", factor=factor, regions=len(regions) if regions else "all")

        def op(crop: np.ndarray) -> np.ndarray:
            return np.clip(crop.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        return self._apply_op_to_regions(image, regions, op, lambda img: op(img))
    
    def contrast_attack(
        self,
        image: np.ndarray,
        factor: Optional[float] = None,
        attack_coordinates: Optional[Union[List[Tuple[int, int]], List[Dict[str, Any]]]] = None,
        detection_result: Optional[Dict[str, Any]] = None,
        target_class: Optional[Union[str, List[str]]] = None,
        strategy: str = "center",
        points_per_bbox: int = 1,
        region_size: Optional[int] = None,
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
            factor = self.get_config_param("black_box_attacks", "contrast", "factor", 0.5)

        regions = self._resolve_regions(image.shape[:2], attack_coordinates, detection_result, target_class, strategy, points_per_bbox, region_size)
        self.log_attack("contrast_attack", factor=factor, regions=len(regions) if regions else "all")

        def op(crop: np.ndarray) -> np.ndarray:
            mean = np.mean(crop, axis=(0, 1))
            return np.clip((crop - mean) * factor + mean, 0, 255).astype(np.uint8)

        return self._apply_op_to_regions(image, regions, op, lambda img: op(img))
    
    def rotation_attack(
        self,
        image: np.ndarray,
        angle: Optional[float] = None,
        attack_coordinates: Optional[Union[List[Tuple[int, int]], List[Dict[str, Any]]]] = None,
        detection_result: Optional[Dict[str, Any]] = None,
        target_class: Optional[Union[str, List[str]]] = None,
        strategy: str = "center",
        points_per_bbox: int = 1,
        region_size: Optional[int] = None,
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
            angle = self.get_config_param("black_box_attacks", "rotation", "angle", 15.0)

        regions = self._resolve_regions(image.shape[:2], attack_coordinates, detection_result, target_class, strategy, points_per_bbox, region_size)
        self.log_attack("rotation_attack", angle=angle, regions=len(regions) if regions else "all")

        def op(crop: np.ndarray) -> np.ndarray:
            h_c, w_c = crop.shape[:2]
            center = (w_c // 2, h_c // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            return cv2.warpAffine(crop, matrix, (w_c, h_c), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        # Whole-image rotation preserves original behavior
        return self._apply_op_to_regions(image, regions, op, lambda img: op(img) if regions else (
            cv2.warpAffine(image, cv2.getRotationMatrix2D((image.shape[1] // 2, image.shape[0] // 2), angle, 1.0), (image.shape[1], image.shape[0]))
        ))
    
    def perspective_transform_attack(
        self,
        image: np.ndarray,
        strength: Optional[float] = None,
        attack_coordinates: Optional[Union[List[Tuple[int, int]], List[Dict[str, Any]]]] = None,
        detection_result: Optional[Dict[str, Any]] = None,
        target_class: Optional[Union[str, List[str]]] = None,
        strategy: str = "center",
        points_per_bbox: int = 1,
        region_size: Optional[int] = None,
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
            strength = self.get_config_param("black_box_attacks", "perspective", "strength", 20)

        regions = self._resolve_regions(image.shape[:2], attack_coordinates, detection_result, target_class, strategy, points_per_bbox, region_size)
        self.log_attack("perspective_transform_attack", strength=strength, regions=len(regions) if regions else "all")

        def op(crop: np.ndarray) -> np.ndarray:
            h_c, w_c = crop.shape[:2]
            pts1 = np.float32([[0, 0], [w_c, 0], [0, h_c], [w_c, h_c]])
            def rand_off():
                return np.random.randint(-int(strength), int(strength) + 1)

            pts2 = np.float32([
                [max(0, min(w_c, 0 + rand_off())), max(0, min(h_c, 0 + rand_off()))],
                [max(0, min(w_c, w_c + rand_off())), max(0, min(h_c, 0 + rand_off()))],
                [max(0, min(w_c, 0 + rand_off())), max(0, min(h_c, h_c + rand_off()))],
                [max(0, min(w_c, w_c + rand_off())), max(0, min(h_c, h_c + rand_off()))],
            ])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            return cv2.warpPerspective(crop, matrix, (w_c, h_c), borderMode=cv2.BORDER_REFLECT)

        def whole(img: np.ndarray) -> np.ndarray:
            h_i, w_i = img.shape[:2]
            pts1 = np.float32([[0, 0], [w_i, 0], [0, h_i], [w_i, h_i]])
            pts2 = np.float32([
                [np.random.randint(0, int(strength)), np.random.randint(0, int(strength))],
                [w_i - np.random.randint(0, int(strength)), np.random.randint(0, int(strength))],
                [np.random.randint(0, int(strength)), h_i - np.random.randint(0, int(strength))],
                [w_i - np.random.randint(0, int(strength)), h_i - np.random.randint(0, int(strength))],
            ])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            return cv2.warpPerspective(img, matrix, (w_i, h_i))

        return self._apply_op_to_regions(image, regions, op, whole)