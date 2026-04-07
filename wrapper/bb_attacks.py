import numpy as np
from typing import Tuple, Optional
from attacks import AdversarialAttacks, get_config
import cv2

class BlackBoxAttacks(AdversarialAttacks):
    
    def single_pixel_attack(self, image: np.ndarray, num_modifications: Optional[int] = None) -> np.ndarray:
        if num_modifications is None:
            config = get_config()
            num_modifications = config.get('black_box_attacks', {}).get('single_pixel', {}).get('num_modifications', 1)
        adversarial = image.copy()
        h, w = image.shape[:2]
        for _ in range(num_modifications):
            y = np.random.randint(0, h)
            x = np.random.randint(0, w)
            adversarial[y, x] = np.random.randint(0, 256, 3)
        return adversarial
   
    def blackout_attack(self, image: np.ndarray) -> np.ndarray:
        """
        Полностью заливает изображение черным цветом
        """
        return np.zeros_like(image, dtype=np.uint8)

    def random_noise_attack(self, image: np.ndarray, noise_level: Optional[float] = None) -> np.ndarray:
        if noise_level is None:
            config = get_config()
            noise_level = config.get('black_box_attacks', {}).get('random_noise', {}).get('noise_level', 0.1)
        noise = np.random.normal(0, noise_level * 255, image.shape)
        adversarial = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return adversarial
    
    def gaussian_blur_attack(self, image: np.ndarray, kernel_size: Optional[int] = None) -> np.ndarray:
        if kernel_size is None:
            config = get_config()
            kernel_size = config.get('black_box_attacks', {}).get('gaussian_blur', {}).get('kernel_size', 5)
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    def patch_attack(self, image: np.ndarray, patch_size: Optional[int] = None, patch_color: Optional[Tuple] = None) -> np.ndarray:
        if patch_size is None or patch_color is None:
            config = get_config()
            patch_config = config.get('black_box_attacks', {}).get('patch', {})
            patch_size = patch_size if patch_size is not None else patch_config.get('patch_size', 32)
            patch_color_list = patch_config.get('patch_color', [255, 0, 0])
            patch_color = tuple(patch_color_list) if patch_color is None else patch_color
        adversarial = image.copy()
        h, w = image.shape[:2]
        y = np.random.randint(0, max(1, h - patch_size))
        x = np.random.randint(0, max(1, w - patch_size))
        y_end = min(y + patch_size, h)
        x_end = min(x + patch_size, w)
        adversarial[y:y_end, x:x_end] = patch_color
        return adversarial
    
    def brightness_attack(self, image: np.ndarray, factor: Optional[float] = None) -> np.ndarray:
        if factor is None:
            config = get_config()
            factor = config.get('black_box_attacks', {}).get('brightness', {}).get('factor', 0.5)
        adversarial = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        return adversarial
    
    def contrast_attack(self, image: np.ndarray, factor: Optional[float] = None) -> np.ndarray:
        if factor is None:
            config = get_config()
            factor = config.get('black_box_attacks', {}).get('contrast', {}).get('factor', 0.5)
        mean = np.mean(image, axis=(0, 1))
        adversarial = np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)
        return adversarial
    
    def rotation_attack(self, image: np.ndarray, angle: Optional[float] = None) -> np.ndarray:
        if angle is None:
            config = get_config()
            angle = config.get('black_box_attacks', {}).get('rotation', {}).get('angle', 15.0)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        adversarial = cv2.warpAffine(image, matrix, (w, h))
        return adversarial
    
    def perspective_transform_attack(self, image: np.ndarray, strength: Optional[float] = None) -> np.ndarray:
        if strength is None:
            config = get_config()
            strength = config.get('black_box_attacks', {}).get('perspective', {}).get('strength', 20)
        h, w = image.shape[:2]
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        pts2 = np.float32([
            [np.random.randint(0, int(strength)), np.random.randint(0, int(strength))],
            [w - np.random.randint(0, int(strength)), np.random.randint(0, int(strength))],
            [np.random.randint(0, int(strength)), h - np.random.randint(0, int(strength))],
            [w - np.random.randint(0, int(strength)), h - np.random.randint(0, int(strength))]
        ])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        adversarial = cv2.warpPerspective(image, matrix, (w, h))
        return adversarial