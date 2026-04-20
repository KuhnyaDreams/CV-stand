import numpy as np
from typing import  Optional
from config import get_config

class WhiteBoxAttacks():
    def fgsm_attack(self, image: np.ndarray, epsilon: Optional[float] = None) -> np.ndarray:
        if epsilon is None:
            config = get_config()
            epsilon = config.get('white_box_attacks', {}).get('fgsm', {}).get('epsilon', 0.03)
        image_normalized = image.astype(np.float32) / 255.0
        noise = np.random.randn(*image.shape) * (epsilon * 255)
        adversarial = image_normalized + noise / 255.0
        adversarial = np.clip(adversarial, 0, 1)
        return (adversarial * 255).astype(np.uint8)
    
    def pgd_attack(self, image: np.ndarray, epsilon: Optional[float] = None, num_steps: Optional[int] = None) -> np.ndarray:
        if epsilon is None or num_steps is None:
            config = get_config()
            pgd_config = config.get('white_box_attacks', {}).get('pgd', {})
            epsilon = epsilon if epsilon is not None else pgd_config.get('epsilon', 0.03)
            num_steps = num_steps if num_steps is not None else pgd_config.get('num_steps', 7)
        image_normalized = image.astype(np.float32) / 255.0
        adversarial = image_normalized.copy()
        for step in range(num_steps):
            noise = np.random.randn(*image.shape) * (epsilon / num_steps * 255)
            adversarial_normalized = adversarial + noise / 255.0
            perturbation = adversarial_normalized - image_normalized
            perturbation = np.clip(perturbation, -epsilon, epsilon)
            adversarial = image_normalized + perturbation
            adversarial = np.clip(adversarial, 0, 1)
        return (adversarial * 255).astype(np.uint8)
    
    def deepfool_attack(self, image: np.ndarray, num_classes: Optional[int] = None) -> np.ndarray:
        if num_classes is None:
            config = get_config()
            num_classes = config.get('white_box_attacks', {}).get('deepfool', {}).get('num_classes', 80)
        image_normalized = image.astype(np.float32) / 255.0
        perturbation = np.random.randn(*image.shape) * 0.02
        adversarial = np.clip(image_normalized + perturbation, 0, 1)
        return (adversarial * 255).astype(np.uint8)
    
    def jsma_attack(self, image: np.ndarray, theta: Optional[float] = None, gamma: Optional[float] = None) -> np.ndarray:
        if theta is None or gamma is None:
            config = get_config()
            jsma_config = config.get('white_box_attacks', {}).get('jsma', {})
            theta = theta if theta is not None else jsma_config.get('theta', 1.0)
            gamma = gamma if gamma is not None else jsma_config.get('gamma', 0.1)
        image_normalized = image.astype(np.float32) / 255.0
        h, w = image.shape[:2]
        saliency = np.random.rand(h, w) * gamma
        for _ in range(int(h * w * 0.01)):
            y, x = np.unravel_index(np.argmax(saliency), saliency.shape)
            image_normalized[y, x] = np.clip(image_normalized[y, x] + theta / 255.0, 0, 1)
        return (image_normalized * 255).astype(np.uint8)