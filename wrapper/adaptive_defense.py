import numpy as np
import cv2
from defense import Defenses
from attack_classifier import AttackClassifier  


class AdaptiveDefense:

    def apply_with_type(self, image, attack_type: str):

        if attack_type == "noise":
            return Defenses.denoise(image)

        elif attack_type == "patch":
            return Defenses.jpeg_compression(image, quality=50)

        elif attack_type == "blur":
            return Defenses.normalize_lighting(image)

        elif attack_type == "single_pixel":
            return Defenses.gaussian_blur(image, kernel_size=3)

        else:
            return Defenses.combined(image)

    def apply(self, image):
        attack_type = AttackClassifier.classify(image)  
        print(f"[INFO] Detected attack: {attack_type}")
        return self.apply_with_type(image, attack_type)