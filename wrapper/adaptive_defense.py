"""
Adaptive defense against adversarial attacks based on attack classification.

Module provides AdaptiveDefense class that:
- Determines attack type on an image
- Automatically applies appropriate defense
"""

import numpy as np
import cv2
from defense import Defenses
from attack_classifier import AttackClassifier


class AdaptiveDefense:
    """
    Adaptive defense class against various types of adversarial attacks.

    Uses attack classification to determine optimal defense:
    - noise -> denoising
    - patch -> JPEG compression
    - blur -> lighting normalization
    - single_pixel -> Gaussian blur
    - rotation -> inverse rotation
    - perspective -> perspective correction
    """

    def apply_with_type(self, image, attack_type: str):
        """
        Apply defense based on detected attack type.

        Args:
            image: Input image as numpy array (BGR)
            attack_type: Attack type ('noise', 'patch', 'blur', 'single_pixel',
                                    'rotation', 'perspective', etc.)

        Returns:
            numpy.ndarray: Defended image
        """

        if attack_type == "noise":
            return Defenses.denoise(
                image,
                h=18,
                template_window_size=7,
                search_window_size=21
    )

        elif attack_type == "patch":
            return Defenses.jpeg_compression(image, quality=50)

        elif attack_type == "blur":
            return Defenses.normalize_lighting(image)

        elif attack_type == "single_pixel":
            return Defenses.gaussian_blur(image, kernel_size=3)
        
        elif attack_type == "rotation":
            h, w = image.shape[:2]
            center = (w // 2, h // 2)

            matrix = cv2.getRotationMatrix2D(center, -15, 1.0)
            return cv2.warpAffine(image, matrix, (w, h))


        elif attack_type == "perspective":
            h, w = image.shape[:2]

            src = np.float32([
                [20, 20],
                [w - 20, 20],
                [20, h - 20],
                [w - 20, h - 20]
            ])

            dst = np.float32([
                [0, 0],
                [w, 0],
                [0, h],
                [w, h]
            ])

            matrix = cv2.getPerspectiveTransform(src, dst)

            return cv2.warpPerspective(image, matrix, (w, h))
        
        else:
            return Defenses.combined(image)

    def apply(self, image):
        """
        Automatically detect attack type and apply appropriate defense.

        Args:
            image: Input image

        Returns:
            numpy.ndarray: Defended image
        """
        attack_type = AttackClassifier.classify(image)
        print(f"[INFO] Detected attack: {attack_type}")
        return self.apply_with_type(image, attack_type)