import unittest

import cv2
import numpy as np

from attacks.bb.bb_attacks import BlackBoxAttacks
from defenses.attack_classifier import AttackClassifier


class AttackClassifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.attacker = BlackBoxAttacks()

        image = np.full((256, 256, 3), 180, dtype=np.uint8)
        cv2.rectangle(image, (40, 40), (216, 216), (30, 30, 220), thickness=3)
        cv2.line(image, (40, 216), (216, 40), (0, 0, 0), thickness=4)
        cv2.circle(image, (128, 128), 40, (0, 220, 0), thickness=3)
        cv2.putText(
            image,
            "PHONE",
            (70, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        self.base_image = image

    def test_classifies_blackout(self) -> None:
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        self.assertEqual(AttackClassifier.classify(image), "blackout")

    def test_classifies_random_noise(self) -> None:
        np.random.seed(0)
        attacked = self.attacker.random_noise_attack(self.base_image, noise_level=0.18)
        self.assertEqual(AttackClassifier.classify(attacked), "random_noise")

    def test_classifies_low_light(self) -> None:
        np.random.seed(0)
        attacked = self.attacker.low_light_attack(
            self.base_image,
            brightness_factor=0.35,
            noise_level=0.12,
        )
        self.assertEqual(AttackClassifier.classify(attacked), "low_light")

    def test_classifies_gaussian_blur(self) -> None:
        attacked = self.attacker.gaussian_blur_attack(self.base_image, kernel_size=17)
        self.assertEqual(AttackClassifier.classify(attacked), "gaussian_blur")

    def test_classifies_frame_drop_when_frame_repeats(self) -> None:
        frame = self.base_image.copy()
        self.assertEqual(
            AttackClassifier.classify(frame, prev_frame=frame.copy()),
            "frame_drop",
        )


if __name__ == "__main__":
    unittest.main()
