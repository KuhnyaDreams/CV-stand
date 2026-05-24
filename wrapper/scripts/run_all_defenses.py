from api.model_functions import detect
from attacks.bb.bb_attacks import BlackBoxAttacks
from defenses.adaptive_defense import AdaptiveDefense
from defenses.attack_classifier import AttackClassifier
from utils.path_utils import PathManager

import argparse
import cv2
import os
import time


attacker = BlackBoxAttacks()
defender = AdaptiveDefense()


ATTACK_MAP = {
    "patch": "patch_attack",
    "single_pixel": "single_pixel_attack",
    "noise": "random_noise_attack",
    "blur": "gaussian_blur_attack",
    "brightness": "brightness_attack",
    "contrast": "contrast_attack",
    "rotation": "rotation_attack",
    "perspective": "perspective_transform_attack",
    "blackout": "blackout_attack",
}


def run_attack_and_defense(image_path: str, attack_name: str = "patch"):
    print("=" * 60)
    print("ADVERSARIAL DEFENSE TEST")
    print("=" * 60)


    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    print(f"[1] Loaded image: {image_path}")

    attack_method_name = ATTACK_MAP.get(attack_name)

    if attack_method_name is None:
        raise ValueError(f"Unknown attack: {attack_name}")

    attack_method = getattr(attacker, attack_method_name)
    attacked_image = attack_method(image)

    print(f"[2] Applied attack: {attack_name}")


    timestamp = time.strftime("%Y%m%d_%H%M%S")

    attacked_filename = f"attacked_{timestamp}.png"
    attacked_path = PathManager.get_temp_image_path(attacked_filename)

    cv2.imwrite(attacked_path, attacked_image)

    detected_attack = AttackClassifier.classify(attacked_image)

    print(f"[3] Detected attack type: {detected_attack}")


    defended_image = defender.apply_with_type(
        attacked_image,
        detected_attack
    )

    print(f"[4] Applied defense for: {detected_attack}")


    defended_filename = f"defended_{timestamp}.png"
    defended_path = PathManager.get_temp_image_path(defended_filename)

    cv2.imwrite(defended_path, defended_image)


    print("[5] Running YOLO detection...")

    report = detect(os.path.basename(defended_path))

    print("[6] Done.")
    print("=" * 60)

    return report



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply one attack, classify it, run the matching defense, and evaluate the result.",
    )
    parser.add_argument("image", help="Path to the input image.")
    parser.add_argument(
        "--attack",
        default="perspective",
        choices=sorted(ATTACK_MAP.keys()),
        help="Attack family to apply before the defense step.",
    )
    args = parser.parse_args()

    run_attack_and_defense(args.image, attack_name=args.attack)


if __name__ == "__main__":
    main()
