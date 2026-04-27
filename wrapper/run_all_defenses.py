from model_functions import detect
import cv2
import os
from adaptive_defense import AdaptiveDefense

dPipeline = AdaptiveDefense()


def run_attack_and_defense(image_path: str, attack_type: str = None):
    image = cv2.imread(image_path)
    if image is None:
        cwd = os.getcwd()
        raise FileNotFoundError(
            f"Cannot load image: {image_path}. cv2.imread returned None. Current working dir: {cwd}"
        )


    if attack_type:
        defended_image = dPipeline.apply_with_type(image, attack_type)
    else:
        attack_type = dPipeline.detect_attack(image)
        print(f"[INFO] Detected attack: {attack_type}")
        defended_image = dPipeline.apply_with_type(image, attack_type)

    """if attack_type is None:
        attack_type = "patch" 
        print(f"[INFO] No attack type specified, using default: {attack_type}")
    
    # Применяем защиту
    defended_image = dPipeline.apply_with_type(image, attack_type)"""

    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    temp_filename = f"defended_{timestamp}.png"

    temp_path = os.path.join("..", "data", temp_filename)
    if not os.path.exists(os.path.join("..", "data")):
        temp_path = os.path.join("data", temp_filename)
        if not os.path.exists("data"):
            temp_path = temp_filename

    cv2.imwrite(temp_path, defended_image)

    report = detect(temp_filename)
    return report



run_attack_and_defense(
    "../results/photo.jpg"
)

run_attack_and_defense(
    "../results/photo.jpg"
)