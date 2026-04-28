from model_functions import detect
import cv2
import os
from adaptive_defense import AdaptiveDefense

dPipeline = AdaptiveDefense()


def run_attack_and_defense(image_path: str, attack_type: str = None):
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(image_path)

    # если вручную указали тип атаки
    if attack_type:
        defended_image = dPipeline.apply_with_type(image, attack_type)

    # если нет — автоматически определить
    else:
        defended_image = dPipeline.apply(image)

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
    "../results/photo2.jpg"
)





"""if attack_type is None:
        attack_type = "patch" 
        print(f"[INFO] No attack type specified, using default: {attack_type}")
    
    # Применяем защиту
    defended_image = dPipeline.apply_with_type(image, attack_type)"""