from defence import DefensePipeline, Defenses
from attack_eval import AttackEvaluator
from detection_functions import detect_image
import cv2
import os

aEval = AttackEvaluator()
defenses = Defenses()
dPipeline = DefensePipeline( defenses)
def run_attack_and_defense(image_path: str):
    image, report = aEval.run_single_attack_image(image_path, 'white_box', 'fgsm_attack')
    print(report)
    defended_image = dPipeline.apply(image)
    # save defended image to data/ and call detect_image with filename (API expects a path under /data/)
    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    temp_filename = f"defended_{timestamp}.png"
    temp_path = os.path.join("..", "data", temp_filename)
    if not os.path.exists(os.path.join("..", "data")):
        temp_path = os.path.join("data", temp_filename)
        if not os.path.exists("data"):
            temp_path = temp_filename
    # adv images in this project are RGB — convert to BGR for OpenCV saving
    try:
        cv2.imwrite(temp_path, cv2.cvtColor(defended_image, cv2.COLOR_RGB2BGR))
    except Exception:
        cv2.imwrite(temp_path, defended_image)
    report = detect_image(temp_filename)
    print("====================")
    print(report)

run_attack_and_defense("../data/test.jpg")