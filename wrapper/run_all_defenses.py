from defense import DefensePipeline, Defenses
from model_functions import detect
import cv2
import os
dPipeline = DefensePipeline( Defenses())
def run_attack_and_defense(image_path: str):
    image = cv2.imread(image_path)
    if image is None:
        cwd = os.getcwd()
        raise FileNotFoundError(
            f"Cannot load image: {image_path}. cv2.imread returned None. Current working dir: {cwd}"
        )
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
    # The pipeline operates on OpenCV images (BGR). Save directly without channel swap.
    cv2.imwrite(temp_path, defended_image)
    report = detect(temp_filename)


run_attack_and_defense("../results/patch_attack_results/adv_single_black_box_patch_attack.png")
run_attack_and_defense("../results/single_pixel_attack_results/adv_single_black_box_single_pixel_attack.png")