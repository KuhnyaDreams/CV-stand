import numpy as np
import cv2
from pathlib import Path
import time
from typing import Dict
import os
import yaml

from detection_functions import detect_image

CONFIG = None

def load_config(config_path: str = "config.yaml") -> Dict:
    global CONFIG
    try:
        with open(config_path, 'r') as f:
            CONFIG = yaml.safe_load(f)
        return CONFIG
    except FileNotFoundError:
        print(f"⚠️  Config file not found: {config_path}")
        print("   Using default parameters...")
        CONFIG = {}
        return CONFIG

def get_config() -> Dict:
    global CONFIG
    if CONFIG is None:
        load_config()
    return CONFIG or {}


def find_image_path(filename: str = "test.jpg") -> str:
    possible_paths = [
        filename,
        f"data/{filename}",
        f"../data/{filename}",
        f"../../data/{filename}",
    ]
    for path in possible_paths:
        if Path(path).exists():
            return path
    return "data/test.jpg"

class AdversarialAttacks:
    def __init__(self, model_url: str = "http://localhost:8000"):
        self.model_url = model_url
        self.results = {}
    
    def load_image(self, image_path: str) -> np.ndarray:
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Cannot load image: {image_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def save_adversarial_image(self, adv_image: np.ndarray, output_path: str, suffix: str = "adv") -> str:
        Path(output_path).mkdir(parents=True, exist_ok=True)
        filename = f"{suffix}_{time.strftime('%Y%m%d_%H%M%S')}.png"
        full_path = os.path.join(output_path, filename)
        adv_bgr = cv2.cvtColor(adv_image.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(full_path, adv_bgr)
        return full_path
    
    def evaluate_attack(self, original_image_path: str, adversarial_image_path: str) -> Dict:
        try:
            original_result = detect_image(original_image_path)
            adversarial_result = detect_image(adversarial_image_path)
            original_detections = len(original_result.get('detections', []))
            adversarial_detections = len(adversarial_result.get('detections', []))
            return {
                'original_detections': original_detections,
                'adversarial_detections': adversarial_detections,
                'reduction': original_detections - adversarial_detections,
                'success': adversarial_detections < original_detections,
                'original_result': original_result,
                'adversarial_result': adversarial_result
            }
        except Exception as e:
            return {'error': str(e)}


