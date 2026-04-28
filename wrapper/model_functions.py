import time
import requests
from pathlib import Path
import os

CORE_URL = os.getenv("CORE_URL", "http://localhost:8000")

def _call_core(task, input_path, class_names = None, save_images = True, show_boxes = False):
    """Универсальный вызов API core-сервиса"""

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = Path(input_path)
    name = path.stem

    output_subdir = {
        'detect': 'detection',
        'estimate': 'estimation',
        'segment': 'segmentation',
        'classify': 'classification',
    }.get(task, 'unknown')

    if output_subdir == 'unknown':
        raise ValueError(f"Неизвестная задача: {task}")

    params = {
        "input_path": f"/data/{input_path}",
        "output_path": f"/results/{output_subdir}/{timestamp}-{name}",
        "task": task,
        "save_images": save_images,
        "show_boxes": show_boxes,
    }
    if task != 'estimate' and task != 'classify':
        params["class_names"] = class_names

    response = requests.post(f"{CORE_URL}/{task}", json=params)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Ошибка {task}: статус {response.status_code}")
        return None

def detect(input_path, class_names = None, save_images = True, show_boxes = True):
    return _call_core('detect', input_path, class_names, save_images, show_boxes)

def estimate(input_path, save_images = True):
    return _call_core('estimate', input_path, None, save_images)

def segment(input_path, class_names = None, save_images = True):
    return _call_core('segment', input_path, class_names, save_images)

def classify(input_path, save_images = True):
    """Классификация изображения/папки."""
    return _call_core('classify', input_path, None, save_images)

def analyze_video_phone(video_path: str, frame_interval: int = 1, conf_thres: float = 0.25, iou_threshold: float = 0.2):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = Path(video_path)
    name = path.stem
    output_path = f"/results/video_analysis/{timestamp}-{name}"
    params = {
        "video_path": f"/data/{video_path}",
        "output_path": output_path,
        "frame_interval": frame_interval,
        "conf_thres": conf_thres,
        "iou_threshold": iou_threshold
    }
    response = requests.post(f"{CORE_URL}/analyze_video", json=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Ошибка анализа видео: {response.status_code}")
        return None