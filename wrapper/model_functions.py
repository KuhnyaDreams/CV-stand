import time
import requests
from pathlib import Path
import os

CORE_URL = os.getenv("CORE_URL", "http://localhost:8000")

def _call_core(task, input_path, class_names = None):
    """Универсальный вызов API core-сервиса"""

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = Path(input_path)
    name = path.stem

    output_subdir = {
        'detect': 'detection',
        'estimate': 'estimation',
        'segment': 'segmentation'
    }.get(task, 'unknown')

    if output_subdir == 'unknown':
        raise ValueError(f"Неизвестная задача: {task}")

    params = {
        "input_path": f"/data/{input_path}",
        "output_path": f"/results/{output_subdir}/{timestamp}-{name}",
    }
    if task != 'estimate':
        params["class_names"] = class_names

    response = requests.post(f"{CORE_URL}/{task}", json=params)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Ошибка {task}: статус {response.status_code}")
        return None

def detect(input_path, class_names = None):
    return _call_core('detect', input_path, class_names)

def estimate(input_path):
    return _call_core('estimate', input_path)

def segment(input_path, class_names = None):
    return _call_core('segment', input_path, class_names)