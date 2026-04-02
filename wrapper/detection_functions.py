from pathlib import Path
import requests
import os
import time

CORE_URL = os.getenv("CORE_URL", "http://localhost:8000")

def detect_image(input_path):
    """Тестируем одно изображение через API ядра"""

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = Path(input_path)
    name, suffix = path.stem, path.suffix

    params = {
        "input_path": f"/data/{input_path}",
        "output_path": f"/results/singles/{timestamp}-{name}",
        "class_names": ["person", "cell phone"]
    }

    response = requests.post(f"{CORE_URL}/detect", json=params)

    if response.status_code == 200:
        report = response.json()
        return report
    else:
        print(f"Ошибка: {response.status_code}")
        return None


def detect_folder(input_path):
    """
    Обрабатывает все изображения в папке через эндпоинт /detect (POST с json-параметрами).
    Результаты (изображения с рамками) сохраняются в /app/results внутри core,
    что соответствует локальной папке results.
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_subdir = f"{input_path}-{timestamp}"

    params = {
        "input_path": f"/data/{input_path}",
        "output_path": f"/results/{output_subdir}",
        "class_names": ["person", "cell phone"]
    }

    response = requests.post(f"{CORE_URL}/detect", json=params)

    if response.status_code == 200:
        report = response.json()
        return report
    else:
        print(f"Ошибка: {response.status_code}")
        return None
    