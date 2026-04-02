from pathlib import Path
import requests
import os
import time

CORE_URL = os.getenv("CORE_URL", "http://localhost:8000")

def estimate_image(input_path):
    """Тестируем одно изображение через API ядра"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = Path(input_path)
    name, suffix = path.stem, path.suffix

    params = {
        "input_path": f"/data/{input_path}",
        "output_path": f"/results/estimation/singles/{timestamp}-{name}",
    }

    response = requests.post(f"{CORE_URL}/estimate", json=params)

    if response.status_code == 200:
        report = response.json()
        return report
    else:
        print(f"Ошибка: {response.status_code}")
        return None