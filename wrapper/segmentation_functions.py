import time
import requests
import os
from pathlib import Path

CORE_URL = os.getenv("CORE_URL", "http://localhost:8000")

def segment(input_path, class_names = None):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = Path(input_path)
    name = path.stem

    params = {
        "input_path": f"/data/{input_path}",
        "output_path": f"/results/segmentation/{timestamp}-{name}",
    }
    if class_names:
        params["class_names"] = class_names

    response = requests.post(f"{CORE_URL}/segment", json=params)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Ошибка сегментации: статус {response.status_code}")
        return None
    
if __name__ == "__main__":
    segment('berzerk')