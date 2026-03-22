from pathlib import Path
import requests
import os
import json
import time

CORE_URL = os.getenv("CORE_URL", "http://localhost:8000")


def test_image(input_path):
    """Тестируем одно изображение через API ядра"""

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = Path(input_path)
    name, suffix = path.stem, path.suffix

    params = {
        "input_path": f"/data/{input_path}",
        "output_path": f"/results/singles/{timestamp}-{name}",
        "class_names": ["person", "cell phone"]
    }

    response = requests.post(f"{CORE_URL}/detect/file", json=params)

    if response.status_code == 200:
        report = response.json()
        with open(f'../../results/singles/{timestamp}-{name}/result.json', 'x', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        return report
    else:
        print(f"Ошибка: {response.status_code}")
        return None


def test_folder(input_path, output_path):
    """
    Обрабатывает все изображения в папке через эндпоинт /detect/folder (POST с query-параметрами).
    Результаты (изображения с рамками) сохраняются в /app/results внутри core,
    что соответствует локальной папке results.
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_subdir = f"{output_path}-{timestamp}"

    params = {
        "input_path": f"/data/{input_path}",
        "output_path": f"/results/{output_subdir}",
        "class_names": ["person", "cell phone"]
    }

    response = requests.post(f"{CORE_URL}/detect/folder", json=params)

    if response.status_code == 200:
        report = response.json()
        with open(f'../results/{output_subdir}', 'x', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        return report
    else:
        print(f"Ошибка: {response.status_code}")
        return None


if __name__ == "__main__":
    response = test_image("phone.jpg")
    print(response)
