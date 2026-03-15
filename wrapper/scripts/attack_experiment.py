import requests
import os
import json
import time
from pathlib import Path

# Адрес ядра (в Docker-сети)
CORE_URL = os.getenv("CORE_URL", "http://core:8000")

def test_single_image(image_path):
    """Тестируем одно изображение через API ядра"""
    print(f"Тестирование {image_path}...")
    
    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{CORE_URL}/detect/file", files=files)
    
    if response.status_code == 200:
        detections = response.json()['detections']
        print(f"Найдено объектов: {len(detections)}")
        for d in detections:
            print(f"  {d['class']} ({d['confidence']:.2f})")
        return detections
    else:
        print(f"Ошибка: {response.status_code}")
        return None

def batch_test_folder(input_folder, output_json=None):
    """Тестируем все изображения в папке"""
    results = {}
    image_files = list(Path(input_folder).glob("*.jpg")) + list(Path(input_folder).glob("*.png"))
    
    for i, img_path in enumerate(image_files):
        print(f"[{i+1}/{len(image_files)}] {img_path.name}")
        detections = test_single_image(str(img_path))
        results[img_path.name] = detections
        time.sleep(0.1)  # небольшая задержка
    
    if output_json:
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Результаты сохранены в {output_json}")
    
    return results

if __name__ == "__main__":
    # Проверяем доступность ядра
    try:
        response = requests.get(f"{CORE_URL}/health")
        print("Ядро доступно!")
        print(requests.get(f"{CORE_URL}/").json())
    except:
        print("Не удалось подключиться к ядру")
        exit(1)
    
    # Тестируем одно изображение
    test_single_image("/data/berzerk/example.jpg")
    
    # Или всю папку
    # batch_test_folder("/data/berzerk", "/results/batch_results.json")