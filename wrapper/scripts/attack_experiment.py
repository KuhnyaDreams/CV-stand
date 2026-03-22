import requests
import os
import json
import time

CORE_URL = os.getenv("CORE_URL", "http://localhost:8000")

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
    """
    Обрабатывает все изображения в папке через эндпоинт /detect/folder (POST с query-параметрами).
    Результаты (изображения с рамками) сохраняются в /app/results внутри core,
    что соответствует локальной папке results.
    """
    print(f"Пакетная обработка папки {input_folder}...")
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_subdir = f"batch_{timestamp}"
    
    params = {
        "input_folder": input_folder,
        "output_folder": f"/app/results/{output_subdir}",
        "conf_thres": 0.25
    }
    
    response = requests.post(f"{CORE_URL}/detect/folder", params=params)
    
    if response.status_code == 200:
        report = response.json()
        print(f"Обработано изображений: {report.get('total_images', 0)}")
        
        if output_json:
            with open(output_json, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"Отчёт сохранён в {output_json}")
        
        print(f"Результаты (изображения) сохранены в локальной папке results/{output_subdir}")
        return report
    else:
        print(f"Ошибка при вызове /detect/folder: {response.status_code}")
        print(response.text)
        return None
    
if __name__ == "__main__":
    try:
        response = requests.get(f"{CORE_URL}/health")
        print("Ядро доступно!")
        print(requests.get(f"{CORE_URL}/").json())
    except:
        print("Не удалось подключиться к ядру")
        exit(1)
    
    batch_test_folder("/data/berzerk", "/results/result.json")
    