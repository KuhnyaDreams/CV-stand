from pathlib import Path
import requests
import os
import time

CORE_URL = os.getenv("CORE_URL", "http://localhost:8000")


def _normalize_path_for_core(input_path: str) -> str:
    path = str(input_path).replace("\\", "/")
    if path.startswith("/data/"):
        return path
    if "/data/" in path:
        path = path.split("/data/", 1)[1]
    elif "\\data\\" in path:
        path = path.split("\\data\\", 1)[1]
    while path.startswith("../"):
        path = path[3:]
    while path.startswith("..\\"):
        path = path[3:]
    return f"/data/{path}"


def check_api_health() -> bool:
    try:
        response = requests.get(f"{CORE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        try:
            response = requests.get(f"{CORE_URL}", timeout=5)
            return response.status_code in [200, 404]
        except:
            return False


def _print_detection_summary(result: dict, title: str = "Detection Results", verbose: bool = True):
    if not result or 'detections' not in result:
        if verbose:
            print(f"⚠️  {title}: No detections data")
        return
    
    detections = result.get('detections', [])
    total = len(detections)
    
    if total == 0 and not verbose:
        return
    
    print(f"\n{'='*60}")
    print(f"📊 {title}")
    print(f"{'='*60}")
    print(f"Total objects detected: {total}")
    
    if total == 0:
        print("✅ No threats detected")
        print(f"{'='*60}\n")
        return
    
    grouped = {}
    if detections and isinstance(detections[0], dict):
        for detection in detections:
            class_name = detection.get('class', 'unknown')
            if class_name not in grouped:
                grouped[class_name] = []
            grouped[class_name].append(detection)
    
    if grouped:
        print(f"\n📁 Grouped by class:")
        for class_name, items in sorted(grouped.items()):
            confidence_avg = sum(item.get('confidence', 0) for item in items) / len(items) if items else 0
            print(f"  • {class_name:20} → {len(items):3} items (avg confidence: {confidence_avg:.2%})")
    else:
        print(f"\n📋 Raw detections:")
        for i, det in enumerate(detections, 1):
            print(f"  {i}. {det}")
    
    print(f"{'='*60}\n")


def detect_image(input_path, verbose: bool = False):
    import time
    from pathlib import Path
    import requests
    import json

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = Path(input_path)
    name, suffix = path.stem, path.suffix

    core_input_path = _normalize_path_for_core(input_path)

    params = {
        "input_path": core_input_path,
        "output_path": f"/results/singles/{timestamp}-{name}",
        "class_names": ["person", "cell phone"]
    }

    print(f"📷 Обрабатываем изображение: {input_path}")
    print(f"   Core path: {core_input_path}")
    print(f"   Параметры запроса: {json.dumps(params, indent=2)}")

    try:
        response = requests.post(f"{CORE_URL}/detect/file", json=params, timeout=30)

        if response.status_code == 200:
            report = response.json()
            print(f"✅ Ответ API получен: {json.dumps(report, indent=2)}")
            if verbose:
                _print_detection_summary(report, f"Image Detection: {name}", verbose=True)
            return report
        elif response.status_code == 500:
            error_msg = response.text[:100] if response.text else "Internal Server Error"
            print(f"⚠️  API Error 500: {error_msg}")
            print(f"   Файл: {core_input_path}")
            return None
        else:
            print(f"❌ Ошибка {response.status_code}: {response.text[:100]}")
            return None
    except requests.exceptions.ConnectionError:
        print(f"❌ Не удалось подключиться к {CORE_URL}")
        print("   Убедитесь, что core запущен: docker-compose up -d")
        return None
    except Exception as e:
        print(f"❌ Ошибка при обработке: {str(e)}")
        return None


def detect_folder(input_path, verbose: bool = False):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_subdir = f"{Path(input_path).name}-{timestamp}"

    core_input_path = _normalize_path_for_core(input_path)

    params = {
        "input_path": core_input_path,
        "output_path": f"/results/{output_subdir}",
        "class_names": ["person", "cell phone"]
    }

    try:
        response = requests.post(f"{CORE_URL}/detect/folder", json=params, timeout=60)

        if response.status_code == 200:
            report = response.json()
            if verbose:
                _print_detection_summary(report, f"Folder Detection: {output_subdir}", verbose=True)
            return report
        elif response.status_code == 500:
            error_msg = response.text[:100] if response.text else "Internal Server Error"
            print(f"⚠️  API Error 500: {error_msg}")
            print(f"   Folder: {core_input_path}")
            return None
        else:
            print(f"❌ Ошибка {response.status_code}: {response.text[:100]}")
            return None
    except requests.exceptions.ConnectionError:
        print(f"❌ Не удалось подключиться к {CORE_URL}")
        print("   Убедитесь что core запущен: docker-compose up -d")
        return None
    except Exception as e:
        print(f"❌ Ошибка при обработке: {str(e)}")
        return None