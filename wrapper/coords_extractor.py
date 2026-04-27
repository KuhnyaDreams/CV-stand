import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union, Literal
import random

def extract_attack_coordinates(
    detection_result: Dict[str, Any],
    strategy: str = "center",
    points_per_bbox: int = 1,
    target_class: Optional[Union[str, List[str]]] = None,
    case_sensitive: bool = False,
    return_patch_info: bool = False,
    patch_size_mode: Literal["fixed", "ratio", "bbox_min", "bbox_max"] = "ratio",
    patch_size_value: Union[int, float] = 0.15  # px для fixed, ratio для ratio
) -> Union[List[Tuple[int, int]], List[Dict[str, Any]]]:
    """
    Извлекает координаты и (опционально) размеры патчей из bbox-ов YOLO для adversarial attack.
    
    Args:
        detection_result: JSON/dict с результатами детекции
        strategy: 'center', 'random', 'corners', 'grid'
        points_per_bbox: Количество точек на bbox (для 'random'/'grid')
        target_class: Фильтр по названию класса (поддерживает частичное совпадение)
        case_sensitive: Учитывать ли регистр при сравнении классов
        return_patch_info: Если True — возвращает список dict {x, y, size, bbox, class, ...}
        patch_size_mode:
            - 'fixed': размер в пикселях (patch_size_value)
            - 'ratio': доля от меньшей стороны bbox (patch_size_value ∈ [0, 1])
            - 'bbox_min': размер = min(bbox_width, bbox_height)
            - 'bbox_max': размер = max(bbox_width, bbox_height)
        patch_size_value: Значение размера (пиксели или доля)
                         
    Returns:
        Если return_patch_info=False: List[Tuple[int, int]] — только координаты
        Если return_patch_info=True: List[Dict] с полями:
            - x, y: координаты центра/угла патча
            - size: рекомендуемый размер патча в пикселях
            - bbox: [x1, y1, x2, y2] исходного объекта
            - class, class_id, confidence: метаданные объекта
    """
    results = []
    
    # Нормализация target_class
    if target_class is not None:
        target_classes = [target_class] if isinstance(target_class, str) else list(target_class)
        if not case_sensitive:
            target_classes = [tc.lower() for tc in target_classes]
    else:
        target_classes = None
    
    for img_data in detection_result.get("images", []):
        for obj in img_data.get("objects", []):
            # 🔍 Фильтрация по классу
            if target_classes is not None:
                obj_class = obj.get("class", "")
                compare_class = obj_class if case_sensitive else obj_class.lower()
                if not any(tc == compare_class or tc in compare_class or compare_class in tc 
                          for tc in target_classes):
                    continue
            
            # 📦 Обработка bbox
            x1, y1, x2, y2 = obj["bbox"]
            x_min, x_max = sorted([int(round(x1)), int(round(x2))])
            y_min, y_max = sorted([int(round(y1)), int(round(y2))])
            bbox_w, bbox_h = x_max - x_min, y_max - y_min
            
            # 📐 Вычисление размера патча
            if patch_size_mode == "fixed":
                patch_size = int(patch_size_value)
            elif patch_size_mode == "ratio":
                patch_size = int(min(bbox_w, bbox_h) * patch_size_value)
            elif patch_size_mode == "bbox_min":
                patch_size = min(bbox_w, bbox_h)
            elif patch_size_mode == "bbox_max":
                patch_size = max(bbox_w, bbox_h)
            else:
                patch_size = int(patch_size_value) if isinstance(patch_size_value, (int, float)) else 32
            
            patch_size = max(1, patch_size)  # защита от 0
            
            # 🎯 Генерация координат по стратегии
            candidate_points = []
            if strategy == "center":
                candidate_points = [((x_min + x_max) // 2, (y_min + y_max) // 2)]
            elif strategy == "random":
                for _ in range(points_per_bbox):
                    px = random.randint(x_min, x_max)
                    py = random.randint(y_min, y_max)
                    candidate_points.append((px, py))
            elif strategy == "corners":
                candidate_points = [
                    (x_min, y_min), (x_max, y_min),
                    (x_min, y_max), (x_max, y_max)
                ]
            elif strategy == "grid":
                step_x = max(1, bbox_w // max(1, points_per_bbox))
                step_y = max(1, bbox_h // max(1, points_per_bbox))
                for x in range(x_min, x_max + 1, step_x):
                    for y in range(y_min, y_max + 1, step_y):
                        candidate_points.append((x, y))
            
            # 📦 Формирование результата
            for (px, py) in candidate_points:
                if return_patch_info:
                    results.append({
                        "x": px,
                        "y": py,
                        "size": patch_size,
                        "bbox": [x_min, y_min, x_max, y_max],
                        "class": obj.get("class"),
                        "class_id": obj.get("class_id"),
                        "confidence": obj.get("confidence")
                    })
                else:
                    results.append((px, py))
                        
    return results