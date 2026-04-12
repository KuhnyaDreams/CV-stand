import datetime
import os
from typing import Any, List

keypoint_names = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle'
}

def create_report(request: Any, results: List, model: Any):
    """Универсальное формирование отчёта для задач детекции, оценки позы и сегментации. Модель YOLO26"""
    if not results:
        return {}

    main_attribute = results[0]
    if main_attribute.keypoints is not None:
        task = 'estimate'
    elif main_attribute.masks is not None:
        task = 'segment'
    elif main_attribute.boxes is not None:
        task = 'detect'
    else:
        return {}

    conf_thres = getattr(model, 'conf_thres', 0.25)

    report = {
        "timestamp": datetime.datetime.now().isoformat(),
        "model": model.model_name,
        "conf_thres": conf_thres,
        "input_folder": request.input_path,
        "output_folder": request.output_path,
        "total_images": len(results),
        "images": []
    }

    for result in results:
        image_data = {
            "path": result.path,
            "filename": os.path.basename(result.path),
            "objects": []
        }

        if task == 'detect':
            for box in result.boxes:
                obj = {
                    "class": result.names[int(box.cls)],
                    "class_id": int(box.cls),
                    "confidence": float(box.conf),
                    "bbox": box.xyxy[0].tolist()
                }
                image_data["objects"].append(obj)

        elif task == 'estimate':
            for person_kpts in result.keypoints.data:
                keypoints = {}
                for kpt_idx, (x, y, conf) in enumerate(person_kpts):
                    point_name = keypoint_names[kpt_idx]
                    keypoints[point_name] = {
                        "x": float(x),
                        "y": float(y),
                        "confidence": float(conf)
                    }
                image_data["objects"].append(keypoints)

        elif task == 'segment':
            if result.masks is not None:
                for i in range(len(result.masks)):
                    cls_id = int(result.boxes.cls[i].item())
                    obj = {
                        "class": result.names[cls_id],
                        "class_id": cls_id,
                        "confidence": float(result.boxes.conf[i].item()),
                        "polygon": result.masks.xy[i].tolist()
                    }
                    image_data["objects"].append(obj)

        report["images"].append(image_data)

    return report