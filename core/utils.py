import datetime
import os
from typing import Any, List
import json

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

    task = model.task

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
            
        elif task == 'classify':
            probs = result.probs
            all_probs = probs.data.cpu().tolist()
            predictions = []
            for class_id, confidence in enumerate(all_probs):
                if confidence > 0:
                    predictions.append({
                        "class": result.names[class_id],
                        "class_id": class_id,
                        "confidence": confidence
                    })
            predictions.sort(key=lambda x: x["confidence"], reverse=True)
            image_data["objects"] = predictions

        report["images"].append(image_data)

    return report

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

def create_video_report(request,
                        total_frames_processed,
                        duration,
                        phone_present_frames):
    
    gap_seconds = request.gap_seconds
    output_path=request.output_path
    video_path=request.video_path
    intervals = []
    
    if phone_present_frames:
        start_t = phone_present_frames[0][0]
        end_t = phone_present_frames[0][0]
        confs = [phone_present_frames[0][1]]
        for i in range(1, len(phone_present_frames)):
            gap = phone_present_frames[i][0] - phone_present_frames[i-1][0]
            if gap <= gap_seconds:
                end_t = phone_present_frames[i][0]
                confs.append(phone_present_frames[i][1])
            else:
                intervals.append({
                    "start_time": round(start_t, 2),
                    "end_time": round(end_t, 2),
                    "avg_phone_confidence": round(sum(confs)/len(confs), 3),
                    "max_phone_confidence": round(max(confs), 3),
                    "frame_count": len(confs)
                })
                start_t = phone_present_frames[i][0]
                end_t = phone_present_frames[i][0]
                confs = [phone_present_frames[i][1]]
        intervals.append({
            "start_time": round(start_t, 2),
            "end_time": round(end_t, 2),
            "avg_phone_confidence": round(sum(confs)/len(confs), 3),
            "max_phone_confidence": round(max(confs), 3),
            "frame_count": len(confs)
        })

    total_time = sum(i["end_time"] - i["start_time"] for i in intervals)
    detection_ratio = total_time / duration if duration > 0 else 0

    report = {
        "video_path": video_path,
        "total_frames_processed": total_frames_processed,
        "duration_seconds": duration,
        "intervals": intervals,
        "total_time_with_phone": total_time,
        "detection_ratio": detection_ratio
    }

    if output_path:
        os.makedirs(output_path, exist_ok=True)
        json_path = os.path.join(output_path, "phone_analysis.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    return report

def process_frame(result, frame_idx: int, fps: float, phi: float = 0.2, iou_thresh: float = 0.15):
    timestamp = frame_idx / fps if fps > 0 else 0
    boxes = result.boxes
    if boxes is None:
        return None, None

    persons = []
    phones = []
    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()
        if cls == 0:
            persons.append(xyxy)
        elif cls == 67: 
            phones.append((xyxy, conf))

    if not persons or not phones:
        return None, None

    for (phone_box, phone_conf) in phones:
        phone_center = ((phone_box[0] + phone_box[2]) / 2,
                        (phone_box[1] + phone_box[3]) / 2)
        for person_box in persons:
            w = person_box[2] - person_box[0]
            h = person_box[3] - person_box[1]
            expanded = [
                person_box[0] - phi * w,
                person_box[1] - phi * h,
                person_box[2] + phi * w,
                person_box[3] + phi * h
            ]
            inside_expanded = (expanded[0] <= phone_center[0] <= expanded[2] and
                               expanded[1] <= phone_center[1] <= expanded[3])
            iou = compute_iou(phone_box, person_box)
            if inside_expanded or iou >= iou_thresh:
                return timestamp, max(phone_conf, 0)
    return None, None