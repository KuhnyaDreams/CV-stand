import datetime
import os

from schemas import EstimateRequest, DetectRequest

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


def create_estimation_report(request: EstimateRequest, results, estimator):
    report = {
        "timestamp": datetime.datetime.now().isoformat(),
        "model": "yolo26s-pose",
        "conf_thres": estimator.conf_thres,
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

        for person_keypoints in result.keypoints.data:
            keypoints = {}
            for kpt_idx, kpt in enumerate(person_keypoints):
                x, y, conf = float(kpt[0]), float(kpt[1]), float(kpt[2])

                point_name = keypoint_names[kpt_idx]
                keypoints[point_name] = {
                    "x": x,
                    "y": y,
                    "confidence": conf
                }

            image_data["objects"].append(keypoints)

        report["images"].append(image_data)

    return report


def create_detection_report(request: DetectRequest, results, detector):
    report = {
        "timestamp": datetime.datetime.now().isoformat(),
        "model": "yolo26x",
        "conf_thres": detector.conf_thres,
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

        for box in result.boxes:
            obj = {
                "class": detector.detector.names[int(box.cls)],
                "class_id": int(box.cls),
                "confidence": float(box.conf),
                "bbox": box.xyxy[0].tolist() if hasattr(box, 'xyxy') else []
            }
            image_data["objects"].append(obj)

        report["images"].append(image_data)

    return report
