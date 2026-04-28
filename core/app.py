import json
import os
import uvicorn
from fastapi import FastAPI, HTTPException, Body
from yolo_model import YOLOModel
from schemas import DetectRequest, EstimateRequest, SegmentRequest, PredictRequest, ClassifyRequest, VideoAnalysisRequest, VideoAnalysisResponse, PhoneWithPersonInterval
from utils import create_report, keypoint_names, compute_iou
import cv2

app = FastAPI(title="YOLO26 CV Core", description="Ядро компьютерного зрения на YOLO26")
detector = YOLOModel(model_path="yolo26x.pt", task='detect')
estimator = YOLOModel(model_path='yolo26x-pose.pt', task='estimate')
segmentor = YOLOModel(model_path="yolo26x-seg.pt", task='segment')
classifier = YOLOModel(model_path="yolo26x-cls.pt", task='classify')

AVAILABLE_TASKS = {
    "detect": detector,
    "estimate": estimator,
    "segment": segmentor,
    "classify": classifier
}

@app.get("/")
def root():
    """Основная информация о сервисе и доступных моделях"""
    return {
        "service": "YOLO26 CV Core",
        "status": "active",
        "models": {
            "detect": detector.model_name,
            "estimate": estimator.model_name,
            "segment": segmentor.model_name
        }
    }

@app.get("/health")
def health():
    """Проверка работоспособности сервиса"""
    return {"status": "ok"}

@app.get("/classes")
def get_classes(model: str = "detect"):
    """
    Возвращает список классов или ключевых точек для указанной модели.
    Параметры:
        model: 'detect' (по умолчанию), 'estimate', 'segment'
    """
    if model == "detect":
        return {
            "model": "detect",
            "classes": detector.detector.names
        }
    elif model == "estimate":
        return {
            "model": "estimate",
            "keypoints": keypoint_names
        }
    elif model == "segment":
        return {
            "model": "segment",
            "classes": segmentor.segmentor.names
        }
    elif model == "classify":
        return {"model": "classify", "classes": classifier.model.names}
    else:
        raise HTTPException(status_code=400, detail="Unknown model. Use 'detect', 'estimate', or 'segment'")

async def predict(request: PredictRequest = Body(...)):
    """
    Универсальная функция для детекции, позы или сегментации.
    Поле task определяет модель: 'detect', 'estimate', 'segment'.
    """
    if not os.path.exists(request.input_path):
        raise HTTPException(status_code=404, detail=f"Путь {request.input_path} не найден")

    if request.task not in AVAILABLE_TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task. Available: {list(AVAILABLE_TASKS.keys())}")
    model = AVAILABLE_TASKS[request.task]

    results = model.predict(
        input_path=request.input_path,
        output_path=request.output_path,
        save_images=request.save_images,
        classes=model.get_class_ids(request.class_names),
        show_boxes=request.show_boxes
    )

    report = create_report(request, results, model)
    json_path = os.path.join(request.output_path, "result.json")
    with open(json_path, 'x', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return report

@app.post("/detect")
async def detect(request: DetectRequest = Body(...)):
    pred_req = PredictRequest(
        task='detect',
        input_path=request.input_path,
        output_path=request.output_path,
        save_images=request.save_images,
        class_names=request.class_names,
        show_boxes=request.show_boxes
    )
    return await predict(pred_req)

@app.post("/estimate")
async def estimate(request: EstimateRequest = Body(...)):
    pred_req = PredictRequest(
        task='estimate',
        input_path=request.input_path,
        output_path=request.output_path,
        save_images=request.save_images
    )
    return await predict(pred_req)

@app.post("/segment")
async def segment_objects(request: SegmentRequest = Body(...)):
    pred_req = PredictRequest(
        task='segment',
        input_path=request.input_path,
        output_path=request.output_path,
        save_images=request.save_images,
        class_names=request.class_names
    )
    return await predict(pred_req)

@app.post("/classify")
async def segment_objects(request: ClassifyRequest = Body(...)):
    pred_req = PredictRequest(
        task='classify',
        input_path=request.input_path,
        output_path=request.output_path,
        save_images=request.save_images,
    )
    return await predict(pred_req)

@app.post("/analyze_video", response_model=VideoAnalysisResponse)
async def analyze_video(request: VideoAnalysisRequest = Body(...)):
    full_video_path = request.video_path
    if not os.path.exists(full_video_path):
        raise HTTPException(404, f"Video not found: {full_video_path}")

    cap = cv2.VideoCapture(full_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    cap.release()

    model = detector.model
    frame_idx = 0
    phone_present_frames = []

    results_gen = model.predict(
        source=full_video_path,
        stream=True,
        conf=request.conf_thres,
        save=False,
        verbose=False
    )

    for result in results_gen:
        if frame_idx % request.frame_interval != 0:
            frame_idx += 1
            continue

        timestamp = frame_idx / fps if fps > 0 else 0
        boxes = result.boxes
        if boxes is None:
            frame_idx += 1
            continue

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


        phi = 0.2
        iou_thresh = 0.15
        phone_belongs = False
        max_phone_conf = 0
        for (phone_box, phone_conf) in phones:
            phone_center = ((phone_box[0]+phone_box[2])/2, (phone_box[1]+phone_box[3])/2)
            for person_box in persons:
                w = person_box[2] - person_box[0]
                h = person_box[3] - person_box[1]
                expanded = [
                    person_box[0] - phi*w,
                    person_box[1] - phi*h,
                    person_box[2] + phi*w,
                    person_box[3] + phi*h
                ]
                inside_expanded = (expanded[0] <= phone_center[0] <= expanded[2] and
                           expanded[1] <= phone_center[1] <= expanded[3])
                iou = compute_iou(phone_box, person_box)
                if inside_expanded or iou >= iou_thresh:
                    phone_belongs = True
                    max_phone_conf = max(max_phone_conf, phone_conf)
                    break
            if phone_belongs:
                break

        if phone_belongs:
            phone_present_frames.append((timestamp, max_phone_conf))

        frame_idx += 1

    intervals = []
    if phone_present_frames:
        start_t = phone_present_frames[0][0]
        end_t = phone_present_frames[0][0]
        confs = [phone_present_frames[0][1]]
        for i in range(1, len(phone_present_frames)):
            gap = phone_present_frames[i][0] - phone_present_frames[i-1][0]
            if gap <= request.gap_seconds:
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

    if request.output_path:
        os.makedirs(request.output_path, exist_ok=True)
        report = {
            "video_path": request.video_path,
            "total_frames_processed": frame_idx,
            "duration_seconds": duration,
            "intervals": intervals,
            "total_time_with_phone": total_time,
            "detection_ratio": detection_ratio
        }
        with open(os.path.join(request.output_path, "phone_analysis.json"), "w") as f:
            json.dump(report, f, indent=2)

    return VideoAnalysisResponse(
        video_path=request.video_path,
        total_frames_processed=frame_idx,
        duration_seconds=duration,
        intervals=intervals,
        total_time_with_phone=total_time,
        detection_ratio=detection_ratio
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
