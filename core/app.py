import json
import os
import uvicorn
from fastapi import FastAPI, HTTPException, Body
from yolo_model import YOLOModel
from schemas import DetectRequest, EstimateRequest, SegmentRequest, PredictRequest, ClassifyRequest, VideoAnalysisRequest
from utils import create_report, keypoint_names, create_video_report, process_frame
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

@app.post("/analyze_video")
async def analyze_video(request: VideoAnalysisRequest = Body(...)):
    full_video_path = request.video_path
    if not os.path.exists(full_video_path):
        raise HTTPException(404, f"Video not found: {full_video_path}")

    cap = cv2.VideoCapture(full_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    cap.release()

    frame_idx = 0
    phone_present_frames = []

    results_gen = detector.model.predict(
        source=full_video_path,
        stream=True,
        vid_stride = request.frame_interval,
        conf=request.conf_thres,
        save=False,
        verbose=False
    )

    for result in results_gen:
        ts, conf = process_frame(result, frame_idx, fps,
                                 phi=0.2, iou_thresh=0.15)
        if ts is not None:
            phone_present_frames.append((ts, conf))
        frame_idx += request.frame_interval

    report = create_video_report(
        request=request,
        total_frames_processed=frame_idx,
        duration=duration,
        phone_present_frames=phone_present_frames
    )
    return report

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
