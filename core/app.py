import json
import os
import uvicorn
from fastapi import FastAPI, HTTPException, Body
from yolo_detector import YOLO26Detector
from yolo_pose import YOLO26PoseEstimator
from yolo_segmentor import YOLO26Segmentor
from schemas import DetectRequest, EstimateRequest, SegmentRequest
from utils import create_report

app = FastAPI(title="YOLO26 CV Core", description="Ядро компьютерного зрения на YOLO26")
detector = YOLO26Detector(model_path="yolo26x.pt")
estimator = YOLO26PoseEstimator(model_path='yolo26x-pose.pt')
segmentor = YOLO26Segmentor(model_path="yolo26x-seg.pt")


@app.get("/")
def root():
    """Основная информация о сервисе и доступных моделях"""
    return {
        "service": "YOLO26 CV Core",
        "status": "active",
        "models": {
            "detect": "yolo26x.pt",
            "pose": "yolo26x-pose.pt",
            "segment": "yolo26x-seg.pt"
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
        model: 'detect' (по умолчанию), 'pose', 'segment'
    """
    if model == "detect":
        return {
            "model": "detect",
            "classes": detector.detector.names
        }
    elif model == "pose":
        return {
            "model": "pose",
            "keypoints": estimator.estimator.names
        }
    elif model == "segment":
        return {
            "model": "segment",
            "classes": segmentor.segmentor.names
        }
    else:
        raise HTTPException(status_code=400, detail="Unknown model. Use 'detect', 'pose', or 'segment'")


@app.post("/detect")
async def detect(request: DetectRequest = Body(...)):
    """Детекция объектов на изображении(ях) или видео.
    Обрабатывает изображение/папку/видеозапись, в результате получается отчет и размеченные изображения"""

    if not os.path.exists(request.input_path):
        raise HTTPException(status_code=404, detail=f"Путь {request.input_path} не найден")

    class_ids = detector.get_class_ids(request.class_names)
    results = detector.detect(request.input_path, request.output_path, request.save_images, classes=class_ids)

    report = create_report(request, results, detector)
    with open(f'{request.output_path}/result.json', 'x', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return report


@app.post("/estimate")
async def estimate(request: EstimateRequest = Body(...)):
    """Определение ключевых точек человека.
    Обрабатывает изображение/папку/видеозапись, в результате получается отчет и размеченные изображения"""

    if not os.path.exists(request.input_path):
        raise HTTPException(status_code=404, detail=f"Путь {request.input_path} не найден")

    results = estimator.estimate(request.input_path, request.output_path, request.save_images)

    report = create_report(request, results, estimator)
    with open(f'{request.output_path}/result.json', 'x', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return report

@app.post("/segment")
async def segment_objects(request: SegmentRequest = Body(...)):
    """Сегментация экземпляров на изображении(ях) или видео.
    Обрабатывает изображение/папку/видеозапись, в результате получается отчет и размеченные изображения"""

    if not os.path.exists(request.input_path):
        raise HTTPException(status_code=404, detail=f"Путь {request.input_path} не найден")

    class_ids = segmentor.get_class_ids(request.class_names)
    results = segmentor.segment(request.input_path, request.output_path, request.save_images, classes=class_ids)

    report = create_report(request, results, segmentor)
    with open(f'{request.output_path}/result.json', "x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return report

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
