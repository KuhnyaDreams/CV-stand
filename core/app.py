import json
import os
import uvicorn
from fastapi import FastAPI, HTTPException, Body
from yolo_detector import YOLO26Detector
from yolo_pose import YOLO26PoseEstimator
from yolo_segmentor import YOLO26Segmentor
from schemas import DetectRequest, EstimateRequest, SegmentRequest
from utils import create_detection_report, create_estimation_report, create_segmentation_report

app = FastAPI(title="YOLO26 CV Core", description="Ядро компьютерного зрения на YOLO26")
detector = YOLO26Detector(model_path="yolo26x.pt")
estimator = YOLO26PoseEstimator(model_path='yolo26s-pose.pt')
segmentor = YOLO26Segmentor(model_path="yolo26x-seg.pt")


@app.get("/")
def root():
    return {
        "service": "YOLO26 CV Core",
        "status": "active",
        "model": "yolo26x",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/classes")
def get_classes():
    """Возвращает список классов, которые умеет распознавать модель"""
    return {"classes": detector.detector.names}


@app.post("/detect")
async def detect(request: DetectRequest = Body(...)):
    """
    Обработать изображение/папку и получить отчет/размеченные изображения
    """
    if not os.path.exists(request.input_path):
        raise HTTPException(status_code=404, detail=f"Путь {request.input_path} не найден")

    class_ids = detector.get_class_ids(request.class_names) or None
    results = detector.detect(request.input_path, request.output_path, classes=class_ids)

    report = create_detection_report(request, results, detector)
    with open(f'{request.output_path}/result.json', 'x', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return report


@app.post("/estimate")
async def estimate(request: EstimateRequest = Body(...)):
    """
    Обработать изображение/видеозапись и получить отчет/размеч
    """
    if not os.path.exists(request.input_path):
        raise HTTPException(status_code=404, detail=f"Путь {request.input_path} не найден")

    results = estimator.estimate(request.input_path, request.output_path)
    report = create_estimation_report(request, results, estimator)
    with open(f'{request.output_path}/result.json', 'x', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return report

@app.post("/segment")
async def segment_objects(request: SegmentRequest = Body(...)):
    """Обрабатывает изображение/папку и возвращает JSON с масками сегментации."""
    if not os.path.exists(request.input_path):
        raise HTTPException(status_code=404, detail=f"Путь {request.input_path} не найден")

    class_ids = segmentor.get_class_ids(request.class_names) if request.class_names else None

    results = segmentor.segment(request.input_path, request.output_path, classes=class_ids)
    report = create_segmentation_report(request, results, segmentor)
    with open(f'{request.output_path}/result.json', "x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
