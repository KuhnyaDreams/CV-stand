import datetime
import json
import os
import uvicorn
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from yolo_detector import YOLO26Detector


app = FastAPI(title="YOLO26 CV Core", description="Ядро компьютерного зрения на YOLO26")
detector = YOLO26Detector(model_path="yolo26x.pt")

class DetectRequest(BaseModel):
    input_path: str
    output_path: str = "results"
    class_names: Optional[List[str]] = None


class EstimateRequest(BaseModel):
    input_path: str
    output_path: str = "results"


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

    report = create_detection_report(request, results)
    with open(f'{request.output_path}/result.json', 'x', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return report


@app.get("/pose")
async def estimate(request: EstimateRequest = Body(...)):
    """
    Обработать изображение/видеозапись и получить отчет/размеч
    """


def create_estimation_report(request: EstimateRequest, results):
    pass


def create_detection_report(request: DetectRequest, results):
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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
