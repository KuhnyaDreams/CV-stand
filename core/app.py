from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import os
import shutil
import uuid
from pathlib import Path
from yolo_core import YOLO26Detector

app = FastAPI(title="YOLO26 CV Core", description="Ядро компьютерного зрения на YOLO26")

# Инициализируем детектор
detector = YOLO26Detector(model_path="yolo26x.pt")

# Создаём временную папку для загружаемых файлов
TEMP_DIR = "/tmp/yolo_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

@app.get("/")
def root():
    return {
        "service": "YOLO26 CV Core",
        "status": "active",
        "model": "yolo26x",
        "classes": len(detector.model.names)
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/classes")
def get_classes():
    """Возвращает список классов, которые умеет распознавать модель"""
    return {"classes": detector.model.names}

@app.post("/detect/file")
async def detect_file(file: UploadFile = File(...), conf_thres: float = 0.25):
    """
    Загрузить одно изображение и получить детекции
    """
    # Сохраняем загруженный файл
    file_ext = Path(file.filename).suffix
    file_id = str(uuid.uuid4())
    temp_path = os.path.join(TEMP_DIR, f"{file_id}{file_ext}")
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Создаём временную выходную папку
    output_dir = os.path.join(TEMP_DIR, f"out_{file_id}")
    
    # Запускаем детекцию
    results = detector.detect_single(temp_path, output_dir)
    
    # Формируем ответ
    response = []
    for result in results:
        for box in result.boxes:
            response.append({
                "class": detector.model.names[int(box.cls)],
                "class_id": int(box.cls),
                "confidence": float(box.conf),
                "bbox": box.xyxy[0].tolist() if hasattr(box, 'xyxy') else []
            })
    
    # Очищаем временные файлы
    os.remove(temp_path)
    shutil.rmtree(output_dir, ignore_errors=True)
    
    return {"detections": response}

@app.post("/detect/folder")
async def detect_folder(input_folder: str, output_folder: str = "results", conf_thres: float = 0.25):
    """
    Обработать целую папку с изображениями
    input_folder должен быть доступен внутри контейнера
    """
    if not os.path.exists(input_folder):
        raise HTTPException(status_code=404, detail=f"Папка {input_folder} не найдена")
    
    report = detector.detect_folder(input_folder, output_folder, conf_thres=conf_thres)
    
    return report

@app.get("/results/{folder_name}/{filename}")
async def get_result_file(folder_name: str, filename: str):
    """Получить файл с результатами"""
    file_path = os.path.join("/app/results", folder_name, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return {"error": "File not found"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)