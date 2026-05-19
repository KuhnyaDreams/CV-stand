from ultralytics import YOLO

model = YOLO("yolo26n.pt")  # загрузка модели
model.export(format="saved_model",keras = True)