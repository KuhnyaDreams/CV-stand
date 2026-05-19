import keras
import tensorflow as tf
import cv2
import numpy as np

model_dir = "../core/yolo26n_saved_model"

# 1. Загрузка как Keras-слой
yolo_layer = keras.layers.TFSMLayer(model_dir, call_endpoint='serving_default')

# 2. Подготовка изображения
img = cv2.imread("../data/test.png")
if img is None:
    raise FileNotFoundError("Image not found")
    
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (640, 640))  # должен совпадать с imgsz при экспорте
x = np.expand_dims(img_resized, axis=0).astype(np.float32)

# В Ultralytics >=8.2.0 нормализация /255 уже встроена в экспорт.
# Если модель даёт аномальные предсказания, раскомментируйте:
# x = x / 255.0

# 3. Инференс
output = yolo_layer(x)

# 🔧 Устойчивое извлечение тензора (dict или прямой tensor)
if isinstance(output, dict):
    predictions = list(output.values())[0]
else:
    predictions = output

print("✅ Output shape:", predictions.shape)  # обычно (1, 8400, 84)