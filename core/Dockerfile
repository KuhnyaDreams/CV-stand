FROM python:3.10-slim

WORKDIR /app

# Устанавливаем системные зависимости (исправленная версия)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Копируем requirements
COPY requirements.txt .

# Устанавливаем Python-зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код ядра
COPY yolo_core.py .
COPY app.py .

# Скачиваем модель YOLO26x (если нужно)
RUN python -c "from ultralytics import YOLO; YOLO('yolo26x.pt')"

# Открываем порт для API
EXPOSE 8000

# Запускаем API сервер
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]