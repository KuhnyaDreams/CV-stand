FROM python:3.10-slim

WORKDIR /workspace

# Устанавливаем системные зависимости (нужны для OpenCV)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем зависимости Python
RUN pip install --no-cache-dir \
    requests \
    numpy \
    pillow \
    matplotlib \
    jupyter \
    opencv-python

# Копируем скрипты
COPY scripts/ ./scripts/

# Открываем порт для Jupyter
EXPOSE 8888

# По умолчанию запускаем интерактивную оболочку
CMD ["python", "-c", "print('Wrapper container ready. Use: docker exec -it wrapper python scripts/attack_experiment.py'); import time; time.sleep(3600)"]