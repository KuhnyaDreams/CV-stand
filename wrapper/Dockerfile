FROM python:3.10-slim

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    requests \
    numpy \
    pillow \
    matplotlib \
    jupyter \
    opencv-python

COPY scripts/ ./scripts/

EXPOSE 8888

CMD ["python", "-c", "print('Wrapper container ready. Use: docker exec -it wrapper python scripts/attack_experiment.py'); import time; time.sleep(3600)"]