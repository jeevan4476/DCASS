# Dockerfile for DCASS Steganography System
FROM python:3.10-slim AS base

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

FROM base AS dependencies

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    gymnasium \
    tensorboard

FROM dependencies AS final

COPY . /app

ENV PYTHONPATH="/app"

RUN mkdir -p /app/data \
    /app/checkpoints \
    /app/logs \
    /app/shared_channel

EXPOSE 8888 6006

CMD ["python", "-u", "run_gan_demo.py"]