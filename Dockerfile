# Dockerfile for DCASS Backend API & Engine
# Multi-stage CPU build aligned with Python 3.11 CI

# ============================================================================
# Stage 1: Base with system dependencies
# ============================================================================
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && rm -rf /tmp/* /var/tmp/*

# ============================================================================
# Stage 2: Dependencies builder
# ============================================================================
FROM base AS builder

COPY requirements.txt .

# Install CPU-only PyTorch first so downstream requirements do not pull CUDA wheels,
# then install runtime requirements (filtering dev/notebook packages) and CLIP.
RUN pip install --no-cache-dir --upgrade pip wheel setuptools && \
    pip install --no-cache-dir \
        torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu && \
    grep -vE '^(torch|torchvision|jupyter|ipykernel|pytest|pytest-cov|black|ruff)([<>=!~ ]|$)' \
        requirements.txt > /tmp/requirements-runtime.txt && \
    pip install --no-cache-dir -r /tmp/requirements-runtime.txt \
        --extra-index-url https://download.pytorch.org/whl/cpu && \
    rm -f /tmp/requirements-runtime.txt && \
    pip install --no-cache-dir ftfy regex && \
    pip install --no-cache-dir --no-deps git+https://github.com/openai/CLIP.git@main && \
    python -c "import nltk; nltk.download('punkt', download_dir='/usr/local/share/nltk_data', quiet=True); nltk.download('punkt_tab', download_dir='/usr/local/share/nltk_data', quiet=True)"

# ============================================================================
# Stage 3: Production image
# ============================================================================
FROM python:3.11-slim AS production

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app" \
    NLTK_DATA="/usr/local/share/nltk_data" \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    DCASS_DB_PATH="/app/storage/db/auth.db"

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /usr/local/share/nltk_data /usr/local/share/nltk_data

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN mkdir -p /app/storage/data /app/storage/models /app/storage/db /app/storage/checkpoints /app/storage/logs /app/storage/shared_channel && \
    chmod -R 755 /app

COPY src/ /app/src/
COPY scripts/ /app/scripts/
COPY requirements.txt /app/

RUN useradd --create-home --shell /bin/bash dcass && \
    chown -R dcass:dcass /app
# Optional: enable when host bind-mount permissions match the dcass UID.
# USER dcass

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)" || exit 1

# Single worker: in-memory transmission session state and _engine_work_lock live per-process.
CMD ["python", "-m", "uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# ============================================================================
# Stage 4: Development image
# ============================================================================
FROM production AS development

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    black \
    ruff

COPY . /app/

CMD ["python", "-m", "uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
