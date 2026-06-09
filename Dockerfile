# Dockerfile for DCASS Steganography System
# Optimized multi-stage build for minimal image size
#
# Build: docker build -t dcass:latest .
# Size: ~2.5GB (optimized from ~4GB+)

# ============================================================================
# Stage 1: Base with system dependencies
# ============================================================================
FROM python:3.10-slim AS base

# Set environment variables for optimization
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install minimal system dependencies in single layer
# Clean up in same layer to reduce image size
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

# Copy requirements first (better layer caching)
COPY requirements.txt .

# Install Python packages with optimizations
# Using --no-cache-dir to prevent pip cache buildup
RUN pip install --no-cache-dir --upgrade pip wheel setuptools && \
    pip install --no-cache-dir -r requirements.txt

# Install PyTorch CPU-only (significantly smaller than full torch)
# Using index-url for CPU-only wheels
RUN pip install --no-cache-dir \
    torch==2.0.1+cpu \
    torchvision==0.15.2+cpu \
    torchaudio==2.0.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install additional RL/GAN dependencies
RUN pip install --no-cache-dir \
    gymnasium==0.29.1 \
    tensorboard==2.14.0

# Install CLIP (lighter installation)
RUN pip install --no-cache-dir \
    ftfy \
    regex \
    && pip install --no-cache-dir git+https://github.com/openai/CLIP.git@main

# ============================================================================
# Stage 3: Production image (minimal)
# ============================================================================
FROM python:3.10-slim AS production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app:${PYTHONPATH}" \
    # Reduce memory usage
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /app

# Copy only the installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Install runtime-only system dependencies (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create necessary directories with proper permissions
RUN mkdir -p /app/storage/data /app/storage/models /app/storage/checkpoints /app/storage/logs /app/storage/shared_channel \
    && chmod -R 755 /app

# Copy application code
# Use .dockerignore to exclude unnecessary files
COPY src/ /app/src/
COPY scripts/ /app/scripts/
COPY config/ /app/config/
COPY requirements.txt /app/

# Create non-root user for security (optional but recommended)
RUN useradd --create-home --shell /bin/bash dcass && \
    chown -R dcass:dcass /app
# Uncomment to run as non-root:
# USER dcass

# Expose ports
EXPOSE 8888 6006

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; print('OK')" || exit 1

# Default command
CMD ["python", "-u", "src/cli/main.py"]

# ============================================================================
# Stage 4: Development image (with all tools)
# ============================================================================
FROM production AS development

# Switch to root for installing dev tools
USER root

# Install development dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Install dev Python packages
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    black \
    ruff \
    ipython

# Copy all source files for development
COPY . /app/

# Restore user
# USER dcass

CMD ["python", "-u", "src/cli/main.py"]
