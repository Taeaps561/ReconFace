# ==============================================================================
# Base Python Stage
# ==============================================================================
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

# Install essential system dependencies for OpenCV and ONNX runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python build dependencies & project dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip hatchling && \
    pip install --no-cache-dir .

# Copy application source code
COPY . .

# ==============================================================================
# API Server Stage
# ==============================================================================
FROM base AS api

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ==============================================================================
# Celery Worker Stage
# ==============================================================================
FROM base AS worker

CMD ["celery", "-A", "app.workers.celery_app.celery_app", "worker", "--loglevel=info", "-Q", "ingest,reconface"]
