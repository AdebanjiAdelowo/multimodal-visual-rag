# API + UI image. By default it installs no neural models and serves the
# deterministic mock generator, so it builds quickly and runs anywhere.
# Build with --build-arg EXTRAS="app,models" for real retrieval and generation (CPU).
FROM python:3.12-slim

ARG EXTRAS="app"
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface

RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
# The CPU wheel index keeps the image free of CUDA libraries.
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu ".[${EXTRAS}]"
COPY configs ./configs
COPY app ./app

RUN useradd --create-home appuser \
    && mkdir -p /app/data/processed /app/uploads /app/artifacts /cache \
    && chown -R appuser /app /cache
USER appuser

EXPOSE 8000 8501
CMD ["uvicorn", "mmrag.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
