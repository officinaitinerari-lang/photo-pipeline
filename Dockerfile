# syntax=docker/dockerfile:1.7

############################ Stage 1: build the ONNX upscaler model ############################
FROM python:3.11-slim AS model-builder
ARG BUILD_UPSCALER_MODEL=true
ARG REALESRGAN_URL=https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY scripts/export_upscaler_onnx.py .
RUN mkdir -p /out && if [ "$BUILD_UPSCALER_MODEL" = "true" ]; then \
      pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
      && pip install --no-cache-dir onnx==1.17.0 "numpy<2.3" \
      && curl -fsSL -o realesr-general-x4v3.pth "$REALESRGAN_URL" \
      && python export_upscaler_onnx.py realesr-general-x4v3.pth /out/realesr-general-x4v3.onnx ; \
    else echo "upscaler model build skipped (Lanczos fallback will be used)"; fi

############################ Stage 2: runtime ############################
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="photo-pipeline" \
      org.opencontainers.image.description="Zero-cost photo pipeline: Etsy digital prints + microstock" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Europe/Rome \
    OMP_NUM_THREADS=2 \
    OPENCV_NUM_THREADS=2 \
    MALLOC_ARENA_MAX=2

RUN apt-get update && apt-get install -y --no-install-recommends \
        libimage-exiftool-perl \
        libglib2.0-0 \
        libgomp1 \
        fonts-dejavu-core \
        tzdata \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Open-source fonts (SIL OFL) for the Gallery Poster typography and the infographic
RUN mkdir -p /app/assets/fonts /app/assets/models \
    && cd /app/assets/fonts \
    && curl -fsSL -o "Cinzel[wght].ttf" "https://raw.githubusercontent.com/google/fonts/main/ofl/cinzel/Cinzel%5Bwght%5D.ttf" \
    && curl -fsSL -o "PlayfairDisplay[wght].ttf" "https://raw.githubusercontent.com/google/fonts/main/ofl/playfairdisplay/PlayfairDisplay%5Bwght%5D.ttf" \
    && curl -fsSL -o "Montserrat[wght].ttf" "https://raw.githubusercontent.com/google/fonts/main/ofl/montserrat/Montserrat%5Bwght%5D.ttf" \
    && curl -fsSL -o "OFL-Cinzel.txt" "https://raw.githubusercontent.com/google/fonts/main/ofl/cinzel/OFL.txt" \
    && curl -fsSL -o "OFL-Montserrat.txt" "https://raw.githubusercontent.com/google/fonts/main/ofl/montserrat/OFL.txt"

COPY --from=model-builder /out/ /app/assets/models/
COPY src/ /app/src/

# Non-root user; UID/GID can be overridden at runtime with `user:` in docker-compose
RUN useradd --uid 1000 --create-home --shell /usr/sbin/nologin pipeline \
    && chown -R pipeline:pipeline /app
USER pipeline

VOLUME ["/data"]
HEALTHCHECK --interval=60s --timeout=10s --start-period=120s --retries=3 \
    CMD ["python", "-m", "src.main", "healthcheck"]

ENTRYPOINT ["python", "-m", "src.main"]
CMD ["run"]
