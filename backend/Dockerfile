FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 CACHE_DIR=/data

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 app && mkdir -p /data && chown app:app /data

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# Learned style model + mood heads (Essentia Discogs-EffNet and its classification heads, CC BY-NC-SA 4.0,
# see README): fetched at build time, each pinned by sha256 in app/audio/embed.py (the only source of truth;
# the script needs only numpy, installed above). EMBEDDINGS=off disables them at runtime.
ENV EMBEDDING_MODEL_PATH=/opt/models/discogs-effnet-bsdynamic-1.onnx
COPY app/audio/embed.py /tmp/fetch_models.py
RUN python /tmp/fetch_models.py download "$EMBEDDING_MODEL_PATH" \
    && chmod 444 /opt/models/*.onnx && rm /tmp/fetch_models.py

COPY app ./app
USER app
EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
