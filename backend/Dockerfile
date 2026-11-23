FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 CACHE_DIR=/data

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 app && mkdir -p /data && chown app:app /data

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# Learned style model (Essentia Discogs-EffNet, CC BY-NC-SA 4.0, see README): fetched at build time,
# pinned by sha256 (keep in sync with app/audio/embed.py). EMBEDDINGS=off disables it at runtime.
ARG EMBEDDING_MODEL_URL=https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs-effnet-bsdynamic-1.onnx
ARG EMBEDDING_MODEL_SHA256=a280825b334797cf677939db8cd5762c0392aedd0ca6415dbc1cd083f045e43c
ENV EMBEDDING_MODEL_PATH=/opt/models/discogs-effnet-bsdynamic-1.onnx
RUN mkdir -p /opt/models \
    && python -c "import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])" \
       "$EMBEDDING_MODEL_URL" "$EMBEDDING_MODEL_PATH" \
    && echo "$EMBEDDING_MODEL_SHA256  $EMBEDDING_MODEL_PATH" | sha256sum -c - \
    && chmod 444 "$EMBEDDING_MODEL_PATH"

COPY app ./app
USER app
EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
