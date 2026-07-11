# AI Bug Triage & Release Operator — FastAPI service with bundled Ollama
# embeddings (free, local, no rate limits). Ollama runs inside this same
# container, so the app talks to it at localhost — no separate service needed.
FROM python:3.12-slim

# Faster, cleaner Python in containers
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps: curl + ca-certificates are needed to install Ollama and to
# health-check it from the entrypoint.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates zstd\
    && rm -rf /var/lib/apt/lists/*

# Install Ollama (the local embedding engine).
RUN curl -fsSL https://ollama.com/install.sh | sh

# Python dependencies first so this layer is cached across code changes
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Bake the embedding model INTO the image so startup is instant and needs no
# network. Start the server briefly, pull the model, then stop it.
RUN ollama serve & \
    until curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; do sleep 1; done && \
    ollama pull nomic-embed-text && \
    pkill -f "ollama serve" || true

# App source + entrypoint
COPY app ./app
COPY docker-entrypoint.sh ./docker-entrypoint.sh
RUN chmod +x ./docker-entrypoint.sh

# Qdrant local mode + ingested data live here; mounted as a volume in compose
RUN mkdir -p data
ENV DATA_DIR=/app/data \
    QDRANT_PATH=/app/data/qdrant

# Use the bundled Ollama for embeddings (overridable at runtime).
ENV EMBEDDING_PROVIDER=ollama \
    OLLAMA_BASE_URL=http://localhost:11434/v1 \
    OLLAMA_EMBED_MODEL=nomic-embed-text

EXPOSE 8077

# Entrypoint starts Ollama, waits for it, then runs uvicorn (bound to 0.0.0.0).
CMD ["./docker-entrypoint.sh"]
