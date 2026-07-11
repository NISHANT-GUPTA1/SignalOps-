#!/bin/sh
# Start the bundled Ollama server (used for free, local, rate-limit-free
# embeddings), wait until it's ready, then launch the FastAPI app.
# The app probes the embedding dimension at startup, so Ollama must be up first.
set -e

echo "[entrypoint] starting Ollama server..."
ollama serve &

echo "[entrypoint] waiting for Ollama to accept connections..."
until curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; do
  sleep 1
done
echo "[entrypoint] Ollama ready. Launching app."

exec uvicorn app.main:app --host 0.0.0.0 --port 8077
