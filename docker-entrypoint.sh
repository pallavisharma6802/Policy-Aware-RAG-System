#!/bin/bash
set -e

echo "Waiting for PostgreSQL..."
until PGPASSWORD=$POSTGRES_PASSWORD psql -h "$POSTGRES_HOST" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c '\q' 2>/dev/null; do
  sleep 2
done

echo "Waiting for Weaviate..."
until curl -sf "$WEAVIATE_URL/v1/.well-known/ready" > /dev/null; do
  sleep 2
done

echo "Waiting for Ollama..."
until curl -sf "$OLLAMA_HOST/api/tags" > /dev/null; do
  sleep 2
done

if ! curl -s "$OLLAMA_HOST/api/tags" | grep -q "\"name\":\"$OLLAMA_MODEL\""; then
  echo "Pulling $OLLAMA_MODEL..."
  curl -X POST "$OLLAMA_HOST/api/pull" -d "{\"name\":\"$OLLAMA_MODEL\"}"
fi

python -c "from db.session import init_db; init_db()"

RAW_DOCS_DIR="/app/data/raw_docs"
CHUNKS_DIR="/app/data/processed_chunks"

if [ -z "$(ls -A "$RAW_DOCS_DIR"/*_metadata.json 2>/dev/null)" ]; then
  python -m ingestion.load_docs
fi

if [ -z "$(ls -A "$CHUNKS_DIR"/*_chunks.json 2>/dev/null)" ]; then
  python -m ingestion.chunk
fi

python -m ingestion.load_to_db

CHUNK_COUNT=$(curl -s "$WEAVIATE_URL/v1/objects?class=PolicyChunk&limit=1" | grep -o '"totalResults":[0-9]*' | grep -o '[0-9]*' || echo "0")
if [ "$CHUNK_COUNT" -eq "0" ]; then
  python -m ingestion.embed
fi

exec uvicorn api.main:app --host 0.0.0.0 --port 8000
