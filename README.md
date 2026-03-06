# Policy-Aware RAG System for Ads & Content Moderation

A production-ready RAG system that answers Google Ads policy questions using hybrid search, local LLM inference, and citation-backed responses — with explicit refusal logic to prevent hallucinations.

## Tech Stack

| Layer | Tools |
|---|---|
| Vector Search | Weaviate 1.23 (semantic) + PostgreSQL 15 (metadata filtering) |
| Embeddings | all-MiniLM-L6-v2 (384-dim) |
| LLM | Ollama + Qwen3 4B (~2.5GB, local inference) |
| API | FastAPI + Uvicorn |
| Orchestration | LangChain |
| Infrastructure | Docker Compose (4-service stack) |

## How It Works
```
Query → Hybrid retrieval (Weaviate + PostgreSQL) → Qwen3 generation → Citation extraction → Response
                                                         ↓
                                           Refusal if sources insufficient
```

**Key design decisions:**
- Hybrid retrieval: vector similarity + SQL metadata filters for precision
- Refusal logic: explicitly declines when retrieved chunks don't support an answer
- Section-specific URLs auto-extracted from policy docs (no hardcoding)
- Full audit trail: every response includes chunk IDs, policy paths, and source URLs

## Sample Response
```json
{
  "answer": "Alcohol advertising is allowed but requires certification...",
  "refused": false,
  "citations": [{
    "chunk_id": "google_ads_overview_chunk_005",
    "policy_path": "Prohibited Content > Alcohol",
    "doc_url": "https://support.google.com/adspolicy/answer/6012382"
  }],
  "latency_ms": 2543.2,
  "num_tokens_generated": 87
}
```

Refusal example (by design): "What products are allowed to advertise?" — policies describe restrictions not allowances, so the system refuses rather than guessing.

## Quick Start
```bash
cp .env.docker .env
docker-compose up -d
# First run: 5-10 min to pull Qwen3 4B
# App at http://localhost:8000 | Docs at http://localhost:8000/docs
```

Startup sequence is fully automated: waits for all services, pulls model if missing, runs ingestion pipeline, then starts the server.

## Tests

90 tests, 100% passing across retrieval, generation guardrails, API, edge cases, and integration.
```bash
pytest tests/ -v
pytest tests/ --cov --cov-report=html
```

## Project Structure
```
├── ingestion/       # load_docs → chunk → embed pipeline
├── app/             # retrieval, generation, citations, schemas
├── api/             # FastAPI routes + vanilla HTML/CSS/JS UI
├── db/              # SQLAlchemy models + session
├── tests/           # 90 tests across 9 files
└── docker-compose.yml
```
