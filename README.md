# Policy-Aware RAG

Answers Google Ads policy questions with retrieved sources and citations. Refuses when the corpus doesn’t support an answer.

## Architecture

```
Policy docs → chunk → Postgres (metadata) + Weaviate (vectors)
                              ↓
Query → hybrid retrieve (k=3) → Ollama (qwen3:1.7b) → cite / refuse
                              ↓
                    JSONL logs → optional Databricks Delta + MLflow
```

| Piece | What |
|---|---|
| Corpus | 5 Google Ads policy pages → 67 chunks |
| Retrieve | MiniLM embeddings, Weaviate near-vector, Postgres filters, H2/H3 rerank |
| Generate | Local Ollama; `[SOURCE:N]` citations validated against retrieved chunks |
| API / UI | FastAPI on `:8000` (query, history, eval) |
| Eval | `data/eval/eval_set.json` (80 Qs); `python -m scripts.run_evaluation [--smoke]` |
| Databricks | Optional: sync events, eval runs, and chunks to Delta (`dbx/`) |

## Run

```bash
cp .env.docker .env
ollama pull qwen3:1.7b
docker compose up -d
# http://localhost:8000
```

```bash
pytest tests/ -q
python -m scripts.run_evaluation --smoke
```

## Databricks (optional)

```bash
pip install -r requirements-databricks.txt
# set DATABRICKS_HOST, DATABRICKS_TOKEN, DATABRICKS_HTTP_PATH in .env
python -m scripts.sync_databricks all
```

## Layout

```
app/          retrieval, generation, citations, metrics, analytics
api/          FastAPI + UI
ingestion/    scrape → chunk → Postgres → Weaviate
db/           SQLAlchemy models
dbx/          Databricks sync
scripts/      eval + Databricks CLI
data/eval/    golden set
tests/
```
