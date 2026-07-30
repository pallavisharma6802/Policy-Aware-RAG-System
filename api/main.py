import json
import os
import sys
import threading
import time
from collections import Counter
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import text
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from api.models import (
    CitationResponse,
    EvalSetMeta,
    EvalStatusResponse,
    HealthResponse,
    QueryHistoryResponse,
    QueryLogEntry,
    QueryRequest,
    QueryResponse,
)
from app.generation import generate_policy_response
from app.analytics import log_query_event
from db.session import engine

load_dotenv()

app = FastAPI(
    title="Policy-Aware RAG System",
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

QUERY_LOG_PATH = Path("data/query_log.jsonl")
EVAL_SET_PATH = Path("data/eval/eval_set.json")
EVAL_RESULTS_PATH = Path("data/eval/eval_results.json")
EVAL_STATUS_PATH = Path("data/eval/eval_status.json")

_eval_lock = threading.Lock()


def _append_query_log(entry: dict) -> None:
    QUERY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(QUERY_LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")
    log_query_event(entry)


def _read_query_log(limit: int = 100) -> list:
    if not QUERY_LOG_PATH.exists():
        return []
    lines = QUERY_LOG_PATH.read_text().strip().splitlines()
    entries = []
    for line in reversed(lines[-limit:]):
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return entries[:limit]


def _run_eval_background(smoke: bool = False) -> None:
    try:
        from scripts.run_evaluation import run_evaluation
        run_evaluation(smoke=smoke)
    except Exception as exc:
        EVAL_STATUS_PATH.write_text(json.dumps({
            "status": "error",
            "progress": 0.0,
            "message": str(exc),
        }))


@app.get("/", include_in_schema=False)
async def root():
    html_file = static_dir / "index.html"
    if html_file.exists():
        return FileResponse(html_file)
    return {"message": "Policy RAG API. See /docs."}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    health = {"status": "healthy", "database": "unknown", "vector_db": "unknown", "llm": "unknown"}

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        health["database"] = "connected"
    except Exception as exc:
        health["database"] = f"error: {exc}"
        health["status"] = "degraded"

    try:
        import weaviate
        weaviate_url = os.getenv("WEAVIATE_URL", "http://localhost:8080")
        client = weaviate.Client(url=weaviate_url, timeout_config=(5, 10))
        client.schema.get()
        health["vector_db"] = "connected"
    except Exception as exc:
        health["vector_db"] = f"error: {exc}"
        health["status"] = "degraded"

    try:
        import requests
        ollama_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        r = requests.get(f"{ollama_url}/api/tags", timeout=2)
        health["llm"] = "connected" if r.status_code == 200 else "unreachable"
        if r.status_code != 200:
            health["status"] = "degraded"
    except Exception as exc:
        health["llm"] = f"error: {exc}"
        health["status"] = "degraded"

    return HealthResponse(**health)


@app.post("/query", response_model=QueryResponse)
async def query_policy(request: QueryRequest):
    try:
        response = await run_in_threadpool(
            generate_policy_response,
            query=request.query,
            limit=request.limit,
            region=request.region,
            content_type=request.content_type,
            policy_source=request.policy_source,
        )

        citations = [
            CitationResponse(
                chunk_id=c.chunk_id,
                policy_path=c.policy_path,
                doc_id=c.doc_id,
                doc_url=c.doc_url,
                score=c.score,
                chunk_text=c.chunk_text,
            )
            for c in response.citations
        ]

        _append_query_log({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "query": request.query,
            "refused": response.refused,
            "latency_ms": response.latency_ms,
            "retrieval_ms": response.retrieval_ms,
            "generation_ms": response.generation_ms,
            "num_citations": len(citations),
            "num_tokens_generated": response.num_tokens_generated,
            "model": response.model,
        })

        return QueryResponse(
            answer=response.answer,
            refused=response.refused,
            citations=citations,
            refusal_reason=response.refusal_reason,
            latency_ms=response.latency_ms,
            retrieval_ms=response.retrieval_ms,
            generation_ms=response.generation_ms,
            num_tokens_generated=response.num_tokens_generated,
            model=response.model,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Internal processing error: {exc}")


@app.get("/history", response_model=QueryHistoryResponse)
async def query_history(limit: int = 50):
    raw = _read_query_log(limit=limit)
    entries = [QueryLogEntry(**{k: e.get(k) for k in QueryLogEntry.model_fields}) for e in raw]
    return QueryHistoryResponse(entries=entries, total=len(entries))


@app.get("/eval", response_model=EvalSetMeta)
async def eval_meta():
    if not EVAL_SET_PATH.exists():
        raise HTTPException(status_code=404, detail="Eval set not found.")
    data = json.loads(EVAL_SET_PATH.read_text())
    cats = Counter(item["category"] for item in data)
    qtypes = Counter(item["question_type"] for item in data)
    return EvalSetMeta(
        total=len(data),
        answerable=sum(1 for i in data if not i["expected_refused"]),
        refusal_expected=sum(1 for i in data if i["expected_refused"]),
        categories=dict(cats),
        question_types=dict(qtypes),
    )


@app.post("/eval/run", response_model=EvalStatusResponse)
async def eval_run(background_tasks: BackgroundTasks, smoke: bool = False):
    with _eval_lock:
        status_data = {}
        if EVAL_STATUS_PATH.exists():
            try:
                status_data = json.loads(EVAL_STATUS_PATH.read_text())
            except Exception:
                pass
        if status_data.get("status") == "running":
            return EvalStatusResponse(
                status="running",
                progress=status_data.get("progress", 0),
                message="Evaluation already in progress.",
            )

        EVAL_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
        EVAL_STATUS_PATH.write_text(json.dumps({
            "status": "running",
            "progress": 0.0,
            "message": "Starting…",
        }))

    background_tasks.add_task(_run_eval_background, smoke)
    return EvalStatusResponse(
        status="running",
        progress=0.0,
        message=f"Evaluation started ({'smoke' if smoke else 'full'}).",
    )


@app.get("/eval/status", response_model=EvalStatusResponse)
async def eval_status():
    if not EVAL_STATUS_PATH.exists():
        return EvalStatusResponse(status="idle", progress=0.0, message="No evaluation has been run yet.")

    try:
        status_data = json.loads(EVAL_STATUS_PATH.read_text())
    except Exception:
        return EvalStatusResponse(status="idle", progress=0.0, message="")

    results = None
    if status_data.get("status") == "complete" and EVAL_RESULTS_PATH.exists():
        try:
            full = json.loads(EVAL_RESULTS_PATH.read_text())
            results = {
                "aggregate": full["aggregate"],
                "ran_at": full.get("ran_at"),
                "num_items": full.get("num_items"),
                "mode": full.get("mode"),
                "model": full.get("model"),
            }
        except Exception:
            pass

    return EvalStatusResponse(
        status=status_data.get("status", "idle"),
        progress=status_data.get("progress", 0.0),
        message=status_data.get("message", ""),
        results=results,
    )


@app.get("/eval/results")
async def eval_results():
    if not EVAL_RESULTS_PATH.exists():
        raise HTTPException(status_code=404, detail="No evaluation results found. Run POST /eval/run first.")
    return json.loads(EVAL_RESULTS_PATH.read_text())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
