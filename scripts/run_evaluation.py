from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

sys.path.append(str(Path(__file__).parent.parent))

from app.retrieval import retrieve_policy_chunks
from app.generation import generate_policy_response, OLLAMA_MODEL
from app.metrics import (
    compute_hit_at_k,
    compute_mrr,
    semantic_similarity,
    compute_faithfulness,
    compute_citation_precision,
    compute_aggregate_metrics,
)
from app.analytics import log_eval_event
from app.citations import extract_citations

EVAL_SET_PATH = Path("data/eval/eval_set.json")
RESULTS_PATH = Path("data/eval/eval_results.json")
STATUS_PATH = Path("data/eval/eval_status.json")
SMOKE_SIZE = 16


def _write_status(status: str, progress: float = 0.0, message: str = "") -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps({
        "status": status,
        "progress": progress,
        "message": message,
    }))


def select_smoke_subset(eval_set: List[Dict], n: int = SMOKE_SIZE) -> List[Dict]:
    answerable = [i for i in eval_set if not i["expected_refused"]]
    refusals = [i for i in eval_set if i["expected_refused"]]
    refusal_n = max(1, round(n * len(refusals) / len(eval_set)))
    answerable_n = n - refusal_n

    by_cat: Dict[str, List[Dict]] = defaultdict(list)
    for item in answerable:
        by_cat[item["category"]].append(item)

    selected: List[Dict] = []
    cats = sorted(by_cat.keys())
    if cats and answerable_n > 0:
        per_cat = max(1, answerable_n // len(cats))
        for cat in cats:
            selected.extend(by_cat[cat][:per_cat])
        selected = selected[:answerable_n]

    selected.extend(refusals[:refusal_n])

    seen = {i["id"] for i in selected}
    for item in eval_set:
        if len(selected) >= n:
            break
        if item["id"] not in seen:
            selected.append(item)
            seen.add(item["id"])

    return selected[:n]


def run_evaluation(
    progress_callback=None,
    *,
    smoke: bool = False,
    limit: Optional[int] = None,
) -> dict:
    _write_status("running", 0.0, "Loading eval set…")

    eval_set = json.loads(EVAL_SET_PATH.read_text())
    mode = "full"
    if smoke:
        eval_set = select_smoke_subset(eval_set, SMOKE_SIZE)
        mode = "smoke"
    elif limit is not None:
        eval_set = eval_set[:limit]
        mode = f"limit_{limit}"

    total = len(eval_set)
    per_query_results = []

    for i, item in enumerate(eval_set):
        pct = i / total
        msg = f"[{i+1}/{total}] {item['question'][:55]}…"
        _write_status("running", pct, msg)
        if progress_callback:
            progress_callback(pct, msg)

        retrieval_start = time.time()
        try:
            retrieval_results = retrieve_policy_chunks(item["question"], limit=3)
        except Exception:
            retrieval_results = []
        retrieval_ms = (time.time() - retrieval_start) * 1000

        retrieved_paths = [r["policy_path"] for r in retrieval_results]
        retrieved_ids = {r["chunk_id"] for r in retrieval_results}
        source_paths = item.get("source_policy_paths", [])

        if not item["expected_refused"] and source_paths:
            hit1 = compute_hit_at_k(retrieved_paths, source_paths, 1)
            hit3 = compute_hit_at_k(retrieved_paths, source_paths, 3)
            hit5 = compute_hit_at_k(retrieved_paths, source_paths, 5)
            mrr = compute_mrr(retrieved_paths, source_paths)
        else:
            hit1 = hit3 = hit5 = mrr = None

        try:
            response = generate_policy_response(
                item["question"],
                limit=3,
                retrieved_results=retrieval_results,
            )
        except Exception as exc:
            per_query_results.append({
                "id": item["id"],
                "question": item["question"],
                "question_type": item["question_type"],
                "category": item["category"],
                "difficulty": item["difficulty"],
                "expected_refused": item["expected_refused"],
                "actual_refused": True,
                "error": str(exc),
                "hit_at_1": hit1, "hit_at_3": hit3, "hit_at_5": hit5,
                "mrr": mrr,
                "answer_similarity": None,
                "faithfulness": None,
                "citation_precision": None,
                "latency_ms": None,
                "retrieval_ms": retrieval_ms,
                "generation_ms": None,
                "num_citations": 0,
            })
            continue

        answer_sim = None
        faithfulness = None
        citation_precision = None
        reference = item.get("reference_answer")

        if not response.refused and not item["expected_refused"]:
            if reference and response.answer:
                answer_sim = round(semantic_similarity(reference, response.answer), 3)
            if retrieval_results and response.answer:
                context_chunks = [r["chunk_text"] for r in retrieval_results]
                faithfulness = round(compute_faithfulness(response.answer, context_chunks), 3)
            cited = extract_citations(response.answer)
            citation_precision = compute_citation_precision(cited, retrieved_ids)
            if citation_precision is not None:
                citation_precision = round(citation_precision, 3)

        per_query_results.append({
            "id": item["id"],
            "question": item["question"],
            "question_type": item["question_type"],
            "category": item["category"],
            "difficulty": item["difficulty"],
            "expected_refused": item["expected_refused"],
            "actual_refused": response.refused,
            "refusal_reason": response.refusal_reason,
            "hit_at_1": hit1,
            "hit_at_3": hit3,
            "hit_at_5": hit5,
            "mrr": mrr,
            "answer_similarity": answer_sim,
            "faithfulness": faithfulness,
            "citation_precision": citation_precision,
            "latency_ms": response.latency_ms,
            "retrieval_ms": response.retrieval_ms if response.retrieval_ms else retrieval_ms,
            "generation_ms": response.generation_ms,
            "num_citations": len(response.citations),
            "model": response.model,
        })

    aggregate = compute_aggregate_metrics(per_query_results)

    results = {
        "ran_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "num_items": total,
        "mode": mode,
        "model": os.getenv("OLLAMA_MODEL", OLLAMA_MODEL),
        "aggregate": aggregate,
        "per_query": per_query_results,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    _write_status("complete", 1.0, f"Done — {total} questions evaluated ({mode}).")

    log_eval_event({
        "run_name": f"policy-rag-{mode}",
        "mode": mode,
        "model": results["model"],
        "num_items": total,
        "ran_at": results["ran_at"],
        "aggregate": aggregate,
    })

    try:
        from dbx import databricks_configured, sync_eval_run, log_eval_to_mlflow
        if databricks_configured():
            sync_eval_run(results)
            log_eval_to_mlflow(results)
    except Exception:
        pass

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run policy RAG evaluation")
    parser.add_argument("--smoke", action="store_true", help="Run stratified smoke subset")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate first N items")
    args = parser.parse_args()

    def _print(pct, msg):
        print(f"  {int(pct*100):3d}%  {msg}")

    print(f"Starting evaluation ({'smoke' if args.smoke else 'full'})…\n")
    results = run_evaluation(progress_callback=_print, smoke=args.smoke, limit=args.limit)
    print("\n=== AGGREGATE METRICS ===")
    print(json.dumps(results["aggregate"], indent=2))
    print(f"\nResults saved to {RESULTS_PATH}")
