"""
Evaluation runner for the Policy-Aware RAG System.

Runs the full eval set (data/eval/eval_set.json) through the RAG pipeline and
computes retrieval + generation metrics, saving results to data/eval/eval_results.json.

Usage:
    python -m scripts.run_evaluation
"""

import json
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from app.retrieval import retrieve_policy_chunks
from app.generation import generate_policy_response
from app.metrics import (
    compute_hit_at_k,
    compute_mrr,
    semantic_similarity,
    compute_faithfulness,
    compute_aggregate_metrics,
)

EVAL_SET_PATH  = Path("data/eval/eval_set.json")
RESULTS_PATH   = Path("data/eval/eval_results.json")
STATUS_PATH    = Path("data/eval/eval_status.json")


def _write_status(status: str, progress: float = 0.0, message: str = "") -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps({
        "status":   status,
        "progress": progress,
        "message":  message,
    }))


def run_evaluation(progress_callback=None) -> dict:
    _write_status("running", 0.0, "Loading eval set…")

    eval_set = json.loads(EVAL_SET_PATH.read_text())
    total    = len(eval_set)
    per_query_results = []

    for i, item in enumerate(eval_set):
        pct = i / total
        msg = f"[{i+1}/{total}] {item['question'][:55]}…"
        _write_status("running", pct, msg)
        if progress_callback:
            progress_callback(pct, msg)

        # ── Retrieval ────────────────────────────────────────────────────────
        try:
            retrieval_results = retrieve_policy_chunks(item["question"], limit=3)
        except Exception:
            retrieval_results = []

        retrieved_paths = [r["policy_path"] for r in retrieval_results]
        source_paths    = item.get("source_policy_paths", [])

        if not item["expected_refused"] and source_paths:
            hit1 = compute_hit_at_k(retrieved_paths, source_paths, 1)
            hit3 = compute_hit_at_k(retrieved_paths, source_paths, 3)
            hit5 = compute_hit_at_k(retrieved_paths, source_paths, 5)
            mrr  = compute_mrr(retrieved_paths, source_paths)
        else:
            hit1 = hit3 = hit5 = mrr = None

        # ── Generation ───────────────────────────────────────────────────────
        try:
            response = generate_policy_response(item["question"], limit=3, retrieved_results=retrieval_results)
        except Exception as exc:
            per_query_results.append({
                "id":              item["id"],
                "question":        item["question"],
                "question_type":   item["question_type"],
                "category":        item["category"],
                "difficulty":      item["difficulty"],
                "expected_refused": item["expected_refused"],
                "actual_refused":  True,
                "error":           str(exc),
                "hit_at_1": hit1, "hit_at_3": hit3, "hit_at_5": hit5,
                "mrr": mrr,
                "answer_similarity": None,
                "faithfulness":      None,
                "latency_ms":        None,
                "num_citations":     0,
            })
            continue

        # ── Quality metrics ──────────────────────────────────────────────────
        answer_sim  = None
        faithfulness = None
        reference   = item.get("reference_answer")

        if not response.refused and not item["expected_refused"]:
            if reference and response.answer:
                answer_sim = round(semantic_similarity(reference, response.answer), 3)
            if retrieval_results and response.answer:
                context_chunks = [r["chunk_text"] for r in retrieval_results]
                faithfulness   = round(compute_faithfulness(response.answer, context_chunks), 3)

        per_query_results.append({
            "id":               item["id"],
            "question":         item["question"],
            "question_type":    item["question_type"],
            "category":         item["category"],
            "difficulty":       item["difficulty"],
            "expected_refused": item["expected_refused"],
            "actual_refused":   response.refused,
            "refusal_reason":   response.refusal_reason,
            "hit_at_1":         hit1,
            "hit_at_3":         hit3,
            "hit_at_5":         hit5,
            "mrr":              mrr,
            "answer_similarity": answer_sim,
            "faithfulness":      faithfulness,
            "latency_ms":        response.latency_ms,
            "num_citations":     len(response.citations),
        })

    aggregate = compute_aggregate_metrics(per_query_results)

    results = {
        "ran_at":     time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "num_items":  total,
        "aggregate":  aggregate,
        "per_query":  per_query_results,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    _write_status("complete", 1.0, f"Done — {total} questions evaluated.")

    return results


if __name__ == "__main__":
    def _print(pct, msg):
        print(f"  {int(pct*100):3d}%  {msg}")

    print("Starting evaluation…\n")
    results = run_evaluation(progress_callback=_print)
    print("\n=== AGGREGATE METRICS ===")
    print(json.dumps(results["aggregate"], indent=2))
    print(f"\nResults saved to {RESULTS_PATH}")
