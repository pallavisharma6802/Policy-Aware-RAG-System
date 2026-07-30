import numpy as np
from typing import List, Dict, Optional, Set
from app.embeddings import get_embedding_model


def _cosine_sim(v1: List[float], v2: List[float]) -> float:
    a, b = np.array(v1), np.array(v2)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def semantic_similarity(text1: str, text2: str) -> float:
    model = get_embedding_model()
    emb1 = model.encode(text1).tolist()
    emb2 = model.encode(text2).tolist()
    return _cosine_sim(emb1, emb2)


def compute_hit_at_k(retrieved_paths: List[str], source_paths: List[str], k: int) -> bool:
    top_k = retrieved_paths[:k]
    for source in source_paths:
        source_lower = source.lower()
        for retrieved in top_k:
            retrieved_lower = retrieved.lower()
            if source_lower in retrieved_lower or retrieved_lower in source_lower:
                return True
    return False


def compute_mrr(retrieved_paths: List[str], source_paths: List[str]) -> float:
    for rank, path in enumerate(retrieved_paths, 1):
        path_lower = path.lower()
        for source in source_paths:
            source_lower = source.lower()
            if source_lower in path_lower or path_lower in source_lower:
                return 1.0 / rank
    return 0.0


def compute_faithfulness(answer: str, context_chunks: List[str]) -> float:
    if not answer or not context_chunks:
        return 0.0
    context = " ".join(context_chunks[:3])
    return semantic_similarity(answer, context)


def compute_citation_precision(cited_ids: Set[str], retrieved_ids: Set[str]) -> Optional[float]:
    if not cited_ids:
        return None
    return len(cited_ids & retrieved_ids) / len(cited_ids)


def compute_refusal_f1(per_query_results: List[Dict]) -> float:
    tp = fp = fn = 0
    for r in per_query_results:
        expected = bool(r.get("expected_refused"))
        actual = bool(r.get("actual_refused"))
        if actual and expected:
            tp += 1
        elif actual and not expected:
            fp += 1
        elif not actual and expected:
            fn += 1
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 3)


def compute_aggregate_metrics(per_query_results: List[Dict]) -> Dict:
    answerable = [r for r in per_query_results if not r["expected_refused"]]
    refusal_expected = [r for r in per_query_results if r["expected_refused"]]

    def _mean_metric(key, rows=None):
        rows = rows if rows is not None else answerable
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(float(np.mean(vals)), 3) if vals else 0.0

    correct_refused = sum(1 for r in refusal_expected if r.get("actual_refused"))
    correct_answered = sum(1 for r in answerable if not r.get("actual_refused"))
    false_refusals = sum(1 for r in answerable if r.get("actual_refused"))
    false_answers = sum(1 for r in refusal_expected if not r.get("actual_refused"))

    refusal_accuracy = correct_refused / len(refusal_expected) if refusal_expected else 0.0
    answer_accuracy = correct_answered / len(answerable) if answerable else 0.0
    false_refusal_rate = false_refusals / len(answerable) if answerable else 0.0
    false_answer_rate = false_answers / len(refusal_expected) if refusal_expected else 0.0

    sim_vals = [
        r["answer_similarity"]
        for r in answerable
        if not r.get("actual_refused") and r.get("answer_similarity") is not None
    ]
    faith_vals = [
        r["faithfulness"]
        for r in answerable
        if not r.get("actual_refused") and r.get("faithfulness") is not None
    ]
    cite_vals = [
        r["citation_precision"]
        for r in answerable
        if not r.get("actual_refused") and r.get("citation_precision") is not None
    ]

    latencies = [r["latency_ms"] for r in per_query_results if r.get("latency_ms")]
    retrieval_lat = [r["retrieval_ms"] for r in per_query_results if r.get("retrieval_ms") is not None]
    generation_lat = [r["generation_ms"] for r in per_query_results if r.get("generation_ms") is not None]

    def _pct(vals, p):
        return round(float(np.percentile(vals, p)), 1) if vals else 0.0

    return {
        "total_questions": len(per_query_results),
        "answerable_questions": len(answerable),
        "refusal_expected_questions": len(refusal_expected),
        "retrieval": {
            "hit_at_1": _mean_metric("hit_at_1"),
            "hit_at_3": _mean_metric("hit_at_3"),
            "hit_at_5": _mean_metric("hit_at_5"),
            "mrr": _mean_metric("mrr"),
        },
        "generation": {
            "refusal_accuracy": round(refusal_accuracy, 3),
            "answer_accuracy": round(answer_accuracy, 3),
            "false_refusal_rate": round(false_refusal_rate, 3),
            "false_answer_rate": round(false_answer_rate, 3),
            "refusal_f1": compute_refusal_f1(per_query_results),
            "avg_answer_semantic_similarity": round(float(np.mean(sim_vals)), 3) if sim_vals else 0.0,
            "avg_faithfulness": round(float(np.mean(faith_vals)), 3) if faith_vals else 0.0,
            "avg_citation_precision": round(float(np.mean(cite_vals)), 3) if cite_vals else 0.0,
        },
        "latency": {
            "avg_ms": round(float(np.mean(latencies)), 1) if latencies else 0.0,
            "p50_ms": _pct(latencies, 50),
            "p95_ms": _pct(latencies, 95),
            "retrieval_p50_ms": _pct(retrieval_lat, 50),
            "generation_p50_ms": _pct(generation_lat, 50),
        },
    }
