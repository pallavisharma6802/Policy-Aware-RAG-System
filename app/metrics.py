import numpy as np
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer

_model: Optional[SentenceTransformer] = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model


def _cosine_sim(v1: List[float], v2: List[float]) -> float:
    a, b = np.array(v1), np.array(v2)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def semantic_similarity(text1: str, text2: str) -> float:
    """Cosine similarity between two texts using the shared embedding model."""
    model = _get_model()
    emb1 = model.encode(text1).tolist()
    emb2 = model.encode(text2).tolist()
    return _cosine_sim(emb1, emb2)


def compute_hit_at_k(retrieved_paths: List[str], source_paths: List[str], k: int) -> bool:
    """True if any source_path appears (substring match) in the top-k retrieved paths."""
    top_k = retrieved_paths[:k]
    for source in source_paths:
        source_lower = source.lower()
        for retrieved in top_k:
            retrieved_lower = retrieved.lower()
            if source_lower in retrieved_lower or retrieved_lower in source_lower:
                return True
    return False


def compute_mrr(retrieved_paths: List[str], source_paths: List[str]) -> float:
    """Mean Reciprocal Rank: 1/rank of first matching retrieved path."""
    for rank, path in enumerate(retrieved_paths, 1):
        path_lower = path.lower()
        for source in source_paths:
            source_lower = source.lower()
            if source_lower in path_lower or path_lower in source_lower:
                return 1.0 / rank
    return 0.0


def compute_faithfulness(answer: str, context_chunks: List[str]) -> float:
    """
    Embedding-based faithfulness: cosine similarity between the answer and
    the concatenated retrieved context. Proxy for how grounded the answer is.
    """
    if not answer or not context_chunks:
        return 0.0
    context = " ".join(context_chunks[:3])
    return semantic_similarity(answer, context)


def compute_aggregate_metrics(per_query_results: List[Dict]) -> Dict:
    """Compute all aggregate metrics from per-query evaluation results."""
    answerable = [r for r in per_query_results if not r["expected_refused"]]
    refusal_expected = [r for r in per_query_results if r["expected_refused"]]

    # Retrieval metrics (answerable questions only)
    def _mean_metric(key):
        vals = [r[key] for r in answerable if r.get(key) is not None]
        return round(float(np.mean(vals)), 3) if vals else 0.0

    hit1 = _mean_metric("hit_at_1")
    hit3 = _mean_metric("hit_at_3")
    hit5 = _mean_metric("hit_at_5")
    mrr  = _mean_metric("mrr")

    # Generation / refusal accuracy
    correct_refused  = sum(1 for r in refusal_expected if r.get("actual_refused"))
    correct_answered = sum(1 for r in answerable       if not r.get("actual_refused"))
    false_refusals   = sum(1 for r in answerable       if r.get("actual_refused"))
    false_answers    = sum(1 for r in refusal_expected if not r.get("actual_refused"))

    refusal_accuracy = correct_refused  / len(refusal_expected) if refusal_expected else 0.0
    answer_accuracy  = correct_answered / len(answerable)        if answerable       else 0.0
    false_refusal_rate = false_refusals / len(answerable)        if answerable       else 0.0
    false_answer_rate  = false_answers  / len(refusal_expected)  if refusal_expected else 0.0

    # Semantic similarity (answered + answerable questions only)
    sim_vals = [
        r["answer_similarity"]
        for r in answerable
        if not r.get("actual_refused") and r.get("answer_similarity") is not None
    ]
    avg_similarity = round(float(np.mean(sim_vals)), 3) if sim_vals else 0.0

    # Faithfulness
    faith_vals = [
        r["faithfulness"]
        for r in answerable
        if not r.get("actual_refused") and r.get("faithfulness") is not None
    ]
    avg_faithfulness = round(float(np.mean(faith_vals)), 3) if faith_vals else 0.0

    # Latency
    latencies = [r["latency_ms"] for r in per_query_results if r.get("latency_ms")]
    lat_p50 = round(float(np.percentile(latencies, 50)), 1) if latencies else 0.0
    lat_p95 = round(float(np.percentile(latencies, 95)), 1) if latencies else 0.0
    lat_avg = round(float(np.mean(latencies)),             1) if latencies else 0.0

    return {
        "total_questions":             len(per_query_results),
        "answerable_questions":        len(answerable),
        "refusal_expected_questions":  len(refusal_expected),
        "retrieval": {
            "hit_at_1": hit1,
            "hit_at_3": hit3,
            "hit_at_5": hit5,
            "mrr":      mrr,
        },
        "generation": {
            "refusal_accuracy":              round(refusal_accuracy,    3),
            "answer_accuracy":               round(answer_accuracy,     3),
            "false_refusal_rate":            round(false_refusal_rate,  3),
            "false_answer_rate":             round(false_answer_rate,   3),
            "avg_answer_semantic_similarity": avg_similarity,
            "avg_faithfulness":              avg_faithfulness,
        },
        "latency": {
            "avg_ms": lat_avg,
            "p50_ms": lat_p50,
            "p95_ms": lat_p95,
        },
    }
