import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from app.metrics import (
    compute_hit_at_k,
    compute_mrr,
    compute_citation_precision,
    compute_refusal_f1,
    compute_aggregate_metrics,
)
from app.analytics import log_event
from dbx import databricks_configured
from scripts.run_evaluation import select_smoke_subset


def test_retrieval_and_citation_metrics():
    retrieved = ["Restricted Products > Alcohol", "Editorial > Clarity"]
    assert compute_hit_at_k(retrieved, ["Alcohol"], 1) is True
    assert compute_mrr(retrieved, ["Alcohol"]) == 1.0
    assert compute_citation_precision({"a", "b"}, {"a", "c"}) == 0.5


def test_aggregate_and_refusal_f1():
    rows = [{
        "expected_refused": False,
        "actual_refused": False,
        "hit_at_1": True,
        "hit_at_3": True,
        "hit_at_5": True,
        "mrr": 1.0,
        "answer_similarity": 0.8,
        "faithfulness": 0.7,
        "citation_precision": 1.0,
        "latency_ms": 1000,
        "retrieval_ms": 100,
        "generation_ms": 900,
    }, {
        "expected_refused": True,
        "actual_refused": True,
        "hit_at_1": None,
        "hit_at_3": None,
        "hit_at_5": None,
        "mrr": None,
        "latency_ms": 50,
        "retrieval_ms": 50,
        "generation_ms": 0,
    }]
    agg = compute_aggregate_metrics(rows)
    assert agg["generation"]["refusal_f1"] == compute_refusal_f1(rows) == 1.0
    assert "retrieval_p50_ms" in agg["latency"]


def test_smoke_subset_stratified():
    eval_set = []
    for i in range(60):
        eval_set.append({
            "id": f"a{i}",
            "expected_refused": False,
            "category": ["editorial", "prohibited", "restricted", "misrep", "overview"][i % 5],
            "question": f"q{i}",
        })
    for i in range(20):
        eval_set.append({
            "id": f"r{i}",
            "expected_refused": True,
            "category": "out_of_scope",
            "question": f"refuse{i}",
        })
    smoke = select_smoke_subset(eval_set, n=16)
    assert len(smoke) == 16
    assert sum(1 for x in smoke if x["expected_refused"]) >= 1


def test_analytics_event_written(tmp_path, monkeypatch):
    path = tmp_path / "events.jsonl"
    import app.analytics as analytics
    monkeypatch.setattr(analytics, "EVENTS_PATH", path)
    log_event("query", {"latency_ms": 12, "refused": False})
    assert path.exists() and "query" in path.read_text()
    assert databricks_configured() is False
