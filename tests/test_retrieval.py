import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).parent.parent))

from app.retrieval import retrieve_policy_chunks


def test_vector_search_returns_scored_results():
    results = retrieve_policy_chunks("Can I advertise alcohol?", limit=3)
    assert 0 < len(results) <= 3
    scores = [r["score"] for r in results]
    assert all(s > 0 for s in scores)
    assert scores == sorted(scores, reverse=True)
    assert {"chunk_id", "chunk_text", "policy_path", "doc_url", "score"} <= set(results[0])


def test_region_filter_enforced():
    results = retrieve_policy_chunks("advertising policy", limit=5, region="global")
    for r in results:
        assert r["region"] == "global"


def test_invalid_region_raises():
    with pytest.raises(ValueError):
        retrieve_policy_chunks("ads", limit=2, region="mars")


def test_alcohol_query_is_topically_relevant():
    results = retrieve_policy_chunks("Can I advertise alcohol?", limit=3)
    blob = " ".join(r["policy_path"].lower() + " " + r["chunk_text"].lower() for r in results)
    assert "alcohol" in blob
