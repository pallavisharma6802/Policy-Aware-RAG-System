import sys
from pathlib import Path
from unittest.mock import patch

sys.path.append(str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from app.schemas import PolicyResponse, Citation
from api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert {"status", "database", "vector_db", "llm"} <= set(data)


def test_query_validation_errors():
    assert client.post("/query", json={"limit": 3}).status_code == 422
    assert client.post("/query", json={"query": "Can I advertise alcohol?", "limit": 0}).status_code == 422
    assert client.post("/query", json={"query": "ab", "limit": 3}).status_code == 422


def test_query_happy_path_mocked():
    fake = PolicyResponse(
        answer="Alcohol advertising is restricted [SOURCE:11111111-1111-1111-1111-111111111111].",
        refused=False,
        citations=[Citation(
            chunk_id="11111111-1111-1111-1111-111111111111",
            policy_path="Restricted Products > Alcohol",
            doc_id="doc-1",
            doc_url="https://example.com",
            score=0.9,
            chunk_text="Alcohol advertising is restricted.",
        )],
        latency_ms=120.0,
        retrieval_ms=20.0,
        generation_ms=100.0,
        num_tokens_generated=12,
        model="mock",
    )
    with patch("api.main.generate_policy_response", return_value=fake):
        response = client.post("/query", json={"query": "Can I advertise alcohol?", "limit": 3})
    assert response.status_code == 200
    data = response.json()
    assert data["refused"] is False
    assert data["retrieval_ms"] == 20.0
    assert data["model"] == "mock"


def test_query_refusal_path_mocked():
    fake = PolicyResponse(
        answer="",
        refused=True,
        refusal_reason="No relevant policies found for this query.",
        latency_ms=15.0,
        retrieval_ms=15.0,
        generation_ms=0.0,
        model="mock",
    )
    with patch("api.main.generate_policy_response", return_value=fake):
        response = client.post("/query", json={"query": "totally unrelated quantum ads", "limit": 3})
    assert response.status_code == 200
    assert response.json()["refused"] is True


def test_eval_meta_endpoint():
    response = client.get("/eval")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 80
    assert data["answerable"] == 60
