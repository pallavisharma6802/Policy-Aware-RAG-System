import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.append(str(Path(__file__).parent.parent))

from app.generation import generate_policy_response, translate_index_citations, format_sources
from app.citations import extract_citations, validate_citations


SAMPLE_CHUNK = {
    "chunk_id": "11111111-1111-1111-1111-111111111111",
    "chunk_text": "Alcohol advertising is restricted and may require certification.",
    "score": 0.82,
    "policy_path": "Restricted Products > Alcohol",
    "doc_id": "doc-1",
    "doc_url": "https://example.com/alcohol",
}


def test_index_citation_translation():
    text, index_map = format_sources([SAMPLE_CHUNK])
    assert "SOURCE 1:" in text
    translated = translate_index_citations("Alcohol is restricted [SOURCE:1].", index_map)
    assert SAMPLE_CHUNK["chunk_id"] in translated
    assert validate_citations(extract_citations(translated), {SAMPLE_CHUNK["chunk_id"]})


def test_generation_refuses_when_no_chunks(monkeypatch):
    monkeypatch.setattr("app.generation.retrieve_policy_chunks", lambda **kwargs: [])
    response = generate_policy_response("quantum teleportation ads")
    assert response.refused is True
    assert response.answer == ""


def test_generation_refuses_low_confidence(monkeypatch):
    monkeypatch.setattr(
        "app.generation.retrieve_policy_chunks",
        lambda **kwargs: [{**SAMPLE_CHUNK, "score": 0.01}],
    )
    response = generate_policy_response("random query")
    assert response.refused is True
    assert "Insufficient confidence" in response.refusal_reason


def test_generation_requires_valid_citations(monkeypatch):
    monkeypatch.setattr("app.generation.retrieve_policy_chunks", lambda **kwargs: [SAMPLE_CHUNK])
    mock_llm = MagicMock()
    mock_llm.model = "mock"
    mock_llm.invoke.return_value = "Alcohol is fine [SOURCE:999]."
    response = generate_policy_response("Can I advertise alcohol?", llm=mock_llm)
    assert response.refused is True
    assert "citation validation" in response.refusal_reason


def test_generation_success_with_valid_index_citation(monkeypatch):
    monkeypatch.setattr("app.generation.retrieve_policy_chunks", lambda **kwargs: [SAMPLE_CHUNK])
    mock_llm = MagicMock()
    mock_llm.model = "mock"
    mock_llm.invoke.return_value = "Alcohol ads are restricted [SOURCE:1]."
    response = generate_policy_response("Can I advertise alcohol?", llm=mock_llm)
    assert response.refused is False
    assert len(response.citations) == 1
    assert response.generation_ms is not None
    assert response.model == "mock"
