from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="User's policy compliance question",
        examples=["Can I advertise alcohol?"]
    )
    limit: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Maximum number of policy chunks to retrieve"
    )
    region: Optional[str] = Field(
        default=None,
        description="Filter by geographic region (e.g., 'US', 'EU')"
    )
    content_type: Optional[str] = Field(
        default=None,
        description="Filter by content type (e.g., 'video', 'text')"
    )
    policy_source: Optional[str] = Field(
        default=None,
        description="Filter by policy source (e.g., 'google_ads', 'youtube')"
    )


class CitationResponse(BaseModel):
    chunk_id: str = Field(description="Unique identifier for the source chunk")
    policy_path: str = Field(description="Human-readable policy hierarchy path")
    doc_id: str = Field(description="Source document identifier")
    doc_url: str = Field(description="URL to the source policy document")
    score: Optional[float] = Field(default=None, description="Retrieval relevance score (0–1)")
    chunk_text: Optional[str] = Field(default=None, description="Raw text of the retrieved chunk")


class QueryResponse(BaseModel):
    answer: str = Field(description="Generated answer with inline citations")
    refused: bool = Field(description="Whether the system refused to answer")
    citations: List[CitationResponse] = Field(
        default_factory=list,
        description="List of sources cited in the answer"
    )
    refusal_reason: Optional[str] = Field(
        default=None,
        description="Explanation if the system refused to answer"
    )
    latency_ms: Optional[float] = Field(
        default=None,
        description="Total processing time in milliseconds"
    )
    num_tokens_generated: Optional[int] = Field(
        default=None,
        description="Approximate number of tokens in the answer"
    )


class HealthResponse(BaseModel):
    status: str = Field(description="Service health status")
    database: str = Field(description="PostgreSQL connection status")
    vector_db: str = Field(description="Weaviate connection status")
    llm: str = Field(description="Ollama service status")


# ── Evaluation / metrics models ───────────────────────────────────────────────

class EvalSetMeta(BaseModel):
    total: int
    answerable: int
    refusal_expected: int
    categories: Dict[str, int]
    question_types: Dict[str, int]


class EvalStatusResponse(BaseModel):
    status: str = Field(description="idle | running | complete | error")
    progress: float = Field(default=0.0, description="0.0 – 1.0")
    message: str = Field(default="")
    results: Optional[Dict[str, Any]] = Field(default=None)


class QueryLogEntry(BaseModel):
    ts: str
    query: str
    refused: bool
    latency_ms: Optional[float]
    num_citations: int
    num_tokens_generated: Optional[int]


class QueryHistoryResponse(BaseModel):
    entries: List[QueryLogEntry]
    total: int
