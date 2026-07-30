from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500, examples=["Can I advertise alcohol?"])
    limit: int = Field(default=3, ge=1, le=20)
    region: Optional[str] = None
    content_type: Optional[str] = None
    policy_source: Optional[str] = None


class CitationResponse(BaseModel):
    chunk_id: str
    policy_path: str
    doc_id: str
    doc_url: str
    score: Optional[float] = None
    chunk_text: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    refused: bool
    citations: List[CitationResponse] = Field(default_factory=list)
    refusal_reason: Optional[str] = None
    latency_ms: Optional[float] = None
    retrieval_ms: Optional[float] = None
    generation_ms: Optional[float] = None
    num_tokens_generated: Optional[int] = None
    model: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    database: str
    vector_db: str
    llm: str


class EvalSetMeta(BaseModel):
    total: int
    answerable: int
    refusal_expected: int
    categories: Dict[str, int]
    question_types: Dict[str, int]


class EvalStatusResponse(BaseModel):
    status: str
    progress: float = 0.0
    message: str = ""
    results: Optional[Dict[str, Any]] = None


class QueryLogEntry(BaseModel):
    ts: str
    query: str
    refused: bool
    latency_ms: Optional[float] = None
    retrieval_ms: Optional[float] = None
    generation_ms: Optional[float] = None
    num_citations: int = 0
    num_tokens_generated: Optional[int] = None
    model: Optional[str] = None


class QueryHistoryResponse(BaseModel):
    entries: List[QueryLogEntry]
    total: int
