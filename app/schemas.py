from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class Citation:
    chunk_id: str
    policy_path: str
    doc_id: str
    doc_url: str
    score: float = 0.0
    chunk_text: str = ""

    def to_dict(self) -> Dict:
        return {
            "chunk_id": self.chunk_id,
            "policy_path": self.policy_path,
            "doc_id": self.doc_id,
            "doc_url": self.doc_url,
            "score": self.score,
            "chunk_text": self.chunk_text,
        }


@dataclass
class PolicyResponse:
    answer: str
    refused: bool
    citations: List[Citation] = field(default_factory=list)
    refusal_reason: Optional[str] = None
    latency_ms: Optional[float] = None
    num_tokens_generated: Optional[int] = None
    
    def to_dict(self) -> Dict:
        response = {
            "answer": self.answer,
            "refused": self.refused,
            "citations": [c.to_dict() for c in self.citations]
        }
        
        if self.refusal_reason:
            response["refusal_reason"] = self.refusal_reason
        
        if self.latency_ms is not None:
            response["latency_ms"] = self.latency_ms
        
        if self.num_tokens_generated is not None:
            response["num_tokens_generated"] = self.num_tokens_generated
        
        return response
