import re
from typing import Set, List, Dict
from app.schemas import Citation

CITATION_PATTERN = re.compile(r"\[SOURCE:([a-f0-9\-]{36})\]")


def extract_citations(text: str) -> Set[str]:
    return set(CITATION_PATTERN.findall(text))


def validate_citations(cited_ids: Set[str], retrieved_ids: Set[str]) -> bool:
    if not cited_ids:
        return False
    return cited_ids.issubset(retrieved_ids)


def build_citations(cited_ids: Set[str], results: List[Dict]) -> List[Citation]:
    result_map = {r["chunk_id"]: r for r in results}
    citations = []
    for chunk_id in cited_ids:
        if chunk_id in result_map:
            chunk = result_map[chunk_id]
            citations.append(Citation(
                chunk_id=chunk_id,
                policy_path=chunk["policy_path"],
                doc_id=chunk["doc_id"],
                doc_url=chunk.get("doc_url", ""),
                score=round(chunk.get("score", 0.0), 4),
                chunk_text=chunk.get("chunk_text", ""),
            ))
    return citations
