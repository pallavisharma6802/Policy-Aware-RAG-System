import weaviate
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session
from typing import List, Dict, Optional
from dataclasses import dataclass
import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from db.session import SessionLocal
from db.models import PolicyChunk, PolicySource, Region, ContentType

WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_retriever_instance = None

def get_retriever() -> 'HybridRetriever':
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = HybridRetriever()
    return _retriever_instance

@dataclass
class RetrievalResult:
    chunk_id: str
    chunk_text: str
    policy_section: str
    policy_path: str
    policy_section_level: str
    doc_id: str
    doc_url: str
    policy_source: str
    region: str
    content_type: str
    score: float
    
    def to_dict(self) -> Dict:
        return {
            "chunk_id": self.chunk_id,
            "chunk_text": self.chunk_text,
            "policy_section": self.policy_section,
            "policy_path": self.policy_path,
            "policy_section_level": self.policy_section_level,
            "doc_id": self.doc_id,
            "doc_url": self.doc_url,
            "policy_source": self.policy_source,
            "region": self.region,
            "content_type": self.content_type,
            "score": self.score
        }

class HybridRetriever:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model = SentenceTransformer(model_name)
        self.weaviate_client = weaviate.Client(url=WEAVIATE_URL, timeout_config=(5, 30))
    
    def vector_search(
        self,
        query: str,
        limit: int = 10
    ) -> List[Dict]:
        query_vector = self.model.encode(query).tolist()
        
        query_builder = self.weaviate_client.query.get(
            "PolicyChunk",
            [
                "chunk_id",
                "chunk_text",
                "policy_section",
                "policy_path",
                "policy_section_level",
                "doc_id",
                "doc_url",
                "policy_source",
                "region",
                "content_type",
                "_additional { distance }"
            ]
        ).with_near_vector({"vector": query_vector}).with_limit(limit)
        
        result = query_builder.do()
        
        chunks = result.get("data", {}).get("Get", {}).get("PolicyChunk", [])
        return chunks
    
    def sql_filter(
        self,
        db: Session,
        chunk_ids: List[str],
        region: Optional[str] = None,
        content_type: Optional[str] = None,
        policy_source: Optional[str] = None
    ) -> List[PolicyChunk]:
        query = db.query(PolicyChunk).filter(PolicyChunk.chunk_id.in_(chunk_ids))
        
        if region:
            region_enum = Region(region.strip().lower())
            query = query.filter(PolicyChunk.region == region_enum)
        
        if content_type:
            content_type_enum = ContentType(content_type.strip().lower())
            query = query.filter(PolicyChunk.content_type == content_type_enum)
        
        if policy_source:
            policy_source_enum = PolicySource(policy_source.strip().lower())
            query = query.filter(PolicyChunk.policy_source == policy_source_enum)
        
        # Preserve vector ranking; SQL used only as a filter
        return query.all()
    
    def retrieve(
        self,
        query: str,
        limit: int = 3,
        region: Optional[str] = None,
        content_type: Optional[str] = None,
        policy_source: Optional[str] = None,
        prefer_specific: bool = True
    ) -> List[RetrievalResult]:
        if limit <= 0:
            return []
        
        overfetch_limit = limit * 3
        
        vector_results = self.vector_search(
            query=query,
            limit=overfetch_limit
        )
        
        if not vector_results:
            return []
        
        chunk_ids = [chunk["chunk_id"] for chunk in vector_results]
        
        db = SessionLocal()
        try:
            sql_results = self.sql_filter(
                db=db,
                chunk_ids=chunk_ids,
                region=region,
                content_type=content_type,
                policy_source=policy_source
            )
            
            sql_chunk_map = {str(chunk.chunk_id): chunk for chunk in sql_results}
            
            results = []
            for chunk in vector_results:
                chunk_id = chunk["chunk_id"]
                if chunk_id not in sql_chunk_map:
                    continue
                
                distance = chunk["_additional"]["distance"]
                score = 1 / (1 + distance)
                
                result = RetrievalResult(
                    chunk_id=chunk_id,
                    chunk_text=chunk["chunk_text"],
                    policy_section=chunk["policy_section"],
                    policy_path=chunk["policy_path"],
                    policy_section_level=chunk["policy_section_level"],
                    doc_id=chunk["doc_id"],
                    doc_url=chunk.get("doc_url", ""),
                    policy_source=chunk["policy_source"],
                    region=chunk["region"],
                    content_type=chunk["content_type"],
                    score=score
                )
                results.append(result)
            
            results = self.rerank_by_hierarchy(results, prefer_specific=prefer_specific)
            
            return results[:limit]
        finally:
            db.close()
    
    def rerank_by_hierarchy(
        self,
        results: List[RetrievalResult],
        prefer_specific: bool = True
    ) -> List[RetrievalResult]:
        h3_boost = 0.1 if prefer_specific else -0.1
        
        for result in results:
            if result.policy_section_level == "H3":
                result.score += h3_boost
            elif result.policy_section_level == "H2":
                result.score -= h3_boost
        
        return sorted(results, key=lambda x: x.score, reverse=True)

def retrieve_policy_chunks(
    query: str,
    limit: int = 3,
    region: Optional[str] = None,
    content_type: Optional[str] = None,
    policy_source: Optional[str] = None,
    prefer_specific: bool = True
) -> List[Dict]:
    retriever = get_retriever()
    
    results = retriever.retrieve(
        query=query,
        limit=limit,
        region=region,
        content_type=content_type,
        policy_source=policy_source,
        prefer_specific=prefer_specific
    )
    
    return [result.to_dict() for result in results]

if __name__ == "__main__":
    print("Testing hybrid retrieval...")
    print()
    
    queries = [
        ("Can I advertise alcohol?", None, None),
        ("cryptocurrency ads rules", "global", None),
        ("misrepresentation policy", None, None)
    ]
    
    for query, region, content_type in queries:
        print(f"Query: {query}")
        if region:
            print(f"Region filter: {region}")
        if content_type:
            print(f"Content type filter: {content_type}")
        print()
        
        results = retrieve_policy_chunks(
            query, 
            limit=2, 
            region=region, 
            content_type=content_type
        )
        
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['policy_path']}")
            print(f"     Level: {result['policy_section_level']}, Score: {result['score']:.4f}")
        