import sys
from pathlib import Path

import weaviate
from sqlalchemy import func

sys.path.append(str(Path(__file__).parent.parent))

from db.session import SessionLocal
from db.models import PolicyChunk
from app.embeddings import get_embedding_model


def test_postgres_weaviate_coverage_aligned():
    db = SessionLocal()
    client = weaviate.Client("http://localhost:8080", timeout_config=(5, 10))
    try:
        pg_count = (
            db.query(func.count(PolicyChunk.chunk_id))
            .filter(~PolicyChunk.doc_id.like("pytest_%"))
            .filter(~PolicyChunk.doc_id.like("test_%"))
            .filter(~PolicyChunk.doc_id.like("dup_doc_%"))
            .scalar()
        )
        result = client.query.aggregate("PolicyChunk").with_meta_count().do()
        wv_count = result["data"]["Aggregate"]["PolicyChunk"][0]["meta"]["count"]
        assert pg_count > 0
        assert wv_count > 0
        assert pg_count == wv_count
    finally:
        db.close()


def test_embedding_dimension_and_id_alignment():
    model = get_embedding_model()
    assert len(model.encode("dimension check")) == 384

    db = SessionLocal()
    client = weaviate.Client("http://localhost:8080", timeout_config=(5, 10))
    try:
        row = (
            db.query(PolicyChunk.chunk_id)
            .filter(~PolicyChunk.doc_id.like("pytest_%"))
            .first()
        )
        pg_chunk_id = str(row[0])
        obj = client.data_object.get_by_id(pg_chunk_id, class_name="PolicyChunk")
        assert obj["id"] == pg_chunk_id
        assert obj["properties"]["chunk_id"] == pg_chunk_id
    finally:
        db.close()


def test_no_duplicate_doc_chunk_pairs():
    db = SessionLocal()
    try:
        rows = (
            db.query(PolicyChunk.doc_id, PolicyChunk.chunk_index)
            .filter(~PolicyChunk.doc_id.like("pytest_%"))
            .all()
        )
        assert len(rows) == len(set(rows))
    finally:
        db.close()
