import sys
import uuid
from pathlib import Path

import pytest
from sqlalchemy.exc import IntegrityError

sys.path.append(str(Path(__file__).parent.parent))

from db.session import SessionLocal
from db.models import PolicyChunk


PREFIX = f"pytest_{uuid.uuid4().hex[:8]}"


def _base_chunk(**overrides):
    data = dict(
        chunk_id=str(uuid.uuid4()),
        doc_id=f"{PREFIX}_doc",
        chunk_index=0,
        chunk_text="Test chunk",
        policy_source="google",
        policy_section="Test Section",
        policy_section_level="H2",
        policy_path="Test > Section",
        region="global",
        content_type="general",
        doc_url="https://test.com",
    )
    data.update(overrides)
    return PolicyChunk(**data)


@pytest.fixture(autouse=True)
def cleanup_pytest_rows():
    yield
    db = SessionLocal()
    try:
        db.query(PolicyChunk).filter(PolicyChunk.doc_id.like(f"{PREFIX}%")).delete(
            synchronize_session=False
        )
        db.commit()
    finally:
        db.close()


def test_duplicate_doc_id_chunk_index_fails():
    db = SessionLocal()
    try:
        db.add(_base_chunk(chunk_index=0))
        db.commit()
        db.add(_base_chunk(chunk_index=0, chunk_text="Different"))
        with pytest.raises(IntegrityError):
            db.commit()
    finally:
        db.rollback()
        db.close()


def test_chunk_text_not_null():
    db = SessionLocal()
    try:
        db.add(_base_chunk(chunk_text=None, chunk_index=1))
        with pytest.raises(IntegrityError):
            db.commit()
    finally:
        db.rollback()
        db.close()


def test_duplicate_chunk_id_fails():
    db = SessionLocal()
    try:
        cid = str(uuid.uuid4())
        db.add(_base_chunk(chunk_id=cid, doc_id=f"{PREFIX}_a", chunk_index=0))
        db.commit()
        db.add(_base_chunk(chunk_id=cid, doc_id=f"{PREFIX}_b", chunk_index=0))
        with pytest.raises(IntegrityError):
            db.commit()
    finally:
        db.rollback()
        db.close()
