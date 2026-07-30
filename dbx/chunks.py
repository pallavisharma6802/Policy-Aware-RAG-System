from __future__ import annotations

from typing import Optional

from dbx.client import sql_connection
from dbx.config import DatabricksConfig, load_config
from dbx.schema import ensure_schema


def sync_policy_chunks(config: Optional[DatabricksConfig] = None, *, ensure: bool = True) -> int:
    """Push Postgres policy_chunks into a Databricks Delta table."""
    cfg = config or load_config()
    if cfg is None:
        raise RuntimeError("Databricks is not configured.")
    if ensure:
        ensure_schema(cfg)

    from db.session import SessionLocal
    from db.models import PolicyChunk

    db = SessionLocal()
    try:
        chunks = db.query(PolicyChunk).all()
        rows = []
        for c in chunks:
            rows.append(
                (
                    str(c.chunk_id),
                    c.doc_id,
                    c.chunk_index,
                    c.chunk_text,
                    c.policy_source.value if hasattr(c.policy_source, "value") else str(c.policy_source),
                    c.policy_section,
                    c.policy_section_level,
                    c.policy_path,
                    c.region.value if hasattr(c.region, "value") else str(c.region),
                    c.content_type.value if hasattr(c.content_type, "value") else str(c.content_type),
                    c.doc_url,
                    c.created_at.isoformat() if c.created_at else None,
                )
            )
    finally:
        db.close()

    if not rows:
        return 0

    with sql_connection(cfg) as conn:
        with conn.cursor() as cursor:
            cursor.execute(f"DELETE FROM {cfg.chunks_fqn}")
            for row in rows:
                cursor.execute(
                    f"""
                    INSERT INTO {cfg.chunks_fqn} (
                      chunk_id, doc_id, chunk_index, chunk_text, policy_source,
                      policy_section, policy_section_level, policy_path,
                      region, content_type, doc_url, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    row,
                )
    return len(rows)
