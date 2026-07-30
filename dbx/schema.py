from __future__ import annotations

from typing import List, Optional

from dbx.client import execute
from dbx.config import DatabricksConfig, load_config


def ensure_schema(config: Optional[DatabricksConfig] = None) -> List[str]:
    cfg = config or load_config()
    if cfg is None:
        raise RuntimeError("Databricks is not configured.")

    stmts = [
        f"CREATE SCHEMA IF NOT EXISTS {cfg.catalog}.{cfg.schema}",
        f"""
        CREATE TABLE IF NOT EXISTS {cfg.events_fqn} (
          ts STRING,
          event_type STRING,
          query STRING,
          refused BOOLEAN,
          latency_ms DOUBLE,
          retrieval_ms DOUBLE,
          generation_ms DOUBLE,
          num_citations INT,
          num_tokens_generated INT,
          model STRING,
          payload STRING
        ) USING DELTA
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {cfg.eval_fqn} (
          ran_at STRING,
          mode STRING,
          model STRING,
          num_items INT,
          hit_at_1 DOUBLE,
          hit_at_3 DOUBLE,
          mrr DOUBLE,
          answer_accuracy DOUBLE,
          refusal_accuracy DOUBLE,
          refusal_f1 DOUBLE,
          citation_precision DOUBLE,
          latency_p50_ms DOUBLE,
          latency_p95_ms DOUBLE,
          payload STRING
        ) USING DELTA
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {cfg.chunks_fqn} (
          chunk_id STRING,
          doc_id STRING,
          chunk_index INT,
          chunk_text STRING,
          policy_source STRING,
          policy_section STRING,
          policy_section_level STRING,
          policy_path STRING,
          region STRING,
          content_type STRING,
          doc_url STRING,
          created_at STRING
        ) USING DELTA
        """,
    ]
    execute(stmts, cfg)
    return [cfg.events_fqn, cfg.eval_fqn, cfg.chunks_fqn]
