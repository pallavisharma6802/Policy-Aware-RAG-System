from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from dbx.client import sql_connection
from dbx.config import DatabricksConfig, load_config
from dbx.schema import ensure_schema

EVENTS_PATH = Path(os.getenv("ANALYTICS_EVENTS_PATH", "data/analytics/events.jsonl"))


def read_local_events(limit: int = 5000) -> List[Dict[str, Any]]:
    if not EVENTS_PATH.exists():
        return []
    rows = []
    for line in EVENTS_PATH.read_text().strip().splitlines()[-limit:]:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def sync_query_events(
    events: Optional[Iterable[Dict[str, Any]]] = None,
    config: Optional[DatabricksConfig] = None,
    *,
    ensure: bool = True,
) -> int:
    cfg = config or load_config()
    if cfg is None:
        raise RuntimeError("Databricks is not configured.")
    if ensure:
        ensure_schema(cfg)

    rows = list(events) if events is not None else read_local_events()
    query_rows = [r for r in rows if r.get("event_type", "query") == "query"]
    if not query_rows:
        return 0

    with sql_connection(cfg) as conn:
        with conn.cursor() as cursor:
            for row in query_rows:
                cursor.execute(
                    f"""
                    INSERT INTO {cfg.events_fqn} (
                      ts, event_type, query, refused, latency_ms, retrieval_ms,
                      generation_ms, num_citations, num_tokens_generated, model, payload
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row.get("ts"),
                        row.get("event_type", "query"),
                        row.get("query"),
                        bool(row.get("refused")),
                        row.get("latency_ms"),
                        row.get("retrieval_ms"),
                        row.get("generation_ms"),
                        row.get("num_citations"),
                        row.get("num_tokens_generated"),
                        row.get("model"),
                        json.dumps(row),
                    ),
                )
    return len(query_rows)


def sync_eval_run(
    result: Dict[str, Any],
    config: Optional[DatabricksConfig] = None,
    *,
    ensure: bool = True,
) -> None:
    cfg = config or load_config()
    if cfg is None:
        raise RuntimeError("Databricks is not configured.")
    if ensure:
        ensure_schema(cfg)

    agg = result.get("aggregate", {})
    retrieval = agg.get("retrieval", {})
    generation = agg.get("generation", {})
    latency = agg.get("latency", {})

    with sql_connection(cfg) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                f"""
                INSERT INTO {cfg.eval_fqn} (
                  ran_at, mode, model, num_items,
                  hit_at_1, hit_at_3, mrr,
                  answer_accuracy, refusal_accuracy, refusal_f1, citation_precision,
                  latency_p50_ms, latency_p95_ms, payload
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.get("ran_at"),
                    result.get("mode"),
                    result.get("model"),
                    result.get("num_items"),
                    retrieval.get("hit_at_1"),
                    retrieval.get("hit_at_3"),
                    retrieval.get("mrr"),
                    generation.get("answer_accuracy"),
                    generation.get("refusal_accuracy"),
                    generation.get("refusal_f1"),
                    generation.get("avg_citation_precision"),
                    latency.get("p50_ms"),
                    latency.get("p95_ms"),
                    json.dumps(agg),
                ),
            )
