import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

EVENTS_PATH = Path(os.getenv("ANALYTICS_EVENTS_PATH", "data/analytics/events.jsonl"))


def log_event(event_type: str, payload: Dict[str, Any], *, ts: Optional[str] = None) -> Dict[str, Any]:
    event = {
        "ts": ts or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "event_type": event_type,
        **payload,
    }
    try:
        EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(EVENTS_PATH, "a") as f:
            f.write(json.dumps(event) + "\n")
    except Exception:
        pass

    if event_type == "query":
        _maybe_sync_query(event)
    elif event_type == "eval_run":
        _maybe_log_mlflow(payload)
    return event


def log_query_event(payload: Dict[str, Any]) -> Dict[str, Any]:
    return log_event("query", payload)


def log_eval_event(payload: Dict[str, Any]) -> Dict[str, Any]:
    return log_event("eval_run", payload)


def _maybe_sync_query(event: Dict[str, Any]) -> None:
    try:
        from dbx.config import load_config
        from dbx.events import sync_query_events
    except Exception:
        return
    cfg = load_config()
    if cfg is None or not cfg.sync_on_query:
        return
    try:
        sync_query_events(events=[event], config=cfg, ensure=True)
    except Exception:
        pass


def _maybe_log_mlflow(payload: Dict[str, Any]) -> None:
    if "aggregate" not in payload:
        return
    try:
        from dbx.mlflow_tracking import log_eval_to_mlflow
        log_eval_to_mlflow({
            "mode": payload.get("mode", "full"),
            "model": payload.get("model", "unknown"),
            "num_items": payload.get("num_items", 0),
            "aggregate": payload["aggregate"],
            "ran_at": payload.get("ran_at"),
        })
    except Exception:
        pass
