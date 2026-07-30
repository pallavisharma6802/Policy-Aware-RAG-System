from dbx.config import DatabricksConfig, databricks_configured, load_config
from dbx.schema import ensure_schema
from dbx.events import read_local_events, sync_eval_run, sync_query_events
from dbx.chunks import sync_policy_chunks
from dbx.mlflow_tracking import configure_mlflow, log_eval_to_mlflow


def sync_all() -> dict:
    cfg = load_config()
    if cfg is None:
        raise RuntimeError("Databricks is not configured.")
    tables = ensure_schema(cfg)
    n_events = sync_query_events(config=cfg, ensure=False)
    n_chunks = sync_policy_chunks(config=cfg, ensure=False)
    return {
        "tables": tables,
        "events_synced": n_events,
        "chunks_synced": n_chunks,
        "catalog_schema": f"{cfg.catalog}.{cfg.schema}",
    }


__all__ = [
    "DatabricksConfig",
    "databricks_configured",
    "load_config",
    "ensure_schema",
    "read_local_events",
    "sync_query_events",
    "sync_eval_run",
    "sync_policy_chunks",
    "configure_mlflow",
    "log_eval_to_mlflow",
    "sync_all",
]
