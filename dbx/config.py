from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class DatabricksConfig:
    host: str
    token: str
    http_path: str
    catalog: str = "main"
    schema: str = "policy_rag"
    events_table: str = "query_events"
    eval_table: str = "eval_runs"
    chunks_table: str = "policy_chunks"
    mlflow_experiment: str = "/Shared/policy-rag-eval"
    sync_on_query: bool = False

    @property
    def server_hostname(self) -> str:
        return self.host.replace("https://", "").replace("http://", "").rstrip("/")

    @property
    def events_fqn(self) -> str:
        return f"{self.catalog}.{self.schema}.{self.events_table}"

    @property
    def eval_fqn(self) -> str:
        return f"{self.catalog}.{self.schema}.{self.eval_table}"

    @property
    def chunks_fqn(self) -> str:
        return f"{self.catalog}.{self.schema}.{self.chunks_table}"


def load_config() -> Optional[DatabricksConfig]:
    host = os.getenv("DATABRICKS_HOST", "").strip()
    token = os.getenv("DATABRICKS_TOKEN", "").strip()
    http_path = os.getenv("DATABRICKS_HTTP_PATH", "").strip()
    if not (host and token and http_path):
        return None

    catalog = os.getenv("DATABRICKS_CATALOG", "main").strip() or "main"
    schema = os.getenv("DATABRICKS_SCHEMA", "policy_rag").strip() or "policy_rag"
    return DatabricksConfig(
        host=host,
        token=token,
        http_path=http_path,
        catalog=catalog,
        schema=schema,
        events_table=os.getenv("DATABRICKS_EVENTS_TABLE", "query_events"),
        eval_table=os.getenv("DATABRICKS_EVAL_TABLE", "eval_runs"),
        chunks_table=os.getenv("DATABRICKS_CHUNKS_TABLE", "policy_chunks"),
        mlflow_experiment=os.getenv("MLFLOW_EXPERIMENT_NAME", "/Shared/policy-rag-eval"),
        sync_on_query=os.getenv("DATABRICKS_SYNC_ON_QUERY", "").lower() in {"1", "true", "yes"},
    )


def databricks_configured() -> bool:
    return load_config() is not None
