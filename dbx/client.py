from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional

from dbx.config import DatabricksConfig, load_config


def _require_connector():
    try:
        from databricks import sql
        return sql
    except ImportError as exc:
        raise RuntimeError(
            "Install databricks-sql-connector: pip install -r requirements-databricks.txt"
        ) from exc


@contextmanager
def sql_connection(config: Optional[DatabricksConfig] = None) -> Iterator:
    cfg = config or load_config()
    if cfg is None:
        raise RuntimeError(
            "Databricks is not configured. Set DATABRICKS_HOST, DATABRICKS_TOKEN, DATABRICKS_HTTP_PATH."
        )

    sql = _require_connector()
    connection = sql.connect(
        server_hostname=cfg.server_hostname,
        http_path=cfg.http_path,
        access_token=cfg.token,
    )
    try:
        yield connection
    finally:
        connection.close()


def execute(statements, config: Optional[DatabricksConfig] = None) -> None:
    if isinstance(statements, str):
        statements = [statements]
    with sql_connection(config) as conn:
        with conn.cursor() as cursor:
            for stmt in statements:
                cursor.execute(stmt)
