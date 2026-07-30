import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.append(str(Path(__file__).parent.parent))

from dbx.config import DatabricksConfig, load_config, databricks_configured


def test_databricks_config_roundtrip(monkeypatch):
    monkeypatch.delenv("DATABRICKS_HOST", raising=False)
    monkeypatch.delenv("DATABRICKS_TOKEN", raising=False)
    monkeypatch.delenv("DATABRICKS_HTTP_PATH", raising=False)
    assert databricks_configured() is False

    monkeypatch.setenv("DATABRICKS_HOST", "https://adb-123.azuredatabricks.net")
    monkeypatch.setenv("DATABRICKS_TOKEN", "dapi-test")
    monkeypatch.setenv("DATABRICKS_HTTP_PATH", "/sql/1.0/warehouses/abc")
    cfg = load_config()
    assert isinstance(cfg, DatabricksConfig)
    assert cfg.events_fqn == "main.policy_rag.query_events"


def test_sync_query_events_mocked(monkeypatch):
    monkeypatch.setenv("DATABRICKS_HOST", "https://adb-123.azuredatabricks.net")
    monkeypatch.setenv("DATABRICKS_TOKEN", "dapi-test")
    monkeypatch.setenv("DATABRICKS_HTTP_PATH", "/sql/1.0/warehouses/abc")

    cursor = MagicMock()
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cursor
    conn.cursor.return_value.__exit__.return_value = False

    with patch("dbx.events.ensure_schema"), patch("dbx.events.sql_connection") as mock_sql:
        mock_sql.return_value.__enter__.return_value = conn
        mock_sql.return_value.__exit__.return_value = False
        from dbx.events import sync_query_events
        n = sync_query_events(
            events=[{
                "ts": "2026-01-01T00:00:00Z",
                "event_type": "query",
                "query": "Can I advertise alcohol?",
                "refused": False,
                "latency_ms": 100.0,
                "retrieval_ms": 10.0,
                "generation_ms": 90.0,
                "num_citations": 1,
                "num_tokens_generated": 20,
                "model": "qwen3:1.7b",
            }],
            ensure=False,
        )
    assert n == 1
    assert cursor.execute.called
