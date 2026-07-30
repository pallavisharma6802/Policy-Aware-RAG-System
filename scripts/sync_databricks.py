"""CLI: python -m scripts.sync_databricks [schema|events|chunks|eval|all]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync policy RAG data to Databricks")
    parser.add_argument(
        "command",
        choices=["schema", "events", "chunks", "eval", "all", "status"],
        help="schema=create tables, events=sync query JSONL, chunks=sync Postgres, eval=push latest eval_results, all=schema+events+chunks, status=config check",
    )
    parser.add_argument(
        "--eval-file",
        default="data/eval/eval_results.json",
        help="Path to eval_results.json for the eval command",
    )
    args = parser.parse_args()

    from dbx import (
        databricks_configured,
        ensure_schema,
        load_config,
        log_eval_to_mlflow,
        sync_all,
        sync_eval_run,
        sync_policy_chunks,
        sync_query_events,
    )

    if args.command == "status":
        cfg = load_config()
        if cfg is None:
            print("configured: false")
            print("Set DATABRICKS_HOST, DATABRICKS_TOKEN, DATABRICKS_HTTP_PATH in .env")
            sys.exit(1)
        print("configured: true")
        print(f"host: {cfg.server_hostname}")
        print(f"schema: {cfg.catalog}.{cfg.schema}")
        print(f"events: {cfg.events_fqn}")
        print(f"eval: {cfg.eval_fqn}")
        print(f"chunks: {cfg.chunks_fqn}")
        print(f"mlflow_experiment: {cfg.mlflow_experiment}")
        return

    if not databricks_configured():
        print("Databricks is not configured. Copy values from dbx/env.example into .env")
        sys.exit(1)

    if args.command == "schema":
        tables = ensure_schema()
        print("created/verified:", ", ".join(tables))
    elif args.command == "events":
        n = sync_query_events()
        print(f"synced {n} query events")
    elif args.command == "chunks":
        n = sync_policy_chunks()
        print(f"synced {n} policy chunks")
    elif args.command == "eval":
        path = Path(args.eval_file)
        if not path.exists():
            print(f"missing {path}; run: python -m scripts.run_evaluation --smoke")
            sys.exit(1)
        result = json.loads(path.read_text())
        sync_eval_run(result)
        run_id = log_eval_to_mlflow(result)
        print(f"synced eval run ({result.get('mode')}, {result.get('num_items')} items)")
        if run_id:
            print(f"mlflow run_id: {run_id}")
    elif args.command == "all":
        summary = sync_all()
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
