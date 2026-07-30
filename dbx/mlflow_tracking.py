from __future__ import annotations

import os
from typing import Any, Dict, Optional

from dbx.config import DatabricksConfig, load_config


def configure_mlflow(config: Optional[DatabricksConfig] = None) -> Optional[str]:
    try:
        import mlflow
    except ImportError:
        return None

    cfg = config or load_config()
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    elif cfg is not None:
        mlflow.set_tracking_uri("databricks")
    else:
        return None

    experiment = (
        cfg.mlflow_experiment if cfg is not None else os.getenv("MLFLOW_EXPERIMENT_NAME", "/Shared/policy-rag-eval")
    )
    mlflow.set_experiment(experiment)
    return experiment


def log_eval_to_mlflow(result: Dict[str, Any], config: Optional[DatabricksConfig] = None) -> Optional[str]:
    try:
        import mlflow
    except ImportError:
        return None

    experiment = configure_mlflow(config)
    if experiment is None:
        return None

    agg = result.get("aggregate", {})
    with mlflow.start_run(run_name=f"policy-rag-{result.get('mode', 'eval')}"):
        mlflow.log_params({
            "model": result.get("model", "unknown"),
            "num_items": result.get("num_items", 0),
            "mode": result.get("mode", "full"),
        })
        for section_name in ("retrieval", "generation", "latency"):
            section = agg.get(section_name, {})
            for key, value in section.items():
                if isinstance(value, (int, float)) and value is not None:
                    mlflow.log_metric(f"{section_name}_{key}", float(value))
        run = mlflow.active_run()
        return run.info.run_id if run else None
