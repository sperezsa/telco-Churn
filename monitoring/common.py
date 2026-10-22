"""Shared date-window and MLflow setup for monitoring entry points."""

import sys
from pathlib import Path
from typing import Any

import mlflow
from mlflow import MlflowClient

from telco_churn.config import DEFAULT_CONFIG_PATH, load_config, project_path
from telco_churn.monitoring_store import initialize_monitoring_store
from telco_churn.monitoring_utils import previous_completed_week


def configure_console() -> None:
    """Allow MLflow's Unicode run links to print on Windows consoles."""
    reconfigure_stdout = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure_stdout):
        reconfigure_stdout(encoding="utf-8", errors="replace")


def initialize_context(
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> tuple[dict[str, Any], Path, str, str]:
    """Load settings, resolve the champion version, and initialize local tables."""
    config = load_config(config_path)
    tracking_uri = config["mlflow"]["tracking_uri"]
    model_name = config["mlflow"]["registered_model_name"]
    model_alias = config["mlflow"]["model_alias"]
    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient(tracking_uri)
    champion = client.get_model_version_by_alias(model_name, model_alias)
    model_version = str(champion.version)
    database_path = project_path(config["monitoring"]["database_path"])
    initialize_monitoring_store(
        database_path=database_path,
        sample_model_name=model_name,
        sample_model_version=model_version,
    )
    return config, database_path, model_name, model_version
