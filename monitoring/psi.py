"""Evaluate top-feature population stability for API calls from last week."""

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.artifacts import download_artifacts
from mlflow.sklearn import load_model
from sklearn.pipeline import Pipeline

from monitoring.common import configure_console, initialize_context, previous_completed_week
from telco_churn.config import DEFAULT_CONFIG_PATH
from telco_churn.evaluate import calculate_population_stability_index
from telco_churn.monitoring_store import load_inference_logs


def _load_psi_reference(run_id: str) -> dict[str, Any]:
    """Download the immutable train-only PSI reference from its source run."""
    reference_path = download_artifacts(
        artifact_uri=f"runs:/{run_id}/explainability/psi_reference.json"
    )
    with Path(reference_path).open(encoding="utf-8") as reference_file:
        return json.load(reference_file)


def _status_for_psi(value: float, ok_below: float, warning_below: float) -> str:
    """Map PSI to the configured OK, Warning, or Alert band."""
    if value < ok_below:
        return "OK"
    if value < warning_below:
        return "Warning"
    return "Alert"


def evaluate_psi(config_path: Path = DEFAULT_CONFIG_PATH) -> list[dict[str, Any]]:
    """Calculate weekly PSI by model version and log values plus audit rows."""
    configure_console()
    config, database_path, champion_name, _ = initialize_context(config_path)
    monitoring_config = config["monitoring"]
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    experiment = mlflow.set_experiment(monitoring_config["experiment_name"])
    start_at, end_at = previous_completed_week()
    inference_logs = load_inference_logs(start_at, end_at, database_path)
    if not inference_logs:
        raise ValueError(
            f"No inference logs found for previous week "
            f"[{start_at.isoformat()}, {end_at.isoformat()})."
        )

    psi_config = monitoring_config["psi"]
    ok_below = float(psi_config["ok_below"])
    warning_below = float(psi_config["warning_below"])
    evaluation_timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    client = MlflowClient(config["mlflow"]["tracking_uri"])
    results: list[dict[str, Any]] = []

    grouped = pd.DataFrame(inference_logs).groupby(
        ["model_name", "model_version"], sort=True
    )
    for (model_name, model_version), group in grouped:
        version = client.get_model_version(str(model_name), str(model_version))
        if version.run_id is None:
            raise ValueError(
                f"MLflow model {model_name} version {model_version} has no source run."
            )
        reference = _load_psi_reference(version.run_id)
        model_uri = f"models:/{model_name}/{model_version}"
        model = load_model(model_uri)
        if not isinstance(model, Pipeline):
            raise TypeError(f"MLflow model {model_uri} is not a scikit-learn Pipeline.")
        if "preprocessor" not in model.named_steps:
            raise ValueError(
                f"MLflow model {model_uri} has no 'preprocessor' pipeline step."
            )

        raw_rows = []
        for record in group.to_dict(orient="records"):
            # Exclude identifiers and labels; transform the same raw fields as inference.
            raw_rows.append(
                {
                    key: value
                    for key, value in record["payload"].items()
                    if key not in {"customerID", "Churn"}
                }
            )
        raw_features = pd.DataFrame(raw_rows)
        transformed = model.named_steps["preprocessor"].transform(raw_features)
        if not isinstance(transformed, pd.DataFrame):
            transformed = pd.DataFrame(
                transformed,
                columns=model.named_steps["preprocessor"].get_feature_names_out(),
            )

        psi_values = calculate_population_stability_index(reference, transformed)
        run_rows: list[dict[str, Any]] = []
        for feature_name, psi_value in psi_values.items():
            status = _status_for_psi(psi_value, ok_below, warning_below)
            result = {
                "timestamp": evaluation_timestamp,
                "model_name": str(model_name),
                "model_version": str(model_version),
                "feature_name": feature_name,
                "psi_value": psi_value,
                "status": status,
            }
            results.append(result)
            run_rows.append(result)

        run_name = f"psi-{model_name}-v{model_version}-{start_at:%Y%m%d}"
        with mlflow.start_run(
            experiment_id=experiment.experiment_id, run_name=run_name
        ):
            mlflow.log_params(
                {
                    "model_name": str(model_name),
                    "model_version": str(model_version),
                    "evaluation_timestamp": evaluation_timestamp,
                    "window_start": start_at.isoformat(),
                    "window_end": end_at.isoformat(),
                    "inference_count": len(group),
                }
            )
            mlflow.set_tags(
                {
                    "monitoring_type": "psi",
                    "source_experiment": config["mlflow"]["experiment_name"],
                    "champion_model_name": champion_name,
                }
            )
            for result in run_rows:
                safe_feature = re.sub(r"[^A-Za-z0-9_]", "_", result["feature_name"])
                mlflow.log_metric(f"psi_{safe_feature}", result["psi_value"])
                mlflow.set_tag(f"psi_status_{safe_feature}", result["status"])

            report = pd.DataFrame(run_rows)
            mlflow.log_text(
                report.to_csv(index=False),
                artifact_file="monitoring/psi_report.csv",
            )

    return results


def main() -> None:
    """Run weekly PSI and print the per-feature result records."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    print(json.dumps(evaluate_psi(args.config), indent=2))


if __name__ == "__main__":
    main()
