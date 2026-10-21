"""Evaluate last week's API predictions against the latest CRM snapshot."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

from monitoring.common import configure_console, initialize_context, previous_completed_week
from telco_churn.config import DEFAULT_CONFIG_PATH
from telco_churn.evaluate import calculate_metrics
from telco_churn.monitoring_store import (
    load_current_crm_snapshot,
    load_inference_logs,
)


def evaluate_performance(
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> list[dict[str, Any]]:
    """Join previous-week inferences to today's CRM truth and log metrics."""
    configure_console()
    config, database_path, _, _ = initialize_context(config_path)
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    experiment = mlflow.set_experiment(config["monitoring"]["experiment_name"])
    evaluation_time = datetime.now(timezone.utc)
    start_at, end_at = previous_completed_week(evaluation_time)
    inference_logs = load_inference_logs(start_at, end_at, database_path)
    if not inference_logs:
        raise ValueError(
            f"No inference logs found for previous week "
            f"[{start_at.isoformat()}, {end_at.isoformat()})."
        )

    snapshot_date, crm_records = load_current_crm_snapshot(
        evaluation_time.date(), database_path
    )
    crm_by_customer = {
        record["customer_id"]: record for record in crm_records
    }
    matched: list[dict[str, Any]] = []
    unmatched_count = 0
    for inference in inference_logs:
        crm_record = crm_by_customer.get(inference["customer_id"])
        if crm_record is None:
            unmatched_count += 1
            continue
        matched.append({**inference, **crm_record})

    if not matched:
        raise ValueError(
            f"No previous-week inference customers match CRM snapshot {snapshot_date}."
        )

    evaluation_timestamp = evaluation_time.isoformat(timespec="seconds")
    results: list[dict[str, Any]] = []
    # Build typed keys explicitly rather than relying on pandas GroupBy, whose
    # iterator exposes each key as Hashable and obscures the numeric threshold type.
    groups: dict[tuple[str, str, float], list[dict[str, Any]]] = {}
    for record in matched:
        group_key = (
            str(record["model_name"]),
            str(record["model_version"]),
            float(record["decision_threshold"]),
        )
        groups.setdefault(group_key, []).append(record)

    for (model_name, model_version, threshold), group_records in sorted(groups.items()):
        group = pd.DataFrame(group_records)
        metrics = calculate_metrics(
            y_true=group["churn_actual"].to_numpy(dtype=int),
            probabilities=group["churn_probability"].to_numpy(dtype=float),
            threshold=threshold,
        )
        result = {
            "timestamp": evaluation_timestamp,
            "model_name": str(model_name),
            "model_version": str(model_version),
            "crm_snapshot_date": snapshot_date,
            "window_start": start_at.isoformat(),
            "window_end": end_at.isoformat(),
            "decision_threshold": threshold,
            "matched_inferences": int(len(group)),
            "unmatched_inferences": unmatched_count,
            **metrics,
        }
        results.append(result)

        run_name = f"performance-{model_name}-v{model_version}-{start_at:%Y%m%d}"
        with mlflow.start_run(
            experiment_id=experiment.experiment_id, run_name=run_name
        ):
            mlflow.log_params(
                {
                    "model_name": str(model_name),
                    "model_version": str(model_version),
                    "evaluation_timestamp": evaluation_timestamp,
                    "crm_snapshot_date": snapshot_date,
                    "window_start": start_at.isoformat(),
                    "window_end": end_at.isoformat(),
                    "decision_threshold": threshold,
                    "matched_inferences": len(group),
                    "unmatched_inferences": unmatched_count,
                }
            )
            mlflow.set_tags(
                {
                    "monitoring_type": "performance",
                    "ground_truth": "latest CRM snapshot",
                }
            )
            mlflow.log_metrics(metrics)

            audit = group[
                [
                    "inference_id",
                    "customer_id",
                    "called_at",
                    "churn_probability",
                    "churn_prediction",
                    "churn_actual",
                    "customer_status",
                    "model_name",
                    "model_version",
                ]
            ].copy()
            mlflow.log_text(
                audit.to_csv(index=False),
                artifact_file="monitoring/performance_matches.csv",
            )

    return results


def main() -> None:
    """Run previous-week performance evaluation and print logged metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    print(json.dumps(evaluate_performance(args.config), indent=2))


if __name__ == "__main__":
    main()
