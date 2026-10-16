"""Train, explain, evaluate, and persist the selected XGBoost churn pipeline."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import shap
from mlflow import register_model
from mlflow.sklearn import log_model as log_sklearn_model
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from telco_churn.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load_config, project_path
from telco_churn.data_loader import load_data
from telco_churn.evaluate import evaluate_model
from telco_churn.feature_store import store_shap_features
from telco_churn.preprocessing import make_preprocessor, split_features_target


def shap_importance(
    model: Pipeline,
    transformed_features: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    """Rank encoded columns by mean absolute positive-class SHAP value."""
    if top_k < 1:
        raise ValueError("The number of SHAP features to select must be positive.")

    # Explain the fitted tree model directly because its input is already encoded.
    classifier = model.named_steps["classifier"]
    explainer = shap.TreeExplainer(classifier)
    shap_values = np.asarray(explainer(transformed_features).values)

    # Binary XGBoost explanations have one SHAP value per sample and feature.
    expected_shape = (len(transformed_features), len(transformed_features.columns))
    if shap_values.ndim != 2 or shap_values.shape != expected_shape:
        raise ValueError(
            "Binary XGBoost SHAP values must have one value per sample and feature; "
            f"received shape {shap_values.shape}."
        )

    importance = pd.DataFrame(
        {
            "feature": transformed_features.columns,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        }
    )
    return importance.sort_values("mean_abs_shap", ascending=False).head(top_k)


def _log_metrics(prefix: str, metrics: dict[str, float]) -> None:
    """Log one metric set with a train/test prefix to the active MLflow run."""
    mlflow.log_metrics({f"{prefix}_{name}": value for name, value in metrics.items()})


def train(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Fit and evaluate XGBoost, then log its artifacts to MLflow and Feast."""
    config = load_config(config_path)
    dataframe = load_data(project_path(config["data"]["path"]))
    features, target, customer_ids = split_features_target(
        dataframe,
        target_column=config["data"]["target"],
        customer_id_column=config["data"]["customer_id"],
    )
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=config["training"]["test_size"],
        random_state=config["training"]["random_state"],
        stratify=target,
    )

    # Derive class balance only from training labels to avoid test-set leakage.
    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())
    if positive_count == 0:
        raise ValueError("Training data must contain positive churn examples.")
    scale_pos_weight = negative_count / positive_count

    # Put preprocessing and the classifier in one pipeline so inference uses
    # the exact encoders/imputers learned from the training split.
    preprocessor = make_preprocessor(X_train)
    model_parameters = dict(config["model"])
    model_parameters.update(
        scale_pos_weight=scale_pos_weight,
        random_state=config["training"]["random_state"],
        eval_metric="logloss",
        n_jobs=-1,
    )
    classifier = XGBClassifier(**model_parameters)
    model = Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    with mlflow.start_run() as run:
        model.fit(X_train, y_train)
        threshold = float(config["training"]["decision_threshold"])
        train_metrics = evaluate_model(model, X_train, y_train.to_numpy(), threshold)
        test_metrics = evaluate_model(model, X_test, y_test.to_numpy(), threshold)
        _log_metrics("train", train_metrics)
        _log_metrics("test", test_metrics)

        mlflow.log_params(
            {
                **config["model"],
                "scale_pos_weight": scale_pos_weight,
                "decision_threshold": threshold,
                "test_size": config["training"]["test_size"],
                "random_state": config["training"]["random_state"],
                "shap_top_k": config["training"]["shap_top_k"],
            }
        )

        # Select features from training examples only; holdout data remains unseen
        # during both model fitting and SHAP-based feature selection.
        transformed_train = model.named_steps["preprocessor"].transform(X_train)
        transformed = model.named_steps["preprocessor"].transform(features)
        feature_names = model.named_steps["preprocessor"].get_feature_names_out()
        if not isinstance(transformed_train, pd.DataFrame):
            transformed_train = pd.DataFrame(
                transformed_train, columns=feature_names, index=X_train.index
            )
        if not isinstance(transformed, pd.DataFrame):
            transformed = pd.DataFrame(
                transformed, columns=feature_names, index=features.index
            )
        importance = shap_importance(
            model,
            transformed_train,
            top_k=int(config["training"]["shap_top_k"]),
        )
        importance_path = PROJECT_ROOT / "models" / "feature_importance.csv"
        importance_path.parent.mkdir(parents=True, exist_ok=True)
        importance.to_csv(importance_path, index=False)
        # Store the selected columns for every source customer after selection
        # has been completed without consulting the held-out test set.
        stored_features = store_shap_features(customer_ids, transformed, importance)

        mlflow.log_artifact(str(importance_path), artifact_path="explainability")
        mlflow.log_text(
            json.dumps(stored_features, indent=2),
            artifact_file="explainability/feast_features.json",
        )
        logged_model = log_sklearn_model(
            model,
            name="xgboost_pipeline",
            skops_trusted_types=[
                "numpy.dtype",
                "telco_churn.preprocessing._encode_business_categories",
                "xgboost.core.Booster",
                "xgboost.sklearn.XGBClassifier",
            ],
        )
        # Register this pipeline as a versioned MLflow model and advance the local
        # development alias so the API can load the same artifact by stable URI.
        registered_model = register_model(
            model_uri=logged_model.model_uri,
            name=config["mlflow"]["registered_model_name"],
        )
        mlflow.MlflowClient().set_registered_model_alias(
            name=config["mlflow"]["registered_model_name"],
            alias=config["mlflow"]["model_alias"],
            version=registered_model.version,
        )

        return {
            "run_id": run.info.run_id,
            "registered_model_name": config["mlflow"]["registered_model_name"],
            "registered_model_version": registered_model.version,
            "model_alias": config["mlflow"]["model_alias"],
            "scale_pos_weight": scale_pos_weight,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "shap_features": stored_features,
            "model_uri": (
                f"models:/{config['mlflow']['registered_model_name']}"
                f"@{config['mlflow']['model_alias']}"
            ),
        }


def main() -> None:
    """Parse CLI options and print the completed training summary."""
    # MLflow writes Unicode run links when a run closes; use a safe console codec on Windows.
    reconfigure_stdout = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure_stdout):
        reconfigure_stdout(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Train the Telco churn XGBoost model.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    print(json.dumps(train(args.config), indent=2))


if __name__ == "__main__":
    main()
