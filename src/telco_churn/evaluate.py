"""Classification metrics used consistently for train and test evaluation."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PSI_EPSILON = 1e-6


def build_psi_reference(
    training_features: pd.DataFrame,
    feature_names: list[str],
    bin_count: int = 10,
    epsilon: float = PSI_EPSILON,
) -> dict[str, Any]:
    """Create reusable train distributions and bin definitions for PSI scoring.

    Features with few distinct values are treated as categorical; all others use
    quantile cut points. The returned split points let a later evaluation apply
    exactly the same bins to test or production data.
    """
    if training_features.empty:
        raise ValueError("Training features must contain at least one row.")
    if not feature_names:
        raise ValueError("At least one feature is required to build a PSI reference.")
    if bin_count < 2:
        raise ValueError("PSI bin_count must be at least 2.")
    if not 0 < epsilon < 1:
        raise ValueError("PSI epsilon must be between 0 and 1.")

    missing_features = set(feature_names).difference(training_features.columns)
    if missing_features:
        raise ValueError(
            f"PSI reference features are missing: {', '.join(sorted(missing_features))}"
        )

    reference: dict[str, Any] = {
        "schema_version": 1,
        "reference_split": "train",
        "reference_rows": int(len(training_features)),
        "epsilon": epsilon,
        "features": {},
    }
    feature_references: dict[str, Any] = reference["features"]

    for feature_name in feature_names:
        values = training_features[feature_name]
        if values.isna().any():
            raise ValueError(f"Feature '{feature_name}' contains missing values.")

        unique_values = sorted(values.unique().tolist())
        if len(unique_values) <= bin_count:
            proportions = values.value_counts(normalize=True).reindex(
                unique_values, fill_value=0
            )
            feature_references[feature_name] = {
                "strategy": "categorical",
                "categories": [float(value) for value in unique_values],
                "bin_labels": [str(value) for value in unique_values],
                "reference_proportions": [float(value) for value in proportions],
            }
            continue

        quantiles = np.linspace(0, 1, bin_count + 1)[1:-1]
        split_points = (
            values.quantile(quantiles).drop_duplicates().sort_values().tolist()
        )
        bin_ids = np.searchsorted(split_points, values.to_numpy(), side="left")
        bin_total = len(split_points) + 1
        proportions = np.bincount(bin_ids, minlength=bin_total) / len(values)
        feature_references[feature_name] = {
            "strategy": "quantile",
            "split_points": [float(value) for value in split_points],
            "bin_labels": [
                f"bin_{index}" for index in range(bin_total)
            ],
            "reference_proportions": [float(value) for value in proportions],
        }

    return reference


def calculate_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Calculate thresholded classification metrics and probability metrics."""
    if not 0 < threshold < 1:
        raise ValueError("Decision threshold must be between 0 and 1.")

    # Threshold only the churn probability; keep raw probabilities for ranking metrics.
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "true_negatives": float(tn),
        "false_positives": float(fp),
        "false_negatives": float(fn),
        "true_positives": float(tp),
    }


def evaluate_model(
    model: Any,
    features: Any,
    target: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Evaluate a fitted probabilistic classifier on one dataset split."""
    probabilities = model.predict_proba(features)[:, 1]
    return calculate_metrics(target, probabilities, threshold)
