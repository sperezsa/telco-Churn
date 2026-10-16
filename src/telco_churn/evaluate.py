"""Classification metrics used consistently for train and test evaluation."""

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


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
