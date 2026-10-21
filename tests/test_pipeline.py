import numpy as np
import pandas as pd
import pytest

from telco_churn.evaluate import build_psi_reference, calculate_metrics
from telco_churn.preprocessing import make_preprocessor, split_features_target


def test_split_features_target_excludes_target_and_identifier() -> None:
    """Keep label and direct customer identifier out of model predictors."""
    dataframe = pd.DataFrame(
        {
            "customerID": ["a", "b"],
            "tenure": [1, 2],
            "Contract": ["Month-to-month", "One year"],
            "Churn": ["No", "Yes"],
        }
    )

    features, target, customer_ids = split_features_target(dataframe)

    assert list(features.columns) == ["tenure", "Contract"]
    assert target.tolist() == [0, 1]
    assert customer_ids.tolist() == ["a", "b"]


def test_preprocessor_handles_unseen_categories_and_returns_named_columns() -> None:
    """Preserve the trained feature schema for unseen nominal values."""
    training = pd.DataFrame(
        {
            "tenure": [0, 10, 20],
            "Contract": ["Month-to-month", "One year", "Two year"],
            "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer"],
        }
    )
    preprocessor = make_preprocessor(training)
    transformed = preprocessor.fit_transform(training)
    unseen = preprocessor.transform(
        pd.DataFrame(
            {
                "tenure": [30],
                "Contract": ["Two year"],
                "PaymentMethod": ["Unknown"],
            }
        )
    )

    assert isinstance(transformed, pd.DataFrame)
    assert isinstance(unseen, pd.DataFrame)
    assert list(transformed.columns) == list(unseen.columns)
    assert np.isfinite(unseen.to_numpy()).all()


def test_preprocessor_rejects_unknown_contract_values() -> None:
    """Reject unseen ordinal labels rather than silently assigning an order."""
    training = pd.DataFrame(
        {"tenure": [0, 10], "Contract": ["Month-to-month", "One year"]}
    )
    preprocessor = make_preprocessor(training).fit(training)

    with pytest.raises(ValueError, match="unsupported value"):
        preprocessor.transform(pd.DataFrame({"tenure": [30], "Contract": ["Unknown"]}))


def test_calculate_metrics_uses_configured_churn_threshold() -> None:
    """Verify that lowering the configured threshold affects false negatives."""
    metrics = calculate_metrics(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.4, 0.3, 0.9]),
        threshold=0.35,
    )

    assert metrics["recall"] == 0.5
    assert metrics["false_negatives"] == 1


def test_calculate_metrics_rejects_invalid_threshold() -> None:
    """Reject threshold values that cannot define a probability decision."""
    with pytest.raises(ValueError, match="between 0 and 1"):
        calculate_metrics(np.array([0, 1]), np.array([0.2, 0.8]), threshold=1)


def test_build_psi_reference_stores_train_proportions_and_reusable_bins() -> None:
    """Store distributions and stable bins for categorical and continuous fields."""
    training = pd.DataFrame(
        {
            "Contract": [1, 1, 2, 3, 3],
            "MonthlyCharges": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )

    reference = build_psi_reference(
        training,
        feature_names=["Contract", "MonthlyCharges"],
        bin_count=4,
    )

    assert reference["reference_split"] == "train"
    assert reference["reference_rows"] == 5
    contract = reference["features"]["Contract"]
    assert contract["strategy"] == "categorical"
    assert contract["categories"] == [1.0, 2.0, 3.0]
    assert sum(contract["reference_proportions"]) == pytest.approx(1)
    charges = reference["features"]["MonthlyCharges"]
    assert charges["strategy"] == "quantile"
    assert len(charges["split_points"]) == 3
    assert len(charges["reference_proportions"]) == 4
    assert sum(charges["reference_proportions"]) == pytest.approx(1)


def test_build_psi_reference_rejects_missing_features() -> None:
    """Fail explicitly if SHAP-selected columns are unavailable."""
    with pytest.raises(ValueError, match="missing"):
        build_psi_reference(
            pd.DataFrame({"tenure": [1, 2]}),
            feature_names=["Contract"],
        )
