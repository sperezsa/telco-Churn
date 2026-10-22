"""Persist and query the top SHAP-ranked features using a local Feast store."""

from datetime import datetime, timedelta, timezone

import pandas as pd
from feast import Entity, Field, FeatureStore, FeatureView, FileSource, ValueType
from feast.types import Float32

from telco_churn.config import PROJECT_ROOT


FEATURE_REPO = PROJECT_ROOT / "feature_store"
FEATURE_DATA_PATH = PROJECT_ROOT / "data" / "feast" / "shap_features.parquet"


def store_shap_features(
    customer_ids: pd.Series,
    transformed_features: pd.DataFrame,
    importance: pd.DataFrame,
) -> list[str]:
    """Write the selected feature values and register/materialize a Feast view."""
    selected_features = importance["feature"].tolist()
    if not selected_features:
        raise ValueError("SHAP did not identify any feature to store in Feast.")

    FEATURE_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Feast stores only the selected columns, keyed by customer and event time.
    feature_data = transformed_features.loc[:, selected_features].copy()
    feature_data.insert(0, "customer_id", customer_ids.to_numpy())
    feature_data["event_timestamp"] = datetime.now(timezone.utc)
    feature_data.to_parquet(FEATURE_DATA_PATH, index=False)

    entity = Entity(
        name="customer_id",
        join_keys=["customer_id"],
        value_type=ValueType.STRING,
    )
    source = FileSource(
        name="telco_shap_features_source",
        path=str(FEATURE_DATA_PATH.resolve()),
        event_timestamp_column="event_timestamp",
    )
    feature_view = FeatureView(
        name="telco_shap_features",
        entities=[entity],
        ttl=timedelta(days=3650),
        schema=[
            Field(name=name, dtype=Float32)
            for name in selected_features
        ],
        source=source,
        online=True,
        tags={"derived_from": "SHAP mean absolute value"},
    )

    FEATURE_REPO.mkdir(parents=True, exist_ok=True)
    feature_store = FeatureStore(repo_path=str(FEATURE_REPO))
    feature_store.apply([entity, source, feature_view])
    feature_store.materialize(
        start_date=datetime.now(timezone.utc) - timedelta(minutes=5),
        end_date=datetime.now(timezone.utc) + timedelta(minutes=1),
    )
    return selected_features


def get_online_features(customer_ids: list[str]) -> list[dict[str, object]]:
    """Read selected customer features from Feast's online SQLite store."""
    feature_store = FeatureStore(repo_path=str(FEATURE_REPO))
    response = feature_store.get_online_features(
        entity_rows=[{"customer_id": customer_id} for customer_id in customer_ids],
        features=[f"telco_shap_features:{name}" for name in _stored_feature_names()],
    ).to_dict()
    return [
        {column: values[index] for column, values in response.items()}
        for index in range(len(customer_ids))
    ]


def _stored_feature_names() -> list[str]:
    """Return the registered Feast feature names or report missing setup."""
    schema_path = FEATURE_REPO / "feature_store.yaml"
    if not schema_path.is_file():
        raise FileNotFoundError(
            f"Feast repository is not initialized at {FEATURE_REPO}; train the model first."
        )

    store = FeatureStore(repo_path=str(FEATURE_REPO))
    views = store.list_feature_views()
    for view in views:
        if view.name == "telco_shap_features":
            return [feature.name for feature in view.features]
    raise RuntimeError("Feast feature view 'telco_shap_features' is not registered.")
