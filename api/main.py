"""FastAPI endpoints for health checks and MLflow-backed churn inference."""

import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow.sklearn import load_model as load_sklearn_model
from pydantic import BaseModel, ConfigDict
from sklearn.pipeline import Pipeline

import mlflow
from telco_churn.config import DEFAULT_CONFIG_PATH, load_config, project_path
from telco_churn.monitoring_store import (
    initialize_monitoring_store,
    log_inference,
)


class CustomerInput(BaseModel):
    """Expected Telco input columns, excluding the training label."""

    model_config = ConfigDict(extra="forbid")

    # Keep the customer identifier for the response, but never pass it to XGBoost.
    customerID: str
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


def create_app(model_uri: str | None = None) -> FastAPI:
    """Create the API and load the registered preprocessing/model pipeline."""
    config = load_config(DEFAULT_CONFIG_PATH)
    mlflow_config = config["mlflow"]
    registered_model_name = os.getenv(
        "MODEL_NAME", mlflow_config["registered_model_name"]
    )
    model_alias = os.getenv("MODEL_ALIAS", mlflow_config["model_alias"])
    resolved_model_uri = model_uri or os.getenv(
        "MODEL_URI", f"models:/{registered_model_name}@{model_alias}"
    )
    decision_threshold = float(config["training"]["decision_threshold"])
    database_path = project_path(config["monitoring"]["database_path"])

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Load the exact versioned MLflow pipeline used for inference."""
        mlflow.set_tracking_uri(mlflow_config["tracking_uri"])
        loaded_model = load_sklearn_model(resolved_model_uri)
        if not isinstance(loaded_model, Pipeline) or "preprocessor" not in loaded_model.named_steps:
            raise TypeError(
                f"MLflow model at {resolved_model_uri} must be a fitted "
                "scikit-learn pipeline with a 'preprocessor' step."
            )
        app.state.model = loaded_model
        app.state.model_uri = resolved_model_uri
        app.state.model_name = registered_model_name
        app.state.model_version = str(
            mlflow.MlflowClient().get_model_version_by_alias(
                registered_model_name, model_alias
            ).version
        )
        initialize_monitoring_store(
            database_path=database_path,
            sample_model_name=registered_model_name,
            sample_model_version=app.state.model_version,
        )
        yield

    app = FastAPI(
        title="Telco Churn Prediction API",
        description="Predictions use the registered MLflow preprocessing and XGBoost pipeline.",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health() -> dict[str, str | None]:
        """Report service readiness and the loaded model URI."""
        return {
            "status": "ok" if getattr(app.state, "model", None) is not None else "starting",
            "model_uri": getattr(app.state, "model_uri", None),
        }

    @app.post("/predict")
    def predict(customer: CustomerInput) -> dict[str, Any]:
        """Transform raw customer input and return its churn probability/class."""
        model = getattr(app.state, "model", None)
        if model is None:
            raise HTTPException(status_code=503, detail="Model is not loaded.")

        called_at = datetime.now(timezone.utc)
        payload = customer.model_dump()
        row = pd.DataFrame([customer.model_dump(exclude={"customerID"})])
        # Pipeline.predict_proba runs the registered preprocessing step before XGBoost.
        churn_probability = float(model.predict_proba(row)[0, 1])
        prediction = int(churn_probability >= decision_threshold)
        log_inference(
            customer_id=customer.customerID,
            payload=payload,
            called_at=called_at,
            churn_probability=churn_probability,
            churn_prediction=prediction,
            decision_threshold=decision_threshold,
            model_name=app.state.model_name,
            model_version=app.state.model_version,
            database_path=database_path,
        )
        return {
            "customerID": customer.customerID,
            "churn_prediction": prediction,
            "churn_probability": churn_probability,
            "decision_threshold": decision_threshold,
        }

    return app


app = create_app()
