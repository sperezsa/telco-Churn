from datetime import date, datetime, timezone

from telco_churn.monitoring_store import (
    import_crm_snapshot,
    initialize_monitoring_store,
    load_current_crm_snapshot,
    load_inference_logs,
    log_inference,
)
from telco_churn.monitoring_utils import previous_completed_week


def test_examples_cover_previous_week_and_current_crm_snapshot(tmp_path) -> None:
    """Seed local SQLite tables with dated examples for t-1 versus CRM today."""
    database_path = tmp_path / "monitoring.db"
    initialize_monitoring_store(
        database_path=database_path,
        sample_model_name="example-model",
        sample_model_version="7",
    )

    logs = load_inference_logs(
        datetime(2026, 10, 12, tzinfo=timezone.utc),
        datetime(2026, 10, 19, tzinfo=timezone.utc),
        database_path,
    )
    snapshot_date, customers = load_current_crm_snapshot(
        date(2026, 10, 21), database_path
    )

    assert len(logs) == 8
    assert {row["model_name"] for row in logs} == {"example-model"}
    assert {row["model_version"] for row in logs} == {"7"}
    assert snapshot_date == "2026-10-21"
    assert len(customers) == 8
    assert {row["churn_actual"] for row in customers} == {0, 1}


def test_log_inference_persists_payload_and_prediction(tmp_path) -> None:
    """Record API data in SQLite without seeding or replacing existing entries."""
    database_path = tmp_path / "monitoring.db"
    called_at = datetime(2026, 10, 20, 11, 30, tzinfo=timezone.utc)
    payload = {"customerID": "CUSTOMER-1", "tenure": 12}

    log_inference(
        customer_id="CUSTOMER-1",
        payload=payload,
        called_at=called_at,
        churn_probability=0.72,
        churn_prediction=1,
        decision_threshold=0.35,
        model_name="telco-churn-xgboost",
        model_version="3",
        database_path=database_path,
    )

    logs = load_inference_logs(
        datetime(2026, 10, 20, tzinfo=timezone.utc),
        datetime(2026, 10, 21, tzinfo=timezone.utc),
        database_path,
    )

    assert len(logs) == 1
    assert logs[0]["payload"] == payload
    assert logs[0]["customer_id"] == "CUSTOMER-1"
    assert logs[0]["churn_prediction"] == 1
    assert logs[0]["churn_probability"] == 0.72
    assert logs[0]["model_version"] == "3"


def test_previous_completed_week_uses_utc_calendar_boundaries() -> None:
    """Evaluate a complete Monday-to-Monday week, never a rolling seven days."""
    start, end = previous_completed_week(
        datetime(2026, 10, 21, 12, tzinfo=timezone.utc)
    )

    assert start == datetime(2026, 10, 12, tzinfo=timezone.utc)
    assert end == datetime(2026, 10, 19, tzinfo=timezone.utc)


def test_import_crm_snapshot_adds_a_new_current_date(tmp_path) -> None:
    """Allow replacing the fixture CRM with a newer dated portfolio snapshot."""
    database_path = tmp_path / "monitoring.db"
    initialize_monitoring_store(database_path=database_path)
    csv_path = tmp_path / "crm_today.csv"
    csv_path.write_text(
        "customer_id,snapshot_date,churn_actual,customer_status\n"
        "SAMPLE-001,2026-10-22,0,active\n",
        encoding="utf-8",
    )

    imported = import_crm_snapshot(csv_path, database_path)
    snapshot_date, customers = load_current_crm_snapshot(
        date(2026, 10, 22), database_path
    )

    assert imported == 1
    assert snapshot_date == "2026-10-22"
    assert customers == [
        {
            "customer_id": "SAMPLE-001",
            "snapshot_date": "2026-10-22",
            "churn_actual": 0,
            "customer_status": "active",
        }
    ]
