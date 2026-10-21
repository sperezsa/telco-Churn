"""SQLite persistence for inference logs and CRM snapshots used in monitoring."""

import csv
import json
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from telco_churn.config import PROJECT_ROOT


DEFAULT_DATABASE_PATH = PROJECT_ROOT / "data" / "monitoring" / "monitoring.db"
EXAMPLE_DATA_PATH = PROJECT_ROOT / "data" / "examples"
INFERENCE_LOG_EXAMPLE = EXAMPLE_DATA_PATH / "log_inferencia.csv"
CRM_EXAMPLE = EXAMPLE_DATA_PATH / "clientes_crm.csv"


def initialize_monitoring_store(
    database_path: Path = DEFAULT_DATABASE_PATH,
    seed_examples: bool = True,
    sample_model_name: str = "telco-churn-xgboost",
    sample_model_version: str = "1",
) -> None:
    """Create local monitoring tables and seed each empty table from CSV examples."""
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS log_inferencia (
                inference_id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id TEXT NOT NULL,
                called_at TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                churn_probability REAL NOT NULL CHECK (
                    churn_probability >= 0 AND churn_probability <= 1
                ),
                churn_prediction INTEGER NOT NULL CHECK (
                    churn_prediction IN (0, 1)
                ),
                decision_threshold REAL NOT NULL CHECK (
                    decision_threshold > 0 AND decision_threshold < 1
                ),
                model_name TEXT NOT NULL,
                model_version TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_log_inferencia_called_at
            ON log_inferencia (called_at)
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS clientes_crm (
                customer_id TEXT NOT NULL,
                snapshot_date TEXT NOT NULL,
                churn_actual INTEGER NOT NULL CHECK (churn_actual IN (0, 1)),
                customer_status TEXT NOT NULL,
                PRIMARY KEY (customer_id, snapshot_date)
            )
            """
        )

        if seed_examples:
            _seed_inference_examples(
                connection,
                sample_model_name=sample_model_name,
                sample_model_version=sample_model_version,
            )
            _seed_crm_examples(connection)


def log_inference(
    customer_id: str,
    payload: dict[str, Any],
    called_at: datetime,
    churn_probability: float,
    churn_prediction: int,
    decision_threshold: float,
    model_name: str,
    model_version: str,
    database_path: Path = DEFAULT_DATABASE_PATH,
) -> None:
    """Persist the request and output values for a completed API prediction."""
    initialize_monitoring_store(database_path, seed_examples=False)
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            INSERT INTO log_inferencia (
                customer_id, called_at, payload_json, churn_probability,
                churn_prediction, decision_threshold, model_name, model_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                customer_id,
                called_at.astimezone(timezone.utc).isoformat(timespec="seconds"),
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
                churn_probability,
                churn_prediction,
                decision_threshold,
                model_name,
                str(model_version),
            ),
        )


def load_inference_logs(
    start_at: datetime,
    end_at: datetime,
    database_path: Path = DEFAULT_DATABASE_PATH,
) -> list[dict[str, Any]]:
    """Return inference records in the half-open UTC interval [start_at, end_at)."""
    if not database_path.is_file():
        raise FileNotFoundError(f"Monitoring database not found: {database_path}")

    start_text = start_at.astimezone(timezone.utc).isoformat(timespec="seconds")
    end_text = end_at.astimezone(timezone.utc).isoformat(timespec="seconds")
    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT inference_id, customer_id, called_at, payload_json,
                   churn_probability, churn_prediction, decision_threshold,
                   model_name, model_version
            FROM log_inferencia
            WHERE called_at >= ? AND called_at < ?
            ORDER BY called_at, inference_id
            """,
            (start_text, end_text),
        ).fetchall()

    records = [dict(row) for row in rows]
    for record in records:
        record["payload"] = json.loads(record.pop("payload_json"))
    return records


def load_current_crm_snapshot(
    as_of: date,
    database_path: Path = DEFAULT_DATABASE_PATH,
) -> tuple[str, list[dict[str, Any]]]:
    """Return all CRM rows from the latest complete snapshot on or before a date."""
    if not database_path.is_file():
        raise FileNotFoundError(f"Monitoring database not found: {database_path}")

    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        latest = connection.execute(
            """
            SELECT MAX(snapshot_date) AS snapshot_date
            FROM clientes_crm
            WHERE snapshot_date <= ?
            """,
            (as_of.isoformat(),),
        ).fetchone()["snapshot_date"]
        if latest is None:
            raise ValueError(f"No CRM snapshot exists on or before {as_of.isoformat()}.")

        rows = connection.execute(
            """
            SELECT customer_id, snapshot_date, churn_actual, customer_status
            FROM clientes_crm
            WHERE snapshot_date = ?
            ORDER BY customer_id
            """,
            (latest,),
        ).fetchall()

    return latest, [dict(row) for row in rows]


def import_crm_snapshot(
    csv_path: Path,
    database_path: Path = DEFAULT_DATABASE_PATH,
) -> int:
    """Insert or update CRM records from one dated CSV snapshot."""
    initialize_monitoring_store(database_path, seed_examples=False)
    required_columns = {
        "customer_id",
        "snapshot_date",
        "churn_actual",
        "customer_status",
    }
    imported = 0
    with csv_path.open(encoding="utf-8", newline="") as snapshot_file:
        reader = csv.DictReader(snapshot_file)
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            missing = required_columns.difference(reader.fieldnames or [])
            raise ValueError(
                f"CRM CSV is missing required columns: {', '.join(sorted(missing))}"
            )

        rows = []
        for row in reader:
            customer_id = row["customer_id"].strip()
            if not customer_id:
                raise ValueError("CRM snapshot contains an empty customer_id.")
            snapshot_date = date.fromisoformat(row["snapshot_date"])
            churn_actual = int(row["churn_actual"])
            customer_status = row["customer_status"].strip()
            if churn_actual not in (0, 1):
                raise ValueError("CRM churn_actual values must be 0 or 1.")
            expected_status = "churned" if churn_actual else "active"
            if customer_status != expected_status:
                raise ValueError(
                    f"CRM status for {customer_id} must be '{expected_status}' "
                    f"when churn_actual={churn_actual}."
                )
            rows.append(
                (customer_id, snapshot_date.isoformat(), churn_actual, customer_status)
            )
        if not rows:
            raise ValueError("CRM snapshot CSV contains no customer rows.")

    with sqlite3.connect(database_path) as connection:
        connection.executemany(
            """
            INSERT OR REPLACE INTO clientes_crm (
                customer_id, snapshot_date, churn_actual, customer_status
            ) VALUES (?, ?, ?, ?)
            """,
            rows,
        )
        imported = len(rows)
    return imported


def _seed_inference_examples(
    connection: sqlite3.Connection,
    sample_model_name: str,
    sample_model_version: str,
) -> None:
    """Load dated example predictions once, without overwriting real API logs."""
    current_count = connection.execute(
        "SELECT COUNT(*) FROM log_inferencia"
    ).fetchone()[0]
    if current_count or not INFERENCE_LOG_EXAMPLE.is_file():
        return

    with INFERENCE_LOG_EXAMPLE.open(encoding="utf-8", newline="") as example_file:
        for row in csv.DictReader(example_file):
            connection.execute(
                """
                INSERT OR IGNORE INTO log_inferencia (
                    customer_id, called_at, payload_json, churn_probability,
                    churn_prediction, decision_threshold, model_name, model_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["customer_id"],
                    row["called_at"],
                    row["payload_json"],
                    float(row["churn_probability"]),
                    int(row["churn_prediction"]),
                    float(row["decision_threshold"]),
                    sample_model_name
                    if row["model_name"] == "@configured"
                    else row["model_name"],
                    sample_model_version
                    if row["model_version"] == "@configured"
                    else row["model_version"],
                ),
            )


def _seed_crm_examples(connection: sqlite3.Connection) -> None:
    """Load one current CRM example snapshot if no CRM records exist yet."""
    current_count = connection.execute(
        "SELECT COUNT(*) FROM clientes_crm"
    ).fetchone()[0]
    if current_count or not CRM_EXAMPLE.is_file():
        return

    with CRM_EXAMPLE.open(encoding="utf-8", newline="") as example_file:
        for row in csv.DictReader(example_file):
            connection.execute(
                """
                INSERT OR IGNORE INTO clientes_crm (
                    customer_id, snapshot_date, churn_actual, customer_status
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    row["customer_id"],
                    row["snapshot_date"],
                    int(row["churn_actual"]),
                    row["customer_status"],
                ),
            )
