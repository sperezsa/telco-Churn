"""Persistence for inference logs and CRM snapshots used in monitoring."""

import csv
from functools import lru_cache
import json
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, TypeAlias

from sqlalchemy import (
    CheckConstraint,
    Column,
    Float,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    func,
    insert,
    select,
)
from sqlalchemy.engine import Connection, Engine, URL

from telco_churn.config import PROJECT_ROOT, project_path


DEFAULT_DATABASE_PATH = PROJECT_ROOT / "data" / "monitoring" / "monitoring.db"
EXAMPLE_DATA_PATH = PROJECT_ROOT / "data" / "examples"
INFERENCE_LOG_EXAMPLE = EXAMPLE_DATA_PATH / "log_inferencia.csv"
CRM_EXAMPLE = EXAMPLE_DATA_PATH / "clientes_crm.csv"
CRM_INSERT_BATCH_SIZE = 500
DatabaseTarget: TypeAlias = Path | str

metadata = MetaData()
inference_logs = Table(
    "log_inferencia",
    metadata,
    Column("inference_id", Integer, primary_key=True, autoincrement=True),
    Column("customer_id", String, nullable=False),
    Column("called_at", Text, nullable=False),
    Column("payload_json", Text, nullable=False),
    Column(
        "churn_probability",
        Float,
        CheckConstraint("churn_probability >= 0 AND churn_probability <= 1"),
        nullable=False,
    ),
    Column(
        "churn_prediction",
        Integer,
        CheckConstraint("churn_prediction IN (0, 1)"),
        nullable=False,
    ),
    Column(
        "decision_threshold",
        Float,
        CheckConstraint("decision_threshold > 0 AND decision_threshold < 1"),
        nullable=False,
    ),
    Column("model_name", String, nullable=False),
    Column("model_version", String, nullable=False),
)
Index("idx_log_inferencia_called_at", inference_logs.c.called_at)
crm_customers = Table(
    "clientes_crm",
    metadata,
    Column("customer_id", String, primary_key=True),
    Column("snapshot_date", String, primary_key=True),
    Column(
        "churn_actual",
        Integer,
        CheckConstraint("churn_actual IN (0, 1)"),
        nullable=False,
    ),
    Column("customer_status", String, nullable=False),
)


def monitoring_database_target(configured_path: str | Path) -> DatabaseTarget:
    """Resolve the monitoring URL from the environment or project configuration."""
    database_url = os.getenv("MONITORING_DATABASE_URL")
    if database_url:
        return database_url
    configured_value = str(configured_path)
    if "://" in configured_value:
        return configured_value
    return project_path(configured_path)


@lru_cache(maxsize=8)
def _postgres_engine(database_url: str) -> Engine:
    return create_engine(database_url, pool_pre_ping=True)


def _create_engine(database_target: DatabaseTarget) -> Engine:
    if isinstance(database_target, Path):
        database_target.parent.mkdir(parents=True, exist_ok=True)
        url = URL.create(
            "sqlite+pysqlite", database=str(database_target.resolve())
        )
    elif "://" in database_target:
        url = database_target
    else:
        path = Path(database_target)
        path.parent.mkdir(parents=True, exist_ok=True)
        url = URL.create("sqlite+pysqlite", database=str(path.resolve()))
    if isinstance(url, str) and url.startswith("postgresql"):
        return _postgres_engine(url)
    return create_engine(
        url,
        connect_args={"check_same_thread": False}
        if isinstance(url, URL) and url.get_backend_name() == "sqlite"
        else {},
    )


def _dispose_engine(engine: Engine) -> None:
    if engine.dialect.name != "postgresql":
        engine.dispose()


def _insert_ignoring_conflicts(
    connection: Connection,
    table: Table,
    values: dict[str, Any],
) -> None:
    if connection.dialect.name == "sqlite":
        from sqlalchemy.dialects.sqlite import insert as dialect_insert
    elif connection.dialect.name == "postgresql":
        from sqlalchemy.dialects.postgresql import insert as dialect_insert
    else:
        connection.execute(insert(table).values(**values))
        return
    statement = dialect_insert(table).values(**values).on_conflict_do_nothing()
    connection.execute(statement)


def initialize_monitoring_store(
    database_path: DatabaseTarget = DEFAULT_DATABASE_PATH,
    seed_examples: bool | None = None,
    sample_model_name: str = "telco-churn-xgboost",
    sample_model_version: str = "1",
) -> None:
    """Create monitoring tables and seed local SQLite tables with examples."""
    if seed_examples is None:
        seed_examples = isinstance(database_path, Path)
    engine = _create_engine(database_path)
    metadata.create_all(engine)
    with engine.begin() as connection:
        if seed_examples:
            _seed_inference_examples(
                connection,
                sample_model_name=sample_model_name,
                sample_model_version=sample_model_version,
            )
            _seed_crm_examples(connection)
    _dispose_engine(engine)


def log_inference(
    customer_id: str,
    payload: dict[str, Any],
    called_at: datetime,
    churn_probability: float,
    churn_prediction: int,
    decision_threshold: float,
    model_name: str,
    model_version: str,
    database_path: DatabaseTarget = DEFAULT_DATABASE_PATH,
) -> None:
    """Persist the request and output values for a completed API prediction."""
    initialize_monitoring_store(database_path, seed_examples=False)
    engine = _create_engine(database_path)
    with engine.begin() as connection:
        connection.execute(
            insert(inference_logs).values(
                customer_id=customer_id,
                called_at=called_at.astimezone(timezone.utc).isoformat(
                    timespec="seconds"
                ),
                payload_json=json.dumps(payload, ensure_ascii=False),
                churn_probability=churn_probability,
                churn_prediction=churn_prediction,
                decision_threshold=decision_threshold,
                model_name=model_name,
                model_version=str(model_version),
            )
        )
    _dispose_engine(engine)


def load_inference_logs(
    start_at: datetime,
    end_at: datetime,
    database_path: DatabaseTarget = DEFAULT_DATABASE_PATH,
) -> list[dict[str, Any]]:
    """Return inference records in the half-open UTC interval [start_at, end_at)."""
    _ensure_database_exists(database_path)

    start_text = start_at.astimezone(timezone.utc).isoformat(timespec="seconds")
    end_text = end_at.astimezone(timezone.utc).isoformat(timespec="seconds")
    engine = _create_engine(database_path)
    with engine.connect() as connection:
        rows = connection.execute(
            select(inference_logs)
            .where(
                inference_logs.c.called_at >= start_text,
                inference_logs.c.called_at < end_text,
            )
            .order_by(inference_logs.c.called_at, inference_logs.c.inference_id)
        ).mappings().all()
    _dispose_engine(engine)

    records = [dict(row) for row in rows]
    for record in records:
        record["payload"] = json.loads(record.pop("payload_json"))
    return records


def load_current_crm_snapshot(
    as_of: date,
    database_path: DatabaseTarget = DEFAULT_DATABASE_PATH,
) -> tuple[str, list[dict[str, Any]]]:
    """Return all CRM rows from the latest complete snapshot on or before a date."""
    _ensure_database_exists(database_path)

    engine = _create_engine(database_path)
    with engine.connect() as connection:
        latest = connection.execute(
            select(func.max(crm_customers.c.snapshot_date).label("snapshot_date"))
            .where(crm_customers.c.snapshot_date <= as_of.isoformat())
        ).mappings().one()["snapshot_date"]
        if latest is None:
            raise ValueError(f"No CRM snapshot exists on or before {as_of.isoformat()}.")

        rows = connection.execute(
            select(crm_customers)
            .where(crm_customers.c.snapshot_date == latest)
            .order_by(crm_customers.c.customer_id)
        ).mappings().all()

    _dispose_engine(engine)
    return latest, [dict(row) for row in rows]


def import_crm_snapshot(
    csv_path: Path,
    database_path: DatabaseTarget = DEFAULT_DATABASE_PATH,
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

    engine = _create_engine(database_path)
    with engine.begin() as connection:
        values = [
            {
                "customer_id": customer_id,
                "snapshot_date": snapshot_date,
                "churn_actual": churn_actual,
                "customer_status": customer_status,
            }
            for customer_id, snapshot_date, churn_actual, customer_status in rows
        ]
        for start in range(0, len(values), CRM_INSERT_BATCH_SIZE):
            batch = values[start : start + CRM_INSERT_BATCH_SIZE]
            if connection.dialect.name == "sqlite":
                from sqlalchemy.dialects.sqlite import insert as dialect_insert
            elif connection.dialect.name == "postgresql":
                from sqlalchemy.dialects.postgresql import insert as dialect_insert
            else:
                connection.execute(
                    insert(crm_customers).values(batch).prefix_with("OR REPLACE")
                )
                continue
            statement = dialect_insert(crm_customers).values(batch)
            connection.execute(
                statement.on_conflict_do_update(
                    index_elements=["customer_id", "snapshot_date"],
                    set_={
                        "churn_actual": statement.excluded.churn_actual,
                        "customer_status": statement.excluded.customer_status,
                    },
                )
            )
        imported = len(rows)
    _dispose_engine(engine)
    return imported


def _ensure_database_exists(database_path: DatabaseTarget) -> None:
    """Keep local SQLite failures explicit while allowing remote database URLs."""
    if isinstance(database_path, Path):
        if not database_path.is_file():
            raise FileNotFoundError(f"Monitoring database not found: {database_path}")
    elif "://" not in database_path and not Path(database_path).is_file():
        raise FileNotFoundError(f"Monitoring database not found: {database_path}")


def _seed_inference_examples(
    connection: Connection,
    sample_model_name: str,
    sample_model_version: str,
) -> None:
    """Load dated example predictions once, without overwriting real API logs."""
    current_count = connection.execute(
        select(func.count()).select_from(inference_logs)
    ).scalar_one()
    if current_count or not INFERENCE_LOG_EXAMPLE.is_file():
        return

    with INFERENCE_LOG_EXAMPLE.open(encoding="utf-8", newline="") as example_file:
        for row in csv.DictReader(example_file):
            _insert_ignoring_conflicts(
                connection,
                inference_logs,
                {
                    "customer_id": row["customer_id"],
                    "called_at": row["called_at"],
                    "payload_json": row["payload_json"],
                    "churn_probability": float(row["churn_probability"]),
                    "churn_prediction": int(row["churn_prediction"]),
                    "decision_threshold": float(row["decision_threshold"]),
                    "model_name": sample_model_name
                    if row["model_name"] == "@configured"
                    else row["model_name"],
                    "model_version": sample_model_version
                    if row["model_version"] == "@configured"
                    else row["model_version"],
                },
            )


def _seed_crm_examples(connection: Connection) -> None:
    """Load one current CRM example snapshot if no CRM records exist yet."""
    current_count = connection.execute(
        select(func.count()).select_from(crm_customers)
    ).scalar_one()
    if current_count or not CRM_EXAMPLE.is_file():
        return

    with CRM_EXAMPLE.open(encoding="utf-8", newline="") as example_file:
        for row in csv.DictReader(example_file):
            _insert_ignoring_conflicts(
                connection,
                crm_customers,
                {
                    "customer_id": row["customer_id"],
                    "snapshot_date": row["snapshot_date"],
                    "churn_actual": int(row["churn_actual"]),
                    "customer_status": row["customer_status"],
                },
            )
