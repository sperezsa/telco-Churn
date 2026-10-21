"""Import a current CRM snapshot CSV into the local monitoring database."""

import argparse
import json
from pathlib import Path

from monitoring.common import initialize_context
from telco_churn.config import DEFAULT_CONFIG_PATH
from telco_churn.monitoring_store import import_crm_snapshot


def main() -> None:
    """Upsert the supplied CRM snapshot and report the number of customers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--file",
        type=Path,
        required=True,
        help="CSV with customer_id, snapshot_date, churn_actual, customer_status.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()

    _, database_path, _, _ = initialize_context(args.config)
    imported = import_crm_snapshot(args.file, database_path)
    print(json.dumps({"database": str(database_path), "imported_customers": imported}))


if __name__ == "__main__":
    main()
