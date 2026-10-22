"""Read and validate the raw Telco churn dataset."""

from pathlib import Path

import pandas as pd


# Keep dataset column names in one place for validation and transformations.
TARGET_COLUMN = "Churn"
CUSTOMER_ID_COLUMN = "customerID"


def load_data(path: str | Path) -> pd.DataFrame:
    """Load the CSV and normalize TotalCharges without hiding invalid data."""
    data_path = Path(path)
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Download the Telco churn CSV "
            "as described in the README."
        )

    dataframe = pd.read_csv(data_path)
    required_columns = {TARGET_COLUMN, CUSTOMER_ID_COLUMN, "TotalCharges", "tenure"}
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {', '.join(sorted(missing_columns))}"
        )

    dataframe["TotalCharges"] = pd.to_numeric(
        dataframe["TotalCharges"].astype("string").str.strip(), errors="coerce"
    )
    # Blank TotalCharges is expected only for customers whose tenure is zero.
    invalid_charges = dataframe["TotalCharges"].isna()
    nonzero_tenure = dataframe["tenure"].ne(0)
    if (invalid_charges & nonzero_tenure).any():
        raise ValueError("TotalCharges is missing for customers with non-zero tenure.")
    dataframe["TotalCharges"] = dataframe["TotalCharges"].fillna(0)

    labels = set(dataframe[TARGET_COLUMN].dropna().unique())
    if not labels.issubset({"Yes", "No"}) or dataframe[TARGET_COLUMN].isna().any():
        raise ValueError(f"{TARGET_COLUMN} must contain only 'Yes' and 'No' labels.")

    return dataframe
