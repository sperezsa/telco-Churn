"""Business-specific encoding and leakage-safe model preprocessing."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder


# Keep these mappings explicit so invalid source labels fail instead of being guessed.
_BINARY_COLUMNS = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
_BINARY_MAPPING = {"No": 0, "Yes": 1}
_CONTRACT_MAPPING = {"Month-to-month": 1, "One year": 2, "Two year": 3}


def _encode_business_categories(features: pd.DataFrame) -> pd.DataFrame:
    """Convert binary and ordered contract labels to numeric values."""
    encoded = features.copy()
    for column in _BINARY_COLUMNS:
        if column in encoded:
            encoded[column] = encoded[column].map(_BINARY_MAPPING)
            if encoded[column].isna().any():
                raise ValueError(f"Column '{column}' contains an unsupported value.")
    if "Contract" in encoded:
        encoded["Contract"] = encoded["Contract"].map(_CONTRACT_MAPPING)
        if encoded["Contract"].isna().any():
            raise ValueError("Column 'Contract' contains an unsupported value.")
    return encoded


def split_features_target(
    dataframe: pd.DataFrame,
    target_column: str = "Churn",
    customer_id_column: str = "customerID",
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Separate raw predictors, binary target, and customer identifiers."""
    if target_column not in dataframe.columns:
        raise ValueError(f"Target column '{target_column}' is not present.")
    if customer_id_column not in dataframe.columns:
        raise ValueError(f"Customer ID column '{customer_id_column}' is not present.")

    target = dataframe[target_column].map({"No": 0, "Yes": 1})
    if target.isna().any():
        raise ValueError(f"Target column '{target_column}' contains unsupported labels.")

    customer_ids = dataframe[customer_id_column].astype(str)
    features = dataframe.drop(columns=[target_column, customer_id_column])
    return features, target.astype("int8"), customer_ids


def make_preprocessor(features: pd.DataFrame) -> Pipeline:
    """Build an imputation and encoding pipeline fitted only on training data."""
    business_numeric = [column for column in [*_BINARY_COLUMNS, "Contract"] if column in features]
    numeric_columns = list(
        dict.fromkeys(
            [*features.select_dtypes(include=["number"]).columns, *business_numeric]
        )
    )
    categorical_columns = [
        column
        for column in features.select_dtypes(exclude=["number"]).columns
        if column not in business_numeric
    ]
    if not numeric_columns and not categorical_columns:
        raise ValueError("No model features were found in the input data.")

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                drop="first",
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                categorical_columns,
            )
        )

    column_transformer = ColumnTransformer(
        transformers=transformers, verbose_feature_names_out=False
    )
    preprocessor = Pipeline(
        [
            (
                "business_encoding",
                FunctionTransformer(
                    _encode_business_categories,
                    feature_names_out="one-to-one",
                    validate=False,
                ),
            ),
            ("columns", column_transformer),
        ]
    )
    preprocessor.set_output(transform="pandas")
    return preprocessor
