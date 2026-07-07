"""
Feature engineering: turns cleaned shipment records into a model-ready
feature matrix. Kept as pure functions so they're independently unit-testable
(see tests/test_features.py) and reusable from both the training pipeline
and the online prediction path in src/api/main.py.
"""
import pandas as pd


CATEGORICAL_COLS = ["origin_port", "dest_port", "commodity", "container_type"]
NUMERIC_COLS = [
    "goods_value_usd", "freight_rate_usd", "fuel_index", "fx_usd_pln",
    "fx_usd_inr", "transit_days", "customs_duty_rate", "insurance_usd", "duty_usd",
]
TARGET_COL = "landed_cost_pln"


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ship_month"] = df["ship_date"].dt.month
    df["ship_dayofweek"] = df["ship_date"].dt.dayofweek
    df["ship_quarter"] = df["ship_date"].dt.quarter
    return df


def add_rolling_fx_features(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    df = df.sort_values("ship_date").copy()
    df["fx_usd_pln_roll_mean"] = df["fx_usd_pln"].rolling(window, min_periods=1).mean()
    df["fx_usd_pln_roll_std"] = df["fx_usd_pln"].rolling(window, min_periods=1).std().fillna(0)
    df["freight_rate_roll_mean"] = df["freight_rate_usd"].rolling(window, min_periods=1).mean()
    return df


def encode_categoricals(df: pd.DataFrame, categories: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """One-hot encode categorical columns. If `categories` is provided (from
    training), align new data to the same columns -- critical so the online
    prediction path never produces a feature matrix with a different shape
    than what the model was trained on."""
    df = df.copy()
    learned_categories = {}
    for col in CATEGORICAL_COLS:
        if categories is not None:
            cats = categories[col]
            df[col] = pd.Categorical(df[col], categories=cats)
        else:
            df[col] = df[col].astype("category")
            learned_categories[col] = list(df[col].cat.categories)

    encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS, prefix=CATEGORICAL_COLS)
    return encoded, (categories or learned_categories)


def build_feature_matrix(df: pd.DataFrame, categories: dict | None = None):
    df = add_calendar_features(df)
    df = add_rolling_fx_features(df)

    feature_cols = NUMERIC_COLS + ["ship_month", "ship_dayofweek", "ship_quarter",
                                    "fx_usd_pln_roll_mean", "fx_usd_pln_roll_std",
                                    "freight_rate_roll_mean"]

    encoded, learned_categories = encode_categoricals(df, categories)
    dummy_cols = [c for c in encoded.columns if any(c.startswith(f"{cat}_") for cat in CATEGORICAL_COLS)]

    X = encoded[feature_cols + dummy_cols]
    y = encoded[TARGET_COL] if TARGET_COL in encoded.columns else None
    return X, y, learned_categories
