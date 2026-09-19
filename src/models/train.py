"""
Trains the landed-cost forecaster and records the run in MLflow.

Evaluation is set up the way the model will be used: trained on the past,
tested on bookings made after a cutoff date.

Two leakage guards:
  1. Time split. Test rows are bookings after the cutoff, so the model never
     sees the period it is scored on.
  2. Label availability. A shipment's landed cost is only known once it has
     arrived and cleared customs, weeks after booking. A row goes into the
     training set only if it had arrived by the cutoff. Rows booked before the
     cutoff but still at sea are excluded from both sets.

The model is always compared with two baselines, so a reported improvement
means something:
  - quote_estimate: the booking-day calculation a trader would do by hand
  - bias_corrected_quote: the same, scaled by the average historical overrun
"""
import json
import logging
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

from src.pipeline.features import build_features, market_state

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "models"
ARTIFACT_DIR = BASE_DIR / "artifacts"

TEST_FRACTION = 0.2
PARAMS = dict(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    random_state=42,
)


def time_split(shipments: pd.DataFrame, test_fraction: float = TEST_FRACTION):
    """Split by booking date, keeping only training rows whose outcome was
    known by the cutoff. Returns (train, test, cutoff)."""
    cutoff = shipments["booking_date"].quantile(1 - test_fraction)
    test = shipments[shipments["booking_date"] > cutoff]
    train = shipments[shipments["realized_arrival_date"] <= cutoff]
    return train, test, cutoff


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    abs_err = np.abs(actual - predicted)
    return {
        "mae_pln": float(mean_absolute_error(actual, predicted)),
        "mape": float(np.mean(abs_err / actual)),
        "p90_abs_error_pln": float(np.quantile(abs_err, 0.9)),
    }


def train(shipments_path: Path, market_path: Path, model_dir: Path,
          experiment_name: str = "tradeflow-landed-cost"):
    shipments = pd.read_parquet(shipments_path)
    market = pd.read_parquet(market_path)
    state = market_state(market)

    train_df, test_df, cutoff = time_split(shipments)
    embargoed = len(shipments) - len(train_df) - len(test_df)
    logger.info(
        "Cutoff %s: %d train, %d test, %d excluded (booked before cutoff, arrived after)",
        cutoff.date(), len(train_df), len(test_df), embargoed,
    )
    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError("time split left an empty train or test set")

    X_train, y_train, _, categories = build_features(train_df, state)
    X_test, _, baseline_test, _ = build_features(test_df, state, categories)
    actual = test_df["landed_cost_pln"].to_numpy()

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        mlflow.log_params({**PARAMS, "cutoff": str(cutoff.date()), "n_train": len(train_df),
                           "n_test": len(test_df), "n_embargoed": embargoed})

        model = XGBRegressor(**PARAMS)
        model.fit(X_train, y_train)
        predicted = baseline_test.to_numpy() * np.exp(model.predict(X_test))

        mean_overrun = float(np.exp(y_train.mean()))
        results = {
            "quote_estimate": _metrics(actual, baseline_test.to_numpy()),
            "bias_corrected_quote": _metrics(actual, baseline_test.to_numpy() * mean_overrun),
            "model": _metrics(actual, predicted),
        }
        for name, m in results.items():
            mlflow.log_metrics({f"{name}_{k}": v for k, v in m.items()})
            logger.info("%-22s MAE %8.0f PLN | MAPE %.4f | P90 %8.0f PLN",
                        name, m["mae_pln"], m["mape"], m["p90_abs_error_pln"])

        mlflow.sklearn.log_model(model, artifact_path="model")

        # Everything serving needs goes in one directory, including the market
        # history the features are computed from.
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_dir / "model.joblib")
        (model_dir / "feature_columns.json").write_text(json.dumps(list(X_train.columns)))
        (model_dir / "categories.json").write_text(json.dumps(categories))
        (model_dir / "metrics.json").write_text(json.dumps(
            {"cutoff": str(cutoff.date()), "n_train": len(train_df), "n_test": len(test_df),
             "n_embargoed": embargoed, "results": results}, indent=2))
        market.to_parquet(model_dir / "market_history.parquet", index=False)

        logger.info("Model saved to %s", model_dir)
        return model, results


if __name__ == "__main__":
    train(ARTIFACT_DIR / "clean_shipments.parquet", ARTIFACT_DIR / "market_history.parquet", MODEL_DIR)
