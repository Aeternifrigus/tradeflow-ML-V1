"""
Trains a gradient-boosted regressor to predict landed cost (PLN) for a
shipment given commodity, route, FX, and freight signals. Tracks params/
metrics/artifacts in MLflow so every training run is reproducible and
comparable -- this is the piece that turns "I trained a model once" into
an auditable experiment history.
"""
import json
import logging
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from pipeline.features import build_feature_matrix  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "models"
ARTIFACT_DIR = BASE_DIR / "artifacts"


def train(data_path: Path, model_out: Path, experiment_name: str = "tradeflow-landed-cost"):
    df = pd.read_parquet(data_path)
    X, y, categories = build_feature_matrix(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        params = dict(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
        mlflow.log_params(params)

        model = XGBRegressor(**params)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        mape = mean_absolute_percentage_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        mlflow.log_metrics({"mae_pln": mae, "mape": mape, "r2": r2})
        logger.info("MAE=%.2f PLN | MAPE=%.4f | R2=%.4f", mae, mape, r2)

        mlflow.sklearn.log_model(model, artifact_path="model")

        model_out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_out)
        with open(model_out.parent / "feature_columns.json", "w") as f:
            json.dump(list(X.columns), f)
        with open(model_out.parent / "categories.json", "w") as f:
            json.dump(categories, f)

        logger.info("Model saved to %s", model_out)
        return model, {"mae_pln": mae, "mape": mape, "r2": r2}


if __name__ == "__main__":
    train(ARTIFACT_DIR / "clean_shipments.parquet", MODEL_DIR / "model.joblib")
