"""
Loads the trained model + frozen feature schema and serves predictions for
new shipment inputs. Kept separate from train.py so the API layer only ever
imports inference code, never training dependencies (mlflow, sklearn's
train_test_split, etc.) -- smaller, faster container image at serve time.
"""
import json
from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "models"


class LandedCostPredictor:
    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model = joblib.load(model_dir / "model.joblib")
        with open(model_dir / "feature_columns.json") as f:
            self.feature_columns = json.load(f)
        with open(model_dir / "categories.json") as f:
            self.categories = json.load(f)

    def predict(self, shipment: dict) -> float:
        from src.pipeline.features import build_feature_matrix  # local import avoids circulars

        df = pd.DataFrame([shipment])
        df["ship_date"] = pd.to_datetime(df["ship_date"])
        X, _, _ = build_feature_matrix(df, categories=self.categories)
        X = X.reindex(columns=self.feature_columns, fill_value=0)
        pred = self.model.predict(X)[0]
        return float(pred)
