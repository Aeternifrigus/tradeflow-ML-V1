"""
Loads a trained model directory and forecasts landed cost for new bookings.

Kept separate from train.py so serving never imports training dependencies.
Features come from the same `build_features` function as training, reading
market state from the market history saved alongside the model, so a request
gets exactly the features the same shipment would have had in training.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.pipeline.features import CATEGORICAL_COLS, build_features, market_state

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "models"


class UnknownCategoryError(ValueError):
    """A route, commodity or container type the model was never trained on."""


class LandedCostPredictor:
    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model = joblib.load(model_dir / "model.joblib")
        self.feature_columns = json.loads((model_dir / "feature_columns.json").read_text())
        self.categories = json.loads((model_dir / "categories.json").read_text())
        market = pd.read_parquet(model_dir / "market_history.parquet")
        self.market_state = market_state(market)
        self.market_as_of = market["date"].max().date()

    def _check_categories(self, df: pd.DataFrame) -> None:
        for col in CATEGORICAL_COLS:
            unknown = set(df[col]) - set(self.categories[col])
            if unknown:
                raise UnknownCategoryError(
                    f"{col} {sorted(unknown)} not seen in training; known: {self.categories[col]}"
                )

    def predict_frame(self, shipments: pd.DataFrame) -> pd.DataFrame:
        """Forecasts for many bookings. Returns forecast and baseline per row."""
        self._check_categories(shipments)
        X, _, baseline, _ = build_features(shipments, self.market_state, self.categories)
        X = X[self.feature_columns]
        forecast = baseline.to_numpy() * np.exp(self.model.predict(X))
        return pd.DataFrame({"forecast_pln": forecast, "quote_estimate_pln": baseline.to_numpy()})

    def predict(self, shipment: dict) -> dict:
        row = self.predict_frame(pd.DataFrame([shipment])).iloc[0]
        return {
            "forecast_landed_cost_pln": float(row["forecast_pln"]),
            "quote_based_estimate_pln": float(row["quote_estimate_pln"]),
            "market_data_as_of": str(self.market_as_of),
        }
