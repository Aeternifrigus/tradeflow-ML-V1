"""
FastAPI serving layer for TradeFlow-ML.

Endpoints:
  GET  /health           -- liveness/readiness probe (used by Cloud Run + CI smoke test)
  POST /predict          -- landed-cost forecast for a shipment at booking time
  POST /parse-document   -- extraction of structured fields from raw document text

Kept deliberately thin: all real logic lives in src/models and src/agent so
it's unit-testable without spinning up the HTTP layer.
"""
import datetime as dt
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.agent.document_parser import parse_document_dict
from src.models.predict import MODEL_DIR, LandedCostPredictor, UnknownCategoryError
from src.pipeline.features import MarketDataError

app = FastAPI(title="TradeFlow-ML", version="0.2.0")

_predictor: LandedCostPredictor | None = None


def model_dir() -> Path:
    return Path(os.environ.get("TRADEFLOW_MODEL_DIR", MODEL_DIR))


def get_predictor() -> LandedCostPredictor:
    global _predictor
    if _predictor is None:
        if not (model_dir() / "model.joblib").exists():
            raise HTTPException(
                status_code=503,
                detail="Model not trained yet. Run `python -m src.models.train` first.",
            )
        _predictor = LandedCostPredictor(model_dir())
    return _predictor


class ShipmentInput(BaseModel):
    """What is known when the shipment is booked."""

    booking_date: dt.date = Field(..., examples=["2026-06-20"])
    lead_time_days: int = Field(..., ge=0, le=90, description="Days from booking to departure")
    origin_port: str = Field(..., examples=["Mundra"])
    dest_port: str = Field(..., examples=["Gdynia"])
    commodity: str = Field(..., examples=["psyllium_husk"])
    container_type: str = Field(..., examples=["40ft_FCL"])
    goods_value_usd: float = Field(..., gt=0)
    customs_duty_rate: float = Field(..., ge=0, le=1)
    planned_transit_days: float = Field(..., gt=0, le=90)
    freight_quote_usd: float = Field(..., gt=0, description="Forwarder's quote on booking day")


class PredictionResponse(BaseModel):
    forecast_landed_cost_pln: float
    quote_based_estimate_pln: float
    market_data_as_of: str


class DocumentInput(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(shipment: ShipmentInput):
    predictor = get_predictor()
    try:
        result = predictor.predict(shipment.model_dump())
    except (UnknownCategoryError, MarketDataError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return PredictionResponse(
        forecast_landed_cost_pln=round(result["forecast_landed_cost_pln"], 2),
        quote_based_estimate_pln=round(result["quote_based_estimate_pln"], 2),
        market_data_as_of=result["market_data_as_of"],
    )


@app.post("/parse-document")
def parse_document_endpoint(payload: DocumentInput):
    return parse_document_dict(payload.text)
