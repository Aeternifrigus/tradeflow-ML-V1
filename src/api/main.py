"""
FastAPI serving layer for TradeFlow-ML.

Endpoints:
  GET  /health           -- liveness/readiness probe (used by Cloud Run + CI smoke test)
  POST /predict          -- landed-cost prediction for a shipment
  POST /parse-document   -- agentic extraction of structured fields from raw doc text

Kept deliberately thin: all real logic lives in src/models and src/agent so
it's unit-testable without spinning up the HTTP layer.
"""
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.agent.document_parser import parse_document_dict
from src.models.predict import LandedCostPredictor

app = FastAPI(title="TradeFlow-ML", version="0.1.0")

_predictor: LandedCostPredictor | None = None


def get_predictor() -> LandedCostPredictor:
    global _predictor
    if _predictor is None:
        model_path = Path(__file__).resolve().parents[2] / "models" / "model.joblib"
        if not model_path.exists():
            raise HTTPException(
                status_code=503,
                detail="Model not trained yet. Run `python -m src.models.train` first.",
            )
        _predictor = LandedCostPredictor()
    return _predictor


class ShipmentInput(BaseModel):
    ship_date: str = Field(..., examples=["2026-07-01"])
    origin_port: str = Field(..., examples=["Mundra"])
    dest_port: str = Field(..., examples=["Gdynia"])
    commodity: str = Field(..., examples=["psyllium_husk"])
    container_type: str = Field(..., examples=["40ft_FCL"])
    goods_value_usd: float
    freight_rate_usd: float
    fuel_index: float
    fx_usd_pln: float
    fx_usd_inr: float
    transit_days: float
    customs_duty_rate: float
    insurance_usd: float
    duty_usd: float


class PredictionResponse(BaseModel):
    predicted_landed_cost_pln: float


class DocumentInput(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(shipment: ShipmentInput):
    predictor = get_predictor()
    pred = predictor.predict(shipment.model_dump())
    return PredictionResponse(predicted_landed_cost_pln=round(pred, 2))


@app.post("/parse-document")
def parse_document_endpoint(payload: DocumentInput):
    return parse_document_dict(payload.text)
