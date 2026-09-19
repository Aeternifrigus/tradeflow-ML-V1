import pytest
from fastapi.testclient import TestClient

from src.models.train import train


@pytest.fixture(scope="module")
def client(dataset, tmp_path_factory):
    """Train into a temporary directory and point the API at it, so tests never
    overwrite the real models/ directory."""
    shipments, market = dataset
    d = tmp_path_factory.mktemp("api")
    shipments.to_parquet(d / "shipments.parquet")
    market.to_parquet(d / "market.parquet")
    train(d / "shipments.parquet", d / "market.parquet", d / "model",
          experiment_name="tradeflow-tests")

    mp = pytest.MonkeyPatch()
    mp.setenv("TRADEFLOW_MODEL_DIR", str(d / "model"))
    import src.api.main as api
    mp.setattr(api, "_predictor", None)
    yield TestClient(api.app)
    mp.undo()


BOOKING = {
    "booking_date": "2026-06-20",
    "lead_time_days": 14,
    "origin_port": "Mundra",
    "dest_port": "Gdynia",
    "commodity": "psyllium_husk",
    "container_type": "40ft_FCL",
    "goods_value_usd": 48000.0,
    "customs_duty_rate": 0.03,
    "planned_transit_days": 28,
    "freight_quote_usd": 4100.0,
}


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_predict_returns_forecast_and_baseline(client):
    resp = client.post("/predict", json=BOOKING)
    assert resp.status_code == 200
    body = resp.json()
    assert body["forecast_landed_cost_pln"] > 0
    assert body["quote_based_estimate_pln"] > 0
    assert body["market_data_as_of"] == "2026-06-30"


def test_unknown_commodity_is_rejected(client):
    resp = client.post("/predict", json={**BOOKING, "commodity": "saffron"})
    assert resp.status_code == 422
    assert "saffron" in resp.json()["detail"]


def test_booking_beyond_market_data_is_rejected(client):
    resp = client.post("/predict", json={**BOOKING, "booking_date": "2027-01-01"})
    assert resp.status_code == 422
    assert "refresh the market data" in resp.json()["detail"]


def test_invalid_input_is_rejected(client):
    resp = client.post("/predict", json={**BOOKING, "goods_value_usd": -5})
    assert resp.status_code == 422


def test_parse_document_endpoint(client):
    resp = client.post("/parse-document", json={"text": "HS Code: 12129921, Quantity: 1000 kg, FOB"})
    assert resp.status_code == 200
    assert resp.json()["hs_code"] == "12129921"
