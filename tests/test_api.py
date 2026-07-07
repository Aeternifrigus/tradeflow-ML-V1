import pytest
from fastapi.testclient import TestClient
from pathlib import Path

from src.pipeline import etl
from src.models import train as train_module


@pytest.fixture(scope="module", autouse=True)
def trained_model(tmp_path_factory):
    """Train a tiny model once for the whole test module so /predict has
    something to load. Uses a small synthetic slice, not the full dataset,
    to keep CI fast."""
    from data.generate_synthetic_data import generate

    tmp_dir = tmp_path_factory.mktemp("api_test_artifacts")
    raw_path = tmp_dir / "raw.csv"
    clean_path = tmp_dir / "clean.parquet"
    model_path = Path(__file__).resolve().parents[1] / "models" / "model.joblib"

    df = generate().sample(300, random_state=1)
    df.to_csv(raw_path, index=False)
    etl.run_etl(raw_path, clean_path)
    train_module.train(clean_path, model_path)
    yield


@pytest.fixture
def client():
    from src.api.main import app
    return TestClient(app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_predict_returns_positive_cost(client):
    payload = {
        "ship_date": "2026-07-01",
        "origin_port": "Mundra",
        "dest_port": "Gdynia",
        "commodity": "psyllium_husk",
        "container_type": "40ft_FCL",
        "goods_value_usd": 26000.0,
        "freight_rate_usd": 2100.0,
        "fuel_index": 110.0,
        "fx_usd_pln": 3.95,
        "fx_usd_inr": 83.1,
        "transit_days": 28.0,
        "customs_duty_rate": 0.03,
        "insurance_usd": 104.0,
        "duty_usd": 780.0,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    assert resp.json()["predicted_landed_cost_pln"] > 0


def test_parse_document_endpoint(client):
    resp = client.post("/parse-document", json={"text": "HS Code: 12129921, Quantity: 1000 kg, FOB"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["hs_code"] == "12129921"
