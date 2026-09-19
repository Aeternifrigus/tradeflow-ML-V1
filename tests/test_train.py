import json

import pytest

from src.models.train import time_split, train


def test_time_split_has_no_overlap_and_only_known_labels(dataset):
    shipments, _ = dataset
    train_df, test_df, cutoff = time_split(shipments)

    assert (test_df["booking_date"] > cutoff).all()
    # Every training label was known (shipment arrived) by the cutoff.
    assert (train_df["realized_arrival_date"] <= cutoff).all()
    assert set(train_df.index).isdisjoint(test_df.index)


@pytest.fixture(scope="module")
def trained(dataset, tmp_path_factory):
    shipments, market = dataset
    d = tmp_path_factory.mktemp("train")
    shipments.to_parquet(d / "shipments.parquet")
    market.to_parquet(d / "market.parquet")
    model_dir = d / "model"
    _, results = train(d / "shipments.parquet", d / "market.parquet", model_dir,
                       experiment_name="tradeflow-tests")
    return model_dir, results


def test_model_beats_both_baselines_on_future_bookings(trained):
    _, results = trained
    model = results["model"]["mae_pln"]
    assert model < results["quote_estimate"]["mae_pln"]
    assert model < results["bias_corrected_quote"]["mae_pln"]


def test_training_writes_everything_serving_needs(trained):
    model_dir, _ = trained
    for name in ["model.joblib", "feature_columns.json", "categories.json",
                 "market_history.parquet", "metrics.json"]:
        assert (model_dir / name).exists(), name
    metrics = json.loads((model_dir / "metrics.json").read_text())
    assert set(metrics["results"]) == {"quote_estimate", "bias_corrected_quote", "model"}
