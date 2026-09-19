import numpy as np
import pandas as pd
import pytest

from src.pipeline.features import (
    MAX_MARKET_STALENESS_DAYS,
    MarketDataError,
    build_features,
    market_state,
)


@pytest.fixture(scope="module")
def state(dataset):
    return market_state(dataset[1])


def test_market_state_never_looks_ahead(dataset):
    """Changing prices after a date must not change that date's features."""
    _, market = dataset
    day = market["date"].iloc[1500]
    shocked = market.copy()
    later = shocked["date"] > day
    shocked.loc[later, ["freight_index_usd", "fx_usd_pln", "fuel_index"]] *= 3.0

    before = market_state(market).loc[:day]
    after = market_state(shocked).loc[:day]
    pd.testing.assert_frame_equal(before, after)


def test_realized_outcome_columns_are_ignored(dataset, state):
    shipments, _ = dataset
    rows = shipments.iloc[:200]
    scrambled = rows.copy()
    for col in [c for c in rows.columns if c.startswith("realized_")]:
        scrambled[col] = scrambled[col].sample(frac=1, random_state=0).to_numpy()

    X1, _, _, cats = build_features(rows, state)
    X2, _, _, _ = build_features(scrambled, state, cats)
    pd.testing.assert_frame_equal(X1, X2)


def test_single_row_features_equal_batch_features(dataset, state):
    """Serving builds features one request at a time; they must match training."""
    shipments, _ = dataset
    rows = shipments.iloc[1000:1040]
    X_batch, _, base_batch, cats = build_features(rows, state)
    for i in range(len(rows)):
        X_one, _, base_one, _ = build_features(rows.iloc[[i]], state, cats)
        pd.testing.assert_series_equal(
            X_one.iloc[0], X_batch.iloc[i], check_names=False
        )
        assert base_one.iloc[0] == pytest.approx(base_batch.iloc[i])


def test_rows_keep_input_order(dataset, state):
    shipments, _ = dataset
    rows = shipments.iloc[:100].sample(frac=1, random_state=3)
    X, y, baseline, _ = build_features(rows, state)
    expected = np.log(rows["landed_cost_pln"].to_numpy() / baseline.to_numpy())
    np.testing.assert_allclose(y.to_numpy(), expected)
    np.testing.assert_allclose(X["log_goods_value_usd"], np.log(rows["goods_value_usd"]))


def test_target_is_absent_at_inference(dataset, state):
    shipments, _ = dataset
    X, y, _, _ = build_features(shipments.iloc[:5].drop(columns=["landed_cost_pln"]), state)
    assert y is None
    assert len(X) == 5


def test_frozen_categories_give_the_same_columns(dataset, state):
    shipments, _ = dataset
    X_all, _, _, cats = build_features(shipments, state)
    X_one, _, _, _ = build_features(shipments.iloc[[0]], state, cats)
    assert list(X_one.columns) == list(X_all.columns)


def test_booking_after_market_history_is_refused(dataset, state):
    shipments, market = dataset
    row = shipments.iloc[[0]].copy()
    row["booking_date"] = market["date"].max() + pd.Timedelta(days=MAX_MARKET_STALENESS_DAYS + 1)
    with pytest.raises(MarketDataError, match="refresh the market data"):
        build_features(row, state)


def test_booking_before_usable_history_is_refused(dataset, state):
    shipments, market = dataset
    row = shipments.iloc[[0]].copy()
    row["booking_date"] = market["date"].min() + pd.Timedelta(days=10)
    with pytest.raises(MarketDataError, match="before usable market history"):
        build_features(row, state)
