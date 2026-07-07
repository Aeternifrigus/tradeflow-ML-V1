import pandas as pd
import pytest

from src.pipeline.features import build_feature_matrix, add_calendar_features


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "ship_date": pd.to_datetime(["2026-01-05", "2026-01-10", "2026-02-01"]),
        "origin_port": ["Mundra", "Mundra", "Nhava_Sheva"],
        "dest_port": ["Gdynia", "Gdynia", "Gdansk"],
        "commodity": ["psyllium_husk", "lentils", "psyllium_husk"],
        "container_type": ["40ft_FCL", "20ft_FCL", "40ft_FCL"],
        "goods_value_usd": [26000.0, 19000.0, 27000.0],
        "freight_rate_usd": [2100.0, 1950.0, 2200.0],
        "fuel_index": [110.0, 108.0, 112.0],
        "fx_usd_pln": [3.95, 3.97, 3.98],
        "fx_usd_inr": [83.1, 83.2, 83.3],
        "transit_days": [28.0, 27.5, 29.0],
        "customs_duty_rate": [0.03, 0.0, 0.03],
        "insurance_usd": [104.0, 76.0, 108.0],
        "duty_usd": [780.0, 0.0, 810.0],
        "landed_cost_pln": [115000.0, 84000.0, 118000.0],
    })


def test_add_calendar_features_adds_expected_columns(sample_df):
    out = add_calendar_features(sample_df)
    assert {"ship_month", "ship_dayofweek", "ship_quarter"}.issubset(out.columns)
    assert out.loc[0, "ship_month"] == 1


def test_build_feature_matrix_shapes(sample_df):
    X, y, categories = build_feature_matrix(sample_df)
    assert len(X) == len(sample_df)
    assert y is not None and len(y) == len(sample_df)
    # one-hot columns for every category value should exist
    assert any(col.startswith("commodity_") for col in X.columns)
    assert "origin_port" in categories


def test_build_feature_matrix_no_target_column_when_absent(sample_df):
    df_no_target = sample_df.drop(columns=["landed_cost_pln"])
    X, y, _ = build_feature_matrix(df_no_target)
    assert y is None
    assert len(X) == len(df_no_target)


def test_categories_alignment_for_unseen_inference_row(sample_df):
    _, _, categories = build_feature_matrix(sample_df)

    new_row = sample_df.iloc[[0]].drop(columns=["landed_cost_pln"])
    X_new, y_new, _ = build_feature_matrix(new_row, categories=categories)

    assert y_new is None
    # feature matrix built with frozen categories should have same dummy columns
    # as one built from full training data (minus target-dependent rows)
    assert "commodity_psyllium_husk" in X_new.columns
