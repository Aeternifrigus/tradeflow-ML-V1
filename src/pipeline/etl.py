"""
ETL stage: load raw shipment records, validate/clean, and write a canonical
parquet dataset that the feature engineering stage consumes.

Design note: this is intentionally file-based (CSV -> Parquet) so it runs
anywhere with no external dependencies. Swap `load_raw` for a warehouse read
(BigQuery / Snowflake / Iceberg table) in production without touching
downstream stages -- that's the point of separating extract from transform.
"""
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "ship_date", "origin_port", "dest_port", "commodity", "container_type",
    "goods_value_usd", "freight_rate_usd", "fuel_index", "fx_usd_pln",
    "fx_usd_inr", "transit_days", "customs_duty_rate", "insurance_usd",
    "duty_usd", "landed_cost_pln",
]


class DataQualityError(Exception):
    pass


def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["ship_date"])
    logger.info("Loaded %d raw rows from %s", len(df), path)
    return df


def validate(df: pd.DataFrame) -> pd.DataFrame:
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise DataQualityError(f"Missing required columns: {missing_cols}")

    before = len(df)
    df = df.dropna(subset=REQUIRED_COLUMNS)
    df = df[df["landed_cost_pln"] > 0]
    df = df[df["transit_days"].between(5, 90)]
    dropped = before - len(df)
    if dropped:
        logger.warning("Dropped %d rows failing quality checks", dropped)

    # Data quality test: no duplicate (date, origin, dest, commodity) combos beyond expected noise
    dup_rate = df.duplicated(subset=["ship_date", "origin_port", "commodity"]).mean()
    if dup_rate > 0.15:
        raise DataQualityError(f"Duplicate rate too high: {dup_rate:.2%}")

    return df.reset_index(drop=True)


def run_etl(raw_path: Path, out_path: Path) -> pd.DataFrame:
    df = load_raw(raw_path)
    df = validate(df)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info("Wrote %d clean rows to %s", len(df), out_path)
    return df


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[2]
    run_etl(base / "data" / "raw_shipments.csv", base / "artifacts" / "clean_shipments.parquet")
