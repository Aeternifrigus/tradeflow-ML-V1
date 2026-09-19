"""
ETL stage: load raw shipments and market history, validate them, and write
canonical parquet files for the feature stage.

File-based (CSV -> Parquet) so it runs anywhere with no external services.
In production `load_*` would read from a warehouse table and a market data
feed; nothing downstream would change.
"""
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SHIPMENT_COLUMNS = [
    "booking_date", "lead_time_days", "origin_port", "dest_port", "commodity",
    "container_type", "goods_value_usd", "customs_duty_rate", "planned_transit_days",
    "freight_quote_usd", "landed_cost_pln", "realized_arrival_date",
]
MARKET_COLUMNS = ["date", "freight_index_usd", "fx_usd_pln", "fuel_index"]


class DataQualityError(Exception):
    pass


def load_shipments(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["booking_date", "realized_arrival_date"])
    logger.info("Loaded %d raw shipments from %s", len(df), path)
    return df


def load_market(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    logger.info("Loaded %d days of market history from %s", len(df), path)
    return df


def validate_shipments(df: pd.DataFrame) -> pd.DataFrame:
    missing = set(SHIPMENT_COLUMNS) - set(df.columns)
    if missing:
        raise DataQualityError(f"Missing shipment columns: {sorted(missing)}")

    before = len(df)
    df = df.dropna(subset=SHIPMENT_COLUMNS)
    df = df[(df["landed_cost_pln"] > 0) & (df["goods_value_usd"] > 0) & (df["freight_quote_usd"] > 0)]
    df = df[df["customs_duty_rate"].between(0, 1)]
    df = df[df["lead_time_days"].between(0, 90) & df["planned_transit_days"].between(5, 90)]
    # A shipment cannot arrive before it was booked.
    df = df[df["realized_arrival_date"] > df["booking_date"]]
    dropped = before - len(df)
    if dropped:
        logger.warning("Dropped %d shipments failing quality checks", dropped)
    if dropped > 0.05 * before:
        raise DataQualityError(f"{dropped} of {before} shipments failed quality checks")

    return df.sort_values("booking_date").reset_index(drop=True)


def validate_market(df: pd.DataFrame) -> pd.DataFrame:
    missing = set(MARKET_COLUMNS) - set(df.columns)
    if missing:
        raise DataQualityError(f"Missing market columns: {sorted(missing)}")
    if df["date"].duplicated().any():
        raise DataQualityError("Market history has duplicate dates")
    if df[MARKET_COLUMNS[1:]].isna().any().any():
        raise DataQualityError("Market history has missing values")
    if (df[MARKET_COLUMNS[1:]] <= 0).any().any():
        raise DataQualityError("Market history has non-positive prices")

    df = df.sort_values("date").reset_index(drop=True)
    gaps = df["date"].diff().dt.days.dropna()
    if (gaps != 1).any():
        # Feature windows count calendar days, so the series must be daily.
        raise DataQualityError(f"Market history is not daily: {int((gaps != 1).sum())} gaps")
    return df


def run_etl(raw_shipments: Path, raw_market: Path, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    shipments = validate_shipments(load_shipments(raw_shipments))
    market = validate_market(load_market(raw_market))

    out_dir.mkdir(parents=True, exist_ok=True)
    shipments.to_parquet(out_dir / "clean_shipments.parquet", index=False)
    market.to_parquet(out_dir / "market_history.parquet", index=False)
    logger.info("Wrote %d shipments and %d market days to %s", len(shipments), len(market), out_dir)
    return shipments, market


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[2]
    run_etl(base / "data" / "raw_shipments.csv", base / "data" / "market_history.csv", base / "artifacts")
