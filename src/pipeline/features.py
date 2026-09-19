"""
Point-in-time feature engineering.

The model answers: "at booking, what will this shipment's landed cost turn out
to be?" Every feature must therefore be something known on the booking date:

- shipment facts from the booking itself (route, goods, container, the
  forwarder's quote, lead time, planned transit)
- market state *as of* the booking date, read from the daily market history
  (never a later day)

The model does not predict landed cost directly. It predicts the log ratio
between the realized cost and the estimate a trader would calculate from the
quote on booking day (`quote_based_estimate_pln`). Two reasons:
  1. The baseline already captures the arithmetic (goods, duty, insurance,
     today's FX); the model only has to learn what moves cost away from it.
  2. Tree models cannot extrapolate. Price levels drift upward over the years,
     but the ratio between outcome and estimate stays in a stable range.

Features are computed per row with no dependence on other rows, so a batch of
training rows and a single API request go through exactly the same code and
get exactly the same values.
"""
import numpy as np
import pandas as pd

from src.pipeline.costs import landed_cost_pln

CATEGORICAL_COLS = ["origin_port", "dest_port", "commodity", "container_type"]

# The only shipment columns the feature pipeline reads. Anything else in the
# input, notably the realized_* outcome columns, is ignored.
BOOKING_COLUMNS = [
    "booking_date", "lead_time_days", "planned_transit_days",
    "goods_value_usd", "customs_duty_rate", "freight_quote_usd",
    *CATEGORICAL_COLS,
]
TARGET_COL = "landed_cost_pln"
MARKET_COLUMNS = ["freight_index_usd", "fx_usd_pln", "fuel_index"]

# Longest look-back used by the market features, and how stale market data may
# be before a booking date is refused.
MARKET_LOOKBACK_DAYS = 180
MAX_MARKET_STALENESS_DAYS = 7


class MarketDataError(ValueError):
    """A booking date the market history cannot describe."""


def market_state(market: pd.DataFrame) -> pd.DataFrame:
    """Daily market features, each computed from that day and earlier only."""
    m = market.set_index("date").sort_index()[MARKET_COLUMNS]
    idx = m["freight_index_usd"]
    out = pd.DataFrame(index=m.index)
    out["fx_usd_pln_spot"] = m["fx_usd_pln"]
    out["fuel_index_spot"] = m["fuel_index"]
    out["freight_index_usd"] = idx
    out["freight_chg_14d"] = idx / idx.shift(14) - 1
    out["freight_chg_30d"] = idx / idx.shift(30) - 1
    out["freight_vs_180d_mean"] = idx / idx.rolling(MARKET_LOOKBACK_DAYS).mean()
    out["freight_vol_30d"] = np.log(idx).diff().rolling(30).std()
    out["fx_chg_30d"] = m["fx_usd_pln"] / m["fx_usd_pln"].shift(30) - 1
    out["fuel_chg_30d"] = m["fuel_index"] / m["fuel_index"].shift(30) - 1
    return out


def market_as_of(dates: pd.Series, state: pd.DataFrame) -> pd.DataFrame:
    """Look up the market state on each date, refusing dates it cannot cover."""
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    valid = state.dropna()
    if valid.empty:
        raise MarketDataError("market history is shorter than the feature look-back")
    first, last = valid.index.min(), valid.index.max()

    too_early = dates < first
    if too_early.any():
        raise MarketDataError(
            f"booking date {dates[too_early].min().date()} is before usable market "
            f"history ({first.date()})"
        )
    too_late = dates > last + pd.Timedelta(days=MAX_MARKET_STALENESS_DAYS)
    if too_late.any():
        raise MarketDataError(
            f"booking date {dates[too_late].max().date()} is more than "
            f"{MAX_MARKET_STALENESS_DAYS} days after the market history ends "
            f"({last.date()}); refresh the market data"
        )

    # merge_asof picks the latest market day on or before each booking date.
    lookup = pd.DataFrame({"booking_date": dates, "_row": np.arange(len(dates))})
    merged = pd.merge_asof(
        lookup.sort_values("booking_date"),
        valid.rename_axis("market_date").reset_index(),
        left_on="booking_date", right_on="market_date", direction="backward",
    )
    return merged.sort_values("_row").drop(columns=["_row", "booking_date"]).reset_index(drop=True)


def quote_based_estimate(shipments: pd.DataFrame, market: pd.DataFrame) -> pd.Series:
    """What a trader would calculate on booking day: the quote, today's fuel
    surcharge and today's FX rate, with no delays."""
    return pd.Series(landed_cost_pln(
        shipments["goods_value_usd"].values,
        shipments["freight_quote_usd"].values,
        market["fuel_index_spot"].values,
        shipments["customs_duty_rate"].values,
        market["fx_usd_pln_spot"].values,
        shipments["container_type"].reset_index(drop=True),
        shipments["dest_port"].reset_index(drop=True),
    ))


def encode_categoricals(df: pd.DataFrame, categories: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """One-hot encode with a fixed category list, learned at training time, so
    inference always produces the same columns as training."""
    df = df.copy()
    learned = {}
    for col in CATEGORICAL_COLS:
        cats = categories[col] if categories is not None else sorted(df[col].unique().tolist())
        df[col] = pd.Categorical(df[col], categories=cats)
        learned[col] = cats
    encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS, prefix=CATEGORICAL_COLS, dtype=float)
    return encoded, learned


def build_features(shipments: pd.DataFrame, market_state_table: pd.DataFrame,
                   categories: dict | None = None):
    """Returns (X, y, baseline, categories).

    y is the log ratio of realized landed cost to the quote-based estimate, or
    None when the input has no target (inference). Rows keep their input order.
    """
    missing = set(BOOKING_COLUMNS) - set(shipments.columns)
    if missing:
        raise ValueError(f"missing booking columns: {sorted(missing)}")

    s = shipments[BOOKING_COLUMNS].reset_index(drop=True).copy()
    s["booking_date"] = pd.to_datetime(s["booking_date"])
    mkt = market_as_of(s["booking_date"], market_state_table)
    baseline = quote_based_estimate(s, mkt)

    departure = s["booking_date"] + pd.to_timedelta(s["lead_time_days"], unit="D")
    feats = pd.DataFrame({
        "lead_time_days": s["lead_time_days"].astype(float),
        "planned_transit_days": s["planned_transit_days"].astype(float),
        "log_goods_value_usd": np.log(s["goods_value_usd"]),
        "customs_duty_rate": s["customs_duty_rate"],
        # How the quote compares with the market index (route and box size show
        # up here too).
        "quote_vs_index": s["freight_quote_usd"] / mkt["freight_index_usd"],
        "freight_share_of_estimate": s["freight_quote_usd"] * mkt["fx_usd_pln_spot"] / baseline,
        "departure_month_sin": np.sin(2 * np.pi * departure.dt.month / 12),
        "departure_month_cos": np.cos(2 * np.pi * departure.dt.month / 12),
        "departure_month": departure.dt.month.astype(float),
        "freight_chg_14d": mkt["freight_chg_14d"],
        "freight_chg_30d": mkt["freight_chg_30d"],
        "freight_vs_180d_mean": mkt["freight_vs_180d_mean"],
        "freight_vol_30d": mkt["freight_vol_30d"],
        "fx_chg_30d": mkt["fx_chg_30d"],
        "fuel_chg_30d": mkt["fuel_chg_30d"],
    })
    encoded, learned = encode_categoricals(s[CATEGORICAL_COLS], categories)
    X = pd.concat([feats, encoded], axis=1)

    y = None
    if TARGET_COL in shipments.columns:
        y = np.log(shipments[TARGET_COL].reset_index(drop=True) / baseline)
    return X, y, baseline, learned
