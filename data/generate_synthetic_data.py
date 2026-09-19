"""
Synthetic data for the Mundra/Nhava Sheva/Chennai -> Gdynia/Gdansk corridor.

Two datasets, shaped like what a trading business actually has:

1. market_history.csv: one row per day with a container freight index, the
   USD/PLN rate and a bunker fuel index. In practice this comes from a data
   vendor (Drewry/Freightos style indices, NBP exchange rates).
2. raw_shipments.csv: one row per booked shipment. The columns a trader knows
   when booking (route, goods, container, the forwarder's freight quote) plus
   the landed cost that only becomes known weeks later, when the container
   has sailed, arrived and cleared customs.

Landed cost is decided *after* booking by things the booking-time quote does
not include:
  - carriers pass freight index moves between booking and departure through
    as rate increases (fully) or reductions (only partly)
  - the bunker surcharge is set at departure
  - delays beyond the free days cost demurrage, and delays cluster in peak
    season and during supply disruptions
  - customs clears at the USD/PLN rate on arrival day, not booking day

The market has structure a model can learn from (seasonality, disruption
regimes that build up over weeks) and noise it cannot (day-to-day FX moves).
The realized_* columns are outcomes for analysis and evaluation only; the
feature pipeline never reads them.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.costs import (
    CONTAINER_FREIGHT_FACTOR,
    DEMURRAGE_FREE_DAYS,
    DEMURRAGE_USD_PER_DAY,
    landed_cost_pln,
)

START = pd.Timestamp("2019-01-01")
END = pd.Timestamp("2026-06-30")
N_SHIPMENTS = 2400

COMMODITIES = ["psyllium_husk", "organic_cumin", "organic_coriander", "lentils", "hotel_textiles"]
COMMODITY_P = [0.28, 0.22, 0.18, 0.20, 0.12]
CONTAINER_TYPES = ["20ft_FCL", "40ft_FCL"]
ORIGIN_PORTS = ["Mundra", "Nhava_Sheva", "Chennai"]
DEST_PORTS = ["Gdynia", "Gdansk"]

GOODS_VALUE_USD_40FT = {
    "psyllium_husk": 48000, "organic_cumin": 44000, "organic_coriander": 39000,
    "lentils": 35000, "hotel_textiles": 42000,
}
DUTY_RATE = {
    "psyllium_husk": 0.03, "organic_cumin": 0.03, "organic_coriander": 0.03,
    "lentils": 0.0, "hotel_textiles": 0.12,
}
ORIGIN_FREIGHT_FACTOR = {"Mundra": 1.0, "Nhava_Sheva": 1.03, "Chennai": 1.12}
DEST_FREIGHT_FACTOR = {"Gdynia": 1.0, "Gdansk": 1.02}
PLANNED_TRANSIT_DAYS = {
    ("Mundra", "Gdynia"): 28, ("Mundra", "Gdansk"): 30,
    ("Nhava_Sheva", "Gdynia"): 29, ("Nhava_Sheva", "Gdansk"): 31,
    ("Chennai", "Gdynia"): 32, ("Chennai", "Gdansk"): 34,
}

# Carriers pass freight increases through in full but reductions only partly.
DOWNWARD_PASS_THROUGH = 0.5


def _disruption_multiplier(n_days: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Supply-chain disruptions (canal closures, port congestion) that build up
    over about a month, plateau, then ease over six weeks. Returns the freight
    multiplier and a 0/1 flag for days inside a disruption."""
    mult = np.ones(n_days)
    active = np.zeros(n_days, dtype=bool)
    day = int(rng.integers(120, 300))
    while day < n_days:
        ramp_up, plateau, ramp_down = 30, int(rng.integers(60, 150)), 45
        peak = rng.uniform(1.6, 2.4)
        shape = np.concatenate([
            np.linspace(1.0, peak, ramp_up),
            np.full(plateau, peak),
            np.linspace(peak, 1.0, ramp_down),
        ])
        end = min(day + len(shape), n_days)
        mult[day:end] = np.maximum(mult[day:end], shape[: end - day])
        active[day:end] = True
        day = end + int(rng.integers(250, 550))
    return mult, active


def generate_market(seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    """Daily market series, plus the disruption flag (not published: in reality
    nobody hands you a clean 'disruption' column, you see it in the prices)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(START, END, freq="D")
    n = len(dates)
    t = np.arange(n)
    doy = dates.dayofyear.values

    # Freight index, USD per 40ft Mundra -> Gdynia: drift, a late-summer peak
    # season, disruption regimes and persistent (AR(1)) noise.
    season = 1.0 + 0.10 * np.sin(2 * np.pi * (doy - 150) / 365)
    disruption, active = _disruption_multiplier(n, rng)
    ar = np.zeros(n)
    for i in range(1, n):
        ar[i] = 0.97 * ar[i - 1] + rng.normal(0, 0.012)
    freight = (1800 + 0.12 * t) * season * disruption * np.exp(ar)

    # USD/PLN: mean-reverting random walk in logs around a slow drift. Mostly
    # unpredictable over a few weeks, as a real exchange rate is.
    log_fx = np.empty(n)
    log_fx[0] = np.log(3.9)
    for i in range(1, n):
        mu = np.log(3.9 + 0.00012 * i)
        log_fx[i] = log_fx[i - 1] + 0.01 * (mu - log_fx[i - 1]) + rng.normal(0, 0.0045)
    fx = np.exp(log_fx)

    # Bunker fuel index: seasonal, persistent noise, and higher when ships are
    # rerouted on longer voyages.
    fuel_ar = np.zeros(n)
    for i in range(1, n):
        fuel_ar[i] = 0.98 * fuel_ar[i - 1] + rng.normal(0, 1.2)
    fuel = (100 + 0.015 * t + 12 * np.sin(2 * np.pi * doy / 365 + 1) + fuel_ar) * (1 + 0.1 * active)

    market = pd.DataFrame({
        "date": dates,
        "freight_index_usd": freight.round(2),
        "fx_usd_pln": fx.round(4),
        "fuel_index": fuel.round(2),
    })
    return market, active


def generate(seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (shipments, market_history)."""
    rng = np.random.default_rng(seed)
    market, disrupted = generate_market(seed)
    m = market.set_index("date")
    disrupted = pd.Series(disrupted, index=m.index)

    # Leave 200 days of history before the first booking (the features look
    # back 180 days) and room at the end for the last shipment to arrive.
    first, last = START + pd.Timedelta(days=200), END - pd.Timedelta(days=90)
    booking = pd.to_datetime(rng.choice(pd.date_range(first, last, freq="D"), N_SHIPMENTS))
    booking = booking.sort_values()

    commodity = rng.choice(COMMODITIES, N_SHIPMENTS, p=COMMODITY_P)
    container = rng.choice(CONTAINER_TYPES, N_SHIPMENTS, p=[0.35, 0.65])
    origin = rng.choice(ORIGIN_PORTS, N_SHIPMENTS, p=[0.6, 0.25, 0.15])
    dest = rng.choice(DEST_PORTS, N_SHIPMENTS, p=[0.7, 0.3])

    lead_days = rng.integers(7, 22, N_SHIPMENTS)
    departure = booking + pd.to_timedelta(lead_days, unit="D")
    planned_transit = np.array([PLANNED_TRANSIT_DAYS[(o, d)] for o, d in zip(origin, dest)])

    cf = np.array([CONTAINER_FREIGHT_FACTOR[c] for c in container])
    route_factor = (np.array([ORIGIN_FREIGHT_FACTOR[o] for o in origin])
                    * np.array([DEST_FREIGHT_FACTOR[d] for d in dest]))

    goods = (np.array([GOODS_VALUE_USD_40FT[c] for c in commodity])
             * np.where(container == "40ft_FCL", 1.0, 0.54)
             * rng.normal(1.0, 0.06, N_SHIPMENTS))
    duty_rate = np.array([DUTY_RATE[c] for c in commodity])

    # What the forwarder quotes on booking day.
    idx_booking = m["freight_index_usd"].reindex(booking).values
    freight_quote = idx_booking * cf * route_factor * rng.normal(1.0, 0.03, N_SHIPMENTS)

    # What is actually charged: the quote, moved by the index between booking
    # and departure (increases in full, reductions in part).
    idx_departure = m["freight_index_usd"].reindex(departure).values
    move = idx_departure / idx_booking - 1
    freight_final = freight_quote * (1 + np.where(move > 0, move, DOWNWARD_PASS_THROUGH * move))

    # Delays: worse in disruptions, in peak season (Aug-Oct) and into Gdansk.
    in_disruption = disrupted.reindex(departure).values
    peak = np.isin(departure.month, [8, 9, 10])
    delay = (rng.exponential(1.5, N_SHIPMENTS)
             + peak * rng.exponential(2.0, N_SHIPMENTS)
             + (dest == "Gdansk") * rng.exponential(1.0, N_SHIPMENTS)
             + in_disruption * rng.normal(12, 3, N_SHIPMENTS).clip(0))
    actual_transit = planned_transit + delay
    arrival = departure + pd.to_timedelta(np.round(actual_transit), unit="D")
    arrival = arrival.where(arrival <= END, END)  # keep lookups inside the market series
    demurrage = (np.clip(delay - DEMURRAGE_FREE_DAYS, 0, None)
                 * np.array([DEMURRAGE_USD_PER_DAY[c] for c in container]))

    fuel_departure = m["fuel_index"].reindex(departure).values
    fx_arrival = m["fx_usd_pln"].reindex(arrival).values

    landed = landed_cost_pln(
        goods, freight_final, fuel_departure, duty_rate, fx_arrival,
        pd.Series(container), pd.Series(dest), demurrage_usd=demurrage,
    )

    shipments = pd.DataFrame({
        # Known at booking
        "booking_date": booking,
        "lead_time_days": lead_days,
        "origin_port": origin,
        "dest_port": dest,
        "commodity": commodity,
        "container_type": container,
        "goods_value_usd": goods.round(2),
        "customs_duty_rate": duty_rate,
        "planned_transit_days": planned_transit,
        "freight_quote_usd": freight_quote.round(2),
        # Target: known only after customs clearance
        "landed_cost_pln": np.round(landed, 2),
        # Realized outcomes, for evaluation and analysis only
        "realized_arrival_date": arrival,
        "realized_freight_usd": freight_final.round(2),
        "realized_delay_days": delay.round(1),
        "realized_fx_usd_pln": fx_arrival.round(4),
    })
    return shipments.reset_index(drop=True), market


if __name__ == "__main__":
    out_dir = Path(__file__).parent
    shipments, market = generate()
    shipments.to_csv(out_dir / "raw_shipments.csv", index=False)
    market.to_csv(out_dir / "market_history.csv", index=False)
    print(f"Wrote {len(shipments)} shipments and {len(market)} days of market history to {out_dir}")
