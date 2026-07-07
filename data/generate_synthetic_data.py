"""
Generates synthetic (but structurally realistic) trade-shipment data modeled on
the Mundra (India) -> Gdynia (Poland) corridor: FCL shipments across multiple
commodities, with freight rates, FX rates, and fuel index driving landed cost.

This stands in for a real historical dataset. Swap this out for actual shipment
records (freight invoices, FX statements, customs declarations) once available --
the ETL and feature pipeline downstream don't care about the source.
"""
import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)

COMMODITIES = ["psyllium_husk", "organic_cumin", "organic_coriander", "lentils", "hotel_textiles"]
CONTAINER_TYPES = ["20ft_FCL", "40ft_FCL"]
ORIGIN_PORTS = ["Mundra", "Nhava_Sheva", "Chennai"]
DEST_PORTS = ["Gdynia", "Gdansk"]

N_ROWS = 2400  # ~ daily shipments over ~6.5 years incl. gaps


def generate() -> pd.DataFrame:
    dates = pd.date_range("2019-01-01", "2026-06-30", freq="D")
    sample_dates = RNG.choice(dates, size=N_ROWS, replace=True)
    sample_dates.sort()

    commodity = RNG.choice(COMMODITIES, size=N_ROWS, p=[0.28, 0.22, 0.18, 0.20, 0.12])
    container = RNG.choice(CONTAINER_TYPES, size=N_ROWS, p=[0.35, 0.65])
    origin = RNG.choice(ORIGIN_PORTS, size=N_ROWS, p=[0.6, 0.25, 0.15])
    dest = RNG.choice(DEST_PORTS, size=N_ROWS, p=[0.7, 0.3])

    t = (pd.to_datetime(sample_dates) - pd.Timestamp("2019-01-01")).days.values

    # Base freight rate with long-run drift + seasonal + shock component (e.g. Red Sea reroutes, fuel spikes)
    base_freight = 1800 + 0.15 * t
    seasonal = 250 * np.sin(2 * np.pi * (t % 365) / 365)
    shocks = np.where(RNG.random(N_ROWS) < 0.03, RNG.uniform(800, 2500, N_ROWS), 0)
    freight_rate_usd = base_freight + seasonal + shocks + RNG.normal(0, 120, N_ROWS)
    freight_rate_usd = np.clip(freight_rate_usd, 900, None)

    # FX rates with a mild trend + noise
    fx_usd_pln = 3.9 + 0.00015 * t + RNG.normal(0, 0.08, N_ROWS)
    fx_usd_inr = 71 + 0.0035 * t + RNG.normal(0, 0.9, N_ROWS)

    fuel_index = 100 + 0.02 * t + 15 * np.sin(2 * np.pi * (t % 365) / 365 + 1) + RNG.normal(0, 5, N_ROWS)

    transit_days = np.where(dest == "Gdynia", RNG.normal(28, 3, N_ROWS), RNG.normal(30, 3, N_ROWS))
    transit_days = np.clip(transit_days, 18, 45)

    commodity_base_value_usd = pd.Series(commodity).map({
        "psyllium_husk": 26000, "organic_cumin": 24000, "organic_coriander": 21000,
        "lentils": 19000, "hotel_textiles": 23000,
    }).values
    container_multiplier = np.where(container == "40ft_FCL", 1.85, 1.0)

    goods_value_usd = commodity_base_value_usd * container_multiplier * RNG.normal(1.0, 0.06, N_ROWS)

    customs_duty_rate = pd.Series(commodity).map({
        "psyllium_husk": 0.03, "organic_cumin": 0.03, "organic_coriander": 0.03,
        "lentils": 0.0, "hotel_textiles": 0.12,
    }).values

    insurance_usd = goods_value_usd * 0.004
    duty_usd = goods_value_usd * customs_duty_rate
    fuel_surcharge_usd = fuel_index * 3.5

    landed_cost_usd = goods_value_usd + freight_rate_usd + insurance_usd + duty_usd + fuel_surcharge_usd
    landed_cost_pln = landed_cost_usd * fx_usd_pln

    df = pd.DataFrame({
        "ship_date": pd.to_datetime(sample_dates),
        "origin_port": origin,
        "dest_port": dest,
        "commodity": commodity,
        "container_type": container,
        "goods_value_usd": goods_value_usd.round(2),
        "freight_rate_usd": freight_rate_usd.round(2),
        "fuel_index": fuel_index.round(2),
        "fx_usd_pln": fx_usd_pln.round(4),
        "fx_usd_inr": fx_usd_inr.round(4),
        "transit_days": transit_days.round(1),
        "customs_duty_rate": customs_duty_rate,
        "insurance_usd": insurance_usd.round(2),
        "duty_usd": duty_usd.round(2),
        "landed_cost_pln": landed_cost_pln.round(2),
    })
    return df.sort_values("ship_date").reset_index(drop=True)


if __name__ == "__main__":
    out_dir = Path(__file__).parent
    df = generate()
    out_path = out_dir / "raw_shipments.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
