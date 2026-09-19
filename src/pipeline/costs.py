"""
Cost rules shared by the data generator and the quote-based baseline.

These are the parts of landed cost a trader can work out at booking time:
fixed rates, tariff rules and terminal charges. What is *not* known at
booking (the freight actually charged at departure, the fuel surcharge, delay
costs and the FX rate on the day customs clears) is what the model forecasts.
"""

INSURANCE_RATE = 0.004            # cargo insurance, share of goods value
BAF_USD_PER_FUEL_POINT = 3.5      # bunker adjustment factor per 40ft container
DEMURRAGE_FREE_DAYS = 4
DEMURRAGE_USD_PER_DAY = {"20ft_FCL": 90.0, "40ft_FCL": 150.0}

# Freight and BAF are quoted per 40ft; a 20ft box costs a fixed share of that.
CONTAINER_FREIGHT_FACTOR = {"20ft_FCL": 0.6, "40ft_FCL": 1.0}

# Terminal handling at the Polish port, charged in PLN.
THC_PLN = {
    ("Gdynia", "20ft_FCL"): 780.0,
    ("Gdynia", "40ft_FCL"): 1100.0,
    ("Gdansk", "20ft_FCL"): 850.0,
    ("Gdansk", "40ft_FCL"): 1250.0,
}


def customs_duty_usd(goods_usd, freight_usd, insurance_usd, duty_rate):
    """EU customs value is CIF: goods plus freight plus insurance."""
    return duty_rate * (goods_usd + freight_usd + insurance_usd)


def landed_cost_pln(goods_usd, freight_usd, fuel_index, duty_rate, fx_usd_pln,
                    container_type, dest_port, demurrage_usd=0.0):
    """Landed cost in PLN from its components. Works on scalars or pandas Series."""
    factor = _lookup(CONTAINER_FREIGHT_FACTOR, container_type)
    insurance = goods_usd * INSURANCE_RATE
    duty = customs_duty_usd(goods_usd, freight_usd, insurance, duty_rate)
    baf = fuel_index * BAF_USD_PER_FUEL_POINT * factor
    usd_total = goods_usd + freight_usd + insurance + duty + baf + demurrage_usd
    thc = _lookup(THC_PLN, list(zip(dest_port, container_type))
                  if not isinstance(dest_port, str) else (dest_port, container_type))
    return usd_total * fx_usd_pln + thc


def _lookup(table, key):
    """Map a scalar key or an iterable of keys through `table`."""
    if isinstance(key, (str, tuple)):
        return table[key]
    import pandas as pd
    return pd.Series([table[k] for k in key]).values
