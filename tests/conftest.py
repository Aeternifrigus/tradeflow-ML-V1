import pandas as pd
import pytest

from data.generate_synthetic_data import generate
from src.pipeline.etl import validate_market, validate_shipments


@pytest.fixture(scope="session")
def dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The full synthetic dataset, validated the same way the ETL does."""
    shipments, market = generate(seed=42)
    return validate_shipments(shipments), validate_market(market)
