"""
Shared pytest fixtures for Phase 1 tests.
All fixtures are offline — no network calls.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tests.fixtures.sample_transcripts import (
    NVDA_NORMAL, MSFT_NO_QA, AAPL_LONG_QA, ACN_BMO, HPE_PRE2016,
)
from tests.fixtures.sample_prices import NVDA_PRICES, ACN_PRICES, HPE_PRICES


@pytest.fixture
def nvda_turns() -> list[dict]:
    return NVDA_NORMAL

@pytest.fixture
def msft_turns() -> list[dict]:
    return MSFT_NO_QA

@pytest.fixture
def aapl_long_turns() -> list[dict]:
    return AAPL_LONG_QA

@pytest.fixture
def acn_turns() -> list[dict]:
    return ACN_BMO

@pytest.fixture
def hpe_turns() -> list[dict]:
    return HPE_PRE2016

@pytest.fixture
def nvda_prices() -> pd.DataFrame:
    return NVDA_PRICES.copy()

@pytest.fixture
def acn_prices() -> pd.DataFrame:
    return ACN_PRICES.copy()

@pytest.fixture
def hpe_prices() -> pd.DataFrame:
    return HPE_PRICES.copy()
