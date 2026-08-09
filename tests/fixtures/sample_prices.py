"""
Synthetic price series for unit testing price_data.py.

All prices are fabricated; values are chosen to produce clean,
predictable forward-return fractions for easy assertion.
"""

from __future__ import annotations

import pandas as pd


def _make_prices(
    start: str,
    n_days: int,
    base_close: float = 100.0,
    daily_step: float = 1.0,
) -> pd.DataFrame:
    """
    Return a DataFrame of trading-day prices starting at *start*.
    Close increases by *daily_step* each day.
    Open = Close - 0.5, Volume = 1_000_000 (constant).
    """
    dates = pd.bdate_range(start=start, periods=n_days)
    closes = [base_close + i * daily_step for i in range(n_days)]
    opens  = [c - 0.5 for c in closes]
    return pd.DataFrame(
        {
            "Open":   opens,
            "High":   [c + 0.25 for c in closes],
            "Low":    [o - 0.25 for o in opens],
            "Close":  closes,
            "Volume": [1_000_000] * n_days,
        },
        index=pd.DatetimeIndex(dates, name="Date"),
    )


# 30 trading days starting 2023-10-23 (Monday)
# Day 0 (index 0) = 2023-10-23
NVDA_PRICES: pd.DataFrame = _make_prices("2023-10-23", n_days=30, base_close=400.0)

# 30 trading days starting 2023-10-23
ACN_PRICES: pd.DataFrame  = _make_prices("2023-10-23", n_days=30, base_close=300.0)

# HPE prices — start before 2016 to test filter behaviour
HPE_PRICES: pd.DataFrame  = _make_prices("2015-11-01", n_days=60, base_close=15.0)
