"""
price_data.py
-------------
Fetch per-ticker OHLCV data from Yahoo Finance and compute forward
returns adjusted for BMO / AMC earnings-call timing.

BMO / AMC return windows (close-to-close)
------------------------------------------
BMO  — market prices the news on *call_date* itself:
    reference  = close on prev_trading_day(call_date)
    fwd_ret_1d = close(call_date)           / reference - 1
    fwd_ret_5d = close(call_date + 4 days)  / reference - 1
    fwd_ret_10d= close(call_date + 9 days)  / reference - 1

AMC  — market prices the news on next_trading_day(call_date):
    reference  = close on call_date  (pre-reaction)
    fwd_ret_1d = close(call_date + 1 day)   / reference - 1
    fwd_ret_5d = close(call_date + 5 days)  / reference - 1
    fwd_ret_10d= close(call_date + 10 days) / reference - 1

All offsets are *trading-day* counts derived from the yfinance price index.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
import yfinance as yf
from loguru import logger

# ── BMO / AMC lookup ──────────────────────────────────────────────────────

CallTiming = Literal["BMO", "AMC"]

CALL_TIMING: dict[str, CallTiming] = {
    "ACN":  "BMO",   # Accenture  — consistently 8 am ET
    "PAYX": "BMO",   # Paychex    — consistently pre-market
    "HPE":  "BMO",   # HP Enterprise — typically pre-market
}
_DEFAULT_TIMING: CallTiming = "AMC"

FORWARD_WINDOWS: tuple[int, ...] = (1, 5, 10)

# ── Timing helpers ────────────────────────────────────────────────────────

def get_call_timing(ticker: str) -> tuple[CallTiming, str]:
    """
    Return (timing, timing_source).

    timing_source values
    --------------------
    "lookup_table"  — ticker found in CALL_TIMING dict
    "assumed_amc"   — ticker not in dict; AMC assumed and logged
    """
    t = ticker.upper()
    if t in CALL_TIMING:
        return CALL_TIMING[t], "lookup_table"
    logger.debug(
        f"{t}: call timing not in lookup table — assuming AMC (timing_source=assumed_amc)."
    )
    return _DEFAULT_TIMING, "assumed_amc"

# ── Price fetching ────────────────────────────────────────────────────────

def fetch_ticker_prices(
    ticker: str,
    start: str = "2014-01-01",
    end: str = "2025-06-01",
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Fetch full OHLCV history via yfinance, optionally caching to parquet.

    Returns a DataFrame indexed by timezone-naive Date with columns:
    Open, High, Low, Close, Volume.
    Returns an empty DataFrame on failure (caller must check).
    """
    if cache_dir is not None:
        path = cache_dir / f"{ticker.upper()}.parquet"
        if path.exists():
            logger.debug(f"Price cache hit: {path}")
            return pd.read_parquet(path)

    logger.info(f"Downloading prices: {ticker} ({start} -> {end})")
    try:
        df = yf.download(
            ticker, start=start, end=end,
            auto_adjust=True, progress=False,
        )
    except Exception as exc:
        logger.error(f"yfinance download failed for {ticker}: {exc}")
        return pd.DataFrame()

    if df.empty:
        logger.warning(f"yfinance returned no data for {ticker}.")
        return df

    # Flatten MultiIndex columns (yfinance >= 0.2 quirk when single ticker)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Ensure timezone-naive index
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
        logger.debug(f"Cached {ticker} -> {path}")

    return df

# ── Trading-day arithmetic ────────────────────────────────────────────────

def _nth_trading_day(
    idx: pd.DatetimeIndex,
    anchor: pd.Timestamp,
    offset: int,
) -> pd.Timestamp | None:
    """
    Return the date that is *offset* trading days from *anchor*.

    * offset = 0  -> anchor itself (or nearest trading day if anchor is not in idx)
    * offset = -1 -> previous trading day
    * offset = +1 -> next trading day

    Returns None if the resulting position is out of bounds.
    """
    if anchor in idx:
        pos = idx.get_loc(anchor)
    else:
        iloc = int(idx.searchsorted(anchor))
        if offset >= 0:
            pos = min(iloc, len(idx) - 1)
        else:
            pos = max(iloc - 1, 0)

    target = pos + offset
    if 0 <= target < len(idx):
        return idx[target]
    return None

# ── Per-call price row ────────────────────────────────────────────────────

def compute_price_row(
    ticker: str,
    call_date: str,
    prices: pd.DataFrame,
) -> dict:
    """
    Compute reference price and forward returns for one earnings call.

    Returns a dict with keys:
        call_timing, timing_source,
        t_minus1_open, t_minus1_close, t_minus1_volume,
        fwd_ret_1d, fwd_ret_5d, fwd_ret_10d,
        price_missing (bool)
    """
    out: dict = {
        "call_timing":    None,
        "timing_source":  None,
        "t_minus1_open":  None,
        "t_minus1_close": None,
        "t_minus1_volume": None,
        "fwd_ret_1d":   None,
        "fwd_ret_5d":   None,
        "fwd_ret_10d":  None,
        "price_missing": False,
    }

    if prices.empty:
        logger.warning(f"{ticker} {call_date}: no price data available.")
        out["price_missing"] = True
        return out

    timing, source = get_call_timing(ticker)
    out["call_timing"]   = timing
    out["timing_source"] = source

    idx     = prices.index
    call_ts = pd.Timestamp(call_date)

    # Reference date and forward-return start offset
    if timing == "BMO":
        ref_ts  = _nth_trading_day(idx, call_ts, offset=-1)
        fwd_base_offset = 0   # fwd_ret_1d -> call_date itself (offset 0)
    else:  # AMC
        ref_ts  = _nth_trading_day(idx, call_ts, offset=0)
        fwd_base_offset = 1   # fwd_ret_1d -> next trading day (offset 1)

    if ref_ts is None or ref_ts not in idx:
        logger.warning(
            f"{ticker} {call_date}: reference trading day not found "
            f"(timing={timing}). Marking price_missing."
        )
        out["price_missing"] = True
        return out

    ref = prices.loc[ref_ts]
    out["t_minus1_open"]   = float(ref.get("Open",   float("nan")))
    out["t_minus1_close"]  = float(ref.get("Close",  float("nan")))
    out["t_minus1_volume"] = float(ref.get("Volume", float("nan")))

    ref_close = out["t_minus1_close"]
    if pd.isna(ref_close) or ref_close == 0:
        logger.warning(f"{ticker} {call_date}: reference close is NaN/0.")
        out["price_missing"] = True
        return out

    # Forward returns
    # For BMO: window 1 -> offset 0, window 5 -> offset 4, window 10 -> offset 9
    # For AMC: window 1 -> offset 1, window 5 -> offset 5, window 10 -> offset 10
    for w in FORWARD_WINDOWS:
        fwd_offset = fwd_base_offset + w - 1
        fwd_ts = _nth_trading_day(idx, call_ts, offset=fwd_offset)
        if fwd_ts is None or fwd_ts not in idx:
            logger.warning(
                f"{ticker} {call_date}: T+{w} price not found "
                f"(offset={fwd_offset}). Possibly near end of data or delisted."
            )
            out[f"fwd_ret_{w}d"] = None
        else:
            fwd_close = float(prices.loc[fwd_ts, "Close"])
            out[f"fwd_ret_{w}d"] = round((fwd_close - ref_close) / ref_close, 8)

    return out

# ── Batch price table ─────────────────────────────────────────────────────

def build_price_table(
    call_records: pd.DataFrame,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Given a DataFrame with at least ``ticker`` and ``call_date`` columns,
    fetch prices for each unique ticker and return a new DataFrame
    with price columns appended.
    """
    tickers = call_records["ticker"].unique().tolist()
    price_cache: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        price_cache[ticker] = fetch_ticker_prices(ticker, cache_dir=cache_dir)

    rows = []
    for _, row in call_records.iterrows():
        rows.append(
            compute_price_row(
                row["ticker"], row["call_date"],
                price_cache.get(row["ticker"], pd.DataFrame()),
            )
        )

    return pd.concat(
        [call_records.reset_index(drop=True), pd.DataFrame(rows)],
        axis=1,
    )
