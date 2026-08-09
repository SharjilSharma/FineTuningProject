"""
preprocessing.py
----------------
Align transcript chunks to price data, apply filters, and write
the final Phase 1 parquet.

Output schema
-------------
transcript_id, chunk_id, ticker, company, sector, call_date, call_timing,
fiscal_quarter, chunk_text, speaker_role,
t_minus1_open, t_minus1_close, t_minus1_volume,
fwd_ret_1d, fwd_ret_5d, fwd_ret_10d

No silent row drops
-------------------
Every removed row is logged with a reason.  Rows with missing price
data are RETAINED (price columns = NaN) so they can be audited.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from src.data.load_transcripts import IT_TICKERS, load_it_transcripts
from src.data.price_data import build_price_table

# ── Schema ────────────────────────────────────────────────────────────────

OUTPUT_COLUMNS: list[str] = [
    "transcript_id", "chunk_id", "ticker", "company", "sector",
    "call_date", "call_timing", "fiscal_quarter",
    "chunk_text", "speaker_role",
    "t_minus1_open", "t_minus1_close", "t_minus1_volume",
    "fwd_ret_1d", "fwd_ret_5d", "fwd_ret_10d",
]

HPE_MIN_DATE: str = "2016-01-01"   # spin-off Nov 2015; data reliable from Q1 2016

# ── Filter helpers ────────────────────────────────────────────────────────

def _filter_unknown_tickers(df: pd.DataFrame) -> pd.DataFrame:
    mask = ~df["ticker"].isin(IT_TICKERS)
    if mask.any():
        bad = df.loc[mask, "ticker"].unique().tolist()
        logger.warning(
            f"Dropping {mask.sum()} rows with unexpected tickers: {bad}. "
            "Reason: not in IT_TICKERS slice."
        )
    return df[~mask].copy()

def _apply_hpe_date_filter(df: pd.DataFrame) -> pd.DataFrame:
    mask = (df["ticker"] == "HPE") & (df["call_date"] < HPE_MIN_DATE)
    n = mask.sum()
    if n:
        logger.info(
            f"HPE filter: dropping {n} rows (call_date < {HPE_MIN_DATE}). "
            "Reason: HPE spin-off data unreliable before Q1 2016."
        )
    return df[~mask].copy()

def _drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    dupes = df.duplicated(subset=["chunk_id"], keep="first")
    if dupes.any():
        logger.warning(
            f"Dropping {dupes.sum()} duplicate chunk_id rows. "
            "Reason: deduplication on chunk_id."
        )
    return df[~dupes].copy()

def _log_price_missing(df: pd.DataFrame) -> None:
    """Log a summary of chunks/transcripts with missing price data (NOT dropped)."""
    if "t_minus1_close" not in df.columns:
        return
    missing = df["t_minus1_close"].isna()
    if not missing.any():
        return
    summary = (
        df.loc[missing]
        .groupby("ticker")["call_date"]
        .nunique()
        .rename("n_calls_missing_price")
        .reset_index()
    )
    logger.warning(
        f"{missing.sum()} chunks ({df.loc[missing, 'transcript_id'].nunique()} transcripts) "
        f"have missing price data (retained with NaN):\n{summary.to_string(index=False)}"
    )

# ── Main pipeline ─────────────────────────────────────────────────────────

def build_aligned_dataset(
    output_path: str | Path = Path("data/processed/phase1_it_transcripts.parquet"),
    price_cache_dir: str | Path | None = Path("data/cache/prices"),
    streaming: bool = True,
) -> pd.DataFrame:
    """
    End-to-end Phase 1 pipeline:
        load transcripts -> filter -> fetch prices -> align -> save parquet.

    Returns the final DataFrame.
    """
    output_path = Path(output_path)
    price_cache_dir = Path(price_cache_dir) if price_cache_dir else None

    # Step 1 — Load chunks
    logger.info("=== Phase 1 | Step 1: Loading transcripts ===")
    df = load_it_transcripts(streaming=streaming)
    logger.info(f"  {len(df):,} raw chunks loaded.")

    # Step 2 — Filters
    logger.info("=== Phase 1 | Step 2: Applying filters ===")
    df = _filter_unknown_tickers(df)
    df = _apply_hpe_date_filter(df)
    df = _drop_duplicates(df)
    logger.info(
        f"  After filters: {len(df):,} chunks, "
        f"{df['transcript_id'].nunique()} transcripts, "
        f"{df['ticker'].nunique()} tickers."
    )

    # Step 3 — Price data
    logger.info("=== Phase 1 | Step 3: Fetching prices ===")
    unique_calls = df[["ticker", "call_date"]].drop_duplicates().reset_index(drop=True)
    logger.info(f"  {len(unique_calls)} unique (ticker, call_date) pairs.")
    price_df = build_price_table(unique_calls, cache_dir=price_cache_dir)

    # Step 4 — Join
    logger.info("=== Phase 1 | Step 4: Joining price data ===")
    df = df.merge(price_df, on=["ticker", "call_date"], how="left", validate="many_to_one")

    # Step 5 — Log missing (do NOT drop)
    _log_price_missing(df)

    # Step 6 — Enforce column order
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[OUTPUT_COLUMNS].copy()

    # Step 7 — Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    # Summary
    logger.info("=== Phase 1 | Complete ===")
    logger.info(f"  Rows:              {len(df):,}")
    logger.info(f"  Unique transcripts:{df['transcript_id'].nunique():,}")
    logger.info(f"  Tickers:           {df['ticker'].nunique()}")
    logger.info(f"  Date range:        {df['call_date'].min()} -> {df['call_date'].max()}")
    logger.info(f"  Output:            {output_path}")
    if "call_timing" in df.columns:
        tc = df.drop_duplicates("transcript_id")["call_timing"].value_counts()
        logger.info(f"  Timing breakdown:\n{tc.to_string()}")

    return df
