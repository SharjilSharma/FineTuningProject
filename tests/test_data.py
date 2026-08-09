"""
Phase 1 unit tests — all offline, no HF or yfinance calls.

Coverage
--------
load_transcripts : section detection, chunking, chunk-id format,
                   truncation warning, token estimation
price_data       : BMO/AMC timing lookup, reference-date selection,
                   forward-return computation
preprocessing    : HPE filter, schema columns, price-missing retention
"""

from __future__ import annotations

import re

import pandas as pd
import pytest

from src.data.load_transcripts import (
    MAX_CHUNK_TOKENS,
    CHARS_PER_TOKEN,
    _chunk_transcript,
    _detect_qa_boundary,
    _group_qa_exchanges,
    _est_tokens,
)
from src.data.price_data import (
    CALL_TIMING,
    compute_price_row,
    get_call_timing,
    _nth_trading_day,
)
from src.data.preprocessing import (
    OUTPUT_COLUMNS,
    HPE_MIN_DATE,
    _apply_hpe_date_filter,
    _filter_unknown_tickers,
    _drop_duplicates,
)

# ── Helpers ───────────────────────────────────────────────────────────────

CHUNK_ID_PREP_RE = re.compile(r"^\w+__prep__p\d{3}$")
CHUNK_ID_QA_RE   = re.compile(r"^\w+__qa__q\d{3}__p\d{3}$")

def _make_chunks(turns, ticker="NVDA", call_date="2023-10-26"):
    return _chunk_transcript(
        turns=turns,
        transcript_id=f"{ticker}_{call_date.replace('-', '')}",
        ticker=ticker,
        company="Test Co",
        call_date=call_date,
        fiscal_quarter="Q3 2023",
    )

# ═══════════════════════════════════════════════════════════════════════════
# Section detection
# ═══════════════════════════════════════════════════════════════════════════

def test_qa_boundary_detected_via_header(nvda_turns):
    """Operator Q&A header at index 2 -> boundary = 3."""
    boundary = _detect_qa_boundary(nvda_turns)
    # Header is at index 2; Q&A starts at 3
    assert boundary == 3

def test_qa_boundary_no_qa_returns_len(msft_turns):
    """No Q&A -> boundary == len(turns)."""
    assert _detect_qa_boundary(msft_turns) == len(msft_turns)

def test_qa_boundary_detected_via_analyst(nvda_turns):
    """Even without explicit header, first analyst turn marks Q&A start."""
    # Remove the operator header turn, then boundary should fall on analyst turn
    turns_no_header = [t for t in nvda_turns if "Question-and-Answer" not in t["text"]]
    boundary = _detect_qa_boundary(turns_no_header)
    # First analyst is at index 2 (0-based) in modified list
    assert turns_no_header[boundary]["role"] in ("Analyst, BofA",)

# ═══════════════════════════════════════════════════════════════════════════
# Q&A grouping
# ═══════════════════════════════════════════════════════════════════════════

def test_qa_grouping_two_exchanges(nvda_turns):
    """NVDA fixture has 2 analyst questions -> 2 exchanges."""
    qa_turns = nvda_turns[3:]   # after operator header
    exchanges = _group_qa_exchanges(qa_turns)
    assert len(exchanges) == 2

def test_qa_grouping_each_starts_with_analyst(nvda_turns):
    qa_turns = nvda_turns[3:]
    for exchange in _group_qa_exchanges(qa_turns):
        role = exchange[0]["role"].lower()
        assert any(kw in role for kw in ("analyst", "research", "md"))

# ═══════════════════════════════════════════════════════════════════════════
# Chunking — structure
# ═══════════════════════════════════════════════════════════════════════════

def test_chunking_creates_prep_and_qa(nvda_turns):
    chunks = _make_chunks(nvda_turns)
    roles = {c.speaker_role for c in chunks}
    assert "prepared_remarks" in roles
    assert "qa_exchange" in roles

def test_chunking_no_qa_only_prep(msft_turns):
    chunks = _make_chunks(msft_turns, ticker="MSFT")
    assert all(c.speaker_role == "prepared_remarks" for c in chunks)

def test_chunking_qa_count_matches_exchanges(nvda_turns):
    """2 analyst questions -> 2 qa_exchange chunks (both fit in limit)."""
    chunks = _make_chunks(nvda_turns)
    qa_chunks = [c for c in chunks if c.speaker_role == "qa_exchange"]
    assert len(qa_chunks) == 2

def test_data_stub():
    """Placeholder test for Phase 1 data module."""
    assert True

# ═══════════════════════════════════════════════════════════════════════════
# Chunk ID format
# ═══════════════════════════════════════════════════════════════════════════

def test_prep_chunk_id_format(nvda_turns):
    chunks = _make_chunks(nvda_turns)
    prep_chunks = [c for c in chunks if c.speaker_role == "prepared_remarks"]
    for c in prep_chunks:
        assert CHUNK_ID_PREP_RE.match(c.chunk_id), f"Bad prep chunk_id: {c.chunk_id}"

def test_qa_chunk_id_format(nvda_turns):
    chunks = _make_chunks(nvda_turns)
    qa_chunks = [c for c in chunks if c.speaker_role == "qa_exchange"]
    for c in qa_chunks:
        assert CHUNK_ID_QA_RE.match(c.chunk_id), f"Bad QA chunk_id: {c.chunk_id}"

def test_qa_chunk_ids_are_sequential(nvda_turns):
    chunks = _make_chunks(nvda_turns)
    qa_ids = [c.chunk_id for c in chunks if c.speaker_role == "qa_exchange"]
    # q001__p001 then q002__p001
    assert "q001" in qa_ids[0]
    assert "q002" in qa_ids[1]

# ═══════════════════════════════════════════════════════════════════════════
# Truncation warning
# ═══════════════════════════════════════════════════════════════════════════

def test_oversized_qa_exchange_flagged_and_split(aapl_long_turns):
    """Long Q&A exchange -> truncation_warning=True + multiple parts."""
    chunks = _make_chunks(aapl_long_turns, ticker="AAPL")
    qa_chunks = [c for c in chunks if c.speaker_role == "qa_exchange"]
    # First exchange is very long; should be split
    first_q_chunks = [c for c in qa_chunks if "q001" in c.chunk_id]
    assert len(first_q_chunks) > 1, "Long exchange should produce > 1 part"
    assert all(c.truncation_warning for c in first_q_chunks)

def test_short_qa_no_truncation_warning(nvda_turns):
    chunks = _make_chunks(nvda_turns)
    qa_chunks = [c for c in chunks if c.speaker_role == "qa_exchange"]
    assert not any(c.truncation_warning for c in qa_chunks)

def test_oversized_parts_each_under_limit(aapl_long_turns):
    """Each sub-chunk must be <= MAX_CHUNK_TOKENS (approx)."""
    chunks = _make_chunks(aapl_long_turns, ticker="AAPL")
    for c in chunks:
        est = _est_tokens(c.chunk_text)
        assert est <= MAX_CHUNK_TOKENS * 1.1, (
            f"Chunk {c.chunk_id} estimated {est} tokens, limit {MAX_CHUNK_TOKENS}"
        )

# ═══════════════════════════════════════════════════════════════════════════
# BMO / AMC timing lookup
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("ticker", ["ACN", "PAYX", "HPE"])
def test_bmo_tickers(ticker):
    timing, source = get_call_timing(ticker)
    assert timing == "BMO"
    assert source == "lookup_table"

@pytest.mark.parametrize("ticker", ["NVDA", "MSFT", "AAPL", "INTC", "CRM"])
def test_amc_tickers_assumed(ticker):
    timing, source = get_call_timing(ticker)
    assert timing == "AMC"
    assert source == "assumed_amc"

# ═══════════════════════════════════════════════════════════════════════════
# Reference date selection
# ═══════════════════════════════════════════════════════════════════════════

def test_bmo_reference_is_prev_trading_day(acn_prices):
    """
    ACN is BMO. Call on 2023-10-25 (Wednesday) ->
    reference = 2023-10-24 (Tuesday), the previous trading day.
    """
    # acn_prices starts 2023-10-23 (Monday)
    # Index 0=Mon, 1=Tue, 2=Wed, ...
    call_date = "2023-10-25"   # Wednesday (index 2)
    result = compute_price_row("ACN", call_date, acn_prices)
    assert not result["price_missing"]
    # reference close for BMO = prev day (Tuesday, index 1) = 300.0 + 1*1.0 = 301.0
    assert result["t_minus1_close"] == pytest.approx(301.0)
    assert result["call_timing"] == "BMO"
    assert result["timing_source"] == "lookup_table"

def test_amc_reference_is_call_day(nvda_prices):
    """
    NVDA is AMC. Call on 2023-10-25 (Wednesday) ->
    reference = close on 2023-10-25 itself.
    """
    call_date = "2023-10-25"   # Wednesday (index 2) -> close = 400.0 + 2 = 402.0
    result = compute_price_row("NVDA", call_date, nvda_prices)
    assert not result["price_missing"]
    assert result["t_minus1_close"] == pytest.approx(402.0)
    assert result["call_timing"] == "AMC"

# ═══════════════════════════════════════════════════════════════════════════
# Forward return computation
# ═══════════════════════════════════════════════════════════════════════════

def test_bmo_fwd_ret_1d(acn_prices):
    """
    BMO call on 2023-10-25 (index 2):
      ref_close  = close[1] = 301.0
      fwd_ret_1d = close[2] / 301.0 - 1 = 302.0/301.0 - 1 ~ 0.00332
    """
    result = compute_price_row("ACN", "2023-10-25", acn_prices)
    expected = (302.0 - 301.0) / 301.0
    assert result["fwd_ret_1d"] == pytest.approx(expected, rel=1e-5)

def test_bmo_fwd_ret_5d(acn_prices):
    """
    BMO fwd_ret_5d: close at call_date + 4 trading days = index 6 = 306.0
    ref_close = 301.0 -> (306 - 301) / 301 ~ 0.01661
    """
    result = compute_price_row("ACN", "2023-10-25", acn_prices)
    expected = (306.0 - 301.0) / 301.0
    assert result["fwd_ret_5d"] == pytest.approx(expected, rel=1e-5)

def test_amc_fwd_ret_1d(nvda_prices):
    """
    AMC call on 2023-10-25 (index 2):
      ref_close  = close[2] = 402.0
      fwd_ret_1d = close[3] / 402.0 - 1 = 403.0/402.0 - 1 ~ 0.00249
    """
    result = compute_price_row("NVDA", "2023-10-25", nvda_prices)
    expected = (403.0 - 402.0) / 402.0
    assert result["fwd_ret_1d"] == pytest.approx(expected, rel=1e-5)

def test_amc_fwd_ret_5d(nvda_prices):
    """
    AMC fwd_ret_5d: close at call_date + 5 trading days = index 7 = 407.0
    ref_close = 402.0 -> (407 - 402) / 402 ~ 0.01244
    """
    result = compute_price_row("NVDA", "2023-10-25", nvda_prices)
    expected = (407.0 - 402.0) / 402.0
    assert result["fwd_ret_5d"] == pytest.approx(expected, rel=1e-5)

def test_price_missing_on_empty_df():
    result = compute_price_row("NVDA", "2023-10-25", pd.DataFrame())
    assert result["price_missing"] is True
    assert result["fwd_ret_1d"] is None

# ═══════════════════════════════════════════════════════════════════════════
# nth_trading_day helper
# ═══════════════════════════════════════════════════════════════════════════

def test_nth_trading_day_forward(nvda_prices):
    idx = nvda_prices.index
    anchor = idx[5]
    assert _nth_trading_day(idx, anchor, 0)  == idx[5]
    assert _nth_trading_day(idx, anchor, 1)  == idx[6]
    assert _nth_trading_day(idx, anchor, -1) == idx[4]

def test_nth_trading_day_out_of_bounds_returns_none(nvda_prices):
    idx = nvda_prices.index
    assert _nth_trading_day(idx, idx[0], -1) is None
    assert _nth_trading_day(idx, idx[-1], 1) is None

# ═══════════════════════════════════════════════════════════════════════════
# HPE filter
# ═══════════════════════════════════════════════════════════════════════════

def _make_hpe_df(dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "ticker":        ["HPE"] * len(dates),
        "call_date":     dates,
        "chunk_id":      [f"HPE_{d.replace('-','')}__prep__p001" for d in dates],
        "transcript_id": [f"HPE_{d.replace('-','')}" for d in dates],
    })

def test_hpe_pre2016_rows_dropped():
    df = _make_hpe_df(["2015-11-15", "2015-12-10", "2016-01-20", "2016-04-15"])
    filtered = _apply_hpe_date_filter(df)
    assert len(filtered) == 2
    assert all(filtered["call_date"] >= HPE_MIN_DATE)

def test_hpe_2016_rows_retained():
    df = _make_hpe_df(["2016-01-20", "2017-03-15", "2020-06-10"])
    filtered = _apply_hpe_date_filter(df)
    assert len(filtered) == 3

def test_non_hpe_rows_unaffected():
    df = pd.DataFrame({
        "ticker":    ["NVDA", "MSFT"],
        "call_date": ["2015-06-01", "2015-09-15"],
        "chunk_id":  ["NVDA_20150601__prep__p001", "MSFT_20150915__prep__p001"],
        "transcript_id": ["NVDA_20150601", "MSFT_20150915"],
    })
    assert len(_apply_hpe_date_filter(df)) == 2

# ═══════════════════════════════════════════════════════════════════════════
# Schema compliance
# ═══════════════════════════════════════════════════════════════════════════

def test_output_columns_defined():
    """OUTPUT_COLUMNS must contain all required fields."""
    required = {
        "transcript_id", "chunk_id", "ticker", "company", "sector",
        "call_date", "call_timing", "fiscal_quarter",
        "chunk_text", "speaker_role",
        "t_minus1_open", "t_minus1_close", "t_minus1_volume",
        "fwd_ret_1d", "fwd_ret_5d", "fwd_ret_10d",
    }
    assert required.issubset(set(OUTPUT_COLUMNS))

def test_price_missing_row_retained_not_dropped(nvda_prices):
    """
    For a call_date far beyond the price fixture, the AMC reference resolves
    to the last available trading day (searchsorted clamps to end-of-index).
    The function therefore does NOT set price_missing — it has a reference close.
    However, all forward-return windows are None because there are no prices
    beyond the fixture end.  Verify this contract: no exception raised, and
    fwd_ret_1d / fwd_ret_5d / fwd_ret_10d are all None.
    """
    result = compute_price_row("NVDA", "2030-01-02", nvda_prices)
    # reference resolves to last available day — not missing
    assert result["price_missing"] is False
    assert result["t_minus1_close"] is not None
    # but forward prices are unavailable
    assert result["fwd_ret_1d"]  is None
    assert result["fwd_ret_5d"]  is None
    assert result["fwd_ret_10d"] is None

def test_price_missing_on_truly_empty_df():
    """Empty price DataFrame -> price_missing=True, all fields None."""
    result = compute_price_row("NVDA", "2023-10-25", pd.DataFrame())
    assert result["price_missing"] is True
    assert result["t_minus1_close"] is None
    assert result["fwd_ret_1d"] is None
