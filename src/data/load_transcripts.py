"""
load_transcripts.py
-------------------
Load earnings-call transcripts from the Hugging Face dataset
``kurry/sp500_earnings_transcripts`` and chunk them for the Phase 1
IT-sector slice (25 companies, 2015-2024).

Chunking strategy
-----------------
Each transcript is segmented into two section types, then chunked:

* Prepared remarks  →  ``{tid}__prep__p{N:03d}``
* Q&A session       →  ``{tid}__qa__q{M:03d}__p{N:03d}``

  One chunk per analyst Q+A exchange; ``p > 1`` only when a single
  exchange exceeds ``MAX_CHUNK_TOKENS``.

Any section that exceeds ``MAX_CHUNK_TOKENS`` is sub-chunked on sentence
boundaries and flagged ``truncation_warning=True``.  Nothing is silently
truncated.

Token estimation
----------------
Uses ``len(text) // CHARS_PER_TOKEN`` (no tokeniser dependency).
Replace with a real tokeniser once the base model is finalised.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pandas as pd
from loguru import logger

# ── Constants ─────────────────────────────────────────────────────────────

IT_TICKERS: frozenset[str] = frozenset({
    "NVDA", "INTC", "QCOM", "TXN", "AVGO", "MU", "AMAT",
    "MSFT", "ORCL", "CRM", "ADBE", "NOW", "INTU", "CDNS", "SNPS",
    "AAPL", "CSCO", "HPE", "NTAP", "KEYS",
    "ACN", "IBM", "FISV", "PAYX", "CTSH",
})

SECTOR: str = "Information Technology"
MIN_DATE: str = "2015-01-01"
MAX_DATE: str = "2024-12-31"
MAX_CHUNK_TOKENS: int = 2_048
CHARS_PER_TOKEN: int = 4   # 1 token ~ 4 chars

_QA_HEADER_RE = re.compile(
    r"question.and.answer|q&a session|open.*floor.*question|now.*take.*question",
    re.IGNORECASE,
)
_ANALYST_ROLE_RE = re.compile(
    r"analyst|research|managing director|\bmd,|portfolio manager",
    re.IGNORECASE,
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# ── Data class ────────────────────────────────────────────────────────────

@dataclass
class TranscriptChunk:
    transcript_id: str
    chunk_id: str
    ticker: str
    company: str
    sector: str
    call_date: str
    fiscal_quarter: str
    chunk_text: str
    speaker_role: str       # "prepared_remarks" | "qa_exchange"
    truncation_warning: bool = False

# ── Low-level helpers ─────────────────────────────────────────────────────

def _est_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN

def _is_analyst(role: str) -> bool:
    return bool(_ANALYST_ROLE_RE.search(role or ""))

def _build_chunk_id(tid: str, section: str, q: int, part: int) -> str:
    if section == "prep":
        return f"{tid}__prep__p{part:03d}"
    return f"{tid}__qa__q{q:03d}__p{part:03d}"

def _split_parts(text: str, max_chars: int) -> list[str]:
    """Split *text* into parts <= max_chars, breaking on sentence boundaries."""
    if len(text) <= max_chars:
        return [text]
    sentences = _SENTENCE_SPLIT_RE.split(text)
    parts: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for sent in sentences:
        if buf_len + len(sent) > max_chars and buf:
            parts.append(" ".join(buf))
            buf, buf_len = [], 0
        buf.append(sent)
        buf_len += len(sent) + 1
    if buf:
        parts.append(" ".join(buf))
    return parts or [text]

# ── Section detection ─────────────────────────────────────────────────────

def _detect_qa_boundary(turns: list[dict]) -> int:
    """
    Return the index of the first Q&A turn.
    Checks for an explicit Q&A header line first, then the first analyst turn.
    Returns ``len(turns)`` if no Q&A section is found.
    """
    for i, t in enumerate(turns):
        if _QA_HEADER_RE.search(t.get("text", "")):
            return i + 1
        if i > 0 and _is_analyst(t.get("role", "")):
            return i
    return len(turns)

def _group_qa_exchanges(qa_turns: list[dict]) -> list[list[dict]]:
    """
    Group Q&A turns into exchanges: each exchange starts at an analyst
    question and includes all management responses until the next analyst.
    """
    exchanges: list[list[dict]] = []
    current: list[dict] = []
    for t in qa_turns:
        if _is_analyst(t.get("role", "")) and current:
            exchanges.append(current)
            current = []
        current.append(t)
    if current:
        exchanges.append(current)
    return exchanges

# ── Chunking ──────────────────────────────────────────────────────────────

def _format_turns(turns: list[dict]) -> str:
    return "\n\n".join(
        f"[{t.get('speaker', 'Unknown')}]: {t.get('text', '')}".strip()
        for t in turns
    )

def _chunk_transcript(
    turns: list[dict],
    transcript_id: str,
    ticker: str,
    company: str,
    call_date: str,
    fiscal_quarter: str,
) -> list[TranscriptChunk]:
    """Convert speaker-turn list for one transcript into TranscriptChunk objects."""
    chunks: list[TranscriptChunk] = []
    max_chars = MAX_CHUNK_TOKENS * CHARS_PER_TOKEN

    qa_idx = _detect_qa_boundary(turns)
    prep_turns = turns[:qa_idx]
    qa_turns   = turns[qa_idx:]

    # ── Prepared remarks ──────────────────────────────────────────────────
    if prep_turns:
        text = _format_turns(prep_turns)
        parts = _split_parts(text, max_chars)
        if len(parts) > 1:
            logger.warning(
                f"{transcript_id} prepared remarks exceed {MAX_CHUNK_TOKENS} tokens "
                f"({_est_tokens(text):,} est.) -> split into {len(parts)} parts."
            )
        for n, part in enumerate(parts, 1):
            chunks.append(TranscriptChunk(
                transcript_id=transcript_id,
                chunk_id=_build_chunk_id(transcript_id, "prep", 0, n),
                ticker=ticker, company=company, sector=SECTOR,
                call_date=call_date, fiscal_quarter=fiscal_quarter,
                chunk_text=part, speaker_role="prepared_remarks",
                truncation_warning=(len(parts) > 1),
            ))

    # ── Q&A ───────────────────────────────────────────────────────────────
    if qa_turns:
        exchanges = _group_qa_exchanges(qa_turns)
        for qn, exchange in enumerate(exchanges, 1):
            text = _format_turns(exchange)
            parts = _split_parts(text, max_chars)
            if len(parts) > 1:
                logger.warning(
                    f"{transcript_id} Q{qn:03d} exchange exceeds "
                    f"{MAX_CHUNK_TOKENS} tokens ({_est_tokens(text):,} est.) "
                    f"-> split into {len(parts)} parts."
                )
            for n, part in enumerate(parts, 1):
                chunks.append(TranscriptChunk(
                    transcript_id=transcript_id,
                    chunk_id=_build_chunk_id(transcript_id, "qa", qn, n),
                    ticker=ticker, company=company, sector=SECTOR,
                    call_date=call_date, fiscal_quarter=fiscal_quarter,
                    chunk_text=part, speaker_role="qa_exchange",
                    truncation_warning=(len(parts) > 1),
                ))

    return chunks

# ── Helpers for dataset ingestion ─────────────────────────────────────────

def _infer_transcript_id(ticker: str, call_date: str) -> str:
    return f"{ticker}_{call_date.replace('-', '')}"

def _infer_fiscal_quarter(call_date: str) -> str:
    try:
        dt = pd.Timestamp(call_date)
        q = (dt.month - 1) // 3 + 1
        return f"Q{q} {dt.year}"
    except Exception:
        return "Unknown"

def _detect_col_map(sample: dict[str, Any]) -> dict[str, str]:
    candidates: dict[str, list[str]] = {
        "ticker":  ["ticker", "symbol", "Ticker", "Symbol"],
        "date":    ["date", "call_date", "Date", "CallDate", "earnings_date"],
        "company": ["company", "Company", "company_name", "name"],
        "speaker": ["speaker", "Speaker", "speaker_name"],
        "role":    ["role", "Role", "title", "Title", "speaker_role", "position"],
        "text":    ["text", "Text", "content", "Content", "transcript"],
    }
    resolved: dict[str, str] = {}
    for field, opts in candidates.items():
        for opt in opts:
            if opt in sample:
                resolved[field] = opt
                break
        if field not in resolved:
            if field in ("ticker", "date", "text"):
                raise ValueError(
                    f"Required column '{field}' not found. "
                    f"Available: {list(sample.keys())}"
                )
            resolved[field] = opts[0]  # graceful default
    return resolved

# ── Public API ────────────────────────────────────────────────────────────

def load_it_transcripts(
    hf_dataset_name: str = "kurry/sp500_earnings_transcripts",
    split: str = "train",
    streaming: bool = True,
) -> pd.DataFrame:
    """
    Load, filter, and chunk transcripts for the 25-company IT slice.

    Returns a DataFrame with columns:
        transcript_id, chunk_id, ticker, company, sector, call_date,
        fiscal_quarter, chunk_text, speaker_role, truncation_warning
    """
    from datasets import load_dataset  # deferred: avoid import cost if unused

    logger.info(f"Loading {hf_dataset_name} (streaming={streaming}) ...")
    ds = load_dataset(
        hf_dataset_name, split=split,
        streaming=streaming, trust_remote_code=True,
    )

    first = next(iter(ds.take(5) if streaming else ds.select(range(5))))  # type: ignore[attr-defined]
    logger.info(f"Dataset columns: {list(first.keys())}")
    col = _detect_col_map(first)
    logger.info(f"Column mapping: {col}")

    # Buffer filtered turns keyed by (ticker, call_date)
    transcript_turns: dict[tuple, dict] = {}

    for row in ds:
        ticker = str(row.get(col["ticker"], "") or "").upper().strip()
        if ticker not in IT_TICKERS:
            continue
        date_raw = str(row.get(col["date"], "") or "")
        try:
            call_date = pd.Timestamp(date_raw).strftime("%Y-%m-%d")
        except Exception:
            logger.debug(f"Unparseable date '{date_raw}' for {ticker} — skipping.")
            continue
        if not (MIN_DATE <= call_date <= MAX_DATE):
            continue

        key = (ticker, call_date)
        if key not in transcript_turns:
            transcript_turns[key] = {
                "company": str(row.get(col["company"], "") or "").strip(),
                "turns": [],
            }
        transcript_turns[key]["turns"].append({
            "speaker": str(row.get(col["speaker"], "") or ""),
            "role":    str(row.get(col["role"], "") or ""),
            "text":    str(row.get(col["text"], "") or ""),
        })

    logger.info(
        f"Buffered {len(transcript_turns)} unique (ticker, date) transcripts."
    )

    all_chunks: list[TranscriptChunk] = []
    for (ticker, call_date), meta in transcript_turns.items():
        tid = _infer_transcript_id(ticker, call_date)
        all_chunks.extend(
            _chunk_transcript(
                turns=meta["turns"],
                transcript_id=tid,
                ticker=ticker,
                company=meta["company"],
                call_date=call_date,
                fiscal_quarter=_infer_fiscal_quarter(call_date),
            )
        )

    df = pd.DataFrame([vars(c) for c in all_chunks])
    n_warn = int(df["truncation_warning"].sum()) if len(df) else 0
    logger.info(
        f"Loaded {len(df):,} chunks from "
        f"{df['transcript_id'].nunique()} transcripts, "
        f"{df['ticker'].nunique()} tickers. "
        f"Truncation warnings: {n_warn}."
    )
    return df
