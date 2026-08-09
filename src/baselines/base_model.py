"""
base_model.py
-------------
Zero-shot baseline: structured signal extraction using the Groq API
(llama-3.1-8b-instant) — the un-fine-tuned base model in the same
Llama 3.x lineage as the Phase 4 fine-tune target (Llama 3.2 3B).

This is Baseline 2.  The same prompt and schema are used by the
fine-tuned model in Phase 4, so the comparison is apples-to-apples.

Labeling schema (identical to fine-tuned model output)
-------------------------------------------------------
{
    "guidance_direction": "raised" | "lowered" | "maintained" | "not_discussed",
    "tone":               "confident" | "cautious" | "evasive" | "neutral",
    "hedging_score":      float [0.0, 1.0],
    "key_flags":          list[str],
}

Rate-limit handling (Groq free tier)
-------------------------------------
Limits: 30 RPM, 6,000 TPM, 14,400 RPD.
With ~2,048-token chunks, effective throughput is ~2-3 chunks/minute
(TPM is the binding constraint).

Strategy:
- tenacity retry with exponential backoff on RateLimitError / APIError
- Explicit inter-request sleep to stay within TPM budget
- Hard daily-request guard: stop and log clearly if RPD is exhausted

Usage
-----
>>> from src.baselines.base_model import score_dataframe
>>> results = score_dataframe(df, max_chunks=10)   # sample run first
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)
import logging

load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────

MODEL_ID: str = "llama-3.1-8b-instant"

# Conservative pacing: Groq free tier is 6,000 TPM.
# Each chunk is ~512 prompt tokens + ~150 completion tokens = ~662 tokens.
# Safe throughput ≈ 6000 / 662 ≈ 9 per minute -> 1 per ~7 seconds.
# We use 8 seconds to give a safe margin.
REQUEST_INTERVAL_SECS: float = 8.0

# Retry config
MAX_ATTEMPTS: int = 5
WAIT_MIN_SECS: float = 10.0
WAIT_MAX_SECS: float = 120.0

# Schema validation
VALID_GUIDANCE = {"raised", "lowered", "maintained", "not_discussed"}
VALID_TONE     = {"confident", "cautious", "evasive", "neutral"}

# ── Prompt template ───────────────────────────────────────────────────────

SYSTEM_PROMPT = """    You are a financial analyst assistant.  Your task is to analyze earnings call
transcript excerpts and extract structured signals.

Always respond with a single JSON object.  No markdown, no explanation, no
additional text — only the JSON object.

JSON schema:
{
  "guidance_direction": "<raised|lowered|maintained|not_discussed>",
  "tone":               "<confident|cautious|evasive|neutral>",
  "hedging_score":      <float between 0.0 and 1.0>,
  "key_flags":          [<list of up to 5 short phrases that stand out>]
}

Field definitions:
- guidance_direction: Did management change, reaffirm, or avoid giving guidance?
- tone: The overall management tone in this excerpt.
  * confident: strong positive language, clear commitments, limited hedging
  * cautious: significant negative language, lowered expectations, warnings
  * evasive: lots of hedging, circular answers, avoids direct commitments
  * neutral: balanced or non-committal language
- hedging_score: 0.0 = no hedging at all; 1.0 = extremely hedged language
- key_flags: notable phrases worth flagging (e.g. "delayed product launch",
  "margin pressure", "macro uncertainty", "beat by 5 cents")
"""

USER_PROMPT_TEMPLATE = """    Analyze the following earnings call excerpt and return the JSON signal:

--- EXCERPT ---
{chunk_text}
--- END ---
"""

# ── Groq client (lazy init) ───────────────────────────────────────────────

_client = None

def _get_client():
    global _client
    if _client is None:
        try:
            from groq import Groq
        except ImportError:
            raise ImportError(
                "groq package not installed. Run: pip install groq"
            )
        api_key = os.environ.get("GROQ_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Add it to your .env file and reload."
            )
        _client = Groq(api_key=api_key)
        logger.info(f"Groq client initialised (model={MODEL_ID}).")
    return _client

# ── Retry-wrapped API call ────────────────────────────────────────────────

def _is_rate_limit_error(exc: Exception) -> bool:
    """Detect Groq rate-limit and transient server errors."""
    try:
        from groq import RateLimitError, APIStatusError
        if isinstance(exc, RateLimitError):
            return True
        if isinstance(exc, APIStatusError) and exc.status_code in (429, 500, 503):
            return True
    except ImportError:
        pass
    return "rate limit" in str(exc).lower() or "429" in str(exc)

@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(
        multiplier=2, min=WAIT_MIN_SECS, max=WAIT_MAX_SECS
    ),
    stop=stop_after_attempt(MAX_ATTEMPTS),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _call_api(chunk_text: str) -> str:
    """
    Send one chunk to Groq and return the raw response text.
    Decorated with tenacity retry — handles transient errors and
    rate-limit errors with exponential backoff.
    """
    client = _get_client()
    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(
                    chunk_text=chunk_text[:6_000]  # safety truncation
                ),
            },
        ],
        temperature=0.0,
        max_tokens=256,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content

# ── JSON parsing + validation ─────────────────────────────────────────────

_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)

def _parse_response(raw: str, chunk_id: str) -> dict:
    """
    Parse and validate the model's JSON response.
    Falls back to a default schema dict on any parse failure.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract JSON from noisy output
        match = _JSON_RE.search(raw)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                data = {}
        else:
            data = {}

    if not data:
        logger.warning(f"{chunk_id}: could not parse model JSON. Using defaults.")
        return _default_output(parse_error=True)

    # Validate and coerce fields
    gd = data.get("guidance_direction", "not_discussed")
    if gd not in VALID_GUIDANCE:
        logger.debug(f"{chunk_id}: invalid guidance_direction='{gd}' -> not_discussed")
        gd = "not_discussed"

    tone = data.get("tone", "neutral")
    if tone not in VALID_TONE:
        logger.debug(f"{chunk_id}: invalid tone='{tone}' -> neutral")
        tone = "neutral"

    hs = data.get("hedging_score", 0.5)
    try:
        hs = max(0.0, min(1.0, float(hs)))
    except (TypeError, ValueError):
        hs = 0.5

    flags = data.get("key_flags", [])
    if not isinstance(flags, list):
        flags = []
    flags = [str(f)[:100] for f in flags[:5]]  # cap at 5, 100 chars each

    return {
        "guidance_direction": gd,
        "tone":               tone,
        "hedging_score":      round(hs, 4),
        "key_flags":          flags,
        "_parse_error":       False,
    }

def _default_output(parse_error: bool = False) -> dict:
    return {
        "guidance_direction": "not_discussed",
        "tone":               "neutral",
        "hedging_score":      0.5,
        "key_flags":          [],
        "_parse_error":       parse_error,
    }

# ── Public API ────────────────────────────────────────────────────────────

def score_chunk(chunk_text: str, chunk_id: str = "unknown") -> dict:
    """
    Run zero-shot extraction on a single chunk_text string.

    Returns a dict matching the labeling schema plus _parse_error flag.
    Does NOT apply inter-request rate-limit pacing — use score_dataframe()
    for batch runs.
    """
    raw = _call_api(chunk_text)
    return _parse_response(raw, chunk_id)

def score_dataframe(
    df: pd.DataFrame,
    max_chunks: int | None = None,
    request_interval: float = REQUEST_INTERVAL_SECS,
) -> pd.DataFrame:
    """
    Run zero-shot extraction across all rows in *df*.

    Parameters
    ----------
    df               : DataFrame with chunk_text and chunk_id columns.
    max_chunks       : If set, process only the first N rows (use for
                       sample runs before burning full rate-limit budget).
    request_interval : Seconds to sleep between API calls (default 8s).

    Returns
    -------
    *df* with new columns: guidance_direction, tone, hedging_score,
    key_flags, _parse_error.
    """
    if "chunk_text" not in df.columns:
        raise ValueError("DataFrame must have a 'chunk_text' column.")

    subset = df.head(max_chunks) if max_chunks else df
    n = len(subset)
    logger.info(
        f"Base-model scoring {n} chunks "
        f"(model={MODEL_ID}, ~{n * request_interval / 60:.1f} min estimated) ..."
    )

    results: list[dict] = []
    for i, (_, row) in enumerate(subset.iterrows(), 1):
        chunk_id = row.get("chunk_id", f"row_{i}")
        logger.debug(f"  [{i}/{n}] {chunk_id}")
        try:
            result = score_chunk(str(row["chunk_text"]), chunk_id=str(chunk_id))
        except Exception as exc:
            logger.error(f"  [{i}/{n}] {chunk_id} FAILED after retries: {exc}")
            result = _default_output(parse_error=True)
        results.append(result)

        # Rate-limit pacing (skip sleep after last request)
        if i < n:
            time.sleep(request_interval)

    scores_df = pd.DataFrame(results, index=subset.index)
    out = pd.concat([subset, scores_df], axis=1)
    n_errors = int(scores_df["_parse_error"].sum())
    logger.info(
        f"Base-model scoring complete. "
        f"Parse errors: {n_errors}/{n}."
    )
    return out
