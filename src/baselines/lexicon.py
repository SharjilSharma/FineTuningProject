"""
lexicon.py
----------
Loughran-McDonald (LM) financial-sentiment lexicon baseline.

This is Baseline 1: pure word-counting, no ML.  Uses the official
Loughran-McDonald Master Dictionary CSV (auto-downloaded on first use
to data/lm_dict/).

Labeling schema output (per chunk)
-----------------------------------
{
    "guidance_direction": "raised" | "lowered" | "maintained" | "not_discussed",
    "tone":               "confident" | "cautious" | "neutral",
    "hedging_score":      float [0, 1],   # uncertainty-word density
    "key_flags":          list[str],       # notable LM-flagged words found
}

Mapping rules
-------------
* tone:
    - net_sentiment > +THRESHOLD_POS -> "confident"
    - net_sentiment < -THRESHOLD_NEG -> "cautious"
    - otherwise                      -> "neutral"

* hedging_score = n_uncertainty / max(n_total_words, 1)

* guidance_direction:
    The LM lexicon has no guidance concept; we apply a simple keyword
    match over the chunk text.  This will produce "not_discussed" for
    most Q&A chunks and only fires on explicit forward-looking phrases.

* key_flags: up to MAX_FLAGS unique LM-flagged words (negative +
    uncertainty) that appeared in the chunk.

Notes
-----
The LM Master Dictionary is a widely-used, freely available academic
resource (see https://sraf.nd.edu/loughranmcdonald-master-dictionary/).
We download the CSV from the authors' public GitHub mirror on first use.
"""

from __future__ import annotations

import re
import urllib.request
from pathlib import Path

import pandas as pd
from loguru import logger

# ── Constants ─────────────────────────────────────────────────────────────

LM_CSV_URL = (
    "https://raw.githubusercontent.com/sedeh/Loughran-McDonald/"
    "master/LoughranMcDonald_MasterDictionary_2018.csv"
)
LM_CACHE_DIR = Path("data/lm_dict")
LM_CACHE_FILE = LM_CACHE_DIR / "LM_MasterDictionary.csv"

THRESHOLD_POS: float = 0.01   # net sentiment > +1% -> confident
THRESHOLD_NEG: float = 0.01   # net sentiment < -1% -> cautious
MAX_FLAGS: int = 10

# Guidance keywords (forward-looking: raised / lowered / maintained)
_RAISE_RE = re.compile(
    r"\b(rais(?:ing|ed|es)|increas(?:ing|ed|es)|upward|above guidance|"
    r"beat(?:ing)? expectation|exceeded|outperform|upgrade)\b",
    re.IGNORECASE,
)
_LOWER_RE = re.compile(
    r"\b(lower(?:ing|ed)?|reduc(?:ing|ed|es)|cut(?:ting)?|below guidance|"
    r"miss(?:ed|ing)?|downward|downgrad|disappoint)\b",
    re.IGNORECASE,
)
_MAINTAIN_RE = re.compile(
    r"\b(in line|on track|reaffirm|maintain(?:ing|ed)?|reiterat|"
    r"consistent with|unchanged|as expect|as guided)\b",
    re.IGNORECASE,
)
_FORWARD_RE = re.compile(
    r"\b(guid(?:ance|ing|ed)?|forecast|outlook|expect(?:ation|ed|s)?|"
    r"project(?:ion|ing|ed)?|next quarter|full.year|fiscal year)\b",
    re.IGNORECASE,
)

# ── LM dictionary loading ─────────────────────────────────────────────────

def _download_lm_dict() -> None:
    """Download LM Master Dictionary CSV to local cache on first use."""
    LM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading LM Master Dictionary -> {LM_CACHE_FILE} ...")
    urllib.request.urlretrieve(LM_CSV_URL, LM_CACHE_FILE)
    logger.info("LM dictionary downloaded.")

def _load_lm_sets() -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    """
    Return (positive_words, negative_words, uncertainty_words) as
    frozensets of uppercase strings.  Downloads the CSV if not cached.
    """
    if not LM_CACHE_FILE.exists():
        _download_lm_dict()

    df = pd.read_csv(LM_CACHE_FILE, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    # Column names vary by CSV version; handle both
    word_col = "Word" if "Word" in df.columns else df.columns[0]
    pos_col  = next((c for c in df.columns if "Positive" in c), None)
    neg_col  = next((c for c in df.columns if "Negative" in c), None)
    unc_col  = next((c for c in df.columns if "Uncertain" in c), None)

    words = df[word_col].str.upper().str.strip()

    pos = frozenset(words[df[pos_col].fillna(0) != 0]) if pos_col else frozenset()
    neg = frozenset(words[df[neg_col].fillna(0) != 0]) if neg_col else frozenset()
    unc = frozenset(words[df[unc_col].fillna(0) != 0]) if unc_col else frozenset()

    logger.info(
        f"LM dictionary loaded: {len(pos)} positive, "
        f"{len(neg)} negative, {len(unc)} uncertainty words."
    )
    return pos, neg, unc

# Lazy-loaded module-level singletons
_LM_POS: frozenset[str] | None = None
_LM_NEG: frozenset[str] | None = None
_LM_UNC: frozenset[str] | None = None

def _get_lm_sets() -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    global _LM_POS, _LM_NEG, _LM_UNC
    if _LM_POS is None:
        _LM_POS, _LM_NEG, _LM_UNC = _load_lm_sets()
    return _LM_POS, _LM_NEG, _LM_UNC

# ── Scoring ───────────────────────────────────────────────────────────────

_WORD_RE = re.compile(r"[A-Za-z]+")

def score_chunk(text: str) -> dict:
    """
    Apply LM lexicon scoring to a single chunk_text string.

    Returns
    -------
    dict matching the labeling schema:
        guidance_direction, tone, hedging_score, key_flags
    """
    lm_pos, lm_neg, lm_unc = _get_lm_sets()

    words = _WORD_RE.findall(text)
    words_upper = [w.upper() for w in words]
    n_total = max(len(words), 1)

    n_pos = sum(1 for w in words_upper if w in lm_pos)
    n_neg = sum(1 for w in words_upper if w in lm_neg)
    n_unc = sum(1 for w in words_upper if w in lm_unc)

    net_sentiment = (n_pos - n_neg) / n_total
    hedging_score = round(n_unc / n_total, 6)

    # Tone classification
    if net_sentiment > THRESHOLD_POS:
        tone = "confident"
    elif net_sentiment < -THRESHOLD_NEG:
        tone = "cautious"
    else:
        tone = "neutral"

    # Guidance direction via keyword match
    has_forward = bool(_FORWARD_RE.search(text))
    if has_forward:
        if _RAISE_RE.search(text):
            guidance_direction = "raised"
        elif _LOWER_RE.search(text):
            guidance_direction = "lowered"
        elif _MAINTAIN_RE.search(text):
            guidance_direction = "maintained"
        else:
            guidance_direction = "not_discussed"
    else:
        guidance_direction = "not_discussed"

    # Key flags: unique LM-negative and LM-uncertainty words found
    flag_words = sorted({
        w for w in words_upper
        if w in lm_neg or w in lm_unc
    })[:MAX_FLAGS]

    return {
        "guidance_direction": guidance_direction,
        "tone":               tone,
        "hedging_score":      hedging_score,
        "key_flags":          flag_words,
        # diagnostic extras (dropped before final schema output if needed)
        "_lm_n_pos":          n_pos,
        "_lm_n_neg":          n_neg,
        "_lm_n_unc":          n_unc,
        "_lm_n_total":        n_total,
        "_lm_net_sentiment":  round(net_sentiment, 6),
    }

def score_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply score_chunk() to every row in *df* (must have a chunk_text column).

    Returns *df* with new columns:
        guidance_direction, tone, hedging_score, key_flags,
        _lm_n_pos, _lm_n_neg, _lm_n_unc, _lm_n_total, _lm_net_sentiment
    """
    if "chunk_text" not in df.columns:
        raise ValueError("DataFrame must have a 'chunk_text' column.")

    logger.info(f"LM scoring {len(df):,} chunks ...")
    results = [score_chunk(text) for text in df["chunk_text"]]
    scores_df = pd.DataFrame(results, index=df.index)
    out = pd.concat([df, scores_df], axis=1)
    logger.info("LM scoring complete.")
    return out
