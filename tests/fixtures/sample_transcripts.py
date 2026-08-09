"""
Synthetic transcript turns for unit testing.

Each fixture represents one transcript as a list of speaker-turn dicts.
"""

from __future__ import annotations

# ── Fixture A: normal call (prep + Q&A with 2 analyst questions) ──────────

NVDA_NORMAL: list[dict] = [
    {"speaker": "CEO",      "role": "CEO",           "text": "Good afternoon everyone. " * 30},
    {"speaker": "CFO",      "role": "CFO",           "text": "Turning to financials. " * 30},
    {"speaker": "Operator", "role": "Operator",      "text": "Question-and-Answer session begins now."},
    {"speaker": "J. Smith", "role": "Analyst, BofA", "text": "Can you speak to GPU demand outlook?"},
    {"speaker": "CEO",      "role": "CEO",           "text": "Demand remains very strong. " * 10},
    {"speaker": "CFO",      "role": "CFO",           "text": "We guide revenue up 20%. " * 5},
    {"speaker": "M. Lee",   "role": "Research Analyst", "text": "What about margin pressure?"},
    {"speaker": "CFO",      "role": "CFO",           "text": "Margins are expected to hold. " * 5},
]

# ── Fixture B: call with NO Q&A section ────────────────────────────────────

MSFT_NO_QA: list[dict] = [
    {"speaker": "CEO", "role": "CEO", "text": "Welcome to our investor update. " * 20},
    {"speaker": "CFO", "role": "CFO", "text": "Revenue grew 12% year over year. " * 20},
]

# ── Fixture C: call where ONE Q&A exchange is very long (> MAX_CHUNK_TOKENS)

def _make_long_text(n_words: int = 3_000) -> str:
    return ("This is a lengthy analyst response discussing many topics. " * (n_words // 10)).strip()

AAPL_LONG_QA: list[dict] = [
    {"speaker": "CEO",      "role": "CEO",           "text": "Good evening. " * 20},
    {"speaker": "Operator", "role": "Operator",      "text": "We will now take questions."},
    {"speaker": "A. Wong",  "role": "Analyst, GS",  "text": "Can you elaborate on services margin?"},
    {"speaker": "CEO",      "role": "CEO",           "text": _make_long_text(3_000)},
    {"speaker": "B. Jones", "role": "MD, Research",  "text": "Follow-up on Mac revenue trajectory?"},
    {"speaker": "CFO",      "role": "CFO",           "text": "Mac revenue will grow. " * 5},
]

# ── Fixture D: BMO company (ACN), simple prep only ───────────────────────

ACN_BMO: list[dict] = [
    {"speaker": "CEO", "role": "CEO", "text": "Accenture delivers strong results. " * 15},
    {"speaker": "CFO", "role": "CFO", "text": "EPS beat consensus by 5 cents. " * 10},
]

# ── Fixture E: HPE call — 2015-12-15 (should be filtered out) ────────────

HPE_PRE2016: list[dict] = [
    {"speaker": "CEO", "role": "CEO", "text": "Welcome to HPE's first standalone call. " * 10},
]
