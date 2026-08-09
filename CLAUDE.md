# CLAUDE.md — Project Rules for Claude Code

This file is read automatically by Claude Code at session start.
Do not delete or rename it.

---

## Project

**Earnings Call Signal Extraction Agent**
A research project testing whether a fine-tuned small LLM extracts structured
signals from earnings call transcripts better than a lexicon baseline or a
zero-shot base model, and whether those signals correlate with forward price returns.

See docs/project-spec.md for full spec. See docs/roadmap.md for phase plan.

---

## Current Phase

**Phase 0 complete — Phase 1 (Data Pipeline) is next.**
Do not skip phases or implement future-phase logic speculatively.

---

## Routing Rules (OmniRoute)

Use these rules when routing tasks through OmniRoute / Headroom:

| Task type | Route to |
|---|---|
| Boilerplate: docstrings, formatting, renaming, comments | Free-tier provider |
| Data wrangling: pandas / yfinance / HF datasets | Free-tier provider |
| Lexicon baseline logic | Free-tier provider |
| LoRA / QLoRA training config or strategy | Anthropic Sonnet (paid) |
| LangGraph agent graph design | Anthropic Sonnet (paid) |
| Architecture decisions, trade-off analysis | Anthropic Sonnet (paid) |
| Eval methodology, statistical interpretation | Anthropic Sonnet (paid) |
| Debugging unknown errors | Anthropic Sonnet (paid) |

Rationale: use paid tier only where reasoning depth pays for itself.

---

## Testing Discipline

- Every src/ module must have a corresponding 	ests/test_<module>.py.
- Tests are written BEFORE or ALONGSIDE implementation (not after).
- Run pytest --cov=src before marking any phase complete.
- No phase is done until tests pass and coverage > 80% for that phase's modules.

---

## Repo Conventions

- All secrets via .env; never hardcode. Use python-dotenv to load.
- Data files (parquet, CSV, model weights) are gitignored — always re-downloadable.
- FAISS index files (.index) are gitignored — always rebuildable.
- Notebooks are for exploration only; production logic lives in src/.
- Use loguru for logging, not print.
- Use pydantic for all data model validation (schemas, API models, configs).

---

## Phase Exit Criteria (summary)

| Phase | Done when |
|---|---|
| 0 | pip install -r requirements.txt clean; folder skeleton present |
| 1 | IT-sector transcripts + forward returns in clean parquet |
| 2 | Both baselines produce valid JSON on 100 sample transcripts |
| 3 | 200-400 labeled examples; test set isolated; >80% spot-check agreement |
| 4 | Fine-tuned model loaded in Ollama; produces valid JSON locally |
| 5 | Per-field accuracy table comparing all three methods |
| 6 | Correlation table with significance, segmented by sector/period |
| 7 | POST /extract triggers full agent pipeline, returns result in <30s |
