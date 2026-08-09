# Build Roadmap — Earnings Call Signal Extraction Agent

Phase-by-phase build order. Each phase must be working end-to-end before the next begins.

---

## Phase 0 — Repo Scaffolding (CURRENT)

**Goal**: skeleton in place; no real logic yet.

- [x] Folder structure (src/, 	ests/, 
otebooks/, docs/)
- [x] equirements.txt (full tech stack)
- [x] .env.example (placeholder secrets)
- [x] .gitignore
- [x] Stub modules with docstrings in every src/ sub-package

**Exit criteria**: git init && pip install -r requirements.txt completes cleanly.

---

## Phase 1 — Data Pipeline

**Goal**: load transcripts and price data, filter to IT sector / 2015-2024 slice.

Tasks:
- src/data/load_transcripts.py — stream kurry/sp500_earnings_transcripts from HF
- src/data/price_data.py — pull T-1 to T+10 OHLCV per ticker via yfinance
- src/data/preprocessing.py — clean, segment, chunk transcripts

**Exit criteria**: a notebook produces a clean DataFrame of IT-sector transcripts
paired with forward returns, cached to parquet.

---

## Phase 2 — Baselines

**Goal**: both baselines producing schema-compliant output on the dataset slice.

Tasks:
- src/baselines/lexicon.py — Loughran-McDonald word count scorer
- src/baselines/base_model.py — zero-shot structured extraction via un-fine-tuned model

**Exit criteria**: both baselines produce valid JSON schema output on a sample of
100 transcripts; baseline metrics recorded in a notebook.

---

## Phase 3 — Labeling

**Goal**: ~200–400 hand-labeled transcripts for training + held-out test set.

Tasks:
- src/labeling/schema.py — Pydantic schema + annotation guidelines
- src/labeling/annotator.py — annotation tooling + LLM bootstrap + review workflow
- PostgreSQL labeled_samples table migration (Alembic)

**Exit criteria**: labeled dataset exported to parquet; test set isolated (most
recent year, time-based split); inter-annotator spot-check > 80% agreement.

---

## Phase 4 — Fine-Tuning

**Goal**: LoRA adapter trained on labeled data, exported to GGUF for local inference.

Tasks:
- src/finetune/config.py — training hyperparameters
- src/finetune/train.py — SFTTrainer loop (Colab / Kaggle T4)
- src/finetune/export.py — merge + GGUF export + Ollama load

**Exit criteria**: fine-tuned model runs locally via Ollama and produces
schema-compliant JSON on new transcript inputs.

---

## Phase 5 — Extraction Quality Evaluation

**Goal**: compare all three methods on the held-out test set.

Tasks:
- src/eval/extraction_quality.py — accuracy / correlation per field
- Comparison table: lexicon vs base model vs fine-tuned

**Exit criteria**: clear, reproducible comparison table showing whether
fine-tuning improves over baselines (even if the answer is "not much").

---

## Phase 6 — Price Correlation Backtest

**Goal**: correlate extracted signals with forward price returns, segmented honestly.

Tasks:
- src/eval/price_correlation.py — Spearman/Pearson correlation per window
- Results segmented by sector and time period

**Exit criteria**: correlation table with significance tests across all three
methods, presented in the evaluation notebook.

---

## Phase 7 — Agent + API (only after Phases 1-6 work end-to-end)

**Goal**: wrap the pipeline as a stateful LangGraph agent with a FastAPI serving layer.

Tasks:
- src/agent/graph.py — LangGraph state machine
- src/agent/nodes.py — trigger / retrieve / extract / validate / store nodes
- src/agent/validators.py — schema + consistency validation
- src/faiss_store/ — embeddings + FAISS index build/query
- src/api/ — FastAPI app + routes + Pydantic models
- PostgreSQL + Redis setup (Docker Compose)

**Exit criteria**: POST /extract on a new transcript triggers the full
agent pipeline and returns a stored ExtractionResult within 30 seconds.
