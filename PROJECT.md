# Earnings Call Signal Extraction Agent

Technical specification.

## 1. Problem Statement

Test whether a fine-tuned small LLM can extract structured, nuanced signals
(management tone, guidance direction, hedging language) from earnings call
transcripts more reliably than a generic base model or a naive lexicon
baseline — and whether that extracted signal shows any measurable
correlation with subsequent stock price movement, evaluated honestly against
baselines.

This is a methodology/research project, not a trading system.

## 2. Datasets

### Primary training corpus
- **Source**: `kurry/sp500_earnings_transcripts` (Hugging Face, MIT license)
- **Coverage**: 33,000+ transcripts, 685 S&P 500 / large-cap companies,
  2005–2025
- **Structure**: speaker-segmented dialogue (each row/segment tagged with
  speaker name/role and text), full verbatim transcripts, metadata
  (ticker, company name, date, unique ID)
- **Sectors**: all 11 GICS sectors represented

### Secondary dataset (financial metrics paired with transcripts)
- **Source**: `glopardo/sp500-earnings-transcripts` (Hugging Face)
- **Coverage**: 2014–2024, S&P 500 companies
- **Adds**: quarterly EPS, P/E, and company fundamentals alongside
  transcript text — useful for correlating extracted signal against
  actual reported numbers, not just price
- Built for ECB working paper "Verba Volant, Transcripta Manent" (2025) —
  same general research direction as this project, good reference paper.

### Price data
- **Source**: `yfinance` (free, no API key)
- Daily OHLCV data per ticker, pulled for the window around each
  transcript's call date (e.g. T-1 to T+10 trading days)

### Live/incremental data (for the agent's trigger step only, not initial training)
- **Source**: API Ninjas Earnings Call Transcript API (free tier, rate-limited)
- Covers 8,000+ companies; used later to pull the newest transcript on
  release, not for bulk historical training data

## 3. Scope (initial build)

- Start with a single GICS sector (e.g. Information Technology) and
  ~20–30 companies to keep labeling effort tractable
- Use 2015–2024 data (denser transcript coverage than pre-2010)
- Expand sector/company coverage only after the full pipeline works
  end-to-end on the small slice

## 4. Labeling Schema

Each transcript (or transcript segment) gets labeled with a structured
output the fine-tuned model will learn to reproduce:

```json
{
  "guidance_direction": "raised | lowered | maintained | not_discussed",
  "tone": "confident | cautious | evasive | neutral",
  "hedging_score": 0-1,
  "key_flags": ["short list of notable phrases/topics, e.g. 'delayed product launch', 'margin pressure'"]
}
```

- Initial labeled set: hand-label a few hundred transcripts personally
  for ground truth
- Optionally bootstrap additional labels with a general-purpose LLM API
  call, then manually review/correct a sample to check label quality
  before trusting them in bulk

## 5. Baselines (must be built before the fine-tuned model, for comparison)

1. **Lexicon baseline**: Loughran-McDonald financial sentiment word
   lists — simple positive/negative word counting, no ML
2. **Base model baseline**: same labeling schema, same prompt, run
   through the un-fine-tuned base model (zero-shot or few-shot prompted)

Both baselines are evaluated with the exact same metrics as the
fine-tuned model — this comparison is the core result of the project.

## 6. Fine-Tuning

- **Base model**: small open-source model, 1–3B params (e.g. Llama 3.2
  3B, Qwen 2.5 1.5B)
- **Method**: LoRA / QLoRA via Hugging Face `peft` + `trl`
- **Compute**: free Colab/Kaggle T4 GPU tier (no local GPU required)
- **Training data**: labeled transcript segments → structured JSON
  output pairs (from section 4)
- **Output**: LoRA adapter weights, merged and quantized (GGUF, 4-bit)
  for local CPU inference via llama.cpp / Ollama

## 7. Evaluation

### Extraction quality (primary metric)
- Agreement between model output and held-out human-labeled test set
  (accuracy per field: guidance_direction, tone; correlation for
  hedging_score)
- Compared across: lexicon baseline vs base model vs fine-tuned model

### Price correlation (secondary, exploratory)
- For each transcript, compute forward price return over multiple
  windows (T+1 day, T+5 days, T+10 days)
- Correlate extracted tone/hedging score against forward returns
- Report correlation coefficient with significance, compared across
  all three methods (lexicon / base / fine-tuned)
- Segment results by sector and time period to check robustness
  (not just one lucky subset)

### Held-out test set
- Time-based split (train on earlier years, test on most recent
  year in scope) to avoid lookahead/leakage, not a random split

## 8. Agent Architecture (build after the core pipeline works)

Pipeline stages, orchestrated as a stateful agent rather than a linear
script:

1. **Trigger** — new transcript becomes available (via API Ninjas
   webhook/poll in production; manual trigger during development)
2. **Retrieve similar past cases** — FAISS vector store of embedded
   historical transcripts/labels, used as grounding context
3. **Extract signal** — fine-tuned LLM produces structured JSON output
4. **Validate** — agent checks output against schema (valid JSON,
   confidence threshold, consistency with retrieved similar cases);
   re-queries the model if validation fails
5. **Store + backtest** — results and running backtest stats written
   to PostgreSQL
6. **Serve** — FastAPI layer exposes results/dashboard endpoints

## 9. Tech Stack

| Component | Tool | Purpose |
|---|---|---|
| Fine-tuning | Hugging Face `peft`, `trl`, LoRA/QLoRA | Efficient fine-tuning on free GPU tier |
| Local inference | llama.cpp / Ollama, GGUF quantization | Run fine-tuned model on CPU/8GB RAM |
| Vector memory | FAISS | Retrieve similar historical transcripts |
| Orchestration | LangGraph | Multi-step agent state machine, retries |
| Storage | PostgreSQL | Labeled data, results, backtest history |
| Caching | Redis | Avoid recomputing embeddings/results |
| API layer | FastAPI | Serve results, trigger analysis |
| Price data | yfinance | Historical/forward price data |
| Baseline sentiment | Loughran-McDonald lexicon | Naive baseline for comparison |

## 10. Build Order

1. Data pipeline: load HF dataset, pull matching price data via yfinance
2. Baselines: lexicon scoring + base model zero-shot extraction
3. Labeling: hand-label initial few hundred examples
4. Fine-tune small model via LoRA on Colab/Kaggle
5. Evaluate extraction quality vs baselines
6. Backtest price correlation vs baselines, segmented by sector/time
7. Only after 1–6 work end-to-end: add FAISS memory, LangGraph
   orchestration, validation loop, Postgres/Redis/FastAPI layer
