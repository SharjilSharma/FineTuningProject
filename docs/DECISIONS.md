# Decision Log

Running record of what was built, what was chosen over what alternative,
and why — updated after every phase. This is the source of truth for
"why did we do it this way" questions, so nothing gets lost to agent
sessions forgetting context or vague memory later.

Format per entry: **What / Why / Alternative considered / Date-ish (phase)**

---

## Phase 0 — Scaffolding

- **Repo structure**: `src/` split into 8 sub-packages (data, baselines,
  labeling, finetune, eval, agent, faiss_store, api) mirroring the
  pipeline stages in the spec. *Why*: keeps each phase's code isolated
  and testable independently. *Alternative*: flat script folder —
  rejected, doesn't scale past Phase 2 without becoming a mess.
- **CLAUDE.md as project instructions**: written to be auto-read by
  Claude Code / agentic tools every session, includes model-routing
  rules and testing discipline. *Why*: avoids re-explaining context
  every session, enforces "no real logic in notebooks" rule.

## Phase 1 — Data Pipeline

- **Dataset**: `kurry/sp500_earnings_transcripts` (Hugging Face).
  *Why*: pre-built, bulk-downloadable, 33k+ transcripts, no scraping
  or per-request API limits. *Alternative*: live API (API Ninjas) —
  reserved for the agent's future "new transcript" trigger only, not
  bulk training data, since it's rate-limited.
- **Scope**: 25 companies, GICS Information Technology sector,
  2015–2024. *Why*: dense, liquid, explicit-guidance sector; small
  enough to keep manual labeling tractable. *Alternative*: full S&P
  500 — deferred to Phase 5 backtest only, for statistical power.
- **AMD excluded** from this slice: too illiquid/volatile in 2015,
  would pollute return calculations. Revisit in later expansion.
- **HPE date filter**: `call_date >= 2016-01-01` (spin-off from HP
  completed Nov 2015, earlier data unreliable).
- **Chunking**: section-level (prepared remarks / Q&A split, ~2-3
  chunks/transcript). *Why*: fits 1-3B model context cleanly, captures
  real tone difference between sections. *Alternative*: speaker-turn
  granularity — rejected, too many tiny rows, slower iteration.
- **BMO/AMC call-timing logic**: static lookup table (Tier 2), with a
  `timing_source` field logged when defaulting. *Why this matters*:
  using the wrong reference day for BMO vs AMC calls bakes the price
  reaction into the "before" baseline, making `fwd_ret_1d` near-zero by
  construction — a real methodology bug this schema was designed to
  avoid.
- **Missing price data**: never dropped, kept as `NaN` and logged
  separately (audit trail), unlike unknown-ticker/HPE-filter rows
  which ARE dropped and logged with reason.

## Phase 2 — Baselines

- **Zero-shot baseline model**: Groq free tier, `llama-3.1-8b-instant`.
  *Why*: free, fast, open-model family (relevant for later comparison
  to the fine-tuned model). Result: 8/8 samples schema-valid, 0 parse
  errors on initial test.
- **Shared eval code**: `src/eval/extraction_quality.py` scores
  lexicon, zero-shot, AND fine-tuned model through identical metric
  functions — no per-method scoring paths. *Why*: this is the
  project's core comparison; divergent scoring would invalidate it.

## Phase 3 — Labeling

- **Circularity problem identified and avoided**: bootstrapping AND
  verifying training labels with the same/similar LLM as the Groq
  baseline would make "fine-tuned beats baseline" untestable by
  construction (you'd just be training it to mimic the baseline).
- **Solution — two-tier labeling**:
  - Training set: fully LLM-bootstrapped via Gemini (`gemini-2.5-flash`,
    later switched mid-run to `gemini-flash-lite-latest` after hitting
    the 20/day limit — flagged as a note, not a re-do, since these are
    bootstrap labels, not ground truth)
  - Eval set (~50-80 chunks): manually reviewed by [you], approve/
    correct/skip via CLI, kept strictly non-overlapping with training
    data, never touched by the bootstrap script
- **Why manual review couldn't be replaced by ChatGPT/Claude
  verification**: still just another LLM's opinion — doesn't solve the
  ground-truth problem, only adds a second correlated model on top of
  the first.
- **Validation script** added to catch typos in categorical fields
  (`guidance_direction`, `tone`) before they silently corrupt Phase 5
  metrics.

## Phase 4 — Fine-Tuning

- **Base model**: `Qwen/Qwen2.5-1.5B-Instruct`. *Why*: strongest
  JSON/structured-output instruction following at this size class
  (per Qwen2.5's training emphasis — not independently benchmarked by
  us, worth caveating if asked), lowest VRAM footprint (~1.5GB at
  4-bit, comfortable on free T4), fast iteration (~10-15 min/run,
  affordable to re-run if something looks wrong), quantizes to a
  ~900MB GGUF that runs in 2-4 sec/call on local CPU for Phase 5.
  *Alternative*: Llama 3.2 3B — solid runner-up, would switch to it
  only if Qwen produces messy JSON in practice or a more widely-
  benchmarked model is needed for writeup credibility.
- *(fill in during/after Phase 4 build)*

## Phase 5 — Evaluation

- *(fill in once run)*

## Phase 6 — Agent Layer

- *(fill in once built)*

---

## Tools whose necessity is genuinely debatable (be honest about these)

- **Redis**: caching for repeated embedding lookups. Useful, not
  load-bearing — the project works without it, just slightly slower
  on repeat runs.
- **Docker (Postgres/Redis)**: convenience/reproducibility, not
  strict necessity — native install would also work.
