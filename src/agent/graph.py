"""
graph.py
--------
LangGraph state machine definition for the end-to-end extraction agent.

Responsibilities (Phase 7 — NOT implemented here)
--------------------------------------------------
Pipeline stages, executed as a stateful graph:
    1. trigger      -- new transcript detected (API Ninjas poll or manual)
    2. retrieve     -- FAISS lookup of similar historical transcripts/labels
    3. extract      -- fine-tuned LLM produces structured JSON
    4. validate     -- schema check + consistency check vs retrieved cases
    5. store        -- write results to PostgreSQL; update backtest stats
    6. serve        -- signal ready to be consumed by FastAPI endpoints

The graph includes a retry edge from validate -> extract on validation failure,
with a maximum retry count to prevent infinite loops.
"""
