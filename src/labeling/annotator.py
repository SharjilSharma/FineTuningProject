"""
annotator.py
------------
Manual and LLM-assisted annotation tooling.

Responsibilities (Phase 3 — NOT implemented here)
--------------------------------------------------
- CLI / notebook helpers to present a transcript chunk and collect a
  human label in the structured schema format.
- LLM-assisted bootstrapping: call a general-purpose LLM API (GPT-4 or
  Claude) to produce draft labels for bulk transcripts.
- Write labeled examples to PostgreSQL (labeled_samples table).
- Track annotator ID and timestamp for auditability.
- Do NOT trust bulk LLM labels blindly — sample-and-review workflow built in.
"""
