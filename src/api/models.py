"""
models.py
---------
Pydantic request / response models for the FastAPI layer.

Responsibilities (Phase 7 — NOT implemented here)
--------------------------------------------------
- ExtractionRequest  : ticker, call_date, raw_transcript (or transcript_id)
- ExtractionResult   : guidance_direction, tone, hedging_score, key_flags,
                       model_version, extracted_at, confidence
- BacktestSummary    : per-ticker / per-method correlation stats
- HealthResponse     : service status, model loaded flag, FAISS index size
"""
