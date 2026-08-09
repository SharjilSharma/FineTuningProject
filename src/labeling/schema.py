"""
schema.py
---------
Canonical labeling schema for the extraction task.

Each transcript (or segment) is labeled with a structured JSON output:

{
    "guidance_direction": "raised | lowered | maintained | not_discussed",
    "tone": "confident | cautious | evasive | neutral",
    "hedging_score": 0.0,   # float in [0, 1] — density of hedging language
    "key_flags": []         # short list of notable phrases / topics
}

Responsibilities (Phase 3 — NOT implemented here)
--------------------------------------------------
- Define Pydantic models for the schema above (validation, serialisation).
- Provide a JSON-schema export for use in prompt engineering and output parsing.
- Include field-level documentation used in annotation guidelines.
"""
