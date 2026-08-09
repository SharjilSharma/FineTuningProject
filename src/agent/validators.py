"""
validators.py
-------------
Output validation logic for the extraction agent.

Responsibilities (Phase 7 — NOT implemented here)
--------------------------------------------------
- Validate raw model output is parseable JSON.
- Validate all required schema fields are present and values are in-enum.
- Check hedging_score is in [0, 1].
- Cross-check extracted tone against retrieved similar cases; flag large
  deviations (signal for agent to re-query or escalate).
- Return a structured ValidationResult with pass/fail and error details.
"""
