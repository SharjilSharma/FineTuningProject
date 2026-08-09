"""
price_correlation.py
--------------------
Correlate extracted signals with subsequent forward price returns.

Responsibilities (Phase 6 — NOT implemented here)
--------------------------------------------------
- For each transcript in the test set:
    - Load forward returns at T+1, T+5, T+10 from price_data module.
    - Pair with extracted tone / hedging_score from each method.
- Compute Pearson / Spearman correlation per window per method.
- Report p-values alongside correlation coefficients.
- Segment by GICS sector and time period to check robustness (no cherry-picking).
- This is a SECONDARY, exploratory metric — do not over-claim.
"""
