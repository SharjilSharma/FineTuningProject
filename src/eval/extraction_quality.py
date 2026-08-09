"""
extraction_quality.py
---------------------
Shared scoring functions used by ALL extraction methods:
    - Lexicon baseline
    - Base model (zero-shot Groq)
    - Fine-tuned model (Phase 4)

There is exactly ONE scoring path.  Do not add per-method branches here.

Metrics
-------
* guidance_direction  : classification accuracy + macro-F1
* tone                : classification accuracy + macro-F1
* hedging_score       : Spearman rank correlation + Pearson r
* key_flags           : token-level F1 (optional; requires ground truth)

Output
------
score_predictions() returns an EvalResult dataclass.
compare_methods()   returns a formatted comparison DataFrame.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

# ── Schema constants ──────────────────────────────────────────────────────

GUIDANCE_LABELS = ["raised", "lowered", "maintained", "not_discussed"]
TONE_LABELS     = ["confident", "cautious", "evasive", "neutral"]

SCHEMA_FIELDS = {
    "guidance_direction": GUIDANCE_LABELS,
    "tone":               TONE_LABELS,
    "hedging_score":      None,   # continuous [0, 1]
    "key_flags":          None,   # list[str]
}

# ── Result dataclass ──────────────────────────────────────────────────────

@dataclass
class EvalResult:
    method_name: str

    # Classification metrics
    guidance_accuracy: float = float("nan")
    guidance_macro_f1: float = float("nan")
    tone_accuracy:     float = float("nan")
    tone_macro_f1:     float = float("nan")

    # Regression metrics
    hedging_spearman:  float = float("nan")
    hedging_pearson:   float = float("nan")
    hedging_spearman_p: float = float("nan")

    # Optional
    key_flags_token_f1: float = float("nan")

    # Sample counts
    n_samples: int = 0
    n_parse_errors: int = 0

    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "method":             self.method_name,
            "n_samples":          self.n_samples,
            "n_parse_errors":     self.n_parse_errors,
            "guidance_accuracy":  round(self.guidance_accuracy, 4),
            "guidance_macro_f1":  round(self.guidance_macro_f1, 4),
            "tone_accuracy":      round(self.tone_accuracy, 4),
            "tone_macro_f1":      round(self.tone_macro_f1, 4),
            "hedging_spearman":   round(self.hedging_spearman, 4),
            "hedging_pearson":    round(self.hedging_pearson, 4),
            "hedging_spearman_p": round(self.hedging_spearman_p, 4),
            "key_flags_f1":       round(self.key_flags_token_f1, 4),
        }

# ── Classification helpers ────────────────────────────────────────────────

def _accuracy(y_true: list, y_pred: list) -> float:
    if not y_true:
        return float("nan")
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)

def _macro_f1(y_true: list, y_pred: list, labels: list[str]) -> float:
    """Macro-averaged F1 across *labels* — no sklearn dependency."""
    f1s = []
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
        fp = sum(t != label and p == label for t, p in zip(y_true, y_pred))
        fn = sum(t == label and p != label for t, p in zip(y_true, y_pred))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        denom     = precision + recall
        f1s.append(2 * precision * recall / denom if denom > 0 else 0.0)
    return float(np.mean(f1s))

# ── Key-flags token F1 ────────────────────────────────────────────────────

def _flags_token_f1(pred_flags: list[list[str]], true_flags: list[list[str]]) -> float:
    """
    Token-level F1 over key_flags lists.
    Tokenises on whitespace, computes per-sample F1, macro-averages.
    """
    sample_f1s = []
    for preds, truths in zip(pred_flags, true_flags):
        pred_toks = set(" ".join(preds).lower().split())
        true_toks = set(" ".join(truths).lower().split())
        if not true_toks and not pred_toks:
            sample_f1s.append(1.0)
            continue
        if not true_toks or not pred_toks:
            sample_f1s.append(0.0)
            continue
        tp = len(pred_toks & true_toks)
        precision = tp / len(pred_toks)
        recall    = tp / len(true_toks)
        denom = precision + recall
        sample_f1s.append(2 * precision * recall / denom if denom > 0 else 0.0)
    return float(np.mean(sample_f1s)) if sample_f1s else float("nan")

# ── Main scoring function ─────────────────────────────────────────────────

def score_predictions(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame | None = None,
    method_name: str = "unnamed",
) -> EvalResult:
    """
    Compute extraction quality metrics for one method's predictions.

    Parameters
    ----------
    predictions  : DataFrame with columns guidance_direction, tone,
                   hedging_score, key_flags (and optionally _parse_error).
    ground_truth : DataFrame with the same schema, aligned by index.
                   If None, only schema-validity stats are computed.
    method_name  : Label for this method in comparison tables.

    Returns
    -------
    EvalResult
    """
    result = EvalResult(method_name=method_name)
    result.n_samples = len(predictions)
    result.n_parse_errors = int(
        predictions.get("_parse_error", pd.Series(False)).sum()
    )

    # Validate schema presence
    for col in ("guidance_direction", "tone", "hedging_score", "key_flags"):
        if col not in predictions.columns:
            logger.warning(f"score_predictions: column '{col}' missing from predictions.")

    if ground_truth is None:
        logger.info(
            f"[{method_name}] No ground truth provided — "
            "schema-validity stats only."
        )
        return result

    if len(predictions) != len(ground_truth):
        raise ValueError(
            f"predictions ({len(predictions)}) and ground_truth "
            f"({len(ground_truth)}) must have the same length."
        )

    # ── Guidance direction ─────────────────────────────────────────────
    if "guidance_direction" in predictions.columns and "guidance_direction" in ground_truth.columns:
        y_pred_g = predictions["guidance_direction"].tolist()
        y_true_g = ground_truth["guidance_direction"].tolist()
        result.guidance_accuracy = _accuracy(y_true_g, y_pred_g)
        result.guidance_macro_f1 = _macro_f1(y_true_g, y_pred_g, GUIDANCE_LABELS)

    # ── Tone ──────────────────────────────────────────────────────────
    if "tone" in predictions.columns and "tone" in ground_truth.columns:
        y_pred_t = predictions["tone"].tolist()
        y_true_t = ground_truth["tone"].tolist()
        result.tone_accuracy = _accuracy(y_true_t, y_pred_t)
        result.tone_macro_f1 = _macro_f1(y_true_t, y_pred_t, TONE_LABELS)

    # ── Hedging score ─────────────────────────────────────────────────
    if "hedging_score" in predictions.columns and "hedging_score" in ground_truth.columns:
        hs_pred = pd.to_numeric(predictions["hedging_score"], errors="coerce").dropna()
        hs_true = pd.to_numeric(ground_truth["hedging_score"], errors="coerce")
        hs_true = hs_true.loc[hs_pred.index]

        if len(hs_pred) >= 2:
            sp = stats.spearmanr(hs_pred, hs_true)
            pe = stats.pearsonr(hs_pred, hs_true)
            result.hedging_spearman   = float(sp.statistic)
            result.hedging_spearman_p = float(sp.pvalue)
            result.hedging_pearson    = float(pe[0])

    # ── Key flags (optional) ──────────────────────────────────────────
    if "key_flags" in predictions.columns and "key_flags" in ground_truth.columns:
        pred_flags = [
            v if isinstance(v, list) else []
            for v in predictions["key_flags"]
        ]
        true_flags = [
            v if isinstance(v, list) else []
            for v in ground_truth["key_flags"]
        ]
        result.key_flags_token_f1 = _flags_token_f1(pred_flags, true_flags)

    logger.info(
        f"[{method_name}] guidance acc={result.guidance_accuracy:.3f}, "
        f"tone acc={result.tone_accuracy:.3f}, "
        f"hedging ρ={result.hedging_spearman:.3f}"
    )
    return result

# ── Comparison table ──────────────────────────────────────────────────────

def compare_methods(results: list[EvalResult]) -> pd.DataFrame:
    """
    Given a list of EvalResult objects (one per method), return a
    formatted comparison DataFrame sorted by guidance_accuracy descending.
    """
    rows = [r.to_dict() for r in results]
    df = pd.DataFrame(rows).set_index("method")
    df = df.sort_values("guidance_accuracy", ascending=False)
    return df

# ── Schema validation (standalone utility) ────────────────────────────────

def validate_schema(output: dict) -> tuple[bool, list[str]]:
    """
    Check that a single extraction output dict matches the labeling schema.

    Returns (is_valid, list_of_errors).
    """
    errors: list[str] = []

    gd = output.get("guidance_direction")
    if gd not in GUIDANCE_LABELS:
        errors.append(f"guidance_direction='{gd}' not in {GUIDANCE_LABELS}")

    tone = output.get("tone")
    if tone not in TONE_LABELS:
        errors.append(f"tone='{tone}' not in {TONE_LABELS}")

    hs = output.get("hedging_score")
    try:
        hs_f = float(hs)
        if not (0.0 <= hs_f <= 1.0):
            errors.append(f"hedging_score={hs_f} out of range [0, 1]")
    except (TypeError, ValueError):
        errors.append(f"hedging_score='{hs}' is not a float")

    flags = output.get("key_flags")
    if not isinstance(flags, list):
        errors.append(f"key_flags must be a list, got {type(flags).__name__}")

    return (len(errors) == 0), errors
