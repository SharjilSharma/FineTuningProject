"""
Phase 2 unit tests — all offline, no Groq API calls.

Coverage
--------
lexicon          : scoring on synthetic text, tone thresholds,
                   guidance direction keyword detection, schema validity
base_model       : JSON parsing, validation, schema coercion, parse errors
extraction_quality: accuracy, macro-F1, Spearman correlation, schema validation
"""

from __future__ import annotations

import json
import unittest.mock as mock

import pandas as pd
import pytest

from src.baselines.lexicon import score_chunk as lm_score, score_dataframe as lm_score_df
from src.baselines.base_model import _parse_response, _default_output, score_dataframe as bm_score_df
from src.eval.extraction_quality import (
    EvalResult,
    validate_schema,
    score_predictions,
    compare_methods,
    _accuracy,
    _macro_f1,
    _flags_token_f1,
    GUIDANCE_LABELS,
    TONE_LABELS,
)

# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _make_schema_dict(
    gd="not_discussed", tone="neutral", hs=0.3, flags=None
) -> dict:
    return {
        "guidance_direction": gd,
        "tone": tone,
        "hedging_score": hs,
        "key_flags": flags or [],
    }

def _make_pred_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)

# ═══════════════════════════════════════════════════════════════════════════
# Lexicon baseline — unit tests (offline, uses mocked LM word sets)
# ═══════════════════════════════════════════════════════════════════════════

_LM_POS_MOCK = frozenset({"STRONG", "GREW", "BEAT", "EXCEEDED", "RECORD"})
_LM_NEG_MOCK = frozenset({"DECLINE", "MISS", "DISAPPOINTED", "RISK", "LOSS"})
_LM_UNC_MOCK = frozenset({"MAY", "MIGHT", "UNCERTAIN", "POSSIBLE", "APPROXIMATELY"})

@pytest.fixture(autouse=True)
def patch_lm_sets(monkeypatch):
    """Prevent any network call to download the LM dictionary."""
    monkeypatch.setattr(
        "src.baselines.lexicon._LM_POS", _LM_POS_MOCK
    )
    monkeypatch.setattr(
        "src.baselines.lexicon._LM_NEG", _LM_NEG_MOCK
    )
    monkeypatch.setattr(
        "src.baselines.lexicon._LM_UNC", _LM_UNC_MOCK
    )

class TestLexiconScoring:
    def test_schema_fields_present(self):
        result = lm_score("Revenue strong grew beat record quarter.")
        assert set(result.keys()) >= {
            "guidance_direction", "tone", "hedging_score", "key_flags"
        }

    def test_confident_tone_on_positive_text(self):
        text = "Revenue strong grew beat exceeded record results this quarter."
        result = lm_score(text)
        assert result["tone"] == "confident"

    def test_cautious_tone_on_negative_text(self):
        # Many negative words, no positive
        text = "We saw decline miss disappointed risk loss across all segments."
        result = lm_score(text)
        assert result["tone"] == "cautious"

    def test_neutral_tone_on_balanced_text(self):
        # Equal positive and negative -> net = 0 -> neutral
        text = "Strong grew versus decline miss in different areas."
        result = lm_score(text)
        assert result["tone"] == "neutral"

    def test_hedging_score_is_float_in_range(self):
        result = lm_score("Revenue may approximately possibly be uncertain.")
        assert isinstance(result["hedging_score"], float)
        assert 0.0 <= result["hedging_score"] <= 1.0

    def test_hedging_score_higher_with_more_uncertainty_words(self):
        low_hedge  = lm_score("Revenue strong grew beat record quarter.")
        high_hedge = lm_score("May might uncertain possible approximately revenue.")
        assert high_hedge["hedging_score"] > low_hedge["hedging_score"]

    def test_guidance_raised_on_raise_keyword(self):
        text = "We are raising our full-year revenue forecast guidance."
        result = lm_score(text)
        assert result["guidance_direction"] == "raised"

    def test_guidance_lowered_on_lower_keyword(self):
        text = "We are lowering our guidance forecast for next quarter."
        result = lm_score(text)
        assert result["guidance_direction"] == "lowered"

    def test_guidance_maintained_on_reaffirm_keyword(self):
        text = "We reaffirm our full-year outlook guidance and forecast."
        result = lm_score(text)
        assert result["guidance_direction"] == "maintained"

    def test_guidance_not_discussed_without_forward_keyword(self):
        text = "Revenue strong grew beat record this quarter."
        result = lm_score(text)
        assert result["guidance_direction"] == "not_discussed"

    def test_key_flags_is_list(self):
        result = lm_score("Revenue decline miss disappointed risk loss.")
        assert isinstance(result["key_flags"], list)

    def test_key_flags_max_count(self):
        # Lots of negative words
        text = " ".join([
            "decline miss disappointed risk loss decline miss disappointed risk loss"
        ] * 5)
        result = lm_score(text)
        assert len(result["key_flags"]) <= 10  # MAX_FLAGS

    def test_score_dataframe_adds_columns(self):
        df = pd.DataFrame({
            "chunk_id": ["c1", "c2"],
            "chunk_text": [
                "Strong grew beat guidance forecast outlook raised.",
                "Decline miss disappointed loss risk.",
            ]
        })
        out = lm_score_df(df)
        for col in ("guidance_direction", "tone", "hedging_score", "key_flags"):
            assert col in out.columns

    def test_score_dataframe_raises_without_chunk_text(self):
        df = pd.DataFrame({"chunk_id": ["c1"]})
        with pytest.raises(ValueError, match="chunk_text"):
            lm_score_df(df)

# ═══════════════════════════════════════════════════════════════════════════
# Base model — unit tests (offline, mocked Groq client)
# ═══════════════════════════════════════════════════════════════════════════

class TestBaseModelParsing:
    def test_valid_json_parsed_correctly(self):
        raw = json.dumps({
            "guidance_direction": "raised",
            "tone": "confident",
            "hedging_score": 0.15,
            "key_flags": ["revenue beat", "margin expansion"],
        })
        result = _parse_response(raw, "test_chunk")
        assert result["guidance_direction"] == "raised"
        assert result["tone"] == "confident"
        assert result["hedging_score"] == pytest.approx(0.15)
        assert "revenue beat" in result["key_flags"]
        assert result["_parse_error"] is False

    def test_invalid_guidance_coerced_to_default(self):
        raw = json.dumps({
            "guidance_direction": "UNKNOWN_VALUE",
            "tone": "neutral",
            "hedging_score": 0.3,
            "key_flags": [],
        })
        result = _parse_response(raw, "test_chunk")
        assert result["guidance_direction"] == "not_discussed"
        assert result["_parse_error"] is False

    def test_invalid_tone_coerced_to_neutral(self):
        raw = json.dumps({
            "guidance_direction": "not_discussed",
            "tone": "bearish",   # not in VALID_TONE
            "hedging_score": 0.4,
            "key_flags": [],
        })
        result = _parse_response(raw, "test_chunk")
        assert result["tone"] == "neutral"

    def test_hedging_score_clamped_to_range(self):
        raw = json.dumps({
            "guidance_direction": "not_discussed",
            "tone": "neutral",
            "hedging_score": 1.99,   # over 1.0
            "key_flags": [],
        })
        result = _parse_response(raw, "test_chunk")
        assert result["hedging_score"] <= 1.0

    def test_key_flags_capped_at_5(self):
        raw = json.dumps({
            "guidance_direction": "not_discussed",
            "tone": "neutral",
            "hedging_score": 0.5,
            "key_flags": [f"flag {i}" for i in range(10)],
        })
        result = _parse_response(raw, "test_chunk")
        assert len(result["key_flags"]) <= 5

    def test_empty_string_returns_parse_error(self):
        result = _parse_response("", "test_chunk")
        assert result["_parse_error"] is True
        assert result["guidance_direction"] == "not_discussed"

    def test_garbage_returns_parse_error(self):
        result = _parse_response("this is not json at all!!!", "test_chunk")
        assert result["_parse_error"] is True

    def test_json_embedded_in_markdown_extracted(self):
        raw = (
            "Sure! Here is the analysis:\n```json\n"
            + json.dumps({
                "guidance_direction": "lowered",
                "tone": "cautious",
                "hedging_score": 0.7,
                "key_flags": ["margin pressure"],
            })
            + "\n```"
        )
        result = _parse_response(raw, "test_chunk")
        # Should extract the embedded JSON
        assert result["_parse_error"] is False
        assert result["guidance_direction"] == "lowered"

class TestBaseModelScoreDataframe:
    def test_score_dataframe_with_mock_api(self, monkeypatch):
        """End-to-end test of score_dataframe with mocked _call_api."""
        mock_response = json.dumps({
            "guidance_direction": "raised",
            "tone": "confident",
            "hedging_score": 0.1,
            "key_flags": ["strong demand"],
        })
        monkeypatch.setattr(
            "src.baselines.base_model._call_api",
            lambda text: mock_response,
        )

        df = pd.DataFrame({
            "chunk_id":   ["NVDA_20231026__prep__p001", "NVDA_20231026__qa__q001__p001"],
            "chunk_text": ["Revenue guidance raised significantly.", "Demand remains very strong."],
        })
        out = bm_score_df(df, max_chunks=2, request_interval=0.0)

        assert len(out) == 2
        assert "guidance_direction" in out.columns
        assert "tone" in out.columns
        assert "hedging_score" in out.columns
        assert "key_flags" in out.columns
        assert out["guidance_direction"].iloc[0] == "raised"

    def test_score_dataframe_raises_without_chunk_text(self):
        df = pd.DataFrame({"chunk_id": ["c1"]})
        with pytest.raises(ValueError, match="chunk_text"):
            bm_score_df(df)

    def test_max_chunks_limits_rows(self, monkeypatch):
        monkeypatch.setattr(
            "src.baselines.base_model._call_api",
            lambda text: json.dumps(_make_schema_dict()),
        )
        df = pd.DataFrame({
            "chunk_id":   [f"c{i}" for i in range(10)],
            "chunk_text": ["Some text."] * 10,
        })
        out = bm_score_df(df, max_chunks=3, request_interval=0.0)
        assert len(out) == 3

# ═══════════════════════════════════════════════════════════════════════════
# Extraction quality metrics
# ═══════════════════════════════════════════════════════════════════════════

class TestExtractionQuality:
    def test_accuracy_perfect(self):
        assert _accuracy(["a", "b", "c"], ["a", "b", "c"]) == pytest.approx(1.0)

    def test_accuracy_zero(self):
        assert _accuracy(["a", "b"], ["b", "a"]) == pytest.approx(0.0)

    def test_accuracy_empty(self):
        import math
        assert math.isnan(_accuracy([], []))

    def test_macro_f1_perfect(self):
        labels = ["raised", "lowered", "maintained", "not_discussed"]
        y = ["raised", "lowered", "maintained", "not_discussed"]
        assert _macro_f1(y, y, labels) == pytest.approx(1.0)

    def test_flags_token_f1_perfect(self):
        pred = [["margin pressure", "strong demand"]]
        true = [["margin pressure", "strong demand"]]
        assert _flags_token_f1(pred, true) == pytest.approx(1.0)

    def test_flags_token_f1_no_overlap(self):
        pred = [["margin pressure"]]
        true = [["revenue beat"]]
        assert _flags_token_f1(pred, true) == pytest.approx(0.0)

    def test_validate_schema_valid(self):
        ok, errors = validate_schema(_make_schema_dict("raised", "confident", 0.2))
        assert ok
        assert errors == []

    def test_validate_schema_bad_guidance(self):
        ok, errors = validate_schema(_make_schema_dict(gd="bullish"))
        assert not ok
        assert any("guidance_direction" in e for e in errors)

    def test_validate_schema_bad_tone(self):
        ok, errors = validate_schema(_make_schema_dict(tone="bearish"))
        assert not ok
        assert any("tone" in e for e in errors)

    def test_validate_schema_hedging_out_of_range(self):
        ok, errors = validate_schema(_make_schema_dict(hs=1.5))
        assert not ok
        assert any("hedging_score" in e for e in errors)

    def test_validate_schema_flags_not_list(self):
        d = _make_schema_dict()
        d["key_flags"] = "not a list"
        ok, errors = validate_schema(d)
        assert not ok

    def test_score_predictions_no_ground_truth(self):
        preds = _make_pred_df([_make_schema_dict("raised", "confident", 0.2)])
        result = score_predictions(preds, method_name="test_method")
        assert result.n_samples == 1
        import math
        assert math.isnan(result.guidance_accuracy)

    def test_score_predictions_with_ground_truth(self):
        preds  = _make_pred_df([
            _make_schema_dict("raised",        "confident", 0.1),
            _make_schema_dict("not_discussed", "neutral",   0.5),
            _make_schema_dict("lowered",       "cautious",  0.8),
        ])
        truths = _make_pred_df([
            _make_schema_dict("raised",        "confident", 0.2),
            _make_schema_dict("not_discussed", "cautious",  0.4),
            _make_schema_dict("maintained",    "cautious",  0.9),
        ])
        result = score_predictions(preds, truths, method_name="test")
        # guidance: 2/3 correct (raised✓, not_discussed✓, lowered✗)
        assert result.guidance_accuracy == pytest.approx(2 / 3)
        assert result.n_samples == 3

    def test_compare_methods_returns_dataframe(self):
        r1 = EvalResult("lexicon",    guidance_accuracy=0.6, tone_accuracy=0.5)
        r2 = EvalResult("base_model", guidance_accuracy=0.75, tone_accuracy=0.65)
        df = compare_methods([r1, r2])
        assert isinstance(df, pd.DataFrame)
        assert df.index[0] == "base_model"   # sorted desc by guidance_accuracy
