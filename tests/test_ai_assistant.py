"""Tests for the AI assistant. No network is used: Gemini calls are faked."""
import json

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from modules import ai_assistant
from modules.ai_assistant import (DEFAULT_MODELS, AIError, _call_gemini, _parse_json_answer,
                                  answer_offline, answer_question, build_privacy_safe_summary,
                                  chart_context, detect_sensitive_columns, generate_insights,
                                  get_api_key, get_model_candidates, offline_chart_explanation,
                                  offline_insights, summary_to_text)
from modules.dashboard import recommend_charts


def make_df() -> pd.DataFrame:
    """120 rows with 4 sensitive columns, a rare category (3 rows) and a clear top region."""
    rng = np.random.default_rng(3)
    n = 120
    region = np.where(np.arange(n) % 3 == 0, "North",
                      np.where(np.arange(n) % 3 == 1, "South", "East"))
    sales = rng.uniform(10, 100, n).round(2)
    sales[region == "North"] += 100                      # North is clearly the biggest
    tier = np.array(["Gold"] * 100 + ["Silver"] * 17 + ["Rare"] * 3)
    return pd.DataFrame({
        "order_id": [f"ORD-{i}" for i in range(n)],
        "customer_email": [f"secret{i}@example.com" for i in range(n)],
        "customer_name": [f"Person Number {i}" for i in range(n)],
        "phone": [f"+91 98765 {10000 + i}" for i in range(n)],
        "date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "region": region,
        "tier": tier,
        "sales": sales,
        "unit_price": rng.uniform(5, 20, n).round(2),
        "age": rng.integers(18, 70, n),
    })


def fake_gemini(reply):
    """A stand-in for _call_gemini that records the prompt and returns a fixed reply."""
    captured = {}

    def fake(prompt, api_key, system_instruction, json_mode=False, models=None):
        captured.update(prompt=prompt, json_mode=json_mode)
        return reply, "fake-model"
    return fake, captured


# ---- privacy ---------------------------------------------------------------
def test_detect_sensitive_columns():
    found = detect_sensitive_columns(make_df())
    assert set(found) == {"order_id", "customer_email", "customer_name", "phone"}
    assert all(isinstance(reason, str) and reason for reason in found.values())


def test_summary_never_contains_raw_values():
    text = summary_to_text(build_privacy_safe_summary(make_df()))
    for marker in ("secret7@example.com", "Person Number 7", "ORD-7", "98765",
                   "customer_email", "customer_name", "order_id", "phone"):
        assert marker not in text


def test_summary_hides_small_groups():
    summary = build_privacy_safe_summary(make_df())
    assert "Rare" not in summary_to_text(summary)
    tier = summary["categorical_columns"]["tier"]
    assert tier["small_groups_hidden"] == 1
    assert [v["value"] for v in tier["top_values"]] == ["Gold", "Silver"]


def test_summary_structure_and_json():
    summary = build_privacy_safe_summary(make_df())
    text = summary_to_text(summary)
    assert json.loads(text) == summary                    # pure JSON types
    assert "NaN" not in text
    assert summary["dataset"]["rows"] == 120
    assert summary["privacy"]["excluded_columns_count"] == 4
    assert summary["privacy"]["raw_rows_included"] is False
    assert set(summary["numeric_columns"]) == {"sales", "unit_price", "age"}
    assert summary["date_columns"]["date"]["start"] == "2024-01-01"
    assert summary["trend"]["measure"] == "sales" and summary["trend"]["periods"] == 4
    assert summary["group_comparison"]["dimension"] == "region"
    assert summary["group_comparison"]["groups"][0]["group"] == "North"


# ---- offline mode ----------------------------------------------------------
def test_offline_insights_shape():
    result = offline_insights(build_privacy_safe_summary(make_df()))
    assert result["source"] == "offline"
    assert 5 <= len(result["insights"]) <= 10
    assert len(result["recommendations"]) >= 3
    assert all(isinstance(t, str) and t
               for t in result["insights"] + result["unusual"] + result["recommendations"])
    assert any("North" in t for t in result["insights"])


def test_offline_flags_unusual_values():
    df = make_df()
    df.loc[0, "age"] = 999                                # an impossible age
    result = offline_insights(build_privacy_safe_summary(df))
    assert any("'age'" in t for t in result["unusual"])


def test_answer_offline_basic():
    df = make_df()
    assert "120" in answer_offline("How many rows are there?", df)
    answer = answer_offline("What is the average of sales?", df)
    assert "sales" in answer.lower() and "average" in answer.lower()
    with pytest.raises(AIError):
        answer_offline("   ", df)


def test_answer_offline_never_uses_excluded_columns():
    df = make_df()
    answer = answer_offline("Tell me about customer_email and phone", df,
                            excluded=list(detect_sensitive_columns(df)))
    assert "secret" not in answer and "98765" not in answer


# ---- chart explanations ----------------------------------------------------
def test_chart_context_and_explanations():
    df = make_df()
    excluded = list(detect_sensitive_columns(df))
    specs = recommend_charts(df)
    assert {s["kind"] for s in specs} >= {"line", "bar", "pie", "histogram", "scatter",
                                          "heatmap"}
    for spec in specs:
        ctx = chart_context(df, spec, excluded)
        text = offline_chart_explanation(ctx)
        assert isinstance(text, str) and len(text) > 40
        assert json.dumps(ctx)                            # JSON-safe


def test_chart_context_blocks_excluded_columns():
    df = make_df()
    with pytest.raises(AIError):
        chart_context(df, {"kind": "bar", "category": "customer_email", "agg": "count"},
                      ["customer_email"])
    with pytest.raises(AIError):
        chart_context(df, {"kind": "bar"}, [])            # 'category' is missing


# ---- Gemini plumbing (no network) -----------------------------------------
def test_parse_json_answer():
    good = '{"insights": ["a", "b"], "unusual": [], "recommendations": ["r"]}'
    assert _parse_json_answer(good)["insights"] == ["a", "b"]
    assert _parse_json_answer("```json\n" + good + "\n```")["recommendations"] == ["r"]
    many = json.dumps({"insights": [f"i{n}" for n in range(15)]})
    assert len(_parse_json_answer(many)["insights"]) == 10     # capped at 10
    with pytest.raises(AIError):
        _parse_json_answer("not json at all")
    with pytest.raises(AIError):
        _parse_json_answer('{"insights": []}')


def test_model_candidates_and_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    assert get_model_candidates() == DEFAULT_MODELS
    monkeypatch.setenv("GEMINI_MODEL", "my-model")
    assert get_model_candidates()[0] == "my-model"
    assert set(DEFAULT_MODELS) <= set(get_model_candidates())
    monkeypatch.setenv("GEMINI_API_KEY", "  abc  ")
    assert get_api_key() == "abc"
    monkeypatch.delenv("GEMINI_API_KEY")
    assert get_api_key() is None


def test_call_gemini_requires_key():
    with pytest.raises(AIError):
        _call_gemini("hello", None, "system")


def test_generate_insights_with_fake_gemini(monkeypatch):
    reply = json.dumps({"insights": ["A", "B"], "unusual": [], "recommendations": ["C"]})
    fake, captured = fake_gemini(reply)
    monkeypatch.setattr(ai_assistant, "_call_gemini", fake)
    result = generate_insights(build_privacy_safe_summary(make_df()), "key")
    assert result["source"] == "gemini" and result["model"] == "fake-model"
    assert result["insights"] == ["A", "B"]
    assert captured["json_mode"] is True
    assert "secret7@example.com" not in captured["prompt"]    # nothing sensitive was sent
    assert '"rows":120' in captured["prompt"]


def test_answer_question_validation_and_fake(monkeypatch):
    summary = build_privacy_safe_summary(make_df())
    with pytest.raises(AIError):
        answer_question("   ", summary, "key")
    with pytest.raises(AIError):
        answer_question("x" * 600, summary, "key")
    fake, _ = fake_gemini("The answer.")
    monkeypatch.setattr(ai_assistant, "_call_gemini", fake)
    answer, model = answer_question("Which region is best?", summary, "key")
    assert answer == "The answer." and model == "fake-model"


# ---- safety ----------------------------------------------------------------
def test_original_is_never_modified():
    df = make_df()
    snapshot = df.copy()
    summary = build_privacy_safe_summary(df)
    offline_insights(summary)
    detect_sensitive_columns(df)
    answer_offline("average of sales", df)
    for spec in recommend_charts(df):
        chart_context(df, spec, [])
    assert_frame_equal(df, snapshot)