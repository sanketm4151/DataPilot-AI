"""Tests for the EDA module."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from pandas.testing import assert_frame_equal

from modules.eda import (EDAError, categorical_columns, categorical_statistics,
                         category_bar_figure, category_counts, comparison_figure,
                         correlation_heatmap, correlation_matrix, crosstab_figure,
                         crosstab_table, date_columns, describe_correlation,
                         distribution_summary, generate_highlights, group_comparison,
                         histogram_figure, missing_bar_figure, missing_pattern_figure,
                         missing_summary, numeric_columns, numeric_statistics,
                         relationship_summary, scatter_figure, top_correlations,
                         trend_figure, trend_over_time, trend_summary, variation_explained,
                         variation_label)


def make_df() -> pd.DataFrame:
    """120 rows. y is almost exactly 2*x; 'maybe' has 12 gaps; 'const' never changes."""
    rng = np.random.default_rng(0)
    n = 120
    x = rng.normal(50, 10, n)
    return pd.DataFrame({
        "x": x,
        "y": 2 * x + rng.normal(0, 1, n),
        "noise": rng.normal(0, 1, n),
        "const": 5.0,
        "group": np.where(np.arange(n) % 2 == 0, "A", "B"),
        "date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "maybe": [np.nan if i % 10 == 0 else 1.0 for i in range(n)],
    })


def make_sales() -> pd.DataFrame:
    return pd.DataFrame({
        "when": pd.to_datetime(["2024-01-05", "2024-01-20", "2024-03-10"]),
        "amount": [10.0, 20.0, 5.0],
    })


# ---- column helpers --------------------------------------------------------
def test_column_helpers():
    df = make_df()
    assert numeric_columns(df) == ["x", "y", "noise", "const", "maybe"]
    assert categorical_columns(df) == ["group"]
    assert date_columns(df) == ["date"]


# ---- correlation -----------------------------------------------------------
def test_correlation_matrix_drops_constant_columns():
    corr = correlation_matrix(make_df())
    assert "const" not in corr.columns
    assert corr.loc["x", "y"] > 0.99


def test_correlation_validation():
    df = make_df()
    with pytest.raises(EDAError):
        correlation_matrix(df, ["x"])             # need two columns
    with pytest.raises(EDAError):
        correlation_matrix(df, ["x", "group"])    # text column
    with pytest.raises(EDAError):
        correlation_matrix(df, ["x", "y"], method="kendall")


def test_describe_correlation():
    assert describe_correlation(0.05) == "negligible"
    assert describe_correlation(0.4) == "moderate positive"
    assert describe_correlation(-0.8) == "very strong negative"


def test_top_correlations():
    pairs = top_correlations(correlation_matrix(make_df(), ["x", "y", "noise"]))
    assert len(pairs) == 3
    assert (pairs.loc[0, "Column A"], pairs.loc[0, "Column B"]) == ("x", "y")
    assert pairs.loc[0, "Strength"] == "very strong positive"


# ---- statistics ------------------------------------------------------------
def test_numeric_statistics():
    stats = numeric_statistics(make_df())
    assert len(stats) == 5
    assert "skew" in stats.columns
    assert stats.loc[stats["Column"] == "maybe", "missing"].iloc[0] == 12


def test_categorical_statistics():
    stats = categorical_statistics(make_df())
    assert stats["Column"].tolist() == ["group"]
    assert stats.loc[0, "Unique"] == 2
    assert stats.loc[0, "Top count"] == 60
    assert stats.loc[0, "Top %"] == 50.0


def test_distribution_summary():
    summary = distribution_summary(pd.Series([1, 1, 1, 1, 100.0]))
    assert "right-skewed" in summary["shape"]
    assert summary["median"] == 1.0
    with pytest.raises(EDAError):
        distribution_summary(pd.Series([1.0, 2.0]))


# ---- missing values --------------------------------------------------------
def test_missing_summary_and_pattern():
    df = make_df()
    summary = missing_summary(df)
    assert summary["Column"].tolist() == ["maybe"]
    assert summary.loc[0, "Missing"] == 12
    assert summary.loc[0, "Missing %"] == 10.0
    assert isinstance(missing_pattern_figure(df), go.Figure)
    assert missing_pattern_figure(df[["x", "y"]]) is None


# ---- categories ------------------------------------------------------------
def test_category_counts_groups_the_rest():
    df = pd.DataFrame({"c": ["a"] * 5 + ["b"] * 3 + ["c"] * 2})
    table = category_counts(df, "c", top_n=1)
    assert table["Category"].tolist() == ["a", "Other (2 categories)"]
    assert table["Count"].tolist() == [5, 5]
    assert table.loc[0, "Percent"] == 50.0
    assert table["Cumulative %"].iloc[-1] == 100.0
    with pytest.raises(EDAError):
        category_counts(df, "nope")


# ---- trends ----------------------------------------------------------------
def test_trend_fills_gaps():
    table = trend_over_time(make_sales(), "when", "amount", freq="M", agg="sum")
    assert table["Value"].tolist() == [30.0, 0.0, 5.0]       # February had no sales
    assert table["Period"].iloc[0] == pd.Timestamp("2024-01-01")
    assert table.loc[1, "Change %"] == -100.0
    assert pd.isna(table.loc[2, "Change %"])                 # change from 0 is undefined
    counts = trend_over_time(make_sales(), "when", None, freq="M", agg="count")
    assert counts["Value"].tolist() == [2.0, 0.0, 1.0]


def test_trend_validation():
    df = make_df()
    with pytest.raises(EDAError):
        trend_over_time(df, "x", "y")                        # x is not a date
    with pytest.raises(EDAError):
        trend_over_time(df, "date", None, agg="sum")         # sum needs a value column
    with pytest.raises(EDAError):
        trend_over_time(df, "date", "x", freq="H")


def test_trend_summary():
    table = trend_over_time(make_sales(), "when", "amount", freq="M", agg="sum")
    info = trend_summary(table)
    assert info["periods"] == 3
    assert info["peak_value"] == 30.0
    assert info["peak_period"] == pd.Timestamp("2024-01-01")
    assert info["low_value"] == 0.0
    assert info["overall_change_pct"] == pytest.approx(-83.3)


# ---- comparisons -----------------------------------------------------------
def test_group_comparison():
    df = pd.DataFrame({"g": ["A", "A", "B", "B"], "v": [1, 3, 10, 20]})
    table = group_comparison(df, "g", "v")
    assert table["Group"].tolist() == ["B", "A"]             # highest average first
    assert table.loc[0, "Mean"] == 15.0
    assert table.loc[0, "Count"] == 2
    with pytest.raises(EDAError):
        group_comparison(df, "g", "g")


def test_variation_explained():
    perfect = pd.DataFrame({"g": ["A", "A", "B", "B"], "v": [1, 1, 5, 5]})
    assert variation_explained(perfect, "g", "v") == pytest.approx(1.0)
    assert variation_explained(make_df(), "group", "noise") < 0.1
    assert variation_label(0.0) == "negligible"
    assert variation_label(1.0) == "large"


def test_relationship_summary():
    df = make_df()
    info = relationship_summary(df, "x", "y")
    assert info["n"] == 120
    assert info["pearson"] > 0.99
    assert info["slope"] == pytest.approx(2, abs=0.1)
    with pytest.raises(EDAError):
        relationship_summary(df, "x", "x")
    with pytest.raises(EDAError):
        relationship_summary(df, "x", "const")               # no variation


def test_scatter_figure():
    df = make_df()
    assert len(scatter_figure(df, "x", "y").data) == 2        # points + trend line
    assert len(scatter_figure(df, "x", "y", color="group").data) == 3
    with pytest.raises(EDAError):
        scatter_figure(df, "x", "y", color="x")


def test_crosstab():
    df = pd.DataFrame({"a": ["x", "x", "y"], "b": ["p", "q", "p"]})
    counts = crosstab_table(df, "a", "b")
    assert counts.loc["x", "p"] == 1 and counts.loc["y", "q"] == 0
    assert crosstab_table(df, "a", "b", percent=True).loc["x", "p"] == 50.0
    with pytest.raises(EDAError):
        crosstab_table(df, "a", "a")


# ---- figures, highlights, safety -------------------------------------------
def test_figures_are_built():
    df = make_df()
    df["tier"] = np.where(df["x"] > 50, "high", "low")
    figures = [
        correlation_heatmap(correlation_matrix(df, ["x", "y", "noise"])),
        histogram_figure(df, "x"),
        histogram_figure(df, "x", group_by="group"),
        missing_bar_figure(missing_summary(df)),
        category_bar_figure(category_counts(df, "group"), "group"),
        trend_figure(trend_over_time(df, "date", "x"), "Total x"),
        comparison_figure(df, "group", "x"),
        comparison_figure(df, "group", "x", kind="bar"),
        crosstab_figure(crosstab_table(df, "group", "tier"), "group", "tier"),
    ]
    assert all(isinstance(fig, go.Figure) for fig in figures)


def test_highlights():
    notes = generate_highlights(make_df())
    assert notes and all(isinstance(note, str) for note in notes)
    assert any("'x' and 'y'" in note for note in notes)       # the strong correlation
    assert any("'maybe'" in note for note in notes)           # the missing values


def test_original_is_never_modified():
    df = make_df()
    snapshot = df.copy()
    numeric_statistics(df)
    correlation_matrix(df)
    missing_summary(df)
    missing_pattern_figure(df)
    trend_over_time(df, "date", "x")
    group_comparison(df, "group", "x")
    scatter_figure(df, "x", "y", color="group")
    generate_highlights(df)
    assert_frame_equal(df, snapshot)