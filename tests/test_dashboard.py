"""Tests for the dashboard module (column roles, filters, KPIs, chart selection)."""
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from pandas.testing import assert_frame_equal

from modules.dashboard import (DashboardError, aggregate_by_category, apply_filters,
                               auto_frequency, build_chart, build_kpis, format_number,
                               get_filter_options, identify_roles, pie_figure,
                               recommend_charts)
from modules.eda import EDAError


def make_df() -> pd.DataFrame:
    """A sales-style dataset: 120 daily rows, one ID, one text column, 3 measures."""
    rng = np.random.default_rng(1)
    n = 120
    return pd.DataFrame({
        "order_id": range(1, n + 1),                                   # ID: not a measure
        "date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "region": np.where(np.arange(n) % 3 == 0, "North",
                           np.where(np.arange(n) % 3 == 1, "South", "East")),
        "segment": np.where(np.arange(n) % 2 == 0, "Retail", "Online"),
        "sales": rng.uniform(10, 100, n).round(2),
        "unit_price": rng.uniform(5, 20, n).round(2),
        "age": rng.integers(18, 70, n),
        "notes": [f"note {i}" for i in range(n)],                      # 120 distinct texts
    })


def make_other_df() -> pd.DataFrame:
    """A NON-sales dataset: no dates, no 'sales-like' column names."""
    n = 30
    return pd.DataFrame({
        "student": [f"s{i}" for i in range(n)],
        "grade": ["A", "B", "C"] * 10,
        "math": (np.arange(n) % 7) * 10 + 40,
        "science": (np.arange(n) * 2) % 11 + 50,
    })


# ---- column roles ----------------------------------------------------------
def test_identify_roles():
    roles = identify_roles(make_df())
    assert roles["measures"] == ["sales", "unit_price", "age"]    # no order_id, no notes
    assert roles["dimensions"] == ["segment", "region"]           # fewest values first
    assert roles["dates"] == ["date"]


def test_auto_frequency():
    assert auto_frequency(pd.Series(pd.date_range("2024-01-01", periods=31))) == "D"
    assert auto_frequency(pd.Series(pd.date_range("2024-01-01", periods=400))) == "M"
    assert auto_frequency(pd.Series(pd.date_range("2024-01-01", periods=3000))) == "Q"
    assert auto_frequency(pd.Series(pd.date_range("2000-01-01", periods=7000))) == "Y"


# ---- filters ---------------------------------------------------------------
def test_filter_options():
    df = make_df()
    options = get_filter_options(df)
    assert list(options["categories"]) == ["segment", "region"]
    assert options["categories"]["region"] == ["East", "North", "South"]
    assert list(options["numeric"]) == ["sales", "unit_price"]
    assert options["numeric"]["sales"] == (df["sales"].min(), df["sales"].max())
    assert options["dates"]["date"] == (df["date"].min().date(), df["date"].max().date())


def test_apply_filters_categories_numbers_dates():
    df = make_df()
    north = apply_filters(df, {"categories": {"region": ["North"]}})
    assert len(north) == 40 and set(north["region"]) == {"North"}

    high = apply_filters(df, {"numeric": {"sales": (50.0, 100.0)}})
    assert len(high) == int((df["sales"] >= 50).sum())

    january = apply_filters(df, {"dates": {"date": (date(2024, 1, 1), date(2024, 1, 10))}})
    assert len(january) == 10                                     # end date is inclusive

    combined = apply_filters(df, {"categories": {"region": ["North", "South"]},
                                  "numeric": {"sales": (50.0, 100.0)}})
    assert set(combined["region"]) <= {"North", "South"}
    assert combined["sales"].min() >= 50
    assert len(apply_filters(df, {})) == len(df)


def test_apply_filters_empty_result_and_errors():
    df = make_df()
    assert apply_filters(df, {"categories": {"region": []}}).empty
    with pytest.raises(EDAError):
        apply_filters(df, {"categories": {"nope": ["x"]}})
    with pytest.raises(EDAError):
        apply_filters(df, {"numeric": {"region": (0, 1)}})        # not a numeric column
    with pytest.raises(EDAError):
        apply_filters(df, {"dates": {"sales": (date(2024, 1, 1), date(2024, 2, 1))}})


# ---- KPI cards -------------------------------------------------------------
def test_format_number():
    assert format_number(1234) == "1,234"
    assert format_number(15000) == "15.00K"
    assert format_number(2_500_000) == "2.50M"
    assert format_number(0.5) == "0.50"
    assert format_number(float("nan")) == "n/a"
    assert format_number(None) == "n/a"


def test_build_kpis():
    kpis = build_kpis(make_df())
    assert [k["label"] for k in kpis] == ["Records", "Total sales", "Average unit_price",
                                          "Average age", "Latest month vs previous"]
    assert kpis[0]["value"] == "120"
    assert kpis[-1]["sub"] == "sales: Apr 2024 vs Mar 2024"


def test_build_kpis_without_dates_or_measures():
    assert [k["label"] for k in build_kpis(make_other_df())] == [
        "Records", "Average math", "Average science"]
    text_only = pd.DataFrame({"c": ["a", "b"] * 5})
    assert [k["label"] for k in build_kpis(text_only)] == ["Records"]


# ---- charts ----------------------------------------------------------------
def test_aggregate_by_category():
    df = pd.DataFrame({"c": ["a", "a", "a", "b", "b", "c"], "v": [1, 2, 3, 10, 20, 100]})
    table = aggregate_by_category(df, "c", "v", "sum", top_n=2)
    assert table["Category"].tolist() == ["c", "b"]
    assert table["Value"].tolist() == [100.0, 30.0]

    counts = aggregate_by_category(df, "c", None, "count", top_n=1, group_rest=True)
    assert counts["Category"].tolist() == ["a", "Other (2 categories)"]
    assert counts["Value"].tolist() == [3.0, 3.0]

    averages = aggregate_by_category(df, "c", "v", "mean", top_n=1, group_rest=True)
    assert len(averages) == 1                                     # no 'Other' for averages


def test_pie_rejects_bad_input():
    df = pd.DataFrame({"c": ["a", "b"], "v": [-1.0, 5.0]})
    with pytest.raises(DashboardError):
        pie_figure(df, "c", "v", "sum")                           # negative value
    with pytest.raises(DashboardError):
        pie_figure(df, "c", "v", "mean")                          # shares of an average


def test_recommend_charts_for_sales_data():
    df = make_df()
    specs = recommend_charts(df)
    assert [s["kind"] for s in specs] == ["line", "bar", "pie", "histogram",
                                          "scatter", "heatmap"]
    assert specs[0]["measure"] == "sales" and specs[0]["agg"] == "sum"
    assert specs[0]["freq"] == "M"
    assert specs[1]["category"] == "segment"
    assert specs[2]["category"] == "region"                       # differs from the bar chart


def test_recommend_charts_for_other_data():
    specs = recommend_charts(make_other_df())
    assert [s["kind"] for s in specs] == ["bar", "pie", "histogram", "scatter"]
    assert specs[0]["measure"] == "math" and specs[0]["agg"] == "mean"
    assert specs[1]["agg"] == "count" and specs[1]["measure"] is None
    assert recommend_charts(pd.DataFrame({"notes": [f"n{i}" for i in range(30)]})) == []


def test_recommended_charts_all_build():
    for df in (make_df(), make_other_df()):
        for spec in recommend_charts(df):
            assert isinstance(build_chart(df, spec), go.Figure)


def test_build_chart_validation():
    df = make_df()
    with pytest.raises(DashboardError):
        build_chart(df, {"kind": "donut"})                        # unknown type
    with pytest.raises(DashboardError):
        build_chart(df, {"kind": "bar"})                          # missing 'category'
    with pytest.raises(EDAError):
        build_chart(df, {"kind": "bar", "category": "region", "measure": "region",
                         "agg": "sum"})                           # text measure
    with pytest.raises(EDAError):
        build_chart(df, {"kind": "line", "date": "sales"})        # not a date column


# ---- safety ----------------------------------------------------------------
def test_original_is_never_modified():
    df = make_df()
    snapshot = df.copy()
    roles = identify_roles(df)
    apply_filters(df, {"categories": {"region": ["North"]}})
    build_kpis(df, roles)
    for spec in recommend_charts(df, roles):
        build_chart(df, spec)
    assert_frame_equal(df, snapshot)