"""Tests for the PDF and Excel report generators."""
import io

import numpy as np
import pandas as pd
import pytest
from openpyxl import load_workbook
from pandas.testing import assert_frame_equal

from modules import report_generator
from modules.report_generator import (_pdf_text, build_excel_report, build_pdf_report,
                                      build_report_context, make_report_charts)

LOG = [{"time": "2026-11-02 10:00:00", "operation": "Fill missing values", "column": "age",
        "details": "Filled 9 missing value(s) in 'age' with the median (44.00)",
        "rows_before": 90, "rows_after": 90, "cells_changed": 9}]


def make_original() -> pd.DataFrame:
    """90 daily rows, 3 numeric columns, 2 categories, 9 missing ages."""
    rng = np.random.default_rng(5)
    n = 90
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "region": np.where(np.arange(n) % 3 == 0, "North",
                           np.where(np.arange(n) % 3 == 1, "South", "East")),
        "segment": np.where(np.arange(n) % 2 == 0, "Retail", "Online"),
        "sales": rng.uniform(10, 100, n).round(2),
        "unit_price": rng.uniform(5, 20, n).round(2),
        "age": rng.integers(18, 70, n).astype(float),
    })
    df.loc[::10, "age"] = np.nan
    return df


def make_cleaned() -> pd.DataFrame:
    df = make_original()
    df["age"] = df["age"].fillna(df["age"].median())
    return df


# ---- text safety -----------------------------------------------------------
def test_pdf_text_cleanup():
    assert _pdf_text("a → b") == "a -> b"
    assert _pdf_text("rocket 🚀") == "rocket ?"
    assert _pdf_text(None) == ""


# ---- context ---------------------------------------------------------------
def test_report_context_contents():
    original, cleaned = make_original(), make_cleaned()
    ctx = build_report_context(original, cleaned, LOG, "sales.csv")
    assert ctx["overview"]["rows"] == 90
    assert ctx["original_overview"]["missing_cells"] == 9
    assert ctx["overview"]["missing_cells"] == 0
    assert ctx["quality"]["overall"] > ctx["original_quality"]["overall"]
    assert not ctx["numeric_stats"].empty
    assert ctx["change"].loc["Missing cells", "Change"] == -9
    assert ctx["cleaning_log"] == LOG
    assert ctx["filename"] == "sales.csv"


def test_context_uses_given_ai_result_or_offline():
    original, cleaned = make_original(), make_cleaned()
    given = {"insights": ["Custom insight"], "unusual": [], "recommendations": ["Do X"],
             "source": "gemini", "model": "m", "fingerprint": "abc"}
    ctx = build_report_context(original, cleaned, [], "f.csv", given)
    assert ctx["ai"]["insights"] == ["Custom insight"]
    assert ctx["ai"]["source"] == "gemini"
    fallback = build_report_context(original, cleaned, [], "f.csv")
    assert fallback["ai"]["source"] == "offline"
    assert len(fallback["ai"]["insights"]) >= 5


# ---- charts ----------------------------------------------------------------
def test_report_charts_are_png():
    charts = make_report_charts(make_cleaned(), 6)
    assert len(charts) == 6                      # line, bar, pie, histogram, scatter, heatmap
    assert all(png.startswith(b"\x89PNG") for _, png in charts)
    assert all(isinstance(title, str) and title for title, _ in charts)
    assert len(make_report_charts(make_cleaned(), 2)) == 2


# ---- PDF -------------------------------------------------------------------
def test_pdf_report_builds():
    original, cleaned = make_original(), make_cleaned()
    ctx = build_report_context(original, cleaned, LOG, "sales.csv")
    pdf = build_pdf_report(ctx, make_report_charts(cleaned, 6))
    assert pdf.startswith(b"%PDF-") and len(pdf) > 5000
    assert build_pdf_report(ctx, []).startswith(b"%PDF-")      # works without charts


def test_pdf_handles_special_characters_and_empty_inputs():
    weird = pd.DataFrame({"A & B <x>": ["Tom & <Jerry>", "rocket 🚀", "=1+1", "plain", "x"] * 4,
                          "n": range(20)})
    texts = pd.DataFrame({"c": ["a", "b"] * 10})                # no numeric columns at all
    for frame in (weird, texts):
        ctx = build_report_context(frame, frame, [], "weird <file>.csv")
        assert build_pdf_report(ctx, []).startswith(b"%PDF-")
        assert build_excel_report(ctx, frame)[:2] == b"PK"      # an .xlsx is a zip file


# ---- Excel -----------------------------------------------------------------
def test_excel_report_sheets():
    original, cleaned = make_original(), make_cleaned()
    ctx = build_report_context(original, cleaned, LOG, "sales.csv")
    xlsx = build_excel_report(ctx, cleaned)
    workbook = pd.ExcelFile(io.BytesIO(xlsx))
    assert workbook.sheet_names[0] == "Summary"
    assert {"Data quality", "Before vs after", "Cleaning log", "Column profile",
            "AI insights", "Cleaned data"} <= set(workbook.sheet_names)
    assert workbook.parse("Cleaned data").shape == cleaned.shape
    assert workbook.parse("Cleaning log").shape[0] == 1


def test_excel_neutralizes_formulas():
    original, cleaned = make_original(), make_cleaned()
    cleaned.loc[0, "region"] = '=HYPERLINK("http://example.com")'
    ctx = build_report_context(original, cleaned, [], "f.csv")
    workbook = load_workbook(io.BytesIO(build_excel_report(ctx, cleaned)))
    column = list(cleaned.columns).index("region") + 1
    cell = workbook["Cleaned data"].cell(row=2, column=column)
    assert cell.data_type == "s"                      # plain text, not a formula
    assert str(cell.value).startswith("'=")


def test_excel_skips_huge_data_sheet(monkeypatch):
    monkeypatch.setattr(report_generator, "EXCEL_DATA_MAX_ROWS", 10)
    original, cleaned = make_original(), make_cleaned()
    ctx = build_report_context(original, cleaned, [], "f.csv")
    workbook = pd.ExcelFile(io.BytesIO(build_excel_report(ctx, cleaned)))
    assert "Cleaned data" not in workbook.sheet_names
    summary = workbook.parse("Summary")
    assert summary["Value"].astype(str).str.contains("Skipped").any()


def test_excel_handles_timezone_dates():
    original, cleaned = make_original(), make_cleaned()
    cleaned["date"] = cleaned["date"].dt.tz_localize("UTC")
    ctx = build_report_context(original, cleaned, [], "f.csv")
    xlsx = build_excel_report(ctx, cleaned)
    assert pd.read_excel(io.BytesIO(xlsx), sheet_name="Cleaned data").shape == cleaned.shape

def test_reports_handle_missing_text_values():
    original, cleaned = make_original(), make_cleaned()
    cleaned.loc[::7, "region"] = np.nan                     # gaps in a text column
    cleaned["note"] = pd.Series([None, "x", "y"] * 30)      # another text column with gaps
    ctx = build_report_context(original, cleaned, [], "f.csv")
    xlsx = build_excel_report(ctx, cleaned)
    assert pd.read_excel(io.BytesIO(xlsx), sheet_name="Cleaned data").shape == cleaned.shape
    assert build_pdf_report(ctx, []).startswith(b"%PDF-")

# ---- safety ----------------------------------------------------------------
def test_original_is_never_modified():
    original, cleaned = make_original(), make_cleaned()
    snap_original, snap_cleaned = original.copy(), cleaned.copy()
    ctx = build_report_context(original, cleaned, LOG, "sales.csv")
    build_pdf_report(ctx, make_report_charts(cleaned, 3))
    build_excel_report(ctx, cleaned)
    assert_frame_equal(original, snap_original)
    assert_frame_equal(cleaned, snap_cleaned)