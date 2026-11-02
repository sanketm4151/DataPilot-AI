"""Tests for formula-safe exports."""
import io

import pandas as pd
from openpyxl import load_workbook
from pandas.testing import assert_frame_equal

from modules.data_transformation import (_is_formula_like, sanitize_for_export, to_csv_bytes,
                                         to_excel_bytes)


def make_df() -> pd.DataFrame:
    return pd.DataFrame({
        "note": ["=SUM(A1)", "+cmd", "@x", "-5", "-abc", "hello", None],
        "n": [1, -2, 3, 4, 5, 6, 7],
        "=bad": range(7),
    })


def test_formula_detection():
    assert _is_formula_like("=1+1") and _is_formula_like("@cmd") and _is_formula_like("-abc")
    assert not _is_formula_like("-5") and not _is_formula_like("-1,200")   # numbers as text
    assert not _is_formula_like("-") and not _is_formula_like("hello")
    assert not _is_formula_like(None) and not _is_formula_like(5)


def test_csv_export_neutralizes_formulas():
    text = to_csv_bytes(make_df()).decode("utf-8-sig")
    assert "'=SUM(A1)" in text and "'+cmd" in text and "'@x" in text and "'-abc" in text
    assert text.splitlines()[0] == "note,n,'=bad"          # dangerous column name too
    assert "'-5" not in text                               # a number as text is left alone
    assert "-2" in text                                    # real negative numbers are untouched


def test_excel_export_neutralizes_formulas():
    workbook = load_workbook(io.BytesIO(to_excel_bytes(make_df())))
    cell = workbook.active.cell(row=2, column=1)           # the '=SUM(A1)' cell
    assert cell.data_type == "s"                           # plain text, not a formula
    assert cell.value == "'=SUM(A1)"


def test_sanitize_does_not_modify_the_input():
    df = make_df()
    snapshot = df.copy()
    sanitize_for_export(df)
    to_csv_bytes(df)
    to_excel_bytes(df)
    assert_frame_equal(df, snapshot)