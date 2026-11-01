"""Tests for the transformation functions."""
import io

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from modules.data_cleaner import CleaningError
from modules.data_transformation import (add_calculated_column, change_dtype, drop_columns,
                                         extract_date_parts, filter_rows, rename_columns,
                                         to_csv_bytes, to_excel_bytes)


def make_df() -> pd.DataFrame:
    return pd.DataFrame({
        "qty": [1, 2, 3, 4],
        "price": [10.0, 20.0, 0.0, 40.0],
        "region": ["North", "South", "North", None],
        "amount": ["5", "6", "7", None],
        "when": pd.to_datetime(["2024-01-05", "2024-02-10", "2024-03-15", None]),
    })


# ---- data types ------------------------------------------------------------
def test_change_to_numeric():
    result, log = change_dtype(make_df(), "amount", "numeric")
    assert result["amount"].iloc[:3].tolist() == [5.0, 6.0, 7.0]
    assert result["amount"].isna().sum() == 1
    assert log is not None


def test_change_to_integer():
    df = pd.DataFrame({"x": [1.0, 2.0, None]})
    result, _ = change_dtype(df, "x", "integer")
    assert str(result["x"].dtype) == "Int64"
    assert result["x"].iloc[0] == 1
    assert result["x"].isna().sum() == 1


def test_integer_rejects_decimals():
    with pytest.raises(CleaningError):
        change_dtype(pd.DataFrame({"x": [1.5, 2.0]}), "x", "integer")


def test_change_to_text_keeps_missing_values():
    df = pd.DataFrame({"v": [1.0, None]})
    result, _ = change_dtype(df, "v", "text")
    assert result["v"].iloc[0] == "1.0"
    assert pd.isna(result["v"].iloc[1])


def test_numbers_cannot_become_dates():
    with pytest.raises(CleaningError):
        change_dtype(make_df(), "qty", "datetime")


# ---- rename / drop ---------------------------------------------------------
def test_rename_columns_including_a_swap():
    result, log = rename_columns(make_df(), {"qty": "quantity"})
    assert "quantity" in result.columns and "qty" not in result.columns
    assert log is not None
    swapped, _ = rename_columns(make_df(), {"qty": "price", "price": "qty"})
    assert swapped.columns[:2].tolist() == ["price", "qty"]
    assert swapped["price"].tolist() == [1, 2, 3, 4]


def test_rename_rejects_bad_names():
    with pytest.raises(CleaningError):
        rename_columns(make_df(), {"qty": "price"})      # would create two 'price' columns
    with pytest.raises(CleaningError):
        rename_columns(make_df(), {"qty": "   "})
    with pytest.raises(CleaningError):
        rename_columns(make_df(), {"nope": "x"})


def test_drop_columns_and_protections():
    result, log = drop_columns(make_df(), ["region"])
    assert "region" not in result.columns
    assert log["operation"] == "Drop columns"
    with pytest.raises(CleaningError):
        drop_columns(make_df(), list(make_df().columns))   # cannot drop everything
    with pytest.raises(CleaningError):
        drop_columns(make_df(), ["nope"])


# ---- filter rows -----------------------------------------------------------
def test_filter_numeric_between():
    result, log = filter_rows(make_df(), "qty", "between", (2, 3))
    assert result["qty"].tolist() == [2, 3]
    assert log["rows_after"] == 2


def test_filter_categories_text_and_missing():
    df = make_df()
    assert len(filter_rows(df, "region", "is one of", ["North"])[0]) == 2
    assert len(filter_rows(df, "region", "contains", "sou")[0]) == 1
    assert len(filter_rows(df, "region", "is missing")[0]) == 1


def test_filter_edge_cases():
    df = make_df()
    with pytest.raises(CleaningError):
        filter_rows(df, "qty", ">", 100)                  # would remove every row
    result, log = filter_rows(df, "qty", ">", 0)          # keeps every row
    assert log is None and len(result) == 4


# ---- calculated columns ----------------------------------------------------
def test_calculated_column():
    result, _ = add_calculated_column(make_df(), "revenue", "qty", "*", "price", True)
    assert result["revenue"].tolist() == [10.0, 40.0, 0.0, 160.0]
    doubled, _ = add_calculated_column(make_df(), "double", "qty", "*", 2, False)
    assert doubled["double"].tolist() == [2.0, 4.0, 6.0, 8.0]


def test_divide_by_zero_becomes_missing():
    result, log = add_calculated_column(make_df(), "ratio", "qty", "/", "price", True)
    assert result["ratio"].iloc[0] == pytest.approx(0.1)
    assert result["ratio"].isna().sum() == 1
    assert "division by zero" in log["details"]
    pct, _ = add_calculated_column(make_df(), "pct", "qty", "pct", "price", True)
    assert pct["pct"].iloc[0] == pytest.approx(10.0)


def test_calculated_column_validation():
    df = make_df()
    with pytest.raises(CleaningError):
        add_calculated_column(df, "qty", "qty", "+", 1, False)       # name already exists
    with pytest.raises(CleaningError):
        add_calculated_column(df, "x", "region", "+", 1, False)      # text column
    with pytest.raises(CleaningError):
        add_calculated_column(df, "  ", "qty", "+", 1, False)        # empty name
    with pytest.raises(CleaningError):
        add_calculated_column(df, "x", "qty", "+", "abc", False)     # not a number


# ---- dates -----------------------------------------------------------------
def test_extract_date_parts():
    parts = ["year", "month_name", "year_month", "is_weekend"]
    result, log = extract_date_parts(make_df(), "when", parts)
    assert result["when_year"].iloc[0] == 2024
    assert result["when_month_name"].iloc[1] == "February"
    assert result["when_year_month"].iloc[2] == "2024-03"
    assert pd.isna(result["when_year_month"].iloc[3])         # missing date stays missing
    assert result["when_is_weekend"].iloc[0] == "Weekday"     # Fri 5 Jan 2024
    assert result["when_is_weekend"].iloc[1] == "Weekend"     # Sat 10 Feb 2024
    assert log["cells_changed"] == 16


def test_date_parts_validation():
    with pytest.raises(CleaningError):
        extract_date_parts(make_df(), "qty", ["year"])        # not a date column
    once, _ = extract_date_parts(make_df(), "when", ["year"])
    with pytest.raises(CleaningError):
        extract_date_parts(once, "when", ["year"])            # column already exists


# ---- export ----------------------------------------------------------------
def test_download_bytes():
    df = make_df()
    csv_bytes = to_csv_bytes(df)
    assert csv_bytes.decode("utf-8-sig").splitlines()[0] == "qty,price,region,amount,when"
    assert pd.read_csv(io.BytesIO(csv_bytes), encoding="utf-8-sig").shape == (4, 5)
    assert pd.read_excel(io.BytesIO(to_excel_bytes(df))).shape == (4, 5)


def test_original_is_never_modified():
    df = make_df()
    snapshot = df.copy()
    change_dtype(df, "amount", "numeric")
    rename_columns(df, {"qty": "quantity"})
    drop_columns(df, ["region"])
    filter_rows(df, "qty", ">", 2)
    add_calculated_column(df, "r", "qty", "*", "price", True)
    extract_date_parts(df, "when", ["year"])
    assert_frame_equal(df, snapshot)