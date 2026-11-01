"""Tests for the data cleaning functions."""
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from modules.data_cleaner import (CleaningError, apply_category_mapping,
                                  apply_missing_treatment, apply_recommended_fills,
                                  convert_to_datetime, convert_to_numeric,
                                  get_column_rename_map, remove_duplicates,
                                  standardize_column_names, standardize_name,
                                  strip_whitespace, suggest_category_mapping,
                                  suggest_missing_treatments, summarize_change)


def make_df() -> pd.DataFrame:
    """Rows 2 and 3 are identical; Score has 1 missing value; text needs trimming."""
    return pd.DataFrame({
        "Order ID ": [1, 2, 3, 3],
        "Region": ["North", "north ", "South", "South"],
        "Amount": ["10%", "20%", "30%", "30%"],
        "Score": [1.0, np.nan, 3.0, 3.0],
        "Name": [" a  b", "c", "d", "d"],
    })


# ---- names -----------------------------------------------------------------
def test_standardize_name():
    assert standardize_name("Unit Price ") == "unit_price"
    assert standardize_name("Order-ID#") == "order_id"
    assert standardize_name("2024 Sales") == "col_2024_sales"
    assert standardize_name("") == "column"


def test_duplicate_column_names_get_suffix():
    df = pd.DataFrame([[1, 2]], columns=["A b", "a_b"])
    assert get_column_rename_map(df) == {"A b": "a_b", "a_b": "a_b_2"}
    result, log = standardize_column_names(df)
    assert list(result.columns) == ["a_b", "a_b_2"]
    assert log is not None


def test_original_dataframe_is_never_modified():
    df = make_df()
    snapshot = df.copy()
    strip_whitespace(df)
    standardize_column_names(df)
    apply_missing_treatment(df, "Score", "median")
    remove_duplicates(df)
    convert_to_numeric(df, "Amount")
    assert_frame_equal(df, snapshot)


# ---- whitespace ------------------------------------------------------------
def test_strip_whitespace():
    result, log = strip_whitespace(make_df())
    assert result["Name"].tolist() == ["a b", "c", "d", "d"]
    assert result["Region"].tolist() == ["North", "north", "South", "South"]
    assert log["cells_changed"] == 2


def test_strip_whitespace_nothing_to_do():
    _, log = strip_whitespace(pd.DataFrame({"a": ["x", "y"], "b": [1, 2]}))
    assert log is None


# ---- missing values --------------------------------------------------------
def test_fill_median():
    result, log = apply_missing_treatment(make_df(), "Score", "median")
    assert result.loc[1, "Score"] == 3.0
    assert log["cells_changed"] == 1


def test_fill_mean():
    result, _ = apply_missing_treatment(make_df(), "Score", "mean")
    assert result.loc[1, "Score"] == pytest.approx(7 / 3)


def test_fill_mode():
    df = pd.DataFrame({"r": ["a", "a", None, "b"]})
    result, _ = apply_missing_treatment(df, "r", "mode")
    assert result["r"].tolist() == ["a", "a", "a", "b"]


def test_fill_custom_number_rejects_text():
    with pytest.raises(CleaningError):
        apply_missing_treatment(make_df(), "Score", "custom", "abc")


def test_mean_rejected_for_text_column():
    with pytest.raises(CleaningError):
        apply_missing_treatment(pd.DataFrame({"r": ["a", None]}), "r", "mean")


def test_drop_rows():
    result, log = apply_missing_treatment(make_df(), "Score", "drop_rows")
    assert len(result) == 3
    assert log["rows_before"] == 4 and log["rows_after"] == 3


# ---- duplicates ------------------------------------------------------------
def test_remove_duplicates_keep_first():
    result, log = remove_duplicates(make_df(), keep="first")
    assert len(result) == 3
    assert log["cells_changed"] == 1


def test_remove_all_copies():
    result, _ = remove_duplicates(make_df(), keep=False)
    assert len(result) == 2


# ---- type conversion -------------------------------------------------------
def test_convert_to_numeric():
    result, log = convert_to_numeric(make_df(), "Amount")
    assert result["Amount"].tolist() == [10, 20, 30, 30]
    assert "%" in log["details"]


def test_convert_to_numeric_counts_failures():
    df = pd.DataFrame({"v": ["5", "x", "7"]})
    result, log = convert_to_numeric(df, "v")
    assert result["v"].isna().sum() == 1
    assert "could not be converted" in log["details"]


def test_convert_to_datetime():
    df = pd.DataFrame({"d": ["2024-01-05", "2024-02-10", "bad", None]})
    result, log = convert_to_datetime(df, "d")
    assert pd.api.types.is_datetime64_any_dtype(result["d"])
    assert result["d"].isna().sum() == 2          # 'bad' and the original None
    assert "1 value" in log["details"]


# ---- categories ------------------------------------------------------------
def test_category_mapping():
    df = pd.DataFrame({"Region": ["North", "North", "north ", "NORTH", "South"]})
    mapping = suggest_category_mapping(df, "Region")
    assert mapping == {"north ": "North", "NORTH": "North"}
    result, log = apply_category_mapping(df, "Region", mapping)
    assert result["Region"].tolist() == ["North", "North", "North", "North", "South"]
    assert log["cells_changed"] == 2


# ---- recommendations -------------------------------------------------------
def test_suggest_missing_treatments():
    df = pd.DataFrame({
        "skewed": [1, 2, 3, 4, 1000, np.nan],
        "even": [1, 2, 3, 4, 5, np.nan],
        "cat": ["a", "a", "b", "a", "b", None],
    })
    types = {"skewed": "numeric", "even": "numeric", "cat": "categorical"}
    methods = {s["column"]: s["method"] for s in suggest_missing_treatments(df, types)}
    assert methods == {"skewed": "median", "even": "mean", "cat": "mode"}


def test_suggest_drop_and_convert_first():
    mostly_empty = pd.DataFrame({"x": [1, None, None, None]})
    assert suggest_missing_treatments(mostly_empty, {"x": "numeric"})[0]["method"] == "drop_column"
    text_numbers = pd.DataFrame({"d": ["10", "20", None]})
    assert suggest_missing_treatments(text_numbers, {"d": "numeric"})[0]["method"] == "convert_first"


def test_apply_recommended_fills():
    df = make_df()
    suggestions = suggest_missing_treatments(df, {"Score": "numeric"})
    result, log = apply_recommended_fills(df, suggestions)
    assert result["Score"].isna().sum() == 0
    assert log["cells_changed"] == 1


# ---- preview summary -------------------------------------------------------
def test_summarize_change():
    df = make_df()
    after, _ = remove_duplicates(df)
    summary = summarize_change(df, after)
    assert summary.loc["Rows", "Change"] == -1
    assert summary.loc["Duplicate rows", "Change"] == -1