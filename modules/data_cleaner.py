"""Automated data cleaning functions (no Streamlit code in this file).

Every cleaning function follows the same contract:

    new_df, log = function(df, ...)

* The input df is NEVER modified: we always work on a copy.
* log is a dict describing what happened, or None if there was nothing to change.
"""
import re
import warnings
from datetime import datetime

import pandas as pd
from pandas.api import types as ptypes

from modules.data_loader import _NUMERIC_JUNK, looks_date, looks_numeric

MISSING_DROP_THRESHOLD = 60   # % missing above which we suggest dropping the column
SKEW_THRESHOLD = 1.0          # |skew| above this -> median is safer than mean

# Friendly names for the missing-value treatments (used in the UI)
METHOD_LABELS = {
    "mean": "Fill with the mean",
    "median": "Fill with the median",
    "mode": "Fill with the most frequent value (mode)",
    "custom": "Fill with a custom value",
    "drop_rows": "Drop rows with a missing value (destructive)",
    "drop_column": "Drop the whole column (destructive)",
    "convert_first": "Convert the data type first",
    "leave_as_is": "Leave as is",
}


class CleaningError(Exception):
    """Raised with a friendly message when a cleaning step cannot be done."""


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def _is_text(series: pd.Series) -> bool:
    return ptypes.is_object_dtype(series) or ptypes.is_string_dtype(series)


def _is_numeric(series: pd.Series) -> bool:
    """Numeric dtype, but True/False columns do not count as numbers."""
    return ptypes.is_numeric_dtype(series) and not ptypes.is_bool_dtype(series)


def _require_column(df: pd.DataFrame, column: str) -> None:
    if column not in df.columns:
        raise CleaningError(f"Column '{column}' was not found.")


def _make_log(operation, column, details, before, after, cells_changed=0) -> dict:
    """Build one cleaning-log entry."""
    return {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "operation": operation,
        "column": column or "-",
        "details": details,
        "rows_before": len(before),
        "rows_after": len(after),
        "cells_changed": int(cells_changed),
    }


def _changed_mask(before_series: pd.Series, after_series: pd.Series) -> pd.Series:
    """True where a value differs between two identically-indexed Series."""
    try:
        same = (before_series == after_series) | (before_series.isna() & after_series.isna())
        return ~same
    except Exception:  # incomparable types: treat every non-empty cell as changed
        return before_series.notna() | after_series.notna()


def count_changed_cells(before: pd.DataFrame, after: pd.DataFrame, column: str) -> int:
    """How many cells in one column differ between before and after."""
    if column not in before.columns or column not in after.columns:
        return 0
    common = before.index.intersection(after.index)
    return int(_changed_mask(before.loc[common, column], after.loc[common, column]).sum())


def changed_columns(before: pd.DataFrame, after: pd.DataFrame) -> list:
    """Names of columns whose values differ between before and after."""
    return [c for c in before.columns
            if c in after.columns and count_changed_cells(before, after, c) > 0]


# --------------------------------------------------------------------------
# Preview helpers (used before the user approves a change)
# --------------------------------------------------------------------------
def summarize_change(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    """Before/after table: rows, columns, missing cells, duplicate rows."""
    def stats(d):
        return {"Rows": len(d), "Columns": d.shape[1],
                "Missing cells": int(d.isna().sum().sum()),
                "Duplicate rows": int(d.duplicated().sum())}
    b, a = stats(before), stats(after)
    return pd.DataFrame({
        "Before": pd.Series(b),
        "After": pd.Series(a),
        "Change": pd.Series({key: a[key] - b[key] for key in b}),
    })


def preview_column_changes(before: pd.DataFrame, after: pd.DataFrame,
                           column: str, limit: int = 10) -> pd.DataFrame:
    """Sample of the values in one column that would change."""
    if column not in before.columns or column not in after.columns:
        return pd.DataFrame()
    common = before.index.intersection(after.index)
    b = before.loc[common, column]
    a = after.loc[common, column]
    mask = _changed_mask(b, a)
    return pd.DataFrame({"Before": b[mask], "After": a[mask]}).head(limit)


# --------------------------------------------------------------------------
# 1. Column names
# --------------------------------------------------------------------------
def standardize_name(name) -> str:
    """'Unit Price ' -> 'unit_price',  'Order-ID#' -> 'order_id'."""
    text = str(name).strip().lower()
    text = re.sub(r"[\W_]+", "_", text).strip("_")   # anything not a letter/digit -> _
    if not text:
        return "column"
    if text[0].isdigit():                            # names should not start with a digit
        text = "col_" + text
    return text


def get_column_rename_map(df: pd.DataFrame) -> dict:
    """{old_name: new_name} for columns whose name would change. Duplicates get _2, _3..."""
    mapping, used = {}, set()
    for column in df.columns:
        base = standardize_name(column)
        new_name, counter = base, 2
        while new_name in used:
            new_name = f"{base}_{counter}"
            counter += 1
        used.add(new_name)
        mapping[column] = new_name
    return {old: new for old, new in mapping.items() if old != new}


def standardize_column_names(df: pd.DataFrame):
    mapping = get_column_rename_map(df)
    if not mapping:
        return df.copy(), None
    result = df.rename(columns=mapping)
    details = f"Renamed {len(mapping)} column(s) to lowercase_with_underscores"
    return result, _make_log("Standardize column names", "multiple", details, df, result)


# --------------------------------------------------------------------------
# 2. Whitespace
# --------------------------------------------------------------------------
def strip_whitespace(df: pd.DataFrame, columns=None):
    """Trim leading/trailing spaces and collapse repeated spaces in text columns."""
    result = df.copy()
    targets = columns if columns else [c for c in result.columns if _is_text(result[c])]
    changed, total = [], 0
    for column in targets:
        series = result[column]
        as_object = series.astype("object")
        is_string = as_object.map(lambda value: isinstance(value, str))
        trimmed = as_object.where(
            ~is_string,   # keep numbers / empty cells untouched
            as_object.astype(str).str.strip().str.replace(r"\s+", " ", regex=True),
        )
        n_changed = int(((trimmed != as_object) & is_string).sum())
        if n_changed:
            result[column] = trimmed
            changed.append(column)
            total += n_changed
    if total == 0:
        return result, None
    details = (f"Trimmed extra spaces in {total} cell(s) across "
               f"{len(changed)} column(s): {', '.join(changed)}")
    return result, _make_log("Trim whitespace", ", ".join(changed), details, df, result, total)


# --------------------------------------------------------------------------
# 3. Data type conversion
# --------------------------------------------------------------------------
def convert_to_numeric(df: pd.DataFrame, column: str):
    """Convert text like '10%', '$1,200' or '45' to numbers. Bad values become missing."""
    _require_column(df, column)
    series = df[column]
    if _is_numeric(series):
        return df.copy(), None
    as_object = series.astype("object")
    cleaned = as_object.where(
        series.isna(), as_object.astype(str).str.replace(_NUMERIC_JUNK, "", regex=True))
    converted = pd.to_numeric(cleaned, errors="coerce")
    failed = int((converted.isna() & series.notna()).sum())
    had_percent = bool(as_object.dropna().astype(str).str.contains("%").any())

    result = df.copy()
    result[column] = converted
    details = f"Converted '{column}' to numeric"
    if had_percent:
        details += " ('%' signs removed: 10% becomes 10)"
    if failed:
        details += f". WARNING: {failed} value(s) could not be converted and became missing"
    return result, _make_log("Convert to numeric", column, details, df, result,
                             converted.notna().sum())


def convert_to_datetime(df: pd.DataFrame, column: str, dayfirst: bool = False):
    """Convert text dates to real datetimes. Unreadable values become missing."""
    _require_column(df, column)
    series = df[column]
    if ptypes.is_datetime64_any_dtype(series):
        return df.copy(), None
    as_object = series.astype("object")
    cleaned = as_object.where(series.isna(), as_object.astype(str).str.strip())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(cleaned, errors="coerce", format="mixed", dayfirst=dayfirst)
    failed = int((parsed.isna() & series.notna()).sum())

    result = df.copy()
    result[column] = parsed
    details = f"Converted '{column}' to datetime"
    if failed:
        details += f". WARNING: {failed} value(s) could not be converted and became missing"
    return result, _make_log("Convert to datetime", column, details, df, result,
                             parsed.notna().sum())


# --------------------------------------------------------------------------
# 4. Inconsistent categories ('North', 'north', 'NORTH ')
# --------------------------------------------------------------------------
def suggest_category_mapping(df: pd.DataFrame, column: str) -> dict:
    """{variant: preferred_label}. The most frequent spelling in each group wins."""
    _require_column(df, column)
    series = df[column].dropna().astype(str)
    normalised = series.str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
    mapping = {}
    for _, group in series.groupby(normalised):
        if group.nunique() < 2:
            continue
        stripped = group.str.strip().str.replace(r"\s+", " ", regex=True)
        counts = stripped.value_counts()
        canonical = sorted(counts[counts == counts.max()].index)[0]   # ties: alphabetical
        for raw in group.unique():
            if raw != canonical:
                mapping[raw] = canonical
    return mapping


def apply_category_mapping(df: pd.DataFrame, column: str, mapping: dict):
    _require_column(df, column)
    clean_map = {k: v for k, v in mapping.items()
                 if v is not None and str(v).strip() != "" and k != v}
    if not clean_map:
        return df.copy(), None
    series = df[column]
    changed = int(series.isin(list(clean_map)).sum())
    if changed == 0:
        return df.copy(), None
    result = df.copy()
    result[column] = series.map(lambda value: clean_map.get(value, value))
    details = f"Standardized {changed} value(s) in '{column}' ({len(clean_map)} variants merged)"
    return result, _make_log("Standardize categories", column, details, df, result, changed)


# --------------------------------------------------------------------------
# 5. Missing values
# --------------------------------------------------------------------------
def get_available_methods(series: pd.Series) -> list:
    """Which treatments make sense for this column."""
    if _is_numeric(series):
        return ["mean", "median", "mode", "custom", "drop_rows", "drop_column"]
    return ["mode", "custom", "drop_rows", "drop_column"]


def suggest_missing_treatments(df: pd.DataFrame, column_types: dict) -> list:
    """Recommend a treatment (and the reason) for every column with missing values."""
    suggestions = []
    for column in df.columns:
        series = df[column]
        missing = int(series.isna().sum())
        if missing == 0:
            continue
        pct = round(missing / len(df) * 100, 2)
        kind = column_types.get(column, "text")
        custom_value = None

        if pct > MISSING_DROP_THRESHOLD:
            method = "drop_column"
            reason = f"{pct}% missing: too little information to fill reliably"
        elif _is_text(series) and (looks_numeric(series) or looks_date(series)):
            method = "convert_first"
            reason = "Numbers/dates are stored as text. Fix the data type first, then fill"
        elif _is_numeric(series):
            skew = series.skew()
            if pd.notna(skew) and abs(skew) > SKEW_THRESHOLD:
                method = "median"
                reason = f"Skewed distribution (skew {skew:.1f}): median is robust to outliers"
            else:
                method = "mean"
                reason = "Roughly symmetric distribution: the mean is a fair estimate"
        elif ptypes.is_datetime64_any_dtype(series):
            if pct < 5:
                method = "drop_rows"
                reason = "Very few dates missing: dropping those rows loses little"
            else:
                method = "leave_as_is"
                reason = "Dates can't be guessed safely: review them manually"
        elif kind in ("categorical", "boolean"):
            method = "mode"
            reason = "Categorical column: the most frequent value is the safest fill"
        else:
            method = "custom"
            custom_value = "Unknown"
            reason = "Free text: fill with an explicit 'Unknown' marker"

        suggestions.append({"column": column, "kind": kind, "missing": missing,
                            "missing_pct": pct, "method": method, "reason": reason,
                            "custom_value": custom_value})
    return suggestions


def _parse_custom_value(series: pd.Series, text):
    """Convert the user's typed value to the column's type."""
    text = str(text).strip()
    if _is_numeric(series):
        try:
            return float(text)
        except ValueError:
            raise CleaningError(f"'{text}' is not a number, but this column is numeric.") from None
    if ptypes.is_datetime64_any_dtype(series):
        parsed = pd.to_datetime(text, errors="coerce")
        if pd.isna(parsed):
            raise CleaningError(f"'{text}' is not a valid date.")
        return parsed
    return text


def apply_missing_treatment(df: pd.DataFrame, column: str, method: str, custom_value=None):
    _require_column(df, column)
    series = df[column]
    missing = int(series.isna().sum())
    if missing == 0:
        return df.copy(), None

    result = df.copy()
    if method in ("mean", "median"):
        if not _is_numeric(series):
            raise CleaningError(
                f"The {method} needs a numeric column. Convert '{column}' to numeric first.")
        value = series.mean() if method == "mean" else series.median()
        if pd.isna(value):
            raise CleaningError(f"Cannot compute the {method}: the column has no values.")
        fill_value, label = value, f"the {method} ({value:.2f})"
    elif method == "mode":
        modes = series.mode(dropna=True)
        if modes.empty:
            raise CleaningError(f"Cannot find a most frequent value in '{column}'.")
        fill_value, label = modes.iloc[0], f"the most frequent value ({modes.iloc[0]})"
    elif method == "custom":
        if custom_value is None or str(custom_value).strip() == "":
            raise CleaningError("Please enter a custom value.")
        fill_value = _parse_custom_value(series, custom_value)
        label = f"the custom value ({custom_value})"
    elif method == "drop_rows":
        result = result.dropna(subset=[column])
        details = f"Dropped {missing} row(s) where '{column}' was missing"
        return result, _make_log("Drop rows with missing values", column, details,
                                 df, result, missing)
    elif method == "drop_column":
        result = result.drop(columns=[column])
        details = f"Dropped column '{column}' ({missing} missing values)"
        return result, _make_log("Drop column", column, details, df, result, missing)
    else:
        raise CleaningError(f"'{METHOD_LABELS.get(method, method)}' is not a fill method here.")

    result[column] = series.fillna(fill_value)
    details = f"Filled {missing} missing value(s) in '{column}' with {label}"
    return result, _make_log("Fill missing values", column, details, df, result, missing)


def apply_recommended_fills(df: pd.DataFrame, suggestions: list):
    """Apply every SAFE recommendation (mean/median/mode/custom) in one go.
    Destructive or uncertain ones (drop, convert-first, leave) are skipped."""
    result, parts, total = df.copy(), [], 0
    for item in suggestions:
        if item["method"] not in ("mean", "median", "mode", "custom"):
            continue
        try:
            result, log = apply_missing_treatment(
                result, item["column"], item["method"], item.get("custom_value"))
        except CleaningError:
            continue   # skip a column we can't fill; keep going
        if log:
            parts.append(f"{item['column']} ({item['method']})")
            total += log["cells_changed"]
    if not parts:
        return result, None
    details = f"Filled {total} missing value(s): " + ", ".join(parts)
    return result, _make_log("Fill missing values (all recommended)", "multiple",
                             details, df, result, total)


# --------------------------------------------------------------------------
# 6. Duplicates
# --------------------------------------------------------------------------
def remove_duplicates(df: pd.DataFrame, subset=None, keep="first"):
    """keep='first' / 'last' keeps one copy; keep=False removes every copy."""
    if keep not in ("first", "last", False):
        raise CleaningError("keep must be 'first', 'last' or False.")
    subset = list(subset) if subset else None
    if subset:
        unknown = [c for c in subset if c not in df.columns]
        if unknown:
            raise CleaningError(f"Unknown column(s): {', '.join(map(str, unknown))}")
    to_remove = int(df.duplicated(subset=subset, keep=keep).sum())
    if to_remove == 0:
        return df.copy(), None
    result = df.drop_duplicates(subset=subset, keep=keep)
    how = {"first": "kept the first copy", "last": "kept the last copy",
           False: "removed every copy"}[keep]
    scope = f"columns {', '.join(subset)}" if subset else "all columns"
    details = f"Removed {to_remove} duplicate row(s) comparing {scope} ({how})"
    return result, _make_log("Remove duplicates", ", ".join(subset) if subset else "all",
                             details, df, result, to_remove)