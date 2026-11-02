"""Data transformation functions (no Streamlit code in this file).

Same contract as data_cleaner:  new_df, log = function(df, ...)
The input df is never modified; log is None when nothing changed.
"""
import io

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from modules.data_cleaner import (CleaningError, _is_numeric, _is_text, _make_log,
                                  _require_column, convert_to_datetime, convert_to_numeric)

EXCEL_MAX_ROWS = 1_048_575   # Excel's hard limit (one row is the header)

DTYPE_TARGETS = {
    "numeric": "Number (decimal)",
    "integer": "Whole number",
    "datetime": "Date / time",
    "text": "Text",
}
CALC_OPERATORS = {
    "+": "Add (+)",
    "-": "Subtract (-)",
    "*": "Multiply (x)",
    "/": "Divide (/)",
    "pct": "Percentage (first / second x 100)",
}
_CALC_SYMBOLS = {"+": "+", "-": "-", "*": "x", "/": "/", "pct": "as % of"}
DATE_PARTS = {
    "year": "Year",
    "quarter": "Quarter (1-4)",
    "month": "Month number",
    "month_name": "Month name",
    "year_month": "Year-Month (e.g. 2024-03)",
    "week": "ISO week number",
    "day": "Day of month",
    "weekday": "Weekday name",
    "is_weekend": "Weekday / Weekend",
}
_COMPARATORS = {
    ">": lambda s, v: s > v, ">=": lambda s, v: s >= v,
    "<": lambda s, v: s < v, "<=": lambda s, v: s <= v,
    "==": lambda s, v: s == v, "!=": lambda s, v: s != v,
}


# --------------------------------------------------------------------------
# 1. Change data type
# --------------------------------------------------------------------------
def _to_integer(df: pd.DataFrame, column: str):
    series = df[column]
    if ptypes.is_integer_dtype(series):
        return df.copy(), None
    converted_df, _ = convert_to_numeric(df, column)
    numbers = converted_df[column]
    non_null = numbers.dropna().astype("float64")
    if not np.isfinite(non_null).all():
        raise CleaningError(f"'{column}' contains infinite values.")
    if (non_null % 1 != 0).any():
        raise CleaningError(f"'{column}' contains decimal values: converting to whole "
                            "numbers would lose information. Use 'Number (decimal)' instead.")
    failed = int(numbers.isna().sum() - series.isna().sum())
    result = df.copy()
    result[column] = numbers.astype("Int64")      # nullable integer: keeps missing values
    details = f"Converted '{column}' to whole numbers"
    if failed:
        details += f". WARNING: {failed} value(s) could not be converted and became missing"
    return result, _make_log("Change type to whole number", column, details, df, result,
                             numbers.notna().sum())


def _to_text(df: pd.DataFrame, column: str):
    series = df[column]
    if _is_text(series):
        return df.copy(), None
    as_object = series.astype("object")
    result = df.copy()
    result[column] = as_object.where(series.isna(), as_object.astype(str))  # keep missing
    return result, _make_log("Change type to text", column, f"Converted '{column}' to text",
                             df, result, series.notna().sum())


def change_dtype(df: pd.DataFrame, column: str, target: str, dayfirst: bool = False):
    """target: 'numeric' | 'integer' | 'datetime' | 'text'."""
    _require_column(df, column)
    series = df[column]
    if target == "numeric":
        return convert_to_numeric(df, column)
    if target == "datetime":
        if _is_numeric(series):
            raise CleaningError(
                "Numbers cannot be reliably converted to dates (is 20240105 a date or a "
                "count?). Convert the column to text first, then to a date.")
        return convert_to_datetime(df, column, dayfirst)
    if target == "integer":
        return _to_integer(df, column)
    if target == "text":
        return _to_text(df, column)
    raise CleaningError(f"Unknown target type '{target}'.")


# --------------------------------------------------------------------------
# 2. Rename and drop columns
# --------------------------------------------------------------------------
def rename_columns(df: pd.DataFrame, mapping: dict):
    """mapping = {old_name: new_name}. New names must be non-empty and unique."""
    clean = {}
    for old, new in mapping.items():
        _require_column(df, old)
        text = "" if new is None or pd.isna(new) else str(new).strip()
        if not text:
            raise CleaningError(f"The new name for '{old}' cannot be empty.")
        if text != old:
            clean[old] = text
    if not clean:
        return df.copy(), None
    new_names = [clean.get(c, c) for c in df.columns]
    duplicates = sorted({n for n in new_names if new_names.count(n) > 1})
    if duplicates:
        raise CleaningError("Column names must be unique. Duplicate name(s): "
                            + ", ".join(map(str, duplicates)))
    result = df.rename(columns=clean)
    details = "Renamed " + ", ".join(f"'{o}' to '{n}'" for o, n in clean.items())
    return result, _make_log("Rename columns", ", ".join(clean), details, df, result)


def drop_columns(df: pd.DataFrame, columns):
    columns = list(dict.fromkeys(columns))          # remove repeats, keep order
    if not columns:
        return df.copy(), None
    for column in columns:
        _require_column(df, column)
    if len(columns) >= df.shape[1]:
        raise CleaningError("You cannot remove every column.")
    result = df.drop(columns=columns)
    details = f"Dropped {len(columns)} column(s): {', '.join(columns)}"
    return result, _make_log("Drop columns", ", ".join(columns), details, df, result,
                             len(df) * len(columns))


# --------------------------------------------------------------------------
# 3. Filter rows
# --------------------------------------------------------------------------
def _coerce(series: pd.Series, value):
    """Convert a UI value to the column's type (float or Timestamp)."""
    if value is None:
        raise ValueError("missing value")
    if ptypes.is_datetime64_any_dtype(series):
        return pd.Timestamp(value)
    return float(value)


def _describe_condition(column, op, value) -> str:
    if op in ("is missing", "is not missing"):
        return f"'{column}' {op}"
    if op == "between":
        return f"'{column}' between {value[0]} and {value[1]}"
    if op in ("is one of", "is not one of"):
        values = list(value)
        shown = ", ".join(map(str, values[:5])) + (", ..." if len(values) > 5 else "")
        return f"'{column}' {op} [{shown}]"
    return f"'{column}' {op} {value}"


def filter_rows(df: pd.DataFrame, column: str, op: str, value=None):
    """KEEP only the rows that satisfy the condition.

    op: > >= < <= == != | between (value=(low, high)) | is one of / is not one of
        (value=list) | contains (value=text) | is missing / is not missing
    """
    _require_column(df, column)
    series = df[column]
    is_number = _is_numeric(series)
    is_date = ptypes.is_datetime64_any_dtype(series)

    if op == "is missing":
        mask = series.isna()
    elif op == "is not missing":
        mask = series.notna()
    elif op in _COMPARATORS or op == "between":
        if not (is_number or is_date):
            raise CleaningError(f"'{op}' needs a numeric or date column.")
        try:
            if op == "between":
                low, high = (_coerce(series, v) for v in value)
                if low > high:
                    raise CleaningError("'From' must not be larger than 'To'.")
                mask = series.between(low, high)          # inclusive on both ends
            else:
                mask = _COMPARATORS[op](series, _coerce(series, value))
        except (TypeError, ValueError):
            raise CleaningError("Please provide a valid value for this filter.") from None
    elif op in ("is one of", "is not one of"):
        values = list(value) if value is not None else []
        if not values:
            raise CleaningError("Select at least one value.")
        mask = series.isin(values)
        if op == "is not one of":
            mask = ~mask
    elif op == "contains":
        text = "" if value is None else str(value)
        if not text.strip():
            raise CleaningError("Type the text to search for.")
        mask = series.astype("string").str.contains(text, case=False, regex=False, na=False)
    else:
        raise CleaningError(f"Unknown filter '{op}'.")

    keep = mask.to_numpy(dtype=bool, na_value=False)
    kept = int(keep.sum())
    if kept == len(df):
        return df.copy(), None
    if kept == 0:
        raise CleaningError("This filter would remove every row. Please adjust the condition.")
    result = df.loc[keep].copy()
    removed = len(df) - kept
    details = (f"Kept {kept} of {len(df)} rows where {_describe_condition(column, op, value)} "
               f"({removed} removed)")
    return result, _make_log("Filter rows", column, details, df, result, removed)


# --------------------------------------------------------------------------
# 4. Calculated columns (structured form: no free-text code is ever executed)
# --------------------------------------------------------------------------
def add_calculated_column(df: pd.DataFrame, new_name: str, left: str, op: str,
                          right, right_is_column: bool = True):
    new_name = str(new_name).strip()
    if not new_name:
        raise CleaningError("Please enter a name for the new column.")
    if new_name in df.columns:
        raise CleaningError(f"A column named '{new_name}' already exists.")
    if op not in CALC_OPERATORS:
        raise CleaningError(f"Unknown operation '{op}'.")
    _require_column(df, left)
    if not _is_numeric(df[left]):
        raise CleaningError(f"'{left}' is not numeric.")
    a = df[left].astype("float64")

    if right_is_column:
        _require_column(df, right)
        if not _is_numeric(df[right]):
            raise CleaningError(f"'{right}' is not numeric.")
        b = df[right].astype("float64")
        missing_input = a.isna() | b.isna()
    else:
        try:
            b = float(right)
        except (TypeError, ValueError):
            raise CleaningError(f"'{right}' is not a valid number.") from None
        missing_input = a.isna()

    if op == "+":
        value = a + b
    elif op == "-":
        value = a - b
    elif op == "*":
        value = a * b
    elif op == "/":
        value = a / b
    else:
        value = a / b * 100
    value = value.replace([np.inf, -np.inf], np.nan)      # division by zero -> missing
    undefined = int((value.isna() & ~missing_input).sum())

    result = df.copy()
    result[new_name] = value.to_numpy()
    details = f"Added '{new_name}' = {left} {_CALC_SYMBOLS[op]} {right}"
    if undefined:
        details += (f". WARNING: {undefined} result(s) could not be calculated "
                    "(division by zero) and are missing")
    return result, _make_log("Add calculated column", new_name, details, df, result,
                             int(value.notna().sum()))


# --------------------------------------------------------------------------
# 5. Date features
# --------------------------------------------------------------------------
def _date_part(series: pd.Series, part: str):
    if part == "year":
        return series.dt.year
    if part == "quarter":
        return series.dt.quarter
    if part == "month":
        return series.dt.month
    if part == "month_name":
        return series.dt.month_name()
    if part == "year_month":
        return series.dt.strftime("%Y-%m")
    if part == "week":
        return series.dt.isocalendar().week.astype("float64")
    if part == "day":
        return series.dt.day
    if part == "weekday":
        return series.dt.day_name()
    # is_weekend -> text label; keep missing dates missing
    labels = np.where(series.dt.day_of_week >= 5, "Weekend", "Weekday")
    return pd.Series(labels, index=series.index).where(series.notna())


def extract_date_parts(df: pd.DataFrame, column: str, parts):
    """Add new columns such as order_date_year, order_date_month_name ..."""
    _require_column(df, column)
    if not ptypes.is_datetime64_any_dtype(df[column]):
        raise CleaningError(f"'{column}' is not a date column. Convert it to Date / time first.")
    parts = list(dict.fromkeys(parts))
    if not parts:
        raise CleaningError("Select at least one date part.")
    unknown = [p for p in parts if p not in DATE_PARTS]
    if unknown:
        raise CleaningError(f"Unknown date part(s): {', '.join(unknown)}")
    new_names = {part: f"{column}_{part}" for part in parts}
    clash = [name for name in new_names.values() if name in df.columns]
    if clash:
        raise CleaningError(f"These columns already exist: {', '.join(clash)}")

    result = df.copy()
    for part, name in new_names.items():
        result[name] = _date_part(df[column], part).to_numpy()
    details = f"Added {len(parts)} column(s) from '{column}': {', '.join(new_names.values())}"
    return result, _make_log("Extract date parts", column, details, df, result,
                             len(df) * len(parts))

# --------------------------------------------------------------------------
# 6. Export (bytes for Streamlit download buttons)
# --------------------------------------------------------------------------
_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def _is_formula_like(value) -> bool:
    """True for text that Excel could run as a formula ('=1+1', '@cmd', '-abc').

    A number stored as text ('-5', '-1,200') is NOT treated as a formula.
    """
    if not isinstance(value, str) or len(value) < 2 or not value.startswith(_FORMULA_PREFIXES):
        return False
    try:
        float(value.replace(",", ""))
        return False
    except ValueError:
        return True


def sanitize_for_export(df: pd.DataFrame) -> pd.DataFrame:
    """A copy that is safe to open in Excel.

    Text that looks like a formula gets a leading apostrophe, so Excel shows it as text
    instead of running it ('formula injection'). The input is never modified.
    """
    out = df.copy()
    out.columns = [f"'{c}" if _is_formula_like(str(c)) else c for c in out.columns]
    for position in range(out.shape[1]):
        series = out.iloc[:, position]
        if ptypes.is_object_dtype(series) or ptypes.is_string_dtype(series):
            flagged = series.map(_is_formula_like).astype(bool)
            if flagged.any():
                out.isetitem(position, series.where(~flagged, "'" + series.astype(str)))
    return out


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """CSV with a BOM ('utf-8-sig') so Excel shows special characters correctly."""
    return sanitize_for_export(df).to_csv(index=False).encode("utf-8-sig")


def to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Cleaned data") -> bytes:
    if len(df) > EXCEL_MAX_ROWS:
        raise CleaningError("This dataset has too many rows for one Excel sheet. Use CSV.")
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        sanitize_for_export(df).to_excel(writer, index=False, sheet_name=sheet_name)
    return buffer.getvalue()