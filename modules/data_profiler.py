"""Automated data profiling: overview, column stats, issues and quality score.

Pure functions only: every function takes a DataFrame and returns plain results.
"""
import pandas as pd
from pandas.api import types as ptypes

from modules.data_loader import looks_date, looks_numeric

# How much each part contributes to the overall data-quality score (sums to 1.0)
SCORE_WEIGHTS = {
    "Completeness": 0.40,   # how few missing cells
    "Uniqueness": 0.20,     # how few duplicate rows
    "Validity": 0.20,       # how many columns have the right data type
    "Consistency": 0.20,    # how few categories with spelling/case variants
}
# Issue kinds that count against the "Validity" score
_VALIDITY_ISSUES = ("type_mismatch", "empty_column")


# --------------------------------------------------------------------------
# Overview and per-column profile
# --------------------------------------------------------------------------
def get_overview(df: pd.DataFrame) -> dict:
    """Headline numbers about the dataset."""
    rows, columns = df.shape
    total_cells = rows * columns
    missing_cells = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    return {
        "rows": rows,
        "columns": columns,
        "total_cells": total_cells,
        "missing_cells": missing_cells,
        "missing_pct": round(missing_cells / total_cells * 100, 2) if total_cells else 0.0,
        "duplicate_rows": duplicate_rows,
        "duplicate_pct": round(duplicate_rows / rows * 100, 2) if rows else 0.0,
        "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 ** 2), 2),
    }


def get_column_profile(df: pd.DataFrame, column_types: dict) -> pd.DataFrame:
    """One row per column: type, dtype, missing %, unique values, example."""
    rows = []
    total = len(df)
    for column in df.columns:
        series = df[column]
        missing = int(series.isna().sum())
        non_null = series.dropna()
        rows.append({
            "Column": column,
            "Detected Type": column_types.get(column, "unknown"),
            "Stored As": str(series.dtype),
            "Missing": missing,
            "Missing %": round(missing / total * 100, 2) if total else 0.0,
            "Unique": int(series.nunique()),
            "Example": str(non_null.iloc[0]) if not non_null.empty else "",
        })
    return pd.DataFrame(rows)


def get_numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """describe() for numeric columns plus skewness."""
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        return pd.DataFrame()
    summary = numeric.describe().T
    summary["skew"] = numeric.skew()
    summary.index.name = "Column"
    return summary.round(2).reset_index()


def get_categorical_summary(df: pd.DataFrame, column_types: dict) -> pd.DataFrame:
    """Most frequent value for each categorical column."""
    rows = []
    for column, kind in column_types.items():
        if kind != "categorical" or column not in df.columns:
            continue
        counts = df[column].value_counts(dropna=True)
        if counts.empty:
            continue
        rows.append({
            "Column": column,
            "Unique": len(counts),
            "Top Value": str(counts.index[0]),
            "Top Count": int(counts.iloc[0]),
            "Top %": round(counts.iloc[0] / counts.sum() * 100, 1),
        })
    return pd.DataFrame(rows)


def find_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Rows that repeat an earlier row (the first copy is not counted)."""
    return df[df.duplicated(keep="first")]


# --------------------------------------------------------------------------
# Issue detection
# --------------------------------------------------------------------------
def detect_type_issues(df: pd.DataFrame) -> list:
    """Find columns whose stored type does not match their content."""
    issues = []
    for column in df.columns:
        series = df[column]
        if series.isna().all():
            issues.append({"column": column, "kind": "empty_column",
                           "issue": "Column is completely empty",
                           "suggestion": "Drop this column"})
        elif ptypes.is_object_dtype(series) or ptypes.is_string_dtype(series):
            if looks_numeric(series):
                issues.append({"column": column, "kind": "type_mismatch",
                               "issue": "Numbers stored as text",
                               "suggestion": "Convert to numeric"})
            elif looks_date(series):
                issues.append({"column": column, "kind": "type_mismatch",
                               "issue": "Dates stored as text",
                               "suggestion": "Convert to datetime"})
        if len(df) > 1 and series.nunique(dropna=True) == 1:
            issues.append({"column": column, "kind": "constant_column",
                           "issue": "Column has only one distinct value",
                           "suggestion": "Consider dropping it (no analytical value)"})
    return issues


def detect_inconsistent_categories(df: pd.DataFrame, max_unique: int = 200) -> dict:
    """Find labels that differ only by case or spacing ('North', 'north ', 'NORTH').

    Returns {column: {normalised_label: [variant1, variant2, ...]}}.
    """
    result = {}
    for column in df.columns:
        series = df[column]
        if not (ptypes.is_object_dtype(series) or ptypes.is_string_dtype(series)):
            continue
        values = series.dropna().astype(str)
        if values.empty or values.nunique() > max_unique:
            continue  # skip IDs / free-text columns
        normalised = (values.str.strip().str.lower()
                      .str.replace(r"\s+", " ", regex=True))
        groups = pd.DataFrame({"raw": values, "norm": normalised}).groupby("norm")["raw"].unique()
        variants = {label: sorted(raw.tolist())
                    for label, raw in groups.items() if len(raw) > 1}
        if variants:
            result[column] = variants
    return result


# --------------------------------------------------------------------------
# Data-quality score
# --------------------------------------------------------------------------
def calculate_quality_score(df: pd.DataFrame, type_issues=None, inconsistent=None) -> dict:
    """Weighted 0-100 score built from four transparent components."""
    overview = get_overview(df)
    if type_issues is None:
        type_issues = detect_type_issues(df)
    if inconsistent is None:
        inconsistent = detect_inconsistent_categories(df)

    n_columns = max(overview["columns"], 1)

    completeness = 100 - overview["missing_pct"]
    uniqueness = 100 - overview["duplicate_pct"]

    invalid_columns = {i["column"] for i in type_issues if i["kind"] in _VALIDITY_ISSUES}
    validity = 100 * (1 - len(invalid_columns) / n_columns)

    text_columns = [c for c in df.columns
                    if ptypes.is_object_dtype(df[c]) or ptypes.is_string_dtype(df[c])]
    consistency = (100 * (1 - len(inconsistent) / len(text_columns))) if text_columns else 100.0

    components = {
        "Completeness": round(completeness, 1),
        "Uniqueness": round(uniqueness, 1),
        "Validity": round(validity, 1),
        "Consistency": round(consistency, 1),
    }
    overall = round(sum(components[name] * SCORE_WEIGHTS[name] for name in components), 1)

    if overall >= 90:
        grade = "Excellent"
    elif overall >= 75:
        grade = "Good"
    elif overall >= 60:
        grade = "Fair"
    else:
        grade = "Poor"

    return {"overall": overall, "grade": grade,
            "components": components, "weights": SCORE_WEIGHTS}