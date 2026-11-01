"""Outlier detection (IQR, Z-score, Isolation Forest) and treatment.

No Streamlit code here. Two steps, same contract as data_cleaner:

    detection = detect_outliers(df, columns, method, ...)    # changes NOTHING
    new_df, log = apply_outlier_action(df, detection, action) # returns a modified copy
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from modules.data_cleaner import CleaningError, _is_numeric, _make_log

OUTLIER_METHODS = {
    "iqr": "IQR (box-plot rule)",
    "zscore": "Z-score",
    "isolation_forest": "Isolation Forest (advanced)",
}
ACTION_LABELS = {
    "keep": "Keep outliers (no change)",
    "remove": "Remove rows containing outliers",
    "cap": "Cap values to the limits",
    "blank": "Replace outliers with missing values",
}
DEFAULT_THRESHOLDS = {"iqr": 1.5, "zscore": 3.0}
IF_MAX_FIT_ROWS = 20_000   # fit Isolation Forest on a sample for big files (speed/memory)
IF_MIN_ROWS = 10


def get_numeric_columns(df: pd.DataFrame) -> list:
    """Columns that hold real numbers (True/False columns do not count)."""
    return [c for c in df.columns if _is_numeric(df[c])]


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------
def _compute_bounds(values: pd.Series, method: str, threshold: float):
    """Return (lower, upper, note). Values outside [lower, upper] are outliers."""
    clean = values.dropna()
    clean = clean[np.isfinite(clean.to_numpy())]
    if len(clean) < 4:
        return None, None, "Fewer than 4 values: not enough data"

    if method == "iqr":
        q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
        spread = q3 - q1                                   # the IQR
        if spread == 0:
            return None, None, "IQR is 0 (most values identical): this method is not informative"
        return q1 - threshold * spread, q3 + threshold * spread, ""

    # Z-score: a value is an outlier if it is more than `threshold` standard
    # deviations away from the mean, i.e. mean +/- threshold * std.
    std = clean.std()
    if pd.isna(std) or std == 0:
        return None, None, "All values identical (standard deviation is 0)"
    mean = clean.mean()
    return mean - threshold * std, mean + threshold * std, ""


def _detect_with_bounds(df, columns, method, threshold):
    """IQR / Z-score: check each column separately against its own limits."""
    flag_data, bounds, rows = {}, {}, []
    for column in columns:
        values = df[column].astype("float64")
        lower, upper, note = _compute_bounds(values, method, threshold)
        if lower is None:
            flagged = np.zeros(len(df), dtype=bool)
        else:
            flagged = ((values < lower) | (values > upper)).to_numpy()  # NaN -> False
            bounds[column] = (float(lower), float(upper))
        non_missing = int(values.notna().sum())
        count = int(flagged.sum())
        rows.append({
            "Column": column,
            "Outliers": count,
            "Outlier %": round(count / non_missing * 100, 2) if non_missing else 0.0,
            "Lower limit": round(lower, 2) if lower is not None else None,
            "Upper limit": round(upper, 2) if upper is not None else None,
            "Min": round(float(values.min()), 2) if non_missing else None,
            "Max": round(float(values.max()), 2) if non_missing else None,
            "Note": note,
        })
        flag_data[column] = flagged
    return pd.DataFrame(flag_data, index=df.index), bounds, pd.DataFrame(rows)


def _detect_isolation_forest(df, columns, contamination, random_state):
    """Isolation Forest looks at all chosen columns TOGETHER and flags unusual rows."""
    if not 0 < contamination <= 0.5:
        raise CleaningError("The expected share of outliers must be between 0 and 0.5.")
    data = df[columns].astype("float64").replace([np.inf, -np.inf], np.nan)
    data = data.dropna(axis=1, how="all")
    if data.shape[1] == 0:
        raise CleaningError("The selected columns contain no values.")
    if len(data) < IF_MIN_ROWS:
        raise CleaningError(f"Isolation Forest needs at least {IF_MIN_ROWS} rows.")

    filled = data.fillna(data.median())          # temporary fill: your data is not changed
    model = IsolationForest(n_estimators=100, contamination=contamination,
                            random_state=random_state)
    train = (filled.sample(IF_MAX_FIT_ROWS, random_state=random_state)
             if len(filled) > IF_MAX_FIT_ROWS else filled)
    model.fit(train)
    flagged = model.predict(filled) == -1        # -1 means "anomaly"

    summary = pd.DataFrame([{
        "Column": ", ".join(data.columns),
        "Outliers": int(flagged.sum()),
        "Outlier %": round(float(flagged.mean()) * 100, 2),
        "Lower limit": None, "Upper limit": None, "Min": None, "Max": None,
        "Note": "Multivariate: rows that are unusual across the chosen columns together",
    }])
    return pd.DataFrame({"isolation_forest": flagged}, index=df.index), summary


def detect_outliers(df: pd.DataFrame, columns, method: str = "iqr", threshold=None,
                    contamination: float = 0.05, random_state: int = 42) -> dict:
    """Find outliers WITHOUT changing anything. Returns a 'detection' dictionary."""
    if method not in OUTLIER_METHODS:
        raise CleaningError(f"Unknown outlier method '{method}'.")
    columns = list(columns)
    if not columns:
        raise CleaningError("Select at least one numeric column.")
    for column in columns:
        if column not in df.columns:
            raise CleaningError(f"Column '{column}' was not found.")
        if not _is_numeric(df[column]):
            raise CleaningError(f"Column '{column}' is not numeric.")

    if method == "isolation_forest":
        threshold, bounds = None, {}
        flags, summary = _detect_isolation_forest(df, columns, contamination, random_state)
    else:
        if threshold is None:
            threshold = DEFAULT_THRESHOLDS[method]
        if threshold <= 0:
            raise CleaningError("The threshold must be greater than 0.")
        flags, bounds, summary = _detect_with_bounds(df, columns, method, threshold)

    any_mask = flags.any(axis=1)
    return {
        "method": method, "columns": columns, "threshold": threshold,
        "contamination": contamination, "flags": flags, "bounds": bounds,
        "summary": summary, "any_mask": any_mask,
        "flagged_rows": int(any_mask.sum()), "total_rows": len(df),
    }


def get_outlier_rows(df: pd.DataFrame, detection: dict, limit: int = 200) -> pd.DataFrame:
    """The flagged rows, with an extra first column saying which column(s) flagged them."""
    positions = np.flatnonzero(detection["any_mask"].to_numpy())[:limit]
    if len(positions) == 0:
        return df.iloc[0:0]
    rows = df.iloc[positions].copy()
    flag_part = detection["flags"].iloc[positions]
    labels = [", ".join(flag_part.columns[row]) for row in flag_part.to_numpy()]
    rows.insert(0, "flagged_in", labels)
    return rows


# --------------------------------------------------------------------------
# Treatment
# --------------------------------------------------------------------------
def apply_outlier_action(df: pd.DataFrame, detection: dict, action: str):
    """action: 'keep' | 'remove' | 'cap' | 'blank'.  Returns (new_df, log)."""
    if action not in ACTION_LABELS:
        raise CleaningError(f"Unknown action '{action}'.")
    if action == "keep" or detection["flagged_rows"] == 0:
        return df.copy(), None
    if detection["total_rows"] != len(df):
        raise CleaningError("The data changed since the detection ran. Please run it again.")

    method_name = OUTLIER_METHODS[detection["method"]]
    flags = detection["flags"]
    columns_text = ", ".join(detection["columns"])

    if action == "remove":
        keep = ~detection["any_mask"].to_numpy()
        if not keep.any():
            raise CleaningError("Removing the outliers would leave no rows.")
        result = df.loc[keep].copy()
        removed = len(df) - len(result)
        details = f"Removed {removed} row(s) flagged by {method_name} in: {columns_text}"
        return result, _make_log("Remove outliers", columns_text, details, df, result, removed)

    # cap / blank work column by column, so Isolation Forest (row-level) cannot use them
    if detection["method"] == "isolation_forest":
        raise CleaningError(
            "Isolation Forest flags unusual rows, not unusual values, so there are no limits "
            "to cap and no single column to blank. Use 'Remove rows' or switch to IQR / Z-score.")

    result = df.copy()
    changed, parts = 0, []
    for column in detection["columns"]:
        count = int(flags[column].sum())
        if count == 0:
            continue
        if action == "cap":
            lower, upper = detection["bounds"][column]
            result[column] = df[column].astype("float64").clip(lower=lower, upper=upper)
            parts.append(f"{column} to [{lower:.2f}, {upper:.2f}] ({count})")
        else:
            result[column] = df[column].mask(flags[column].to_numpy())
            parts.append(f"{column} ({count})")
        changed += count

    if action == "cap":
        details = f"Capped {changed} value(s) at the {method_name} limits: " + "; ".join(parts)
        return result, _make_log("Cap outliers", columns_text, details, df, result, changed)
    details = f"Set {changed} outlier value(s) to missing ({method_name}): " + "; ".join(parts)
    return result, _make_log("Set outliers to missing", columns_text, details, df, result, changed)