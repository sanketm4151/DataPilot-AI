"""Safe reading of CSV / Excel files and automatic column-type detection.

This module contains NO Streamlit code, so it can be tested and reused anywhere.
"""
import csv
import io
import os
import warnings
import pandas as pd
from pandas.api import types as ptypes
from dotenv import load_dotenv

load_dotenv()   # lets DATAPILOT_* settings come from .env when running locally


def _env_int(name: str, default: int) -> int:
    """A positive whole number from an environment variable, or the default."""
    try:
        value = int(os.getenv(name, "").strip())
    except ValueError:
        return default
    return value if value > 0 else default


# Limits protect small free hosts. Override with environment variables or Streamlit secrets.
MAX_FILE_SIZE_MB = _env_int("DATAPILOT_MAX_FILE_MB", 50)
MAX_ROWS = _env_int("DATAPILOT_MAX_ROWS", 1_000_000)
MAX_COLUMNS = 300
SUPPORTED_EXTENSIONS = (".csv", ".xlsx")
# Encodings tried in order. utf-8-sig also handles files with an invisible BOM mark.
CSV_ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")
# Characters removed before testing "is this text really a number?" ("$1,200" or "15%")
_NUMERIC_JUNK = r"[,$€£₹%\s]"


class DataLoadError(Exception):
    """Raised with a friendly message when a file cannot be loaded."""


# --------------------------------------------------------------------------
# Reading files
# --------------------------------------------------------------------------
def _check_size(file_bytes: bytes) -> None:
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise DataLoadError(
            f"File is {size_mb:.1f} MB. The limit is {MAX_FILE_SIZE_MB} MB."
        )


def get_excel_sheets(file_bytes: bytes) -> list:
    """Return the sheet names of an Excel file."""
    try:
        with pd.ExcelFile(io.BytesIO(file_bytes), engine="openpyxl") as workbook:
            return workbook.sheet_names
    except Exception as error:
        raise DataLoadError(f"Could not open the Excel file: {error}") from error


def _read_csv(file_bytes: bytes) -> pd.DataFrame:
    """Read a CSV, trying several encodings and auto-detecting the separator."""
    for encoding in CSV_ENCODINGS:
        try:
            try:
                # sep=None lets pandas guess the delimiter (comma, semicolon, tab...)
                return pd.read_csv(io.BytesIO(file_bytes), encoding=encoding,
                                   sep=None, engine="python")
            except csv.Error:
                # Guessing failed (e.g. a one-column file): fall back to a comma
                return pd.read_csv(io.BytesIO(file_bytes), encoding=encoding)
        except UnicodeDecodeError:
            continue  # wrong encoding, try the next one
        except pd.errors.EmptyDataError as error:
            raise DataLoadError("The CSV file is empty.") from error
        except pd.errors.ParserError as error:
            raise DataLoadError(f"The CSV file is malformed: {error}") from error
    raise DataLoadError("Could not decode the CSV file with any known encoding.")


def load_file(file_bytes: bytes, filename: str, sheet_name=None) -> pd.DataFrame:
    """Load a CSV or Excel file (given as raw bytes) into a DataFrame.

    The uploaded file itself is never written to or changed. Cleanup actions taken while
    loading (empty rows removed...) are listed in df.attrs["load_notes"].
    """
    extension = os.path.splitext(filename)[1].lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise DataLoadError(
            f"Unsupported file type '{extension}'. Please upload a .csv or .xlsx file."
        )
    _check_size(file_bytes)

    try:
        if extension == ".csv":
            df = _read_csv(file_bytes)
        else:
            df = pd.read_excel(io.BytesIO(file_bytes),
                               sheet_name=sheet_name if sheet_name else 0,
                               engine="openpyxl")
    except DataLoadError:
        raise
    except Exception as error:
        raise DataLoadError(f"Could not read '{filename}': {error}") from error

    if df.shape[1] > MAX_COLUMNS:
        raise DataLoadError(f"The file has {df.shape[1]} columns. The limit is {MAX_COLUMNS}.")

    notes = []
    empty_rows = int(df.isna().all(axis=1).sum())
    if empty_rows:
        df = df.dropna(how="all")
        notes.append(f"Removed {empty_rows} completely empty row(s) while loading.")
    unnamed_empty = [c for c in df.columns
                     if str(c).startswith("Unnamed:") and df[c].isna().all()]
    if unnamed_empty:
        df = df.drop(columns=unnamed_empty)
        notes.append(f"Removed {len(unnamed_empty)} empty column(s) without a header.")

    if df.empty or df.shape[1] == 0:
        raise DataLoadError("The file has no data rows.")
    if df.shape[0] > MAX_ROWS:
        raise DataLoadError(f"The file has {df.shape[0]:,} rows. The limit is {MAX_ROWS:,}.")

    df = df.reset_index(drop=True)
    df.columns = [str(column) for column in df.columns]     # column names -> text
    df.attrs["load_notes"] = notes
    return df


# --------------------------------------------------------------------------
# Detecting column types
# --------------------------------------------------------------------------
def _sample(values: pd.Series, size: int = 1000) -> pd.Series:
    """Look at a sample only, so detection stays fast on big files."""
    return values.sample(size, random_state=0) if len(values) > size else values


def looks_numeric(series: pd.Series, threshold: float = 0.9) -> bool:
    """True if a text column is really numbers (e.g. '10%', '$1,200', '45')."""
    values = series.dropna()
    if values.empty:
        return False
    cleaned = _sample(values).astype(str).str.replace(_NUMERIC_JUNK, "", regex=True)
    converted = pd.to_numeric(cleaned, errors="coerce")
    return bool(converted.notna().mean() >= threshold)


def looks_date(series: pd.Series, threshold: float = 0.8) -> bool:
    """True if a text column is really dates (e.g. '2024-03-15')."""
    values = series.dropna()
    if values.empty:
        return False
    values = _sample(values).astype(str).str.strip()
    # Dates contain digits; this also stops words like 'May' counting as dates
    if values.str.contains(r"\d").mean() < threshold:
        return False
    # Plain numbers such as '2024' or '12' are numbers, not dates
    if looks_numeric(values):
        return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(values, errors="coerce", format="mixed")
    return bool(parsed.notna().mean() >= threshold)


def classify_columns(df: pd.DataFrame) -> dict:
    """Return {column_name: 'numeric' | 'categorical' | 'date' | 'text' | 'boolean'}."""
    column_types = {}
    for column in df.columns:
        series = df[column]
        if ptypes.is_bool_dtype(series):
            column_types[column] = "boolean"
        elif ptypes.is_datetime64_any_dtype(series):
            column_types[column] = "date"
        elif ptypes.is_numeric_dtype(series):
            column_types[column] = "numeric"
        else:
            non_null = series.dropna()
            if non_null.empty:
                column_types[column] = "text"
            elif looks_numeric(series):
                column_types[column] = "numeric"   # numbers stored as text
            elif looks_date(series):
                column_types[column] = "date"      # dates stored as text
            else:
                unique_count = non_null.nunique()
                unique_ratio = unique_count / len(non_null)
                average_length = non_null.astype(str).str.len().mean()
                # Few distinct, short values -> category. Otherwise free text / IDs.
                if unique_count <= 20 or (unique_ratio <= 0.2 and average_length <= 50):
                    column_types[column] = "categorical"
                else:
                    column_types[column] = "text"
    return column_types