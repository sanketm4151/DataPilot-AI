"""SQLite history of cleaning sessions (standard library only: nothing to install).

PRIVACY: only METADATA and the cleaning log are stored (file name, row counts, quality
scores, which operations ran). Dataset values are NEVER written to the database.
All SQL uses '?' placeholders, so user-supplied text can never change a query.
"""
import os
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

import pandas as pd

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "datapilot_history.sqlite"

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at       TEXT NOT NULL,
    filename         TEXT NOT NULL,
    rows_original    INTEGER,
    columns_original INTEGER,
    rows_current     INTEGER,
    columns_current  INTEGER,
    quality_original REAL,
    quality_current  REAL,
    operations_count INTEGER
);
CREATE TABLE IF NOT EXISTS operations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    step          INTEGER NOT NULL,
    time          TEXT,
    operation     TEXT,
    column_name   TEXT,
    details       TEXT,
    rows_before   INTEGER,
    rows_after    INTEGER,
    cells_changed INTEGER
);
"""

SESSION_COLUMNS = ["id", "created_at", "filename", "rows_original", "rows_current",
                   "quality_original", "quality_current", "operations_count"]
OPERATION_COLUMNS = ["step", "time", "operation", "column_name", "details",
                     "rows_before", "rows_after", "cells_changed"]


class DatabaseError(Exception):
    """Raised with a friendly message when the history cannot be read or written."""


def history_enabled() -> bool:
    """History is OFF unless DATAPILOT_ENABLE_HISTORY is set (see the module notes above)."""
    return os.getenv("DATAPILOT_ENABLE_HISTORY", "").strip().lower() in ("1", "true", "yes", "on")


def _connect(db_path=None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else Path(os.getenv("DATAPILOT_DB") or DEFAULT_DB_PATH)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA foreign_keys = ON")      # needed for ON DELETE CASCADE
    connection.executescript(SCHEMA)                    # creates tables only if missing
    return connection


def _to_int(value) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def save_session(filename, original_overview: dict, overview: dict, quality_original: float,
                 quality_current: float, cleaning_log, db_path=None) -> int:
    """Store one session and its cleaning steps. Returns the new session id."""
    log = list(cleaning_log or [])
    try:
        with closing(_connect(db_path)) as connection, connection:   # 'with' = one transaction
            cursor = connection.execute(
                "INSERT INTO sessions (created_at, filename, rows_original, columns_original, "
                "rows_current, columns_current, quality_original, quality_current, "
                "operations_count) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), str(filename)[:255],
                 _to_int(original_overview.get("rows")), _to_int(original_overview.get("columns")),
                 _to_int(overview.get("rows")), _to_int(overview.get("columns")),
                 float(quality_original), float(quality_current), len(log)))
            session_id = cursor.lastrowid
            connection.executemany(
                "INSERT INTO operations (session_id, step, time, operation, column_name, "
                "details, rows_before, rows_after, cells_changed) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [(session_id, step, str(entry.get("time", "")), str(entry.get("operation", "")),
                  str(entry.get("column", "")), str(entry.get("details", "")),
                  _to_int(entry.get("rows_before")), _to_int(entry.get("rows_after")),
                  _to_int(entry.get("cells_changed")))
                 for step, entry in enumerate(log, start=1)])
        return int(session_id)
    except sqlite3.Error as error:
        raise DatabaseError(f"Could not save the history: {error}") from error


def list_sessions(limit: int = 50, db_path=None) -> pd.DataFrame:
    """The most recent sessions, newest first."""
    try:
        with closing(_connect(db_path)) as connection:
            rows = connection.execute(
                f"SELECT {', '.join(SESSION_COLUMNS)} FROM sessions "
                "ORDER BY id DESC LIMIT ?", (int(limit),)).fetchall()
    except sqlite3.Error as error:
        raise DatabaseError(f"Could not read the history: {error}") from error
    return pd.DataFrame([tuple(row) for row in rows], columns=SESSION_COLUMNS)


def get_operations(session_id: int, db_path=None) -> pd.DataFrame:
    """The cleaning steps of one session, in the order they were applied."""
    try:
        with closing(_connect(db_path)) as connection:
            rows = connection.execute(
                f"SELECT {', '.join(OPERATION_COLUMNS)} FROM operations "
                "WHERE session_id = ? ORDER BY step", (int(session_id),)).fetchall()
    except sqlite3.Error as error:
        raise DatabaseError(f"Could not read the history: {error}") from error
    return pd.DataFrame([tuple(row) for row in rows], columns=OPERATION_COLUMNS)


def delete_session(session_id: int, db_path=None) -> bool:
    """Delete one session (its steps go with it). True if something was deleted."""
    try:
        with closing(_connect(db_path)) as connection, connection:
            cursor = connection.execute("DELETE FROM sessions WHERE id = ?", (int(session_id),))
            return cursor.rowcount > 0
    except sqlite3.Error as error:
        raise DatabaseError(f"Could not delete the session: {error}") from error


def clear_history(db_path=None) -> int:
    """Delete every session. Returns how many were removed."""
    try:
        with closing(_connect(db_path)) as connection, connection:
            return connection.execute("DELETE FROM sessions").rowcount
    except sqlite3.Error as error:
        raise DatabaseError(f"Could not clear the history: {error}") from error