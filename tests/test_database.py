"""Tests for the SQLite history (each test uses its own temporary database file)."""
import pytest

from modules.database import (clear_history, delete_session, get_operations,
                              history_enabled, list_sessions, save_session)

ORIGINAL = {"rows": 520, "columns": 14}
CURRENT = {"rows": 500, "columns": 14}
LOG = [
    {"time": "2026-11-02 10:00:00", "operation": "Trim whitespace", "column": "region",
     "details": "Trimmed 12 cells", "rows_before": 520, "rows_after": 520, "cells_changed": 12},
    {"time": "2026-11-02 10:01:00", "operation": "Remove duplicates", "column": "all",
     "details": "Removed 20 duplicate row(s)", "rows_before": 520, "rows_after": 500,
     "cells_changed": 20},
]


def test_save_and_list_sessions(tmp_path):
    db = tmp_path / "history.sqlite"
    first = save_session("a.csv", ORIGINAL, CURRENT, 80.5, 99.0, LOG, db_path=db)
    second = save_session("b.csv", ORIGINAL, CURRENT, 70.0, 95.0, [], db_path=db)
    sessions = list_sessions(db_path=db)
    assert sessions["id"].tolist() == [second, first]            # newest first
    row = sessions[sessions["id"] == first].iloc[0]
    assert row["filename"] == "a.csv"
    assert row["rows_original"] == 520 and row["rows_current"] == 500
    assert row["quality_current"] == pytest.approx(99.0)
    assert row["operations_count"] == 2


def test_operations_roundtrip_and_order(tmp_path):
    db = tmp_path / "history.sqlite"
    session_id = save_session("a.csv", ORIGINAL, CURRENT, 80.0, 99.0, LOG, db_path=db)
    steps = get_operations(session_id, db_path=db)
    assert steps["operation"].tolist() == ["Trim whitespace", "Remove duplicates"]
    assert steps["step"].tolist() == [1, 2]
    assert steps.loc[1, "rows_after"] == 500


def test_delete_cascade_and_clear(tmp_path):
    db = tmp_path / "history.sqlite"
    first = save_session("a.csv", ORIGINAL, CURRENT, 80.0, 99.0, LOG, db_path=db)
    save_session("b.csv", ORIGINAL, CURRENT, 80.0, 99.0, LOG, db_path=db)
    assert delete_session(first, db_path=db) is True
    assert get_operations(first, db_path=db).empty               # steps were deleted too
    assert delete_session(first, db_path=db) is False            # already gone
    assert len(list_sessions(db_path=db)) == 1
    assert clear_history(db_path=db) == 1
    assert list_sessions(db_path=db).empty


def test_sql_injection_is_harmless(tmp_path):
    db = tmp_path / "history.sqlite"
    nasty = "x'); DROP TABLE sessions;--"
    save_session(nasty, ORIGINAL, CURRENT, 80.0, 99.0, LOG, db_path=db)
    sessions = list_sessions(db_path=db)                         # the table still exists
    assert sessions.loc[0, "filename"] == nasty


def test_history_enabled_flag(monkeypatch):
    monkeypatch.delenv("DATAPILOT_ENABLE_HISTORY", raising=False)
    assert history_enabled() is False
    monkeypatch.setenv("DATAPILOT_ENABLE_HISTORY", "1")
    assert history_enabled() is True
    monkeypatch.setenv("DATAPILOT_ENABLE_HISTORY", "off")
    assert history_enabled() is False


def test_handles_incomplete_log_entries(tmp_path):
    db = tmp_path / "history.sqlite"
    session_id = save_session("a.csv", ORIGINAL, CURRENT, 80.0, 99.0,
                              [{"operation": "Only a name"}], db_path=db)
    steps = get_operations(session_id, db_path=db)
    assert steps.loc[0, "operation"] == "Only a name"
    assert steps.loc[0, "rows_before"] == 0