"""Tests for safe file loading and validation."""
import io

import pandas as pd
import pytest

from modules import data_loader
from modules.data_loader import DataLoadError, get_excel_sheets, load_file


def test_load_csv_basic():
    df = load_file(b"a,b\n1,x\n2,y\n", "t.csv")
    assert df.shape == (2, 2)
    assert list(df.columns) == ["a", "b"]
    assert df.attrs["load_notes"] == []


def test_load_csv_semicolon_and_latin1():
    semicolon = load_file(b"name;amount;city\nAna;1;Pune\nBo;2;Goa\n", "t.csv")
    assert semicolon.shape == (2, 3)
    latin = "city,price\nZ\u00fcrich,10\nM\u00fcnchen,20\n".encode("cp1252")
    assert load_file(latin, "t.csv")["city"].tolist() == ["Z\u00fcrich", "M\u00fcnchen"]


def test_load_rejects_bad_extension_empty_and_header_only():
    for payload, name in ((b"x", "t.txt"), (b"", "t.csv"), (b"a,b\n", "t.csv")):
        with pytest.raises(DataLoadError):
            load_file(payload, name)


def test_load_rejects_oversize_and_too_wide(monkeypatch):
    monkeypatch.setattr(data_loader, "MAX_FILE_SIZE_MB", 0)
    with pytest.raises(DataLoadError):
        load_file(b"a,b\n1,2\n", "t.csv")
    monkeypatch.setattr(data_loader, "MAX_FILE_SIZE_MB", 50)
    monkeypatch.setattr(data_loader, "MAX_COLUMNS", 3)
    with pytest.raises(DataLoadError):
        load_file(b"a,b,c,d,e\n1,2,3,4,5\n", "t.csv")


def test_load_drops_empty_rows_and_unnamed_columns():
    df = load_file(b"a,b,\n1,2,\n,,\n3,4,\n", "t.csv")
    assert df.shape == (2, 2)
    assert list(df.columns) == ["a", "b"]
    assert len(df.attrs["load_notes"]) == 2


def test_load_excel_sheets():
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame({"a": [1, 2]}).to_excel(writer, sheet_name="first", index=False)
        pd.DataFrame({"b": [3, 4, 5]}).to_excel(writer, sheet_name="second", index=False)
    data = buffer.getvalue()
    assert get_excel_sheets(data) == ["first", "second"]
    assert load_file(data, "t.xlsx").shape == (2, 1)
    assert load_file(data, "t.xlsx", "second").shape == (3, 1)





def test_env_int_settings(monkeypatch):
    monkeypatch.delenv("DATAPILOT_TEST_LIMIT", raising=False)
    assert data_loader._env_int("DATAPILOT_TEST_LIMIT", 7) == 7
    monkeypatch.setenv("DATAPILOT_TEST_LIMIT", "123")
    assert data_loader._env_int("DATAPILOT_TEST_LIMIT", 7) == 123
    for bad in ("abc", "-4", "0", ""):
        monkeypatch.setenv("DATAPILOT_TEST_LIMIT", bad)
        assert data_loader._env_int("DATAPILOT_TEST_LIMIT", 7) == 7