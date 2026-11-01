"""Tests for the data loader's type detection and the profiler."""
import pandas as pd

from modules.data_loader import classify_columns
from modules.data_profiler import (calculate_quality_score, detect_inconsistent_categories,
                                   detect_type_issues, get_overview)


def make_dirty_df() -> pd.DataFrame:
    """Small dataset with known problems: 1 duplicate, 1 missing, text numbers, mixed labels."""
    return pd.DataFrame({
        "id": [1, 2, 3, 3],
        "region": ["North", "north ", "South", "South"],
        "amount": ["10", "20", "30", "30"],        # numbers stored as text
        "score": [1.0, None, 3.0, 3.0],
    })


def test_overview_counts():
    overview = get_overview(make_dirty_df())
    assert overview["rows"] == 4
    assert overview["columns"] == 4
    assert overview["missing_cells"] == 1
    assert overview["duplicate_rows"] == 1


def test_classify_columns():
    types = classify_columns(make_dirty_df())
    assert types["id"] == "numeric"
    assert types["score"] == "numeric"
    assert types["amount"] == "numeric"      # detected despite being stored as text
    assert types["region"] == "categorical"


def test_detects_numeric_stored_as_text():
    issues = detect_type_issues(make_dirty_df())
    flagged = [issue["column"] for issue in issues if issue["kind"] == "type_mismatch"]
    assert flagged == ["amount"]


def test_detects_inconsistent_categories():
    result = detect_inconsistent_categories(make_dirty_df())
    assert "region" in result
    assert "amount" not in result
    assert result["region"]["north"] == ["North", "north "]


def test_quality_score_perfect_data():
    clean = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    assert calculate_quality_score(clean)["overall"] == 100.0


def test_quality_score_drops_for_dirty_data():
    score = calculate_quality_score(make_dirty_df())
    assert 0 <= score["overall"] < 100