"""Tests for outlier detection and treatment."""
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from modules.data_cleaner import CleaningError
from modules.outlier_detection import (apply_outlier_action, detect_outliers,
                                       get_numeric_columns, get_outlier_rows)


def make_df() -> pd.DataFrame:
    """30 ordinary values (10 or 11) plus one huge value in the last row."""
    return pd.DataFrame({"v": [10.0] * 15 + [11.0] * 15 + [1000.0], "label": ["x"] * 31})


def make_cloud() -> pd.DataFrame:
    """200 well-behaved 2-D points plus two extreme rows at the end."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "a": np.append(rng.normal(0, 1, 200), [50, 60]),
        "b": np.append(rng.normal(0, 1, 200), [50, -60]),
    })


def test_get_numeric_columns():
    assert get_numeric_columns(make_df()) == ["v"]


def test_iqr_flags_extreme_value():
    detection = detect_outliers(make_df(), ["v"], "iqr")
    assert detection["flagged_rows"] == 1
    assert bool(detection["flags"]["v"].iloc[-1])
    assert detection["summary"].loc[0, "Outliers"] == 1
    assert detection["bounds"]["v"] == pytest.approx((8.5, 12.5))


def test_zscore_flags_extreme_value():
    detection = detect_outliers(make_df(), ["v"], "zscore")
    assert detection["flagged_rows"] == 1
    assert bool(detection["flags"]["v"].iloc[-1])


def test_iqr_zero_spread_is_skipped_with_a_note():
    df = pd.DataFrame({"v": [5.0] * 20 + [100.0]})
    detection = detect_outliers(df, ["v"], "iqr")
    assert detection["flagged_rows"] == 0
    assert "v" not in detection["bounds"]
    assert "IQR is 0" in detection["summary"].loc[0, "Note"]


def test_remove_outliers():
    df = make_df()
    detection = detect_outliers(df, ["v"], "iqr")
    result, log = apply_outlier_action(df, detection, "remove")
    assert len(result) == 30
    assert 1000.0 not in result["v"].tolist()
    assert log["rows_before"] == 31 and log["rows_after"] == 30


def test_cap_outliers():
    df = make_df()
    detection = detect_outliers(df, ["v"], "iqr")
    result, log = apply_outlier_action(df, detection, "cap")
    assert len(result) == 31
    assert result["v"].max() == pytest.approx(12.5)
    assert log["cells_changed"] == 1


def test_set_outliers_to_missing():
    df = make_df()
    detection = detect_outliers(df, ["v"], "iqr")
    result, log = apply_outlier_action(df, detection, "blank")
    assert len(result) == 31
    assert result["v"].isna().sum() == 1
    assert pd.isna(result["v"].iloc[-1])


def test_keep_does_nothing():
    df = make_df()
    detection = detect_outliers(df, ["v"], "iqr")
    result, log = apply_outlier_action(df, detection, "keep")
    assert log is None
    assert_frame_equal(result, df)


def test_isolation_forest_flags_extreme_rows():
    df = make_cloud()
    detection = detect_outliers(df, ["a", "b"], "isolation_forest", contamination=0.02)
    assert detection["flags"].shape[1] == 1
    assert detection["any_mask"].iloc[-2:].all()
    assert 2 <= detection["flagged_rows"] <= 8
    result, _ = apply_outlier_action(df, detection, "remove")
    assert len(result) == len(df) - detection["flagged_rows"]


def test_isolation_forest_cannot_cap():
    df = make_cloud()
    detection = detect_outliers(df, ["a", "b"], "isolation_forest", contamination=0.02)
    with pytest.raises(CleaningError):
        apply_outlier_action(df, detection, "cap")


def test_validation_errors():
    df = make_df()
    with pytest.raises(CleaningError):
        detect_outliers(df, ["v"], "magic")
    with pytest.raises(CleaningError):
        detect_outliers(df, ["label"], "iqr")      # text column
    with pytest.raises(CleaningError):
        detect_outliers(df, [], "iqr")
    with pytest.raises(CleaningError):
        detect_outliers(df, ["nope"], "iqr")


def test_original_is_never_modified():
    df = make_df()
    snapshot = df.copy()
    for action in ("remove", "cap", "blank", "keep"):
        detection = detect_outliers(df, ["v"], "iqr")
        apply_outlier_action(df, detection, action)
    assert_frame_equal(df, snapshot)


def test_get_outlier_rows():
    df = make_df()
    rows = get_outlier_rows(df, detect_outliers(df, ["v"], "iqr"))
    assert len(rows) == 1
    assert rows.columns[0] == "flagged_in"
    assert rows["flagged_in"].iloc[0] == "v"