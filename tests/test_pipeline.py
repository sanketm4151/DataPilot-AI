"""End-to-end test: the whole pipeline on the sample dataset, with no Streamlit involved."""
import io
from pathlib import Path

import pandas as pd
import pytest
from pandas.api import types as ptypes
from pandas.testing import assert_frame_equal

from modules.data_cleaner import (apply_category_mapping, apply_recommended_fills,
                                  convert_to_datetime, convert_to_numeric, remove_duplicates,
                                  standardize_column_names, strip_whitespace,
                                  suggest_category_mapping, suggest_missing_treatments)
from modules.data_loader import classify_columns, load_file
from modules.data_profiler import (calculate_quality_score, detect_inconsistent_categories,
                                   get_overview)
from modules.data_transformation import to_csv_bytes
from modules.database import get_operations, save_session
from modules.outlier_detection import apply_outlier_action, detect_outliers
from modules.report_generator import (build_excel_report, build_pdf_report,
                                      build_report_context, make_report_charts)

SAMPLE = Path(__file__).resolve().parent.parent / "sample_data" / "sample_sales.csv"


@pytest.mark.skipif(not SAMPLE.exists(),
                    reason="run: python sample_data\\generate_sample_data.py")
def test_full_pipeline_on_sample_data(tmp_path):
    original = load_file(SAMPLE.read_bytes(), SAMPLE.name)
    if original.shape != (520, 14):
        pytest.skip("sample_sales.csv was changed: this test expects the generated file")
    snapshot = original.copy()
    log = []

    def step(frame, function, *args):
        new_frame, entry = function(frame, *args)
        if entry:
            log.append(entry)
        return new_frame

    # ---- clean exactly like a user would on the Data Cleaning page ----
    df = step(original, standardize_column_names)
    df = step(df, strip_whitespace)
    df = step(df, convert_to_datetime, "order_date")
    df = step(df, convert_to_numeric, "discount")
    for column in ("region", "payment_method"):
        df = step(df, apply_category_mapping, column, suggest_category_mapping(df, column))
    df = step(df, apply_recommended_fills,
              suggest_missing_treatments(df, classify_columns(df)))
    df = step(df, remove_duplicates)

    assert len(df) == 500                                      # 20 duplicates removed
    assert int(df.isna().sum().sum()) == 0
    assert int(df.duplicated().sum()) == 0
    assert detect_inconsistent_categories(df) == {}
    assert ptypes.is_datetime64_any_dtype(df["order_date"])
    assert ptypes.is_numeric_dtype(df["discount"])
    assert set(df["region"]) == {"North", "South", "East", "West"}

    # ---- outliers: cap impossible ages ----
    detection = detect_outliers(df, ["customer_age"], "iqr")
    assert detection["flagged_rows"] >= 1
    df = step(df, apply_outlier_action, detection, "cap")
    assert df["customer_age"].max() <= detection["bounds"]["customer_age"][1] + 1e-9
    assert len(df) == 500                                      # capping never removes rows

    # ---- the original upload was never touched ----
    assert_frame_equal(original, snapshot)

    # ---- quality improved ----
    before = calculate_quality_score(original)["overall"]
    after = calculate_quality_score(df)["overall"]
    assert after > before and after >= 95

    # ---- reports and exports ----
    context = build_report_context(original, df, log, SAMPLE.name)
    assert build_pdf_report(context, make_report_charts(df, 3)).startswith(b"%PDF-")
    assert build_excel_report(context, df)[:2] == b"PK"
    assert pd.read_csv(io.BytesIO(to_csv_bytes(df)), encoding="utf-8-sig").shape == df.shape

    # ---- history ----
    db = tmp_path / "history.sqlite"
    session_id = save_session(SAMPLE.name, get_overview(original), get_overview(df),
                              before, after, log, db_path=db)
    assert len(get_operations(session_id, db_path=db)) == len(log) >= 6