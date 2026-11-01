"""DataPilot AI - main Streamlit application.

app.py only handles the user interface. All real logic lives in modules/.
"""
from pathlib import Path

import pandas as pd
from pandas.api import types as ptypes
import plotly.express as px
import streamlit as st

from modules.data_transformation import (CALC_OPERATORS, DATE_PARTS, DTYPE_TARGETS,
                                         add_calculated_column, change_dtype, drop_columns,
                                         extract_date_parts, filter_rows, rename_columns,
                                         to_csv_bytes, to_excel_bytes)
from modules.eda import (CORRELATION_METHODS, TREND_AGGREGATIONS, TREND_FREQUENCIES, EDAError,
                         categorical_columns, categorical_statistics, category_bar_figure,
                         category_counts, comparison_figure, correlation_heatmap,
                         correlation_matrix, crosstab_figure, crosstab_table, date_columns,
                         distribution_summary, generate_highlights, group_comparison,
                         histogram_figure, missing_bar_figure, missing_pattern_figure,
                         missing_summary, numeric_columns, numeric_statistics,
                         relationship_summary, scatter_figure, top_correlations,
                         trend_figure, trend_over_time, trend_summary, variation_explained,
                         variation_label)
from modules.outlier_detection import (ACTION_LABELS, DEFAULT_THRESHOLDS, OUTLIER_METHODS,
                                       apply_outlier_action, detect_outliers,
                                       get_numeric_columns, get_outlier_rows)
from modules.data_loader import (MAX_FILE_SIZE_MB, DataLoadError,
                                 classify_columns, get_excel_sheets, load_file)

from modules.data_cleaner import (METHOD_LABELS, CleaningError, apply_category_mapping,
                                  apply_missing_treatment, apply_recommended_fills,
                                  changed_columns, convert_to_datetime, convert_to_numeric,
                                  get_available_methods, get_column_rename_map,
                                  preview_column_changes, remove_duplicates,
                                  standardize_column_names, strip_whitespace,
                                  suggest_category_mapping, suggest_missing_treatments,
                                  summarize_change)

from modules.data_profiler import (calculate_quality_score, detect_inconsistent_categories,
                                   detect_type_issues, find_duplicates, get_categorical_summary,
                                   get_column_profile, get_numeric_summary, get_overview)

BASE_DIR = Path(__file__).parent
SAMPLE_PATH = BASE_DIR / "sample_data" / "sample_sales.csv"

st.set_page_config(page_title="DataPilot AI", page_icon="🧭", layout="wide")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def load_css() -> None:
    """Inject our custom CSS file into the page."""
    css_path = BASE_DIR / "assets" / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>",
                    unsafe_allow_html=True)


def init_state() -> None:
    """Create the session-state keys the app uses (only if they don't exist yet)."""
    defaults = {
        "original_df": None,    # untouched copy of the upload - NEVER modified
        "working_df": None,     # the copy we clean and transform
        "filename": None,
        "column_types": {},
        "cleaning_log": [],     # filled in Phase 4
        "loaded_key": None,
        "last_upload_key": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def store_dataset(df: pd.DataFrame, filename: str, key: str) -> None:
    """Save a freshly loaded dataset into session state."""
    st.session_state["original_df"] = df              # preserved forever
    st.session_state["working_df"] = df.copy()        # all edits happen here
    st.session_state["filename"] = filename
    st.session_state["column_types"] = classify_columns(df)
    st.session_state["cleaning_log"] = []
    st.session_state["history"] = []          # <-- NEW: undo history
    st.session_state["loaded_key"] = key


def kpi_card(label: str, value, sub: str = "") -> None:
    """Draw one KPI card."""
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>{sub_html}</div>',
        unsafe_allow_html=True,
    )


def has_data() -> bool:
    return st.session_state["working_df"] is not None


# --------------------------------------------------------------------------
# Page 1: Upload & Preview
# --------------------------------------------------------------------------
def render_upload_page() -> None:
    st.header("📤 Upload & Preview")
    st.caption(f"Upload a CSV or Excel (.xlsx) file. Maximum size: {MAX_FILE_SIZE_MB} MB. "
               "Your original file is never modified.")

    uploaded = st.file_uploader("Choose a file", type=["csv", "xlsx"])

    if st.button("Or try the sample sales dataset"):
        if not SAMPLE_PATH.exists():
            st.error("Sample file not found. Run: python sample_data\\generate_sample_data.py")
        else:
            try:
                df = load_file(SAMPLE_PATH.read_bytes(), SAMPLE_PATH.name)
                store_dataset(df, SAMPLE_PATH.name, f"sample-{SAMPLE_PATH.stat().st_mtime}")
            except DataLoadError as error:
                st.error(str(error))

    if uploaded is not None:
        file_bytes = uploaded.getvalue()
        sheet = None
        if uploaded.name.lower().endswith(".xlsx"):
            try:
                sheets = get_excel_sheets(file_bytes)
                sheet = sheets[0] if len(sheets) == 1 else st.selectbox("Select a sheet", sheets)
            except DataLoadError as error:
                st.error(str(error))
                return

        upload_key = f"{uploaded.name}-{uploaded.size}-{sheet}"
        # Only (re)load when the upload actually changed
        if upload_key != st.session_state["last_upload_key"]:
            try:
                with st.spinner("Reading file..."):
                    df = load_file(file_bytes, uploaded.name, sheet)
                store_dataset(df, uploaded.name, upload_key)
                st.session_state["last_upload_key"] = upload_key
            except DataLoadError as error:
                st.error(str(error))
                return

    if not has_data():
        st.info("👆 Upload a file or click the sample-data button to begin.")
        return

    df = st.session_state["working_df"]
    types = st.session_state["column_types"]

    st.success(f"Loaded **{st.session_state['filename']}**")

    # Dimension KPIs
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Rows", f"{df.shape[0]:,}")
    with c2:
        kpi_card("Columns", f"{df.shape[1]:,}")
    with c3:
        kpi_card("Numeric columns", sum(t == "numeric" for t in types.values()))
    with c4:
        kpi_card("Date columns", sum(t == "date" for t in types.values()))

    st.subheader("First 10 records")
    st.dataframe(df.head(10))

    st.subheader("Detected column types")
    type_table = pd.DataFrame({
        "Column": df.columns,
        "Detected Type": [types.get(c, "unknown") for c in df.columns],
        "Stored As": [str(df[c].dtype) for c in df.columns],
    })
    st.dataframe(type_table)
    st.caption("'Detected Type' is what the column really contains; 'Stored As' is how it "
               "is currently stored. A mismatch (e.g. numeric vs object) means it needs cleaning.")


# --------------------------------------------------------------------------
# Page 2: Data Profile
# --------------------------------------------------------------------------
def render_profile_page() -> None:
    st.header("📊 Data Profile")
    if not has_data():
        st.info("Please upload a dataset first (Upload & Preview page).")
        return

    df = st.session_state["working_df"]
    types = st.session_state["column_types"]

    overview = get_overview(df)
    issues = detect_type_issues(df)
    inconsistent = detect_inconsistent_categories(df)
    score = calculate_quality_score(df, issues, inconsistent)

    # ---- Quality score ----
    st.subheader("Data quality score")
    cols = st.columns(5)
    with cols[0]:
        kpi_card("Overall score", f"{score['overall']}/100", score["grade"])
    for col, (name, value) in zip(cols[1:], score["components"].items()):
        with col:
            kpi_card(name, f"{value}%", f"weight {int(score['weights'][name] * 100)}%")
    with st.expander("How is the score calculated?"):
        st.markdown(
            "- **Completeness (40%)**: 100 minus the % of missing cells\n"
            "- **Uniqueness (20%)**: 100 minus the % of duplicate rows\n"
            "- **Validity (20%)**: % of columns whose stored type matches their content\n"
            "- **Consistency (20%)**: % of text columns without case/spacing variants"
        )

    # ---- Overview ----
    st.subheader("Overview")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Rows × Columns", f"{overview['rows']:,} × {overview['columns']}")
    with c2:
        kpi_card("Missing cells", f"{overview['missing_cells']:,}", f"{overview['missing_pct']}% of all cells")
    with c3:
        kpi_card("Duplicate rows", f"{overview['duplicate_rows']:,}", f"{overview['duplicate_pct']}% of rows")
    with c4:
        kpi_card("Memory", f"{overview['memory_mb']} MB")

    # ---- Detail tabs ----
    tab_cols, tab_missing, tab_dups, tab_stats, tab_issues = st.tabs(
        ["Columns", "Missing values", "Duplicates", "Statistics", "Issues found"])

    with tab_cols:
        st.dataframe(get_column_profile(df, types))

    with tab_missing:
        profile = get_column_profile(df, types)
        missing = profile[profile["Missing"] > 0][["Column", "Missing", "Missing %"]]
        if missing.empty:
            st.success("No missing values found 🎉")
        else:
            fig = px.bar(missing.sort_values("Missing %"), x="Missing %", y="Column",
                         orientation="h", title="Missing values by column (%)")
            st.plotly_chart(fig)
            st.dataframe(missing)

    with tab_dups:
        if overview["duplicate_rows"] == 0:
            st.success("No duplicate rows found 🎉")
        else:
            st.warning(f"{overview['duplicate_rows']} duplicate rows found. "
                       "They are NOT removed yet; you decide in the Cleaning phase.")
            st.dataframe(find_duplicates(df).head(100))

    with tab_stats:
        st.markdown("**Numeric columns**")
        numeric_summary = get_numeric_summary(df)
        if numeric_summary.empty:
            st.info("No numeric columns stored as numbers.")
        else:
            st.dataframe(numeric_summary)
        st.markdown("**Categorical columns**")
        categorical_summary = get_categorical_summary(df, types)
        if categorical_summary.empty:
            st.info("No categorical columns detected.")
        else:
            st.dataframe(categorical_summary)

    with tab_issues:
        st.markdown("**Data type problems**")
        if issues:
            st.dataframe(pd.DataFrame(issues)[["column", "issue", "suggestion"]])
        else:
            st.success("No data type problems detected.")
        st.markdown("**Inconsistent category labels**")
        if inconsistent:
            for column, groups in inconsistent.items():
                with st.expander(f"Column: {column}"):
                    for label, variants in groups.items():
                        shown = ", ".join(f"'{v}'" for v in variants)
                        st.write(f"These all mean **{label}**: {shown}")
        else:
            st.success("No inconsistent labels detected.")

# --------------------------------------------------------------------------
# Page 3: Data Cleaning
# --------------------------------------------------------------------------
MAX_UNDO_STEPS = 5   # how many changes can be undone


def refresh_types() -> None:
    """Re-detect column types after the data changed."""
    st.session_state["column_types"] = classify_columns(st.session_state["working_df"])


def commit_change(new_df: pd.DataFrame, log: dict) -> None:
    """Called when the user clicks Apply: save the change and record it in the log."""
    history = st.session_state.setdefault("history", [])
    history.append((st.session_state["working_df"], list(st.session_state["cleaning_log"])))
    del history[:-MAX_UNDO_STEPS]                      # keep only the latest steps
    st.session_state["working_df"] = new_df
    st.session_state["cleaning_log"].append(log)
    refresh_types()


def undo_last() -> None:
    history = st.session_state.get("history", [])
    if history:
        previous_df, previous_log = history.pop()
        st.session_state["working_df"] = previous_df
        st.session_state["cleaning_log"] = previous_log
        refresh_types()


def reset_to_original() -> None:
    st.session_state["working_df"] = st.session_state["original_df"].copy()
    st.session_state["cleaning_log"] = []
    st.session_state["history"] = []
    refresh_types()


def safe_run(function, *args, **kwargs):
    """Run a cleaner function; show a friendly error instead of crashing."""
    try:
        return function(*args, **kwargs)
    except CleaningError as error:
        st.error(str(error))
        return None, None


def show_preview(before, after, log, key, column=None, confirm_text=None) -> None:
    """Preview of a change plus the Apply button. Nothing is saved until Apply is clicked."""
    st.markdown("**👀 Preview: nothing changes until you click Apply**")
    st.info(log["details"])
    st.dataframe(summarize_change(before, after))
    if column:
        sample = preview_column_changes(before, after, column)
        if not sample.empty:
            st.caption(f"Sample of changed values in '{column}':")
            st.dataframe(sample)
    approved = True
    if confirm_text:                                   # extra safety for destructive steps
        approved = st.checkbox(confirm_text, key=f"confirm_{key}")
    st.button("✅ Apply this change", key=f"apply_{key}", on_click=commit_change,
              args=(after, log), disabled=not approved)


def render_cleaning_page() -> None:
    st.header("🧹 Data Cleaning")
    if not has_data():
        st.info("Please upload a dataset first (Upload & Preview page).")
        return

    df = st.session_state["working_df"]
    types = st.session_state["column_types"]

    # ---- Top bar: undo / reset ----
    left, middle, right = st.columns([1, 1, 3])
    left.button("↩️ Undo last", on_click=undo_last,
                disabled=not st.session_state.get("history"))
    middle.button("🔄 Reset to original", on_click=reset_to_original)
    right.caption("Your original upload is always preserved. Work through the tabs from "
                  "left to right: each change is previewed first and only applied when you approve.")

    # ---- Current state KPIs (original score is calculated once per upload) ----
    if st.session_state.get("orig_score_key") != st.session_state["loaded_key"]:
        st.session_state["orig_score"] = calculate_quality_score(
            st.session_state["original_df"])["overall"]
        st.session_state["orig_score_key"] = st.session_state["loaded_key"]
    overview = get_overview(df)
    score_now = calculate_quality_score(df)["overall"]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Quality score", f"{score_now}/100", f"original: {st.session_state['orig_score']}")
    with c2:
        kpi_card("Rows", f"{overview['rows']:,}")
    with c3:
        kpi_card("Missing cells", f"{overview['missing_cells']:,}", f"{overview['missing_pct']}%")
    with c4:
        kpi_card("Duplicate rows", f"{overview['duplicate_rows']:,}")

    tab_names, tab_types, tab_cats, tab_missing, tab_dups, tab_log = st.tabs([
        "🔤 Names & whitespace", "🔁 Data types", "🏷️ Categories",
        "❓ Missing values", "♊ Duplicates", "📜 Cleaning log"])

    # ============ Names & whitespace ============
    with tab_names:
        st.markdown("#### Standardize column names")
        rename_map = get_column_rename_map(df)
        if not rename_map:
            st.success("Column names are already standardized.")
        else:
            st.dataframe(pd.DataFrame({"Current name": list(rename_map),
                                       "New name": list(rename_map.values())}))
            after, log = safe_run(standardize_column_names, df)
            if after is not None and log is not None:
                show_preview(df, after, log, key="names")

        st.markdown("#### Remove unnecessary whitespace")
        after, log = safe_run(strip_whitespace, df)
        if after is not None:
            if log is None:
                st.success("No extra spaces found in text values.")
            else:
                changed = changed_columns(df, after)
                show_preview(df, after, log, key="whitespace",
                             column=changed[0] if changed else None)

    # ============ Data types ============
    with tab_types:
        mismatches = [i for i in detect_type_issues(df) if i["kind"] == "type_mismatch"]
        if not mismatches:
            st.success("No data type problems detected.")
        else:
            st.dataframe(pd.DataFrame(mismatches)[["column", "issue", "suggestion"]])
            column = st.selectbox("Column to convert", [i["column"] for i in mismatches],
                                  key="type_column")
            issue = next(i for i in mismatches if i["column"] == column)
            if issue["suggestion"] == "Convert to datetime":
                dayfirst = st.checkbox("Dates are day-first (e.g. 31/12/2024)",
                                       key=f"dayfirst_{column}")
                after, log = safe_run(convert_to_datetime, df, column, dayfirst)
            else:
                after, log = safe_run(convert_to_numeric, df, column)
            if after is not None and log is not None:
                show_preview(df, after, log, key="types", column=column)

    # ============ Inconsistent categories ============
    with tab_cats:
        inconsistent = detect_inconsistent_categories(df)
        if not inconsistent:
            st.success("No inconsistent category labels detected.")
        else:
            st.write("These columns contain labels that differ only by case or spacing.")
            column = st.selectbox("Column", list(inconsistent), key="cat_column")
            mapping = suggest_category_mapping(df, column)
            counts = df[column].value_counts()
            table = pd.DataFrame({"Original": list(mapping),
                                  "Standardize to": list(mapping.values()),
                                  "Rows": [int(counts.get(k, 0)) for k in mapping]})
            st.caption("You can edit the 'Standardize to' column. Clear it to skip a variant.")
            edited = st.data_editor(table, key=f"cat_editor_{column}", hide_index=True,
                                    disabled=["Original", "Rows"])
            final_map = {row["Original"]: row["Standardize to"] for _, row in edited.iterrows()
                         if str(row["Standardize to"]).strip()}
            after, log = safe_run(apply_category_mapping, df, column, final_map)
            if after is not None and log is not None:
                show_preview(df, after, log, key="cats", column=column)

    # ============ Missing values ============
    with tab_missing:
        suggestions = suggest_missing_treatments(df, types)
        if not suggestions:
            st.success("No missing values 🎉")
        else:
            table = pd.DataFrame(suggestions)[
                ["column", "kind", "missing", "missing_pct", "method", "reason"]]
            table["method"] = table["method"].map(METHOD_LABELS)
            table.columns = ["Column", "Type", "Missing", "Missing %", "Recommended", "Why"]
            st.dataframe(table)

            st.markdown("#### Option A: apply all safe recommendations at once")
            bulk_after, bulk_log = apply_recommended_fills(df, suggestions)
            if bulk_log is None:
                st.caption("No column has a safe automatic fill. Use Option B below.")
            else:
                show_preview(df, bulk_after, bulk_log, key="bulk_missing")

            st.markdown("#### Option B: choose the treatment for one column")
            column = st.selectbox("Column", [s["column"] for s in suggestions], key="miss_column")
            rec = next(s for s in suggestions if s["column"] == column)
            st.info(f"Recommended: **{METHOD_LABELS[rec['method']]}**. {rec['reason']}")
            methods = get_available_methods(df[column])
            default = methods.index(rec["method"]) if rec["method"] in methods else 0
            method = st.selectbox("Treatment", methods, index=default,
                                  format_func=METHOD_LABELS.get, key=f"miss_method_{column}")
            custom_value = None
            if method == "custom":
                custom_value = st.text_input("Custom value", value=rec.get("custom_value") or "",
                                             key=f"miss_custom_{column}")
            if method == "custom" and not custom_value.strip():
                st.info("Type a value to see the preview.")
            else:
                after, log = safe_run(apply_missing_treatment, df, column, method, custom_value)
                if after is not None and log is not None:
                    destructive = method in ("drop_rows", "drop_column")
                    show_preview(df, after, log, key="single_missing", column=column,
                                 confirm_text="I understand this removes data from the working "
                                              "copy (the original stays safe)" if destructive else None)

    # ============ Duplicates ============
    with tab_dups:
        st.write(f"Fully identical rows: **{int(df.duplicated().sum())}**")
        st.caption("Removing duplicates is optional. If repeats are legitimate (e.g. repeat "
                   "purchases), simply don't apply anything.")
        use_subset = st.checkbox("Compare only some columns (e.g. an ID column)", key="dup_subset_on")
        subset = None
        if use_subset:
            subset = st.multiselect("Columns that define a duplicate", list(df.columns),
                                    key="dup_subset")
        choice = st.radio("Which copies should be kept?",
                          ["Keep the first copy", "Keep the last copy", "Remove all copies"],
                          key="dup_keep")
        keep = {"Keep the first copy": "first", "Keep the last copy": "last",
                "Remove all copies": False}[choice]
        if use_subset and not subset:
            st.info("Select at least one column.")
        else:
            after, log = safe_run(remove_duplicates, df, subset, keep)
            if after is not None:
                if log is None:
                    st.success("No duplicates found for this definition.")
                else:
                    removed = df.loc[~df.index.isin(after.index)]
                    with st.expander("Rows that would be removed"):
                        st.dataframe(removed.head(100))
                    show_preview(df, after, log, key="dups",
                                 confirm_text="I understand these rows will be removed from the "
                                              "working data (the original stays safe)")

    # ============ Cleaning log ============
    with tab_log:
        log_entries = st.session_state["cleaning_log"]
        if not log_entries:
            st.info("No cleaning operations applied yet.")
        else:
            st.dataframe(pd.DataFrame(log_entries)[
                ["time", "operation", "column", "details", "rows_before",
                 "rows_after", "cells_changed"]])
        st.markdown("**Original vs. current data**")
        st.dataframe(summarize_change(st.session_state["original_df"], df))


# --------------------------------------------------------------------------
# Shared helpers for Phase 5 pages
# --------------------------------------------------------------------------
def history_bar(prefix: str) -> None:
    """Undo / reset buttons (the prefix keeps widget keys unique per page)."""
    left, middle, right = st.columns([1, 1, 3])
    left.button("↩️ Undo last", on_click=undo_last, key=f"{prefix}_undo",
                disabled=not st.session_state.get("history"))
    middle.button("🔄 Reset to original", on_click=reset_to_original, key=f"{prefix}_reset")
    right.caption("Every change is previewed first and only applied when you approve. "
                  "Your original upload is always preserved.")


def data_signature(df: pd.DataFrame) -> int:
    """Changes whenever columns or row count change; used in widget keys so that
    dropdowns reset cleanly after you apply a change."""
    return abs(hash((tuple(df.columns), len(df))))


@st.cache_data(show_spinner=False)
def cached_csv(df: pd.DataFrame) -> bytes:
    return to_csv_bytes(df)


@st.cache_data(show_spinner="Building Excel file...")
def cached_excel(df: pd.DataFrame) -> bytes:
    return to_excel_bytes(df)


# --------------------------------------------------------------------------
# Page 4: Outlier Detection
# --------------------------------------------------------------------------
def render_outlier_page() -> None:
    st.header("🎯 Outlier Detection")
    if not has_data():
        st.info("Please upload a dataset first (Upload & Preview page).")
        return

    df = st.session_state["working_df"]
    numeric_cols = get_numeric_columns(df)
    if not numeric_cols:
        st.info("No numeric columns found. If numbers are stored as text, convert them on "
                "the Data Cleaning page first.")
        return
    sig = data_signature(df)
    history_bar("outliers")
    st.caption("An outlier is an unusual value, but not necessarily a mistake (a very large "
               "order can be real). Review the flagged rows before changing anything.")

    # ---- Settings ----
    method = st.radio("Detection method", list(OUTLIER_METHODS), horizontal=True,
                      format_func=OUTLIER_METHODS.get, key=f"out_method_{sig}")
    columns = st.multiselect("Numeric columns to check", numeric_cols, default=numeric_cols,
                             key=f"out_cols_{sig}")
    threshold, contamination = None, 0.05
    if method == "iqr":
        threshold = st.slider("IQR multiplier (1.5 = standard, 3.0 = only extreme values)",
                              1.0, 3.0, DEFAULT_THRESHOLDS["iqr"], 0.1, key=f"out_thr_iqr_{sig}")
    elif method == "zscore":
        threshold = st.slider("Z-score threshold (3.0 = standard)",
                              2.0, 5.0, DEFAULT_THRESHOLDS["zscore"], 0.1,
                              key=f"out_thr_z_{sig}")
    else:
        contamination = st.slider("Expected share of outliers", 0.01, 0.20, 0.05, 0.01,
                                  key=f"out_contam_{sig}")
        st.caption("Isolation Forest always flags about this share of rows as the most unusual "
                   "ones, so raise or lower it to match your expectations.")
    if not columns:
        st.info("Select at least one column.")
        return

    try:
        detection = detect_outliers(df, columns, method, threshold=threshold,
                                    contamination=contamination)
    except CleaningError as error:
        st.error(str(error))
        return

    flagged, total = detection["flagged_rows"], detection["total_rows"]
    summary = detection["summary"]

    # ---- Results ----
    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card("Rows flagged", f"{flagged:,}", f"{flagged / total * 100:.1f}% of {total:,} rows")
    with c2:
        kpi_card("Columns with outliers", int((summary["Outliers"] > 0).sum()))
    with c3:
        kpi_card("Method", OUTLIER_METHODS[method])

    st.subheader("Summary per column")
    st.dataframe(summary)

    chart_options = detection["columns"]
    chart_col = st.selectbox("Column to visualize", chart_options,
                             key=f"out_chart_{sig}_{'|'.join(chart_options)}")
    if chart_col in detection["flags"].columns:
        flagged_values = detection["flags"][chart_col].to_numpy()
    else:                                       # Isolation Forest: one flag for the whole row
        flagged_values = detection["any_mask"].to_numpy()
    plot_df = pd.DataFrame({"row": range(len(df)),
                            "value": df[chart_col].astype("float64").to_numpy(),
                            "status": pd.Series(flagged_values).map(
                                {True: "Potential outlier", False: "Normal"})})
    normal, odd = plot_df[plot_df["status"] == "Normal"], plot_df[plot_df["status"] != "Normal"]
    if len(normal) > 5000:                      # keep the chart fast; ALL outliers are kept
        normal = normal.sample(5000, random_state=0)
    fig = px.scatter(pd.concat([normal, odd]), x="row", y="value", color="status",
                     color_discrete_map={"Normal": "#4F8BF9", "Potential outlier": "#E63946"},
                     title=f"{chart_col}: normal values vs potential outliers", opacity=0.7)
    st.plotly_chart(fig)

    if flagged == 0:
        st.success("No outliers detected with these settings 🎉")
        return

    st.subheader("Flagged rows")
    st.caption("Showing up to 200 rows. 'flagged_in' says which column(s) triggered the flag.")
    st.dataframe(get_outlier_rows(df, detection, limit=200))

    # ---- Treatment ----
    st.subheader("What should happen to these outliers?")
    actions = ["keep", "remove"] if method == "isolation_forest" else list(ACTION_LABELS)
    action = st.radio("Treatment", actions, format_func=ACTION_LABELS.get,
                      key=f"out_action_{sig}_{method}")
    if action == "keep":
        st.info("Outliers are kept: nothing changes.")
        return
    after, log = safe_run(apply_outlier_action, df, detection, action)
    if after is None or log is None:
        return
    show_preview(df, after, log, key=f"outliers_{action}",
                 column=chart_col if action in ("cap", "blank") else None,
                 confirm_text=("I understand these rows are removed from the working copy "
                               "(the original stays safe)") if action == "remove" else None)


# --------------------------------------------------------------------------
# Page 5: Transformation & Export
# --------------------------------------------------------------------------
def filter_value_input(series: pd.Series, op: str, key: str):
    """Draw the right input widget(s) for a filter. Returns (value, is_ready)."""
    if op in ("is missing", "is not missing"):
        return None, True
    non_null = series.dropna()
    is_num = ptypes.is_numeric_dtype(series) and not ptypes.is_bool_dtype(series)
    is_date = ptypes.is_datetime64_any_dtype(series)

    if is_num or is_date:
        if non_null.empty:
            st.warning("This column has no values to filter on.")
            return None, False
        low = float(non_null.min()) if is_num else non_null.min().date()
        high = float(non_null.max()) if is_num else non_null.max().date()

        def pick(label, default, suffix, container):
            if is_num:
                return container.number_input(label, value=default, key=f"{key}_{suffix}")
            return container.date_input(label, value=default, key=f"{key}_{suffix}")

        if op == "between":
            left, right = st.columns(2)
            return (pick("From", low, "from", left), pick("To", high, "to", right)), True
        return pick("Value", low, "value", st), True

    if op in ("is one of", "is not one of"):
        options = sorted(non_null.unique().tolist(), key=str)
        if len(options) > 500:
            st.info("Too many distinct values to list. Use 'contains' instead.")
            return None, False
        chosen = st.multiselect("Values", options, key=f"{key}_values")
        return chosen, bool(chosen)

    text = st.text_input("Text to look for (not case-sensitive)", key=f"{key}_text")
    return text, bool(text.strip())


def render_transformation_page() -> None:
    st.header("🔧 Transformation & Export")
    if not has_data():
        st.info("Please upload a dataset first (Upload & Preview page).")
        return

    df = st.session_state["working_df"]
    sig = data_signature(df)
    history_bar("transform")
    st.caption(f"Current data: {df.shape[0]:,} rows × {df.shape[1]} columns")

    (tab_types, tab_rename, tab_drop, tab_filter,
     tab_calc, tab_dates, tab_download) = st.tabs([
        "🔁 Data types", "✏️ Rename", "🗑️ Drop columns", "🔍 Filter rows",
        "➕ Calculated column", "📅 Date features", "⬇️ Download"])

    # ============ Data types ============
    with tab_types:
        column = st.selectbox("Column", list(df.columns), key=f"tf_type_col_{sig}")
        st.caption(f"Currently stored as: {df[column].dtype}")
        target = st.selectbox("Convert to", list(DTYPE_TARGETS), format_func=DTYPE_TARGETS.get,
                              key=f"tf_type_target_{sig}")
        dayfirst = False
        if target == "datetime":
            dayfirst = st.checkbox("Dates are day-first (e.g. 31/12/2024)",
                                   key=f"tf_dayfirst_{sig}")
        after, log = safe_run(change_dtype, df, column, target, dayfirst)
        if after is not None:
            if log is None:
                st.success("This column already has that type.")
            else:
                show_preview(df, after, log, key="tf_types", column=column)

    # ============ Rename ============
    with tab_rename:
        st.caption("Edit the 'New name' cells you want to change.")
        table = pd.DataFrame({"Current name": list(df.columns), "New name": list(df.columns)})
        edited = st.data_editor(table, hide_index=True, disabled=["Current name"],
                                key=f"tf_rename_{sig}")
        mapping = {row["Current name"]: row["New name"] for _, row in edited.iterrows()
                   if row["New name"] != row["Current name"]}
        if not mapping:
            st.info("Change a name above to see the preview.")
        else:
            after, log = safe_run(rename_columns, df, mapping)
            if after is not None and log is not None:
                show_preview(df, after, log, key="tf_rename")

    # ============ Drop columns ============
    with tab_drop:
        hints = [i["column"] for i in detect_type_issues(df)
                 if i["kind"] in ("empty_column", "constant_column")]
        if hints:
            st.caption("Columns with no analytical value (empty or a single value): "
                       + ", ".join(dict.fromkeys(hints)))
        to_drop = st.multiselect("Columns to remove", list(df.columns), key=f"tf_drop_{sig}")
        if to_drop:
            after, log = safe_run(drop_columns, df, to_drop)
            if after is not None and log is not None:
                show_preview(df, after, log, key="tf_drop",
                             confirm_text="I understand these columns are removed from the "
                                          "working copy (the original stays safe)")
        else:
            st.info("Select one or more columns to see the preview.")

    # ============ Filter rows ============
    with tab_filter:
        column = st.selectbox("Column", list(df.columns), key=f"tf_filter_col_{sig}")
        series = df[column]
        is_num = ptypes.is_numeric_dtype(series) and not ptypes.is_bool_dtype(series)
        if is_num:
            operators = ["between", ">", ">=", "<", "<=", "==", "!=",
                         "is missing", "is not missing"]
        elif ptypes.is_datetime64_any_dtype(series):
            operators = ["between", ">", ">=", "<", "<=", "is missing", "is not missing"]
        else:
            operators = ["is one of", "is not one of", "contains", "is missing", "is not missing"]
        op = st.selectbox("Keep rows where the value", operators,
                          key=f"tf_filter_op_{sig}_{column}")
        value, ready = filter_value_input(series, op, key=f"tf_filter_{sig}_{column}_{op}")
        if ready:
            after, log = safe_run(filter_rows, df, column, op, value)
            if after is not None:
                if log is None:
                    st.success("All rows match this condition, so nothing would be removed.")
                else:
                    show_preview(df, after, log, key="tf_filter",
                                 confirm_text="I understand rows that do not match are removed "
                                              "from the working copy (the original stays safe)")
        else:
            st.info("Enter a value to see the preview.")

    # ============ Calculated column ============
    with tab_calc:
        numeric_cols = get_numeric_columns(df)
        if not numeric_cols:
            st.info("This dataset has no numeric columns.")
        else:
            left = st.selectbox("First column", numeric_cols, key=f"tf_calc_left_{sig}")
            calc_op = st.selectbox("Operation", list(CALC_OPERATORS),
                                   format_func=CALC_OPERATORS.get, key=f"tf_calc_op_{sig}")
            mode = st.radio("The second value is", ["Another column", "A fixed number"],
                            horizontal=True, key=f"tf_calc_mode_{sig}")
            if mode == "Another column":
                right = st.selectbox("Second column", numeric_cols, key=f"tf_calc_right_{sig}")
                right_is_column = True
            else:
                right = st.number_input("Number", value=1.0, key=f"tf_calc_number_{sig}")
                right_is_column = False
            new_name = st.text_input("Name of the new column", placeholder="e.g. gross_revenue",
                                     key=f"tf_calc_name_{sig}")
            if not new_name.strip():
                st.info("Enter a name for the new column to see the preview.")
            else:
                after, log = safe_run(add_calculated_column, df, new_name, left, calc_op,
                                      right, right_is_column)
                if after is not None and log is not None:
                    show_preview(df, after, log, key="tf_calc")
                    shown = [left] + ([right] if right_is_column and right != left else []) \
                        + [new_name.strip()]
                    st.caption("First rows of the new column:")
                    st.dataframe(after[shown].head(5))

    # ============ Date features ============
    with tab_dates:
        date_cols = [c for c in df.columns if ptypes.is_datetime64_any_dtype(df[c])]
        if not date_cols:
            st.info("No real date columns yet. Convert text dates to 'Date / time' in the "
                    "Data types tab (or on the Data Cleaning page) first.")
        else:
            column = st.selectbox("Date column", date_cols, key=f"tf_date_col_{sig}")
            parts = st.multiselect("Parts to extract", list(DATE_PARTS),
                                   default=["year", "month_name", "year_month"],
                                   format_func=DATE_PARTS.get, key=f"tf_date_parts_{sig}")
            if not parts:
                st.info("Select at least one part.")
            else:
                after, log = safe_run(extract_date_parts, df, column, parts)
                if after is not None and log is not None:
                    show_preview(df, after, log, key="tf_dates")
                    new_cols = [c for c in after.columns if c not in df.columns]
                    st.dataframe(after[[column] + new_cols].head(5))

    # ============ Download ============
    with tab_download:
        st.write("Download the current (cleaned and transformed) data. "
                 "Your original file is not touched.")
        stem = Path(st.session_state["filename"]).stem
        d1, d2 = st.columns(2)
        d1.download_button("⬇️ Download CSV", data=cached_csv(df), file_name=f"{stem}_cleaned.csv",
                           mime="text/csv", key=f"dl_csv_{sig}")
        try:
            d2.download_button(
                "⬇️ Download Excel", data=cached_excel(df), file_name=f"{stem}_cleaned.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"dl_xlsx_{sig}")
        except CleaningError as error:
            d2.warning(str(error))
        if st.session_state["cleaning_log"]:
            log_csv = pd.DataFrame(st.session_state["cleaning_log"]).to_csv(index=False)
            st.download_button("⬇️ Download cleaning log (CSV)", data=log_csv.encode("utf-8"),
                               file_name=f"{stem}_cleaning_log.csv", mime="text/csv",
                               key=f"dl_log_{sig}")
        st.markdown("**Original vs. current data**")
        st.dataframe(summarize_change(st.session_state["original_df"], df))


# --------------------------------------------------------------------------
# Page 6: Exploratory Data Analysis
# --------------------------------------------------------------------------
def eda_run(function, *args, **kwargs):
    """Run an EDA function; show a friendly error instead of crashing."""
    try:
        return function(*args, **kwargs)
    except EDAError as error:
        st.error(str(error))
        return None


def render_eda_page() -> None:
    st.header("🔍 Exploratory Analysis")
    if not has_data():
        st.info("Please upload a dataset first (Upload & Preview page).")
        return

    df = st.session_state["working_df"]
    sig = data_signature(df)
    num_cols, cat_cols, date_cols = numeric_columns(df), categorical_columns(df), date_columns(df)
    st.caption("EDA only reads your data: nothing here changes it. For the best results, "
               "clean the data first (Data Cleaning page).")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Rows", f"{df.shape[0]:,}")
    with c2:
        kpi_card("Numeric columns", len(num_cols))
    with c3:
        kpi_card("Categorical columns", len(cat_cols))
    with c4:
        kpi_card("Date columns", len(date_cols))

    with st.expander("💡 Automatic highlights", expanded=True):
        for note in generate_highlights(df):
            st.markdown(f"- {note}")

    (tab_stats, tab_corr, tab_dist, tab_missing,
     tab_cats, tab_trend, tab_compare) = st.tabs([
        "📋 Statistics", "🔗 Correlation", "📈 Distributions", "❓ Missing values",
        "🏷️ Categories", "📅 Trends", "⚖️ Comparisons"])

    # ============ Statistics ============
    with tab_stats:
        st.markdown("**Numeric columns**")
        stats = numeric_statistics(df)
        if stats.empty:
            st.info("No numeric columns. If numbers are stored as text, convert them on the "
                    "Data Cleaning page.")
        else:
            st.dataframe(stats)
            st.caption("skew: 0 = symmetric; above +1 or below -1 = strongly skewed. "
                       "kurtosis: high values mean many extreme values.")
        st.markdown("**Categorical columns**")
        cat_stats = categorical_statistics(df)
        if cat_stats.empty:
            st.info("No categorical columns detected.")
        else:
            st.dataframe(cat_stats)

    # ============ Correlation ============
    with tab_corr:
        if len(num_cols) < 2:
            st.info("At least two numeric columns are needed.")
        else:
            method = st.radio("Method", list(CORRELATION_METHODS), horizontal=True,
                              format_func=CORRELATION_METHODS.get, key=f"eda_corr_method_{sig}")
            chosen = st.multiselect("Columns", num_cols, default=num_cols[:15],
                                    key=f"eda_corr_cols_{sig}")
            corr = eda_run(correlation_matrix, df, chosen, method)
            if corr is not None:
                st.plotly_chart(correlation_heatmap(corr))
                skipped = [c for c in chosen if c not in corr.columns]
                if skipped:
                    st.caption("Skipped (only one distinct value, so correlation is "
                               f"undefined): {', '.join(skipped)}")
                st.markdown("**Strongest relationships**")
                st.dataframe(top_correlations(corr, 10))
                st.caption("Correlation shows that two columns move together. It does not "
                           "prove that one causes the other.")

    # ============ Distributions ============
    with tab_dist:
        if not num_cols:
            st.info("No numeric columns to analyze.")
        else:
            column = st.selectbox("Numeric column", num_cols, key=f"eda_dist_col_{sig}")
            bins = st.slider("Number of bins", 5, 100, 30, key=f"eda_dist_bins_{sig}")
            group_options = ["(none)"] + [c for c in cat_cols if df[c].nunique() <= 15]
            group = st.selectbox("Split by (optional)", group_options,
                                 key=f"eda_dist_group_{sig}")
            summary = eda_run(distribution_summary, df[column])
            if summary is not None:
                d1, d2, d3, d4 = st.columns(4)
                with d1:
                    kpi_card("Mean", f"{summary['mean']:,.2f}")
                with d2:
                    kpi_card("Median", f"{summary['median']:,.2f}")
                with d3:
                    kpi_card("Std deviation", f"{summary['std']:,.2f}")
                with d4:
                    kpi_card("Skew", summary["skew"])
                st.info(f"Shape: **{summary['shape']}**")
            fig = eda_run(histogram_figure, df, column, bins,
                          None if group == "(none)" else group)
            if fig is not None:
                st.plotly_chart(fig)

    # ============ Missing values ============
    with tab_missing:
        missing = missing_summary(df)
        if missing.empty:
            st.success("No missing values 🎉")
        else:
            st.plotly_chart(missing_bar_figure(missing))
            pattern = missing_pattern_figure(df)
            if pattern is not None:
                st.plotly_chart(pattern)
                st.caption("Gaps that line up vertically are missing together (e.g. the same "
                           "customers skipped several questions).")
            st.dataframe(missing)

    # ============ Categories ============
    with tab_cats:
        if not cat_cols:
            st.info("No categorical columns detected.")
        else:
            column = st.selectbox("Categorical column", cat_cols, key=f"eda_cat_col_{sig}")
            top_n = st.slider("Show the top N categories", 3, 30, 10, key=f"eda_cat_top_{sig}")
            table = eda_run(category_counts, df, column, top_n)
            if table is not None:
                st.plotly_chart(category_bar_figure(table, column))
                st.dataframe(table)

    # ============ Trends ============
    with tab_trend:
        if not date_cols:
            st.info("No real date columns yet. Convert text dates to 'Date / time' on the "
                    "Data Cleaning page (Data types tab) first.")
        else:
            t1, t2, t3 = st.columns(3)
            date_col = t1.selectbox("Date column", date_cols, key=f"eda_trend_date_{sig}")
            freq = t2.selectbox("Time step", list(TREND_FREQUENCIES), index=2,
                                format_func=TREND_FREQUENCIES.get, key=f"eda_trend_freq_{sig}")
            agg = t3.selectbox("Measure", list(TREND_AGGREGATIONS),
                               format_func=TREND_AGGREGATIONS.get, key=f"eda_trend_agg_{sig}")
            value_col, label = None, TREND_AGGREGATIONS["count"]
            if agg != "count":
                if not num_cols:
                    st.info("No numeric column to measure. Choose 'Number of rows'.")
                    return_early = True
                else:
                    value_col = st.selectbox("Numeric column", num_cols,
                                             key=f"eda_trend_val_{sig}")
                    label = f"{TREND_AGGREGATIONS[agg]} of {value_col}"
                    return_early = False
            else:
                return_early = False
            if not return_early:
                trend = eda_run(trend_over_time, df, date_col, value_col, freq, agg)
                if trend is not None:
                    info = trend_summary(trend)
                    k1, k2, k3, k4 = st.columns(4)
                    with k1:
                        kpi_card("Periods", info["periods"])
                    with k2:
                        change = info["overall_change_pct"]
                        kpi_card("First → last", f"{change:+.1f}%" if change is not None else "n/a")
                    with k3:
                        kpi_card("Peak", f"{info['peak_value']:,.0f}",
                                 info["peak_period"].strftime("%Y-%m-%d"))
                    with k4:
                        kpi_card("Lowest", f"{info['low_value']:,.0f}",
                                 info["low_period"].strftime("%Y-%m-%d"))
                    st.plotly_chart(trend_figure(trend, label, f"{label} per period"))
                    with st.expander("Trend table"):
                        st.dataframe(trend)

    # ============ Comparisons ============
    with tab_compare:
        mode = st.radio("Compare", ["Numeric by category", "Numeric vs numeric",
                                    "Category vs category"], horizontal=True,
                        key=f"eda_cmp_mode_{sig}")

        if mode == "Numeric by category":
            if not cat_cols or not num_cols:
                st.info("This needs at least one categorical and one numeric column.")
            else:
                a, b = st.columns(2)
                category = a.selectbox("Category", cat_cols, key=f"eda_cmp_cat_{sig}")
                value = b.selectbox("Numeric column", num_cols, key=f"eda_cmp_val_{sig}")
                kind = st.radio("Chart", ["box", "bar"], horizontal=True,
                                format_func={"box": "Box plot (spread)",
                                             "bar": "Bar (average)"}.get,
                                key=f"eda_cmp_kind_{sig}")
                table = eda_run(group_comparison, df, category, value)
                fig = eda_run(comparison_figure, df, category, value, kind)
                if table is not None and fig is not None:
                    eta = variation_explained(df, category, value)
                    st.info(f"'{category}' explains **{eta * 100:.1f}%** of the variation in "
                            f"'{value}' (effect: **{variation_label(eta)}**).")
                    st.plotly_chart(fig)
                    st.dataframe(table)

        elif mode == "Numeric vs numeric":
            if len(num_cols) < 2:
                st.info("At least two numeric columns are needed.")
            else:
                a, b, c = st.columns(3)
                x = a.selectbox("X axis", num_cols, key=f"eda_sc_x_{sig}")
                y = b.selectbox("Y axis", num_cols, index=1, key=f"eda_sc_y_{sig}")
                colors = ["(none)"] + [c_ for c_ in cat_cols if df[c_].nunique() <= 15]
                color = c.selectbox("Colour by (optional)", colors, key=f"eda_sc_color_{sig}")
                info = eda_run(relationship_summary, df, x, y)
                fig = eda_run(scatter_figure, df, x, y, None if color == "(none)" else color)
                if info is not None and fig is not None:
                    st.info(f"Relationship: **{info['strength']}** "
                            f"(Pearson {info['pearson']}, Spearman {info['spearman']}). "
                            f"On average, each +1 in '{x}' goes with "
                            f"{info['slope']:+.3f} in '{y}'.")
                    st.plotly_chart(fig)

        else:
            if len(cat_cols) < 2:
                st.info("At least two categorical columns are needed.")
            else:
                a, b, c = st.columns(3)
                first = a.selectbox("Rows", cat_cols, key=f"eda_ct_a_{sig}")
                second = b.selectbox("Columns", cat_cols, index=1, key=f"eda_ct_b_{sig}")
                percent = c.checkbox("Show % of each row", key=f"eda_ct_pct_{sig}")
                table = eda_run(crosstab_table, df, first, second, percent)
                if table is not None:
                    st.plotly_chart(crosstab_figure(table, first, second, percent))
                    st.dataframe(table)

# --------------------------------------------------------------------------
# Main: sidebar navigation
# --------------------------------------------------------------------------



def main() -> None:
    init_state()
    load_css()

    st.sidebar.title("🧭 DataPilot AI")
    st.sidebar.caption("Automated Data Cleaning & Analytics")
    page = st.sidebar.radio("Navigation", ["📤 Upload & Preview", "📊 Data Profile",
                                           "🧹 Data Cleaning", "🎯 Outlier Detection",
                                           "🔧 Transformation & Export",
                                           "🔍 Exploratory Analysis"])
    if has_data():
        st.sidebar.divider()
        st.sidebar.caption(f"Dataset: {st.session_state['filename']}")
        df = st.session_state["working_df"]
        st.sidebar.caption(f"{df.shape[0]:,} rows × {df.shape[1]} columns")

    if page == "📤 Upload & Preview":
        render_upload_page()
    elif page == "📊 Data Profile":
        render_profile_page()
    elif page == "🧹 Data Cleaning":
        render_cleaning_page()
    elif page == "🎯 Outlier Detection":
        render_outlier_page()
    elif page == "🔧 Transformation & Export":
        render_transformation_page()
    else:
        render_eda_page()

main()