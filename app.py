"""DataPilot AI - main Streamlit application.

app.py only handles the user interface. All real logic lives in modules/.
"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

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
# Main: sidebar navigation
# --------------------------------------------------------------------------



def main() -> None:
    init_state()
    load_css()

    st.sidebar.title("🧭 DataPilot AI")
    st.sidebar.caption("Automated Data Cleaning & Analytics")
    page = st.sidebar.radio("Navigation", ["📤 Upload & Preview", "📊 Data Profile",
                                           "🧹 Data Cleaning"])
    if has_data():
        st.sidebar.divider()
        st.sidebar.caption(f"Dataset: {st.session_state['filename']}")
        df = st.session_state["working_df"]
        st.sidebar.caption(f"{df.shape[0]:,} rows × {df.shape[1]} columns")

    if page == "📤 Upload & Preview":
        render_upload_page()
    elif page == "📊 Data Profile":
        render_profile_page()
    else:
        render_cleaning_page()



main()