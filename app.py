"""DataPilot AI - main Streamlit application.

app.py only handles the user interface. All real logic lives in modules/.
"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from modules.data_loader import (MAX_FILE_SIZE_MB, DataLoadError,
                                 classify_columns, get_excel_sheets, load_file)
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
# Main: sidebar navigation
# --------------------------------------------------------------------------
def main() -> None:
    init_state()
    load_css()

    st.sidebar.title("🧭 DataPilot AI")
    st.sidebar.caption("Automated Data Cleaning & Analytics")
    page = st.sidebar.radio("Navigation", ["📤 Upload & Preview", "📊 Data Profile"])

    if has_data():
        st.sidebar.divider()
        st.sidebar.caption(f"Dataset: {st.session_state['filename']}")
        df = st.session_state["working_df"]
        st.sidebar.caption(f"{df.shape[0]:,} rows × {df.shape[1]} columns")

    if page == "📤 Upload & Preview":
        render_upload_page()
    else:
        render_profile_page()


main()