"""PDF (reportlab) and Excel (openpyxl) report generation.

No Streamlit code here. Nothing in this file changes the DataFrames it receives.
Charts are drawn with Matplotlib/Seaborn straight into memory (no files, no browser).
"""
import io
import re
from datetime import datetime
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")          # draw into memory: never opens a window

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from pandas.api import types as ptypes
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (CondPageBreak, Image, KeepTogether, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

from modules.ai_assistant import build_privacy_safe_summary, offline_insights
from modules.dashboard import (_value_label, aggregate_by_category, format_number,
                               identify_roles, recommend_charts)
from modules.data_cleaner import summarize_change
from modules.data_loader import classify_columns
from modules.data_profiler import (calculate_quality_score, detect_inconsistent_categories,
                                   detect_type_issues, get_categorical_summary,
                                   get_column_profile, get_numeric_summary, get_overview)
from modules.eda import (correlation_matrix, missing_summary, relationship_summary,
                         trend_over_time)

EXCEL_DATA_MAX_ROWS = 200_000      # above this the Excel report skips the data sheet
CONTENT_WIDTH = A4[0] - 4 * cm     # printable width (2 cm margins)
BLUE, RED, NAVY = "#4F8BF9", "#E63946", "#1F3A68"


# --------------------------------------------------------------------------
# 1. Collect everything the reports need (one dictionary, built once)
# --------------------------------------------------------------------------
def build_report_context(original_df: pd.DataFrame, df: pd.DataFrame, cleaning_log,
                         filename: str, ai_result=None) -> dict:
    """Gather all facts for the report. ai_result = insights from the AI page (optional)."""
    types = classify_columns(df)
    issues = detect_type_issues(df)
    inconsistent = detect_inconsistent_categories(df)
    log = list(cleaning_log or [])

    if ai_result and ai_result.get("insights"):
        ai = {key: list(ai_result.get(key, []))
              for key in ("insights", "unusual", "recommendations")}
        ai.update(source=ai_result.get("source", "offline"), model=ai_result.get("model"))
    else:                                    # no AI insights available: use the offline ones
        ai = offline_insights(build_privacy_safe_summary(df, None, log))

    return {
        "filename": str(filename),
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "overview": get_overview(df),
        "original_overview": get_overview(original_df),
        "quality": calculate_quality_score(df, issues, inconsistent),
        "original_quality": calculate_quality_score(original_df),
        "profile": get_column_profile(df, types),
        "numeric_stats": get_numeric_summary(df),
        "categorical_stats": get_categorical_summary(df, types),
        "missing": missing_summary(df),
        "change": summarize_change(original_df, df),
        "cleaning_log": log,
        "remaining_issues": issues,
        "inconsistent_columns": list(inconsistent),
        "ai": ai,
    }


# --------------------------------------------------------------------------
# 2. Static charts for the PDF (Matplotlib / Seaborn)
# --------------------------------------------------------------------------
def _png(fig: Figure) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    return buffer.getvalue()


def _render_bar(df, spec):
    calc = spec.get("agg", "sum")
    table = aggregate_by_category(df, spec["category"], spec.get("measure"), calc,
                                  spec.get("top_n", 10))
    label = _value_label(spec.get("measure"), calc)
    title = f"{label} by {spec['category']}"
    fig = Figure(figsize=(8, 4.2))
    ax = fig.subplots()
    ordered = table.iloc[::-1]                         # biggest bar on top
    ax.barh(ordered["Category"], ordered["Value"], color=BLUE)
    ax.set_title(title)
    ax.set_xlabel(label)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    return title, fig


def _render_pie(df, spec):
    calc = spec.get("agg", "count")
    table = aggregate_by_category(df, spec["category"], spec.get("measure"), calc, 7,
                                  group_rest=True)
    title = f"{_value_label(spec.get('measure'), calc)} by {spec['category']}"
    fig = Figure(figsize=(6.5, 4.5))
    ax = fig.subplots()
    ax.pie(table["Value"], labels=table["Category"], autopct="%1.0f%%", startangle=90,
           wedgeprops={"width": 0.55})
    ax.set_title(title)
    return title, fig


def _render_line(df, spec):
    measure = spec.get("measure")
    calc = spec.get("agg", "sum") if measure else "count"
    table = trend_over_time(df, spec["date"], measure if calc != "count" else None,
                            spec.get("freq", "M"), calc)
    label = _value_label(measure, calc)
    title = f"{label} over time"
    fig = Figure(figsize=(8, 4.2))
    ax = fig.subplots()
    ax.plot(table["Period"], table["Value"], color=BLUE, marker="o", markersize=3, label=label)
    if len(table) >= 3:
        ax.plot(table["Period"], table["Moving average (3)"], color=RED, linestyle="--",
                label="3-period moving average")
        ax.legend()
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    return title, fig


def _render_histogram(df, spec):
    values = df[spec["measure"]].dropna().astype("float64")
    title = f"Distribution of {spec['measure']}"
    fig = Figure(figsize=(8, 4.2))
    ax = fig.subplots()
    sns.histplot(values, bins=spec.get("bins", 30), kde=bool(values.nunique() > 1),
                 color=BLUE, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(spec["measure"])
    return title, fig


def _render_scatter(df, spec):
    x, y = spec["x"], spec["y"]
    info = relationship_summary(df, x, y)              # also validates the columns
    data = df[[x, y]].dropna().astype("float64")
    shown = data.sample(2000, random_state=0) if len(data) > 2000 else data
    title = f"{y} vs {x} (correlation r = {info['pearson']})"
    fig = Figure(figsize=(8, 4.6))
    ax = fig.subplots()
    ax.scatter(shown[x], shown[y], s=12, alpha=0.5, color=BLUE)
    xs = np.array([data[x].min(), data[x].max()])
    ax.plot(xs, info["intercept"] + info["slope"] * xs, color=RED, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.grid(alpha=0.3)
    return title, fig


def _render_heatmap(df, spec):
    corr = correlation_matrix(df, spec["columns"])
    title = "Correlation heatmap"
    fig = Figure(figsize=(7, 6))
    ax = fig.subplots()
    sns.heatmap(corr, annot=len(corr) <= 10, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1,
                annot_kws={"size": 7}, ax=ax)
    ax.set_title(title)
    return title, fig


_RENDERERS = {"bar": _render_bar, "pie": _render_pie, "line": _render_line,
              "histogram": _render_histogram, "scatter": _render_scatter,
              "heatmap": _render_heatmap}


def make_report_charts(df: pd.DataFrame, max_charts: int = 6) -> list:
    """[(title, png_bytes), ...] for the most useful charts. A chart that fails is skipped."""
    charts = []
    for spec in recommend_charts(df, identify_roles(df), max_charts):
        renderer = _RENDERERS.get(spec["kind"])
        if renderer is None:
            continue
        try:
            title, fig = renderer(df, spec)
            charts.append((title, _png(fig)))
        except Exception:      # one bad chart must never stop the whole report
            continue
    return charts


# --------------------------------------------------------------------------
# 3. PDF helpers
# --------------------------------------------------------------------------
_PDF_REPLACEMENTS = {"→": "->", "←": "<-", "≥": ">=", "≤": "<=", "≈": "~", "✓": "OK",
                     "\u202f": " "}


def _pdf_text(value) -> str:
    """Make text safe for the PDF's built-in font (Latin characters only)."""
    text = "" if value is None else str(value)
    for old, new in _PDF_REPLACEMENTS.items():
        text = text.replace(old, new)
    text = re.sub(r"[\x00-\x08\x0b-\x1f]", " ", text)           # control characters
    return text.encode("cp1252", errors="replace").decode("cp1252")   # others become '?'


def _p(text, style) -> Paragraph:
    """A Paragraph with user text escaped, so '&' or '<' can never break the PDF."""
    return Paragraph(escape(_pdf_text(text)), style)


def _fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return str(value)
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        return "" if pd.isna(value) else format_number(value)
    return str(value)


def _shorten(text, limit: int) -> str:
    text = str(text)
    return text if len(text) <= limit else text[:limit - 3] + "..."


def _styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("DPTitle", parent=base["Title"], fontSize=22, leading=26,
                                alignment=0, textColor=colors.HexColor(NAVY), spaceAfter=6),
        "h1": ParagraphStyle("DPH1", parent=base["Heading1"], fontSize=15, leading=18,
                             textColor=colors.HexColor(NAVY), spaceBefore=14, spaceAfter=6),
        "h2": ParagraphStyle("DPH2", parent=base["Heading2"], fontSize=11, leading=14,
                             textColor=colors.HexColor(NAVY), spaceBefore=8, spaceAfter=4),
        "body": ParagraphStyle("DPBody", parent=base["BodyText"], fontSize=9.5, leading=13),
        "small": ParagraphStyle("DPSmall", parent=base["BodyText"], fontSize=8, leading=10,
                                textColor=colors.grey),
        "cell": ParagraphStyle("DPCell", parent=base["BodyText"], fontSize=8, leading=10),
        "cell_head": ParagraphStyle("DPCellHead", parent=base["BodyText"], fontSize=8,
                                    leading=10, textColor=colors.white,
                                    fontName="Helvetica-Bold"),
        "bullet": ParagraphStyle("DPBullet", parent=base["BodyText"], fontSize=9.5,
                                 leading=13, leftIndent=14, bulletIndent=4, spaceAfter=3),
    }


def _table(rows: list, widths: list, styles: dict) -> Table:
    """rows[0] is the header. widths are relative column widths."""
    data = [[_p(cell, styles["cell_head"] if r == 0 else styles["cell"]) for cell in row]
            for r, row in enumerate(rows)]
    scale = CONTENT_WIDTH / sum(widths)
    table = Table(data, colWidths=[w * scale for w in widths], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(BLUE)),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#C9CED6")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F6FB")]),
    ]))
    return table


def _bullets(items, styles) -> list:
    return [Paragraph(escape(_pdf_text(item)), styles["bullet"], bulletText="•")
            for item in items]


def _footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillGray(0.4)
    canvas.drawString(2 * cm, 1.2 * cm, "DataPilot AI | automated data analysis report")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


# --------------------------------------------------------------------------
# 4. The PDF report
# --------------------------------------------------------------------------
def build_pdf_report(context: dict, charts=None, title: str = "Data Analysis Report") -> bytes:
    """Build the PDF in memory and return its bytes."""
    styles, charts = _styles(), charts or []
    ov, orig = context["overview"], context["original_overview"]
    q, oq = context["quality"], context["original_quality"]
    story = [
        _p("DataPilot AI", styles["small"]),
        _p(title, styles["title"]),
        _p(f"File: {context['filename']}   |   Generated: {context['generated']}",
           styles["body"]),
        Spacer(1, 6),
        _p("Generated automatically. The original upload was never modified: all cleaning "
           "was done on a working copy.", styles["small"]),
    ]

    # ---- 1. Overview ----
    story.append(_p("1. Dataset overview", styles["h1"]))
    story.append(_table([
        ["Metric", "Original upload", "Current data"],
        ["Rows", _fmt(orig["rows"]), _fmt(ov["rows"])],
        ["Columns", _fmt(orig["columns"]), _fmt(ov["columns"])],
        ["Missing cells", f"{_fmt(orig['missing_cells'])} ({orig['missing_pct']}%)",
         f"{_fmt(ov['missing_cells'])} ({ov['missing_pct']}%)"],
        ["Duplicate rows", f"{_fmt(orig['duplicate_rows'])} ({orig['duplicate_pct']}%)",
         f"{_fmt(ov['duplicate_rows'])} ({ov['duplicate_pct']}%)"],
        ["Memory (MB)", _fmt(orig["memory_mb"]), _fmt(ov["memory_mb"])],
    ], [3, 3, 3], styles))
    profile = context["profile"]
    counts = profile["Detected Type"].value_counts()
    story += [Spacer(1, 6),
              _p("Column types: " + ", ".join(f"{n} {t}" for t, n in counts.items()) + ".",
                 styles["body"]),
              Spacer(1, 4)]
    rows = [["Column", "Detected type", "Stored as", "Missing %", "Unique values"]]
    rows += [[r["Column"], r["Detected Type"], r["Stored As"], _fmt(r["Missing %"]),
              _fmt(r["Unique"])] for _, r in profile.head(30).iterrows()]
    story.append(_table(rows, [4, 2, 2, 1.6, 2], styles))
    if len(profile) > 30:
        story.append(_p(f"... and {len(profile) - 30} more columns.", styles["small"]))

    # ---- 2. Data quality ----
    story += [CondPageBreak(6 * cm), _p("2. Data quality", styles["h1"])]
    rows = [["Component", "Original", "Current", "Weight"],
            ["Overall score", f"{oq['overall']} ({oq['grade']})",
             f"{q['overall']} ({q['grade']})", "100%"]]
    for name, weight in q["weights"].items():
        rows.append([name, f"{oq['components'][name]}%", f"{q['components'][name]}%",
                     f"{round(weight * 100)}%"])
    story.append(_table(rows, [3, 2.5, 2.5, 1.5], styles))
    story += [Spacer(1, 6), _p(
        "Score = 40% completeness (few empty cells) + 20% uniqueness (few duplicate rows) + "
        "20% validity (columns stored with the right type) + 20% consistency (no spelling or "
        "case variants in labels).", styles["small"])]
    notes = [f"{i['column']}: {i['issue']}" for i in context["remaining_issues"]
             if i["kind"] == "type_mismatch"]
    if context["inconsistent_columns"]:
        notes.append("Inconsistent category labels remain in: "
                     + ", ".join(map(str, context["inconsistent_columns"])))
    story.append(_p("Remaining issues", styles["h2"]))
    story += _bullets(notes, styles) if notes else [
        _p("No remaining data-type or label problems were detected.", styles["body"])]

    # ---- 3. Cleaning operations ----
    story += [CondPageBreak(6 * cm), _p("3. Cleaning operations performed", styles["h1"])]
    log = context["cleaning_log"]
    if not log:
        story.append(_p("No cleaning operations were applied.", styles["body"]))
    else:
        rows = [["#", "Operation", "Column", "Details", "Rows", "Cells changed"]]
        for number, entry in enumerate(log[:40], start=1):
            rows.append([number, entry.get("operation", ""), entry.get("column", ""),
                         _shorten(entry.get("details", ""), 230),
                         f"{entry.get('rows_before', 0):,} -> {entry.get('rows_after', 0):,}",
                         _fmt(entry.get("cells_changed", 0))])
        story.append(_table(rows, [0.6, 2.4, 2, 6, 2, 1.4], styles))
        if len(log) > 40:
            story.append(_p(f"... and {len(log) - 40} more operations.", styles["small"]))
    story += [Spacer(1, 8), _p("Before vs after", styles["h2"])]
    rows = [["Metric", "Before", "After", "Change"]]
    rows += [[metric, _fmt(r["Before"]), _fmt(r["After"]), f"{int(r['Change']):+,}"]
             for metric, r in context["change"].iterrows()]
    story.append(_table(rows, [3, 2, 2, 2], styles))

    # ---- 4. Statistical summary ----
    story += [CondPageBreak(6 * cm), _p("4. Statistical summary", styles["h1"])]
    numeric = context["numeric_stats"]
    if numeric.empty:
        story.append(_p("There are no numeric columns.", styles["body"]))
    else:
        story.append(_p("Numeric columns", styles["h2"]))
        rows = [["Column", "Count", "Mean", "Std", "Min", "Median", "Max", "Skew"]]
        rows += [[r["Column"], _fmt(r["count"]), _fmt(r["mean"]), _fmt(r["std"]),
                  _fmt(r["min"]), _fmt(r["50%"]), _fmt(r["max"]), _fmt(r["skew"])]
                 for _, r in numeric.head(15).iterrows()]
        story.append(_table(rows, [3, 1.3, 1.5, 1.5, 1.5, 1.5, 1.5, 1.2], styles))
        if len(numeric) > 15:
            story.append(_p(f"... and {len(numeric) - 15} more numeric columns.",
                            styles["small"]))
    categorical = context["categorical_stats"]
    if not categorical.empty:
        story.append(_p("Categorical columns", styles["h2"]))
        rows = [["Column", "Unique", "Most common value", "Count", "Share %"]]
        rows += [[r["Column"], _fmt(r["Unique"]), r["Top Value"], _fmt(r["Top Count"]),
                  _fmt(r["Top %"])] for _, r in categorical.head(15).iterrows()]
        story.append(_table(rows, [3, 1.3, 4, 1.5, 1.5], styles))
    missing = context["missing"]
    if not missing.empty:
        story.append(_p("Remaining missing values", styles["h2"]))
        rows = [["Column", "Missing", "Missing %"]]
        rows += [[r["Column"], _fmt(r["Missing"]), _fmt(r["Missing %"])]
                 for _, r in missing.head(20).iterrows()]
        story.append(_table(rows, [4, 2, 2], styles))

    # ---- 5. Visualizations ----
    story += [CondPageBreak(8 * cm), _p("5. Key visualizations", styles["h1"])]
    if not charts:
        story.append(_p("No charts were included.", styles["body"]))
    for chart_title, png in charts:
        width, height = ImageReader(io.BytesIO(png)).getSize()
        scale = min(CONTENT_WIDTH / width, 11 * cm / height)
        story.append(KeepTogether([
            _p(chart_title, styles["h2"]),
            Image(io.BytesIO(png), width=width * scale, height=height * scale),
            Spacer(1, 8)]))

    # ---- 6. Insights ----
    ai = context["ai"]
    story += [CondPageBreak(8 * cm), _p("6. Insights and business recommendations",
                                        styles["h1"])]
    source = (f"Gemini ({ai.get('model')})" if ai.get("source") == "gemini"
              else "offline rule-based analysis")
    story += [_p(f"Generated by {source} from aggregated statistics only (no raw rows). "
                 "Automated insights can contain mistakes: verify important figures before "
                 "acting on them.", styles["small"]), Spacer(1, 4)]
    for heading, key in (("Key insights", "insights"), ("Unusual observations", "unusual"),
                         ("Business recommendations", "recommendations")):
        if ai.get(key):
            story.append(_p(heading, styles["h2"]))
            story += _bullets(ai[key], styles)

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=2 * cm, bottomMargin=2 * cm, title=_pdf_text(title),
                            author="DataPilot AI")
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buffer.getvalue()


# --------------------------------------------------------------------------
# 5. The Excel report
# --------------------------------------------------------------------------
def _excel_safe(frame: pd.DataFrame) -> pd.DataFrame:
    """A copy that is safe to write to Excel.

    * Text starting with '=' would be saved as a live formula, so it gets a leading
      apostrophe and stays plain text (protects against 'formula injection').
    * Excel cannot store timezone-aware datetimes, so the timezone is dropped.
    """
    out = frame.copy()
    out.columns = [f"'{c}" if str(c).startswith("=") else c for c in out.columns]
    for column in out.columns:
        series = out[column]
        if isinstance(series.dtype, pd.DatetimeTZDtype):
            out[column] = series.dt.tz_localize(None)
        elif series.dtype == object or ptypes.is_string_dtype(series):
            is_formula = series.map(lambda v: isinstance(v, str) and v.startswith("=")) \
                .astype(bool)
            if is_formula.any():
                out[column] = series.where(~is_formula, "'" + series.astype(str))
    return out


def _cell_text(value) -> str:
    """Text used to measure column width. Missing values (NaN, None, NaT) become ''."""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def _style_sheet(ws, frame: pd.DataFrame, wrap: bool = False) -> None:
    """Bold coloured header, frozen top row and sensible column widths."""
    for cell in ws[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="4F8BF9")
        cell.alignment = Alignment(vertical="center")
    ws.freeze_panes = "A2"
    for position in range(frame.shape[1]):          # by position: safe even with repeated names
        header = str(frame.columns[position])
        lengths = [len(_cell_text(value)) for value in frame.iloc[:200, position]]
        width = max([len(header)] + lengths) + 2
        ws.column_dimensions[get_column_letter(position + 1)].width = min(max(width, 8), 60)
    if wrap:
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(wrap_text=True, vertical="top")

def build_excel_report(context: dict, df: pd.DataFrame) -> bytes:
    """Multi-sheet Excel workbook as bytes."""
    ov, q, oq, ai = (context["overview"], context["quality"], context["original_quality"],
                     context["ai"])
    include_data = len(df) <= EXCEL_DATA_MAX_ROWS
    source = f"Gemini ({ai.get('model')})" if ai.get("source") == "gemini" \
        else "Offline rule-based"

    summary = pd.DataFrame([
        ("File", context["filename"]),
        ("Generated", context["generated"]),
        ("Rows", f"{ov['rows']:,}"),
        ("Columns", ov["columns"]),
        ("Missing cells", f"{ov['missing_cells']:,} ({ov['missing_pct']}%)"),
        ("Duplicate rows", f"{ov['duplicate_rows']:,}"),
        ("Quality score (original)", f"{oq['overall']} ({oq['grade']})"),
        ("Quality score (current)", f"{q['overall']} ({q['grade']})"),
        ("Cleaning operations applied", len(context["cleaning_log"])),
        ("Insights source", source),
        ("Cleaned data sheet", "Included" if include_data else
         f"Skipped: more than {EXCEL_DATA_MAX_ROWS:,} rows (use the CSV download)"),
    ], columns=["Item", "Value"])

    quality = pd.DataFrame(
        [{"Component": "Overall score", "Original": oq["overall"], "Current": q["overall"],
          "Weight %": 100}]
        + [{"Component": name, "Original": oq["components"][name],
            "Current": q["components"][name], "Weight %": round(weight * 100)}
           for name, weight in q["weights"].items()])

    change = context["change"].reset_index().rename(columns={"index": "Metric"})
    log = (pd.DataFrame(context["cleaning_log"]) if context["cleaning_log"]
           else pd.DataFrame({"Note": ["No cleaning operations were applied."]}))

    ai_rows = [(section, text)
               for section, key in (("Key insight", "insights"), ("Unusual", "unusual"),
                                    ("Recommendation", "recommendations"))
               for text in ai.get(key, [])]
    ai_sheet = pd.DataFrame(ai_rows or [("Note", "No insights available.")],
                            columns=["Section", "Text"])

    sheets = [("Summary", summary), ("Data quality", quality), ("Before vs after", change),
              ("Cleaning log", log), ("Column profile", context["profile"]),
              ("Numeric statistics", context["numeric_stats"]),
              ("Categorical statistics", context["categorical_stats"]),
              ("Missing values", context["missing"]), ("AI insights", ai_sheet)]
    if include_data:
        sheets.append(("Cleaned data", df))

    wrapped = {"Summary", "Cleaning log", "AI insights"}
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, frame in sheets:
            if frame is None or frame.empty:
                continue
            safe = _excel_safe(frame)
            safe.to_excel(writer, sheet_name=name, index=False)
            _style_sheet(writer.sheets[name], safe, wrap=name in wrapped)
    return buffer.getvalue()