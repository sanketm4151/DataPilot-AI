"""Automatic dashboard generation: column roles, filters, KPI cards and charts.

No Streamlit code here. Nothing in this file changes the DataFrame it receives.
The same functions work for ANY dataset, not just sales data.
"""
import re

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pandas.api import types as ptypes

from modules.eda import (PALETTE, TREND_FREQUENCIES, EDAError, _check_column, _check_numeric,
                         categorical_columns, correlation_heatmap, correlation_matrix,
                         date_columns, histogram_figure, numeric_columns, scatter_figure,
                         top_correlations, trend_figure, trend_over_time)

MAX_DIMENSION_UNIQUE = 20     # a category with more distinct values is not charted
MAX_CATEGORY_FILTERS = 4      # keep the filter panel tidy
MAX_NUMERIC_FILTERS = 2
MAX_PIE_SLICES = 7            # the rest is grouped as "Other"

# Column-name hints. They only influence ORDER and Total-vs-Average, never eligibility.
MEASURE_HINTS = ("sales", "revenue", "amount", "total", "profit", "income", "spend",
                 "cost", "price", "quantity", "qty", "value")
ADDITIVE_HINTS = ("sales", "revenue", "amount", "total", "profit", "income", "spend",
                  "cost", "quantity", "qty")        # adding these up makes sense

AGG_LABELS = {"sum": "Total", "mean": "Average", "median": "Median", "count": "Number of rows"}
CHART_KINDS = {"bar": "Bar chart", "line": "Line chart (trend)", "pie": "Pie chart",
               "histogram": "Histogram", "scatter": "Scatter plot",
               "heatmap": "Correlation heatmap"}


class DashboardError(EDAError):
    """A friendly error. It is an EDAError, so the app's existing handler shows it."""


# --------------------------------------------------------------------------
# 1. Column roles
# --------------------------------------------------------------------------
def _name_has(name, hints) -> bool:
    lowered = str(name).lower()
    return any(hint in lowered for hint in hints)


def _is_additive(name) -> bool:
    return _name_has(name, ADDITIVE_HINTS)


def _measure_rank(name) -> int:
    """Lower rank = more likely to be the main business measure."""
    lowered = str(name).lower()
    for position, hint in enumerate(MEASURE_HINTS):
        if hint in lowered:
            return position
    return len(MEASURE_HINTS)


def _is_id_like(name, series: pd.Series) -> bool:
    """IDs are numbers too, but summing or averaging them is meaningless."""
    if re.search(r"(^|[\W_])id($|[\W_])", str(name).lower()):
        return True
    values = series.dropna()
    return (len(values) >= 10 and ptypes.is_integer_dtype(series)
            and values.nunique() == len(values))


def identify_roles(df: pd.DataFrame, max_dimension_unique: int = MAX_DIMENSION_UNIQUE) -> dict:
    """Return {'measures': [...], 'dimensions': [...], 'dates': [...]}."""
    measures = [c for c in numeric_columns(df)
                if not _is_id_like(c, df[c]) and df[c].nunique(dropna=True) > 1]
    measures.sort(key=_measure_rank)                       # stable: ties keep column order
    dimensions = sorted(categorical_columns(df, max_dimension_unique),
                        key=lambda c: df[c].nunique(dropna=True))   # fewest values first
    return {"measures": measures, "dimensions": dimensions, "dates": date_columns(df)}


def auto_frequency(series: pd.Series) -> str:
    """Pick a sensible time step so a line chart has a readable number of points."""
    values = series.dropna()
    if values.empty:
        return "M"
    span_days = (values.max() - values.min()).days
    if span_days <= 90:
        return "D"
    if span_days <= 1825:        # up to 5 years
        return "M"
    if span_days <= 5475:        # up to 15 years
        return "Q"
    return "Y"


# --------------------------------------------------------------------------
# 2. Filters
# --------------------------------------------------------------------------
def get_filter_options(df: pd.DataFrame, roles=None) -> dict:
    """What the filter panel should offer, built from the column roles."""
    if roles is None:
        roles = identify_roles(df)
    categories = {c: sorted(df[c].dropna().unique().tolist(), key=str)
                  for c in roles["dimensions"][:MAX_CATEGORY_FILTERS]}
    numeric = {}
    for column in roles["measures"][:MAX_NUMERIC_FILTERS]:
        values = df[column].dropna()
        numeric[column] = (float(values.min()), float(values.max()))
    dates = {}
    for column in roles["dates"][:1]:
        values = df[column].dropna()
        if not values.empty:
            dates[column] = (values.min().date(), values.max().date())
    return {"categories": categories, "numeric": numeric, "dates": dates}


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Return a filtered COPY.

    filters = {"categories": {col: [allowed values]},
               "numeric":    {col: (low, high)},
               "dates":      {col: (first_day, last_day)}}      (end dates are inclusive)
    Rows with a missing value in a filtered column are hidden while that filter is active.
    """
    mask = pd.Series(True, index=df.index)
    for column, values in (filters.get("categories") or {}).items():
        _check_column(df, column)
        mask &= df[column].isin(list(values))
    for column, (low, high) in (filters.get("numeric") or {}).items():
        _check_numeric(df, column)
        mask &= df[column].between(low, high).fillna(False)
    for column, (start, end) in (filters.get("dates") or {}).items():
        _check_column(df, column)
        if not ptypes.is_datetime64_any_dtype(df[column]):
            raise DashboardError(f"'{column}' is not a date column.")
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        mask &= (df[column] >= start) & (df[column] < end + pd.Timedelta(days=1))
    return df.loc[mask.to_numpy(dtype=bool, na_value=False)].copy()


# --------------------------------------------------------------------------
# 3. KPI cards
# --------------------------------------------------------------------------
def format_number(value) -> str:
    """1234 -> '1,234',  2500000 -> '2.50M',  0.5 -> '0.50'."""
    if value is None or pd.isna(value):
        return "n/a"
    value = float(value)
    magnitude = abs(value)
    for limit, divisor, suffix in ((1e9, 1e9, "B"), (1e6, 1e6, "M"), (1e4, 1e3, "K")):
        if magnitude >= limit:
            return f"{value / divisor:.2f}{suffix}"
    if value.is_integer():
        return f"{int(value):,}"
    return f"{value:,.2f}"


def build_kpis(df: pd.DataFrame, roles=None, max_measures: int = 3) -> list:
    """List of {'label', 'value', 'sub'} cards: row count, main measures, latest change."""
    if roles is None:
        roles = identify_roles(df)
    kpis = [{"label": "Records", "value": f"{len(df):,}", "sub": f"{df.shape[1]} columns"}]

    for measure in roles["measures"][:max_measures]:
        if measure not in df.columns:
            continue
        values = df[measure].dropna()
        if values.empty:
            continue
        additive = _is_additive(measure)
        main = values.sum() if additive else values.mean()
        kpis.append({
            "label": f"{'Total' if additive else 'Average'} {measure}",
            "value": format_number(main),
            "sub": f"median {format_number(values.median())} · max {format_number(values.max())}",
        })

    if roles["dates"] and roles["measures"]:
        date_column, primary = roles["dates"][0], roles["measures"][0]
        agg = "sum" if _is_additive(primary) else "mean"
        try:
            table = trend_over_time(df, date_column, primary, "M", agg)
            if len(table) >= 2 and pd.notna(table["Change %"].iloc[-1]):
                latest, previous = table["Period"].iloc[-1], table["Period"].iloc[-2]
                kpis.append({
                    "label": "Latest month vs previous",
                    "value": f"{table['Change %'].iloc[-1]:+.1f}%",
                    "sub": f"{primary}: {latest:%b %Y} vs {previous:%b %Y}",
                })
        except EDAError:
            pass     # not enough dated data for a comparison
    return kpis


# --------------------------------------------------------------------------
# 4. Chart builders
# --------------------------------------------------------------------------
def _check_agg(agg: str) -> None:
    if agg not in AGG_LABELS:
        raise DashboardError(f"Unknown calculation '{agg}'.")


def _value_label(measure, agg: str) -> str:
    if agg == "count" or not measure:
        return AGG_LABELS["count"]
    return f"{AGG_LABELS[agg]} of {measure}"


def aggregate_by_category(df: pd.DataFrame, category: str, measure=None, agg: str = "sum",
                          top_n: int = 10, group_rest: bool = False) -> pd.DataFrame:
    """Table [Category, Value]: one row per category, biggest first.

    group_rest=True folds the categories beyond top_n into 'Other' (sum/count only,
    because an 'Other' average would be meaningless).
    """
    _check_column(df, category)
    _check_agg(agg)
    top_n = max(int(top_n), 1)
    if agg == "count" or not measure:
        agg = "count"
        values = df[category].value_counts(dropna=True)
    else:
        _check_numeric(df, measure)
        if measure == category:
            raise DashboardError("Choose two different columns.")
        data = df[[category, measure]].dropna()
        values = data.groupby(category, observed=True)[measure].agg(agg).astype("float64")
    values = values.sort_values(ascending=False)
    if values.empty:
        raise DashboardError(f"'{category}' has no values to chart.")

    shown, rest = values.head(top_n), values.iloc[top_n:]
    table = pd.DataFrame({"Category": shown.index.astype(str),
                          "Value": shown.to_numpy(dtype="float64")})
    if group_rest and len(rest) and agg in ("sum", "count"):
        other = pd.DataFrame([{"Category": f"Other ({len(rest)} categories)",
                               "Value": float(rest.sum())}])
        table = pd.concat([table, other], ignore_index=True)
    return table


def bar_figure(df, category, measure=None, agg="sum", top_n=10) -> go.Figure:
    table = aggregate_by_category(df, category, measure, agg, top_n)
    label = _value_label(measure, agg)
    fig = px.bar(table, x="Value", y="Category", orientation="h", text_auto=".3s",
                 title=f"{label} by {category}")
    fig.update_traces(marker_color=PALETTE["main"])
    fig.update_layout(yaxis={"categoryorder": "total ascending"},
                      xaxis_title=label, yaxis_title=category)
    return fig


def pie_figure(df, category, measure=None, agg="count",
               max_slices: int = MAX_PIE_SLICES) -> go.Figure:
    if agg not in ("sum", "count"):
        raise DashboardError("A pie chart shows shares, so it needs 'Total' or "
                             "'Number of rows'. Use a bar chart for averages.")
    table = aggregate_by_category(df, category, measure, agg, max_slices, group_rest=True)
    if (table["Value"] < 0).any():
        raise DashboardError("A pie chart cannot show negative values. Use a bar chart.")
    if table["Value"].sum() <= 0:
        raise DashboardError("There is nothing to show in a pie chart.")
    label = _value_label(measure, agg)
    fig = px.pie(table, names="Category", values="Value", hole=0.4,
                 title=f"{label} by {category}")
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return fig


def line_figure(df, date_column, measure=None, agg="sum", freq="M") -> go.Figure:
    _check_agg(agg)
    if not measure:
        agg = "count"
    table = trend_over_time(df, date_column, measure if agg != "count" else None, freq, agg)
    label = _value_label(measure, agg)
    return trend_figure(table, label,
                        f"{label} over time ({TREND_FREQUENCIES[freq].lower()})")


def _need(spec: dict, key: str):
    """Fetch a required chart setting or raise a friendly error."""
    if spec.get(key) in (None, ""):
        raise DashboardError(f"The chart setting '{key}' is missing.")
    return spec[key]


def build_chart(df: pd.DataFrame, spec: dict) -> go.Figure:
    """Turn a chart specification (a plain dict) into a Plotly figure."""
    kind = spec.get("kind")
    if kind == "bar":
        fig = bar_figure(df, _need(spec, "category"), spec.get("measure"),
                         spec.get("agg", "sum"), spec.get("top_n", 10))
    elif kind == "pie":
        fig = pie_figure(df, _need(spec, "category"), spec.get("measure"),
                         spec.get("agg", "count"))
    elif kind == "line":
        fig = line_figure(df, _need(spec, "date"), spec.get("measure"),
                          spec.get("agg", "sum"), spec.get("freq", "M"))
    elif kind == "histogram":
        fig = histogram_figure(df, _need(spec, "measure"), spec.get("bins", 30))
    elif kind == "scatter":
        fig = scatter_figure(df, _need(spec, "x"), _need(spec, "y"), spec.get("color"))
    elif kind == "heatmap":
        fig = correlation_heatmap(correlation_matrix(df, _need(spec, "columns")))
    else:
        raise DashboardError(f"Unknown chart type '{kind}'.")
    if spec.get("title"):
        fig.update_layout(title=spec["title"])
    fig.update_layout(margin={"l": 10, "r": 10, "t": 60, "b": 10}, height=400)
    return fig


# --------------------------------------------------------------------------
# 5. Automatic chart selection
# --------------------------------------------------------------------------
def recommend_charts(df: pd.DataFrame, roles=None, max_charts: int = 8) -> list:
    """Choose charts from the column roles. Returns a list of chart specifications."""
    if roles is None:
        roles = identify_roles(df)
    measures, dims, dates = roles["measures"], roles["dimensions"], roles["dates"]

    primary = measures[0] if measures else None
    additive = bool(primary) and _is_additive(primary)
    agg = "sum" if additive else ("mean" if primary else "count")
    measure = primary if agg != "count" else None
    charts = []

    # 1. Trend over time
    if dates:
        charts.append({"kind": "line", "date": dates[0], "measure": measure, "agg": agg,
                       "freq": auto_frequency(df[dates[0]])})

    # 2. Bar chart: the category with the fewest values
    bar_dim = dims[0] if dims else None
    if bar_dim:
        charts.append({"kind": "bar", "category": bar_dim, "measure": measure,
                       "agg": agg, "top_n": 10})

    # 3. Pie chart: a different small category when possible. Shares need Total or Count.
    pie_options = [d for d in dims if d != bar_dim and 2 <= df[d].nunique() <= 8]
    if pie_options:
        pie_dim = pie_options[0]
    else:
        pie_dim = bar_dim if bar_dim and df[bar_dim].nunique() <= 8 else None
    if pie_dim:
        charts.append({"kind": "pie", "category": pie_dim,
                       "measure": primary if additive else None,
                       "agg": "sum" if additive else "count"})

    # 4. Distribution of the main measure
    if primary:
        charts.append({"kind": "histogram", "measure": primary, "bins": 30})

    # 5. Scatter of the most strongly related pair of measures
    if len(measures) >= 2:
        pair = (measures[0], measures[1])
        try:
            top = top_correlations(correlation_matrix(df, measures[:15]), n=1)
            if not top.empty:
                pair = (top.loc[0, "Column A"], top.loc[0, "Column B"])
        except EDAError:
            pass
        charts.append({"kind": "scatter", "x": pair[0], "y": pair[1]})

    # 6. Correlation heatmap
    if len(measures) >= 3:
        charts.append({"kind": "heatmap", "columns": measures[:10]})

    # 7. One more category if there is one left
    spare = [d for d in dims if d not in (bar_dim, pie_dim)]
    if spare:
        charts.append({"kind": "bar", "category": spare[0], "measure": measure,
                       "agg": agg, "top_n": 10})
    return charts[:max_charts]