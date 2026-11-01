"""Exploratory data analysis (EDA): statistics, correlations, distributions,
missing values, categories, trends and comparisons.

No Streamlit code here. Functions return DataFrames, dicts or Plotly figures,
and NONE of them modify the DataFrame they receive.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pandas.api import types as ptypes

from modules.data_cleaner import _is_numeric

MAX_CATEGORY_UNIQUE = 50      # more distinct values than this -> not treated as a category
MAX_GROUPS_IN_COLOR = 15      # more groups than this makes a chart unreadable
SCATTER_MAX_POINTS = 5000     # plot a sample above this (the trend line uses ALL rows)
MAX_TREND_POINTS = 3000
PALETTE = {"main": "#4F8BF9", "accent": "#E63946"}

CORRELATION_METHODS = {
    "pearson": "Pearson (straight-line relationship)",
    "spearman": "Spearman (rank-based, robust to outliers)",
}
# Period codes understood by pandas' to_period()
TREND_FREQUENCIES = {"D": "Daily", "W": "Weekly", "M": "Monthly",
                     "Q": "Quarterly", "Y": "Yearly"}
TREND_AGGREGATIONS = {"sum": "Total (sum)", "mean": "Average", "median": "Median",
                      "count": "Number of rows"}


class EDAError(Exception):
    """Raised with a friendly message when an analysis cannot be done."""


# --------------------------------------------------------------------------
# Column helpers and validation
# --------------------------------------------------------------------------
def numeric_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if _is_numeric(df[c])]


def date_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if ptypes.is_datetime64_any_dtype(df[c])]


def categorical_columns(df: pd.DataFrame, max_unique: int = MAX_CATEGORY_UNIQUE) -> list:
    """Non-numeric, non-date columns with between 2 and max_unique distinct values."""
    result = []
    for column in df.columns:
        series = df[column]
        if _is_numeric(series) or ptypes.is_datetime64_any_dtype(series):
            continue
        if 2 <= series.nunique(dropna=True) <= max_unique:
            result.append(column)
    return result


def _check_column(df: pd.DataFrame, column) -> None:
    if column not in df.columns:
        raise EDAError(f"Column '{column}' was not found.")


def _check_numeric(df: pd.DataFrame, column) -> None:
    _check_column(df, column)
    if not _is_numeric(df[column]):
        raise EDAError(f"Column '{column}' is not numeric.")


# --------------------------------------------------------------------------
# 1. Descriptive statistics
# --------------------------------------------------------------------------
def numeric_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """describe() for numeric columns plus missing count, skewness and kurtosis."""
    columns = numeric_columns(df)
    if not columns:
        return pd.DataFrame()
    data = df[columns].astype("float64")
    stats = data.describe().T
    stats["missing"] = data.isna().sum()
    stats["skew"] = data.skew()
    stats["kurtosis"] = data.kurt()
    stats.index.name = "Column"
    return stats.round(2).reset_index()


def categorical_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """One row per categorical column: distinct values and the most common one."""
    rows = []
    for column in categorical_columns(df):
        counts = df[column].value_counts()
        rows.append({
            "Column": column,
            "Unique": len(counts),
            "Top value": str(counts.index[0]),
            "Top count": int(counts.iloc[0]),
            "Top %": round(counts.iloc[0] / counts.sum() * 100, 1),
            "Missing": int(df[column].isna().sum()),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 2. Correlation analysis
# --------------------------------------------------------------------------
def correlation_matrix(df: pd.DataFrame, columns=None, method: str = "pearson") -> pd.DataFrame:
    if method not in CORRELATION_METHODS:
        raise EDAError(f"Unknown correlation method '{method}'.")
    columns = list(columns) if columns is not None else numeric_columns(df)
    for column in columns:
        _check_numeric(df, column)
    data = df[columns].astype("float64")
    data = data.loc[:, data.nunique() > 1]     # a constant column has no correlation
    if data.shape[1] < 2:
        raise EDAError("Select at least two numeric columns that have more than one "
                       "distinct value.")
    return data.corr(method=method)


def describe_correlation(r: float) -> str:
    """Turn a correlation coefficient into words, e.g. 0.8 -> 'very strong positive'."""
    size = abs(r)
    if size < 0.1:
        return "negligible"
    if size < 0.3:
        label = "weak"
    elif size < 0.5:
        label = "moderate"
    elif size < 0.7:
        label = "strong"
    else:
        label = "very strong"
    return f"{label} {'positive' if r > 0 else 'negative'}"


def top_correlations(corr: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """The n strongest column pairs (each pair listed once), strongest first."""
    names, rows = list(corr.columns), []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            value = corr.iat[i, j]
            if pd.notna(value):
                rows.append((names[i], names[j], float(value)))
    pairs = pd.DataFrame(rows, columns=["Column A", "Column B", "Correlation"])
    if pairs.empty:
        return pairs
    order = pairs["Correlation"].abs().sort_values(ascending=False).index
    pairs = pairs.reindex(order).head(n).reset_index(drop=True)
    pairs["Strength"] = pairs["Correlation"].map(describe_correlation)
    pairs["Correlation"] = pairs["Correlation"].round(3)
    return pairs


def correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    fig = px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, aspect="auto",
                    color_continuous_scale="RdBu_r", title="Correlation heatmap")
    return fig


# --------------------------------------------------------------------------
# 3. Distribution analysis
# --------------------------------------------------------------------------
def distribution_summary(series: pd.Series) -> dict:
    values = series.dropna().astype("float64")
    if len(values) < 3:
        raise EDAError("At least 3 values are needed to describe a distribution.")
    skew = float(values.skew())
    if abs(skew) < 0.5:
        shape = "roughly symmetric"
    elif skew > 0:
        shape = ("strongly right-skewed (a few very high values pull the mean up)"
                 if skew >= 1 else "moderately right-skewed")
    else:
        shape = ("strongly left-skewed (a few very low values pull the mean down)"
                 if skew <= -1 else "moderately left-skewed")
    return {"count": int(len(values)), "mean": float(values.mean()),
            "median": float(values.median()), "std": float(values.std()),
            "min": float(values.min()), "max": float(values.max()),
            "skew": round(skew, 2), "shape": shape}


def histogram_figure(df: pd.DataFrame, column: str, bins: int = 30, group_by=None) -> go.Figure:
    """Histogram with a box plot on top; optionally split by a category."""
    _check_numeric(df, column)
    columns = [column]
    if group_by:
        _check_column(df, group_by)
        if group_by == column:
            raise EDAError("Choose a different column to split by.")
        if df[group_by].nunique() > MAX_GROUPS_IN_COLOR:
            raise EDAError(f"'{group_by}' has more than {MAX_GROUPS_IN_COLOR} groups: "
                           "choose a column with fewer groups.")
        columns.append(group_by)
    data = df[columns].dropna().copy()
    if data.empty:
        raise EDAError(f"'{column}' has no values to plot.")
    data[column] = data[column].astype("float64")
    if group_by:
        data[group_by] = data[group_by].astype(str)
    fig = px.histogram(data, x=column, color=group_by, nbins=int(bins), marginal="box",
                       opacity=0.75 if group_by else 1.0,
                       title=f"Distribution of {column}")
    if group_by:
        fig.update_layout(barmode="overlay")
    return fig


# --------------------------------------------------------------------------
# 4. Missing-value analysis
# --------------------------------------------------------------------------
def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Only the columns that have gaps, worst first."""
    missing = df.isna().sum()
    table = pd.DataFrame({"Column": missing.index, "Missing": missing.to_numpy().astype(int)})
    total = len(df)
    table["Missing %"] = (table["Missing"] / total * 100).round(2) if total else 0.0
    table = table[table["Missing"] > 0].sort_values("Missing", ascending=False)
    return table.reset_index(drop=True)


def missing_bar_figure(summary: pd.DataFrame) -> go.Figure:
    fig = px.bar(summary.sort_values("Missing %"), x="Missing %", y="Column",
                 orientation="h", text="Missing %", title="Missing values by column (%)")
    fig.update_traces(marker_color=PALETTE["accent"])
    return fig


def missing_pattern_figure(df: pd.DataFrame, max_rows: int = 300):
    """Grid of rows (x) by columns (y): red = missing. Returns None if nothing is missing.

    Shows whether gaps are scattered randomly or appear together.
    """
    columns = [c for c in df.columns if df[c].isna().any()]
    if not columns:
        return None
    sample = df[columns]
    if len(sample) > max_rows:      # evenly spaced rows keep the original order
        sample = sample.iloc[np.linspace(0, len(sample) - 1, max_rows).astype(int)]
    matrix = sample.isna().to_numpy().T.astype(int)
    fig = px.imshow(matrix, x=list(range(1, matrix.shape[1] + 1)),
                    y=[str(c) for c in columns], aspect="auto",
                    color_continuous_scale=[[0, "#DCE6F7"], [1, PALETTE["accent"]]],
                    title="Missing-value pattern (red = missing)")
    fig.update_coloraxes(showscale=False)
    fig.update_xaxes(title="Row (sampled)")
    return fig


# --------------------------------------------------------------------------
# 5. Category analysis
# --------------------------------------------------------------------------
def category_counts(df: pd.DataFrame, column: str, top_n: int = 10) -> pd.DataFrame:
    """Counts and percentages; categories beyond top_n are grouped as 'Other'."""
    _check_column(df, column)
    counts = df[column].value_counts(dropna=True)
    if counts.empty:
        raise EDAError(f"'{column}' has no values.")
    total, top_n = int(counts.sum()), max(int(top_n), 1)
    shown, rest = counts.head(top_n), counts.iloc[top_n:]
    table = pd.DataFrame({"Category": shown.index.astype(str),
                          "Count": shown.to_numpy().astype(int)})
    if len(rest):
        other = pd.DataFrame([{"Category": f"Other ({len(rest)} categories)",
                               "Count": int(rest.sum())}])
        table = pd.concat([table, other], ignore_index=True)
    table["Percent"] = (table["Count"] / total * 100).round(1)
    table["Cumulative %"] = table["Percent"].cumsum().round(1)
    return table


def category_bar_figure(table: pd.DataFrame, column: str) -> go.Figure:
    fig = px.bar(table, x="Count", y="Category", orientation="h", text="Percent",
                 title=f"Most common values of {column}")
    fig.update_traces(texttemplate="%{text}%", marker_color=PALETTE["main"])
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    return fig


# --------------------------------------------------------------------------
# 6. Trend analysis
# --------------------------------------------------------------------------
def trend_over_time(df: pd.DataFrame, date_column: str, value_column=None,
                    freq: str = "M", agg: str = "sum") -> pd.DataFrame:
    """Aggregate a value per time period. Periods with no data are filled in."""
    _check_column(df, date_column)
    if not ptypes.is_datetime64_any_dtype(df[date_column]):
        raise EDAError(f"'{date_column}' is not a date column. Convert it to Date / time "
                       "on the Data Cleaning or Transformation page first.")
    if freq not in TREND_FREQUENCIES:
        raise EDAError(f"Unknown time step '{freq}'.")
    if agg not in TREND_AGGREGATIONS:
        raise EDAError(f"Unknown aggregation '{agg}'.")
    columns = [date_column]
    if agg != "count":
        if not value_column:
            raise EDAError("Choose a numeric column to aggregate.")
        _check_numeric(df, value_column)
        columns.append(value_column)

    data = df[columns].dropna().copy()
    if data.empty:
        raise EDAError("There are no rows with both a date and a value.")
    data["_period"] = data[date_column].dt.to_period(freq)
    if agg == "count":
        result = data.groupby("_period").size().astype("float64")
    else:
        data[value_column] = data[value_column].astype("float64")
        result = data.groupby("_period")[value_column].agg(agg)

    full_range = pd.period_range(result.index.min(), result.index.max())
    if len(full_range) > MAX_TREND_POINTS:
        raise EDAError(f"This would create {len(full_range):,} points. "
                       "Choose a coarser time step (e.g. Monthly).")
    result = result.reindex(full_range)
    if agg in ("sum", "count"):
        result = result.fillna(0.0)          # no activity in a period means zero

    table = pd.DataFrame({"Period": full_range.to_timestamp(),
                          "Value": result.to_numpy(dtype=float)})
    change = table["Value"].pct_change(fill_method=None) * 100
    table["Change %"] = change.replace([np.inf, -np.inf], np.nan).round(1)
    table["Moving average (3)"] = table["Value"].rolling(3, min_periods=1).mean()
    return table


def trend_summary(table: pd.DataFrame) -> dict:
    values = table["Value"]
    first, last = float(values.iloc[0]), float(values.iloc[-1])
    return {
        "periods": len(table),
        "first": first, "last": last,
        "overall_change_pct": round((last - first) / first * 100, 1) if first else None,
        "average": float(values.mean()),
        "peak_period": table.loc[values.idxmax(), "Period"],
        "peak_value": float(values.max()),
        "low_period": table.loc[values.idxmin(), "Period"],
        "low_value": float(values.min()),
    }


def trend_figure(table: pd.DataFrame, value_label: str, title=None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=table["Period"], y=table["Value"], mode="lines+markers",
                             name=value_label, line={"color": PALETTE["main"]}))
    if len(table) >= 3:
        fig.add_trace(go.Scatter(x=table["Period"], y=table["Moving average (3)"],
                                 mode="lines", name="3-period moving average",
                                 line={"color": PALETTE["accent"], "dash": "dash"}))
    fig.update_layout(title=title or "Trend over time", xaxis_title="Period",
                      yaxis_title=value_label, hovermode="x unified")
    return fig


# --------------------------------------------------------------------------
# 7. Comparisons
# --------------------------------------------------------------------------
def _prepare_groups(df, category, value, max_groups=None) -> pd.DataFrame:
    _check_column(df, category)
    _check_numeric(df, value)
    if category == value:
        raise EDAError("Choose two different columns.")
    data = df[[category, value]].dropna().copy()
    if data.empty:
        raise EDAError("There are no rows with both values.")
    data[value] = data[value].astype("float64")
    data[category] = data[category].astype(str)
    if max_groups:
        biggest = data[category].value_counts().head(max_groups).index
        data = data[data[category].isin(biggest)]
    return data


def group_comparison(df: pd.DataFrame, category: str, value: str,
                     max_groups: int = 15) -> pd.DataFrame:
    """Statistics of a numeric column for each category, highest average first."""
    data = _prepare_groups(df, category, value, max_groups)
    stats = data.groupby(category)[value].agg(
        ["count", "mean", "median", "std", "min", "max", "sum"])
    stats = stats.sort_values("mean", ascending=False).round(2)
    stats.index.name = "Group"
    stats = stats.reset_index()
    stats.columns = ["Group", "Count", "Mean", "Median", "Std", "Min", "Max", "Sum"]
    return stats


def variation_explained(df: pd.DataFrame, category: str, value: str) -> float:
    """Share (0-1) of the variation in `value` explained by the groups (eta-squared)."""
    data = _prepare_groups(df, category, value)
    grand_mean = data[value].mean()
    total = float(((data[value] - grand_mean) ** 2).sum())
    if total == 0:
        return 0.0
    groups = data.groupby(category)[value].agg(["count", "mean"])
    between = float((groups["count"] * (groups["mean"] - grand_mean) ** 2).sum())
    return between / total


def variation_label(eta_squared: float) -> str:
    if eta_squared < 0.01:
        return "negligible"
    if eta_squared < 0.06:
        return "small"
    if eta_squared < 0.14:
        return "medium"
    return "large"


def comparison_figure(df: pd.DataFrame, category: str, value: str,
                      kind: str = "box", max_groups: int = 15) -> go.Figure:
    data = _prepare_groups(df, category, value, max_groups)
    if kind == "box":
        return px.box(data, x=category, y=value, title=f"{value} by {category}")
    if kind == "bar":
        means = (data.groupby(category)[value].mean()
                 .sort_values(ascending=False).reset_index())
        fig = px.bar(means, x=category, y=value, title=f"Average {value} by {category}")
        fig.update_traces(marker_color=PALETTE["main"])
        return fig
    raise EDAError(f"Unknown chart type '{kind}'.")


def relationship_summary(df: pd.DataFrame, x: str, y: str) -> dict:
    """How two numeric columns relate: correlations and a best-fit straight line."""
    _check_numeric(df, x)
    _check_numeric(df, y)
    if x == y:
        raise EDAError("Choose two different columns.")
    data = df[[x, y]].dropna().astype("float64")
    if len(data) < 3:
        raise EDAError("At least 3 rows with both values are needed.")
    if data[x].nunique() < 2 or data[y].nunique() < 2:
        raise EDAError("Both columns need at least 2 distinct values.")
    pearson = float(data[x].corr(data[y]))
    spearman = float(data[x].corr(data[y], method="spearman"))
    slope, intercept = np.polyfit(data[x], data[y], 1)
    return {"n": int(len(data)), "pearson": round(pearson, 3),
            "spearman": round(spearman, 3), "slope": float(slope),
            "intercept": float(intercept), "strength": describe_correlation(pearson)}


def scatter_figure(df: pd.DataFrame, x: str, y: str, color=None,
                   max_points: int = SCATTER_MAX_POINTS) -> go.Figure:
    summary = relationship_summary(df, x, y)       # also validates the columns
    columns = [x, y]
    if color:
        _check_column(df, color)
        if color in (x, y):
            raise EDAError("The colour column must differ from the x and y columns.")
        if df[color].nunique() > MAX_GROUPS_IN_COLOR:
            raise EDAError(f"'{color}' has more than {MAX_GROUPS_IN_COLOR} groups.")
        columns.append(color)
    data = df[columns].dropna().copy()
    data[x], data[y] = data[x].astype("float64"), data[y].astype("float64")
    if color:
        data[color] = data[color].astype(str)
    plot_data = data.sample(max_points, random_state=0) if len(data) > max_points else data
    fig = px.scatter(plot_data, x=x, y=y, color=color, opacity=0.65,
                     title=f"{y} vs {x}  (correlation r = {summary['pearson']})")
    x_range = np.array([data[x].min(), data[x].max()])
    fig.add_trace(go.Scatter(x=x_range, y=summary["intercept"] + summary["slope"] * x_range,
                             mode="lines", name="Trend line",
                             line={"color": PALETTE["accent"], "dash": "dash"}))
    return fig


def crosstab_table(df: pd.DataFrame, a: str, b: str, percent: bool = False,
                   max_levels: int = 15) -> pd.DataFrame:
    """Counts of every combination of two categorical columns (percent = % of each row)."""
    _check_column(df, a)
    _check_column(df, b)
    if a == b:
        raise EDAError("Choose two different columns.")
    data = df[[a, b]].dropna().astype(str)
    if data.empty:
        raise EDAError("There are no rows with both values.")
    keep_a = data[a].value_counts().head(max_levels).index
    keep_b = data[b].value_counts().head(max_levels).index
    data = data[data[a].isin(keep_a) & data[b].isin(keep_b)]
    table = pd.crosstab(data[a], data[b])
    if percent:
        table = (table.div(table.sum(axis=1), axis=0) * 100).round(1)
    return table


def crosstab_figure(table: pd.DataFrame, a: str, b: str, percent: bool = False) -> go.Figure:
    return px.imshow(table, text_auto=".1f" if percent else True, aspect="auto",
                     color_continuous_scale="Blues",
                     labels={"x": b, "y": a, "color": "% of row" if percent else "Rows"},
                     title=f"{a} vs {b}")


# --------------------------------------------------------------------------
# 8. Automatic highlights (plain-English findings)
# --------------------------------------------------------------------------
def generate_highlights(df: pd.DataFrame, max_items: int = 8) -> list:
    notes = []

    summary = missing_summary(df)
    if not summary.empty:
        worst = summary.iloc[0]
        notes.append(f"'{worst['Column']}' has the most missing values "
                     f"({worst['Missing %']}%). {len(summary)} column(s) have gaps in total.")

    stats = numeric_statistics(df)
    if not stats.empty:
        skewed = stats[stats["skew"].abs() > 1].sort_values(
            "skew", key=lambda s: s.abs(), ascending=False)
        for _, row in skewed.head(2).iterrows():
            side = "right" if row["skew"] > 0 else "left"
            notes.append(f"'{row['Column']}' is strongly {side}-skewed (skew {row['skew']}): "
                         "the median describes a typical value better than the mean.")

    try:
        pairs = top_correlations(correlation_matrix(df), n=1)
        if not pairs.empty and abs(pairs.loc[0, "Correlation"]) >= 0.5:
            row = pairs.loc[0]
            notes.append(f"'{row['Column A']}' and '{row['Column B']}' have a "
                         f"{row['Strength']} correlation (r = {row['Correlation']}).")
    except EDAError:
        pass    # fewer than two usable numeric columns

    dominant = 0
    for column in categorical_columns(df):
        counts = df[column].value_counts()
        share = counts.iloc[0] / counts.sum() * 100
        if share >= 60:
            notes.append(f"'{column}' is dominated by '{counts.index[0]}' ({share:.0f}% of rows).")
            dominant += 1
            if dominant == 2:
                break

    if not notes:
        notes.append("No strong patterns stood out. Explore the tabs below for details.")
    return notes[:max_items]