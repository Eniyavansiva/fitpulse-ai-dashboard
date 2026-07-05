"""Plotly chart and KPI card builders for the FitPulse AI dashboard."""

from __future__ import annotations

from html import escape
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# ── Theme configuration ────────────────────────────────────────────────────

THEMES: dict[str, dict[str, str]] = {
    "dark": {
        "background": "#0D0D0D",
        "card": "#1A1A2E",
        "grid": "#2B2B40",
        "text": "#FFFFFF",
        "muted": "#A0A0B0",
        "mint": "#00F5A0",
        "coral": "#FF6B6B",
        "purple": "#7B61FF",
        "blue": "#4DA3FF",
    },
    "light": {
        "background": "#F5F7FB",
        "card": "#FFFFFF",
        "grid": "#DDE3EF",
        "text": "#172033",
        "muted": "#667085",
        "mint": "#00A86B",
        "coral": "#D9485F",
        "purple": "#6847E8",
        "blue": "#2775D8",
    },
}


def get_theme_colors(theme: str) -> dict[str, str]:
    """Return the requested palette, defaulting safely to FitPulse dark mode."""
    return THEMES.get(theme, THEMES["dark"])


def _base_figure(title: str, theme: str, height: int = 330) -> go.Figure:
    """Create a consistently styled base figure for FitPulse visualizations."""
    colors = get_theme_colors(theme)
    figure = go.Figure()
    figure.update_layout(
        title={"text": title, "font": {"size": 19, "family": "Space Grotesk, sans-serif", "color": colors["text"]}},
        height=height,
        margin={"l": 14, "r": 14, "t": 52, "b": 18},
        paper_bgcolor=colors["card"],
        plot_bgcolor=colors["card"],
        font={"family": "Inter, sans-serif", "color": colors["text"]},
        hoverlabel={"bgcolor": colors["background"], "font_color": colors["text"]},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "font": {"color": colors["text"], "size": 14},
        },
    )
    figure.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor=colors["grid"],
        tickfont={"color": colors["muted"], "size": 14},
        title_font={"color": colors["text"], "size": 15},
    )
    figure.update_yaxes(
        gridcolor=colors["grid"],
        zeroline=False,
        tickfont={"color": colors["muted"], "size": 14},
        title_font={"color": colors["text"], "size": 15},
    )
    return figure


def _empty_chart(title: str, theme: str, message: str, height: int = 330) -> go.Figure:
    """Create a styled chart placeholder when filters have no matching records."""
    colors = get_theme_colors(theme)
    figure = _base_figure(title, theme, height)
    figure.add_annotation(
        text=message,
        showarrow=False,
        font={"size": 14, "color": colors["muted"]},
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
    )
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    return figure


def _daily_average(dataframe: pd.DataFrame, date_column: str, value_column: str) -> pd.DataFrame:
    """Aggregate a dataframe into a sorted date-and-value frame for time-series charts."""
    if dataframe.empty or date_column not in dataframe or value_column not in dataframe:
        return pd.DataFrame(columns=[date_column, value_column])
    return (
        dataframe.dropna(subset=[date_column, value_column])
        .groupby(date_column, as_index=False)[value_column]
        .mean()
        .sort_values(date_column)
    )


# ── Activity charts ────────────────────────────────────────────────────────

def build_steps_trend(activity: pd.DataFrame, theme: str) -> go.Figure:
    """Build the daily steps line chart with rolling average and 10,000-step goal."""
    if activity.empty:
        return _empty_chart("Daily Steps", theme, "No activity records in this date range.")
    colors = get_theme_colors(theme)
    daily = _daily_average(activity, "ActivityDate", "TotalSteps")
    if daily.empty:
        return _empty_chart("Daily Steps", theme, "No step records available for this profile.")
    daily["RollingAverage"] = daily["TotalSteps"].rolling(7, min_periods=1).mean()

    figure = _base_figure("Daily Steps", theme)
    figure.add_trace(
        go.Scatter(
            x=daily["ActivityDate"],
            y=daily["TotalSteps"],
            name="Steps",
            mode="lines+markers",
            line={"color": colors["mint"], "width": 3},
            marker={"size": 5, "color": colors["mint"]},
            fill="tozeroy",
            fillcolor="rgba(0, 245, 160, 0.10)",
            hovertemplate="%{x|%b %d}<br><b>%{y:,.0f} steps</b><extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=daily["ActivityDate"],
            y=daily["RollingAverage"],
            name="7-day average",
            mode="lines",
            line={"color": colors["purple"], "width": 2, "dash": "dash"},
            hovertemplate="%{x|%b %d}<br><b>%{y:,.0f} average steps</b><extra></extra>",
        )
    )
    figure.add_hline(
        y=10_000,
        line_dash="dot",
        line_color=colors["coral"],
        annotation_text="10,000 goal",
        annotation_font_color=colors["coral"],
    )
    figure.update_yaxes(title_text="Steps", rangemode="tozero")
    return figure


def build_activity_donut(activity: pd.DataFrame, theme: str) -> go.Figure:
    """Build a donut chart for very active, fairly active, light, and sedentary time."""
    if activity.empty:
        return _empty_chart("Activity Breakdown", theme, "No activity minutes in this date range.")
    colors = get_theme_colors(theme)
    categories = {
        "Very active": "VeryActiveMinutes",
        "Fairly active": "FairlyActiveMinutes",
        "Lightly active": "LightlyActiveMinutes",
        "Sedentary": "SedentaryMinutes",
    }
    values = [float(activity[column].fillna(0).mean()) if column in activity else 0.0 for column in categories.values()]
    if not any(values):
        return _empty_chart("Activity Breakdown", theme, "No activity-minute records available.")

    total_minutes = int(sum(values))
    figure = go.Figure(
        go.Pie(
            labels=list(categories.keys()),
            values=values,
            hole=0.68,
            sort=False,
            marker={"colors": [colors["mint"], colors["blue"], colors["purple"], colors["muted"]]},
            textinfo="none",
            hovertemplate="<b>%{label}</b><br>%{value:,.0f} min (%{percent})<extra></extra>",
        )
    )
    figure.update_layout(
        title={"text": "Activity Breakdown", "font": {"size": 17, "family": "Space Grotesk, sans-serif"}},
        height=330,
        margin={"l": 10, "r": 10, "t": 52, "b": 16},
        paper_bgcolor=colors["card"],
        font={"family": "Inter, sans-serif", "color": colors["text"]},
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.04, "xanchor": "center", "x": 0.5},
        annotations=[
            {
                "text": f"<b>{total_minutes:,}</b><br><span style='font-size:11px'>avg. daily minutes</span>",
                "showarrow": False,
                "font": {"color": colors["text"], "size": 18},
            }
        ],
    )
    return figure


def build_segment_distribution_chart(
    segments: dict[str, int],
    title: str,
    theme: str,
) -> go.Figure:
    """Build a horizontal bar chart showing how many users fall into each cohort segment."""
    if not segments or not any(segments.values()):
        return _empty_chart(title, theme, "No user-level records are available for segmentation.")
    colors = get_theme_colors(theme)
    labels = list(segments.keys())
    values = list(segments.values())
    bar_colors = [colors["coral"], colors["purple"], colors["mint"]]
    figure = _base_figure(title, theme)
    figure.add_trace(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=bar_colors[: len(values)],
            text=values,
            textposition="outside",
            textfont={"color": colors["text"], "size": 15},
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>%{x} users<extra></extra>",
        )
    )
    figure.update_layout(showlegend=False, margin={"l": 22, "r": 42, "t": 52, "b": 18})
    figure.update_xaxes(title_text="Users", dtick=1, rangemode="tozero")
    figure.update_yaxes(categoryorder="array", categoryarray=labels[::-1], tickfont={"color": colors["muted"], "size": 14})
    return figure


# ── Calorie charts ─────────────────────────────────────────────────────────

def build_hourly_calories_heatmap(calories: pd.DataFrame, theme: str) -> go.Figure:
    """Build a weekday-by-hour heatmap showing average calorie burn patterns."""
    if calories.empty:
        return _empty_chart("Hourly Calorie Heatmap", theme, "No hourly calorie records in this date range.")
    colors = get_theme_colors(theme)
    prepared = calories.dropna(subset=["ActivityHour", "Calories"]).copy()
    if prepared.empty:
        return _empty_chart("Hourly Calorie Heatmap", theme, "No hourly calorie records available.")
    prepared["Weekday"] = pd.to_datetime(prepared["ActivityHour"]).dt.day_name()
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    heatmap = prepared.pivot_table(index="Weekday", columns="Hour", values="Calories", aggfunc="mean")
    heatmap = heatmap.reindex(index=weekday_order, columns=range(24))

    figure = _base_figure("Hourly Calorie Heatmap", theme)
    figure.add_trace(
        go.Heatmap(
            z=heatmap.to_numpy(),
            x=list(range(24)),
            y=weekday_order,
            colorscale=[[0, colors["card"]], [0.3, colors["purple"]], [1, colors["mint"]]],
            colorbar={"title": "Calories", "tickfont": {"color": colors["muted"]}},
            hovertemplate="<b>%{y}</b> · %{x}:00<br>%{z:.0f} calories<extra></extra>",
        )
    )
    figure.update_xaxes(title_text="Hour of day", dtick=2)
    figure.update_yaxes(autorange="reversed")
    return figure


def build_daily_calories_chart(activity: pd.DataFrame, theme: str) -> go.Figure:
    """Build daily calories bars with a seven-day rolling average overlay."""
    if activity.empty:
        return _empty_chart("Daily Calories", theme, "No daily calorie records in this date range.")
    colors = get_theme_colors(theme)
    daily = _daily_average(activity, "ActivityDate", "Calories")
    if daily.empty:
        return _empty_chart("Daily Calories", theme, "No daily calorie records available.")
    daily["RollingAverage"] = daily["Calories"].rolling(7, min_periods=1).mean()

    figure = _base_figure("Daily Calories", theme)
    figure.add_trace(
        go.Bar(
            x=daily["ActivityDate"],
            y=daily["Calories"],
            name="Calories",
            marker_color=colors["coral"],
            opacity=0.78,
            hovertemplate="%{x|%b %d}<br><b>%{y:,.0f} calories</b><extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=daily["ActivityDate"],
            y=daily["RollingAverage"],
            name="7-day average",
            mode="lines",
            line={"color": colors["mint"], "width": 3},
            hovertemplate="%{x|%b %d}<br><b>%{y:,.0f} average calories</b><extra></extra>",
        )
    )
    figure.update_yaxes(title_text="Calories", rangemode="tozero")
    return figure


# ── Sleep charts ───────────────────────────────────────────────────────────

def build_sleep_duration_chart(sleep: pd.DataFrame, theme: str) -> go.Figure:
    """Build grouped bars comparing time asleep with total time spent in bed."""
    if sleep.empty:
        return _empty_chart("Sleep Duration vs. Time in Bed", theme, "No sleep records in this date range.")
    colors = get_theme_colors(theme)
    daily = sleep.groupby("SleepDay", as_index=False)[["TotalMinutesAsleep", "TotalTimeInBed"]].mean().sort_values("SleepDay")
    if daily.empty:
        return _empty_chart("Sleep Duration vs. Time in Bed", theme, "No sleep records available.")
    daily["AsleepHours"] = daily["TotalMinutesAsleep"] / 60
    daily["InBedHours"] = daily["TotalTimeInBed"] / 60

    figure = _base_figure("Sleep Duration vs. Time in Bed", theme)
    figure.add_trace(
        go.Bar(
            x=daily["SleepDay"],
            y=daily["AsleepHours"],
            name="Asleep",
            marker_color=colors["purple"],
            hovertemplate="%{x|%b %d}<br><b>%{y:.1f} hours asleep</b><extra></extra>",
        )
    )
    figure.add_trace(
        go.Bar(
            x=daily["SleepDay"],
            y=daily["InBedHours"],
            name="Time in bed",
            marker_color=colors["blue"],
            opacity=0.64,
            hovertemplate="%{x|%b %d}<br><b>%{y:.1f} hours in bed</b><extra></extra>",
        )
    )
    figure.update_layout(barmode="group")
    figure.update_yaxes(title_text="Hours", rangemode="tozero")
    return figure


def build_sleep_efficiency_gauge(sleep: pd.DataFrame, theme: str) -> go.Figure:
    """Build a 0–100% gauge summarizing average sleep efficiency."""
    if sleep.empty or sleep["SleepEfficiency"].dropna().empty:
        return _empty_chart("Sleep Efficiency", theme, "No sleep-efficiency records in this date range.")
    colors = get_theme_colors(theme)
    efficiency = float(sleep["SleepEfficiency"].mean())
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=efficiency,
            number={"suffix": "%", "font": {"size": 40, "color": colors["text"]}},
            title={"text": "Average efficiency", "font": {"size": 14, "color": colors["muted"]}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": colors["muted"]},
                "bar": {"color": colors["purple"]},
                "bgcolor": colors["grid"],
                "steps": [
                    {"range": [0, 70], "color": "rgba(255, 107, 107, 0.22)"},
                    {"range": [70, 85], "color": "rgba(255, 193, 7, 0.18)"},
                    {"range": [85, 100], "color": "rgba(0, 245, 160, 0.18)"},
                ],
            },
        )
    )
    figure.update_layout(
        title={"text": "Sleep Efficiency", "font": {"size": 17, "family": "Space Grotesk, sans-serif"}},
        height=330,
        margin={"l": 20, "r": 20, "t": 55, "b": 18},
        paper_bgcolor=colors["card"],
        font={"family": "Inter, sans-serif", "color": colors["text"]},
    )
    return figure


def build_sleep_distribution_chart(sleep: pd.DataFrame, theme: str) -> go.Figure:
    """Build a sleep-efficiency histogram with a scaled normal-curve overlay."""
    if sleep.empty:
        return _empty_chart("Sleep Score Distribution", theme, "No sleep records in this date range.")
    colors = get_theme_colors(theme)
    values = sleep["SleepEfficiency"].dropna().to_numpy(dtype=float)
    if values.size == 0:
        return _empty_chart("Sleep Score Distribution", theme, "No sleep scores available.")

    figure = _base_figure("Sleep Score Distribution", theme)
    bin_size = max(5, min(10, int(np.ptp(values) / 4) if values.size > 1 else 10))
    figure.add_trace(
        go.Histogram(
            x=values,
            name="Sleep scores",
            marker_color=colors["purple"],
            opacity=0.75,
            xbins={"size": bin_size},
            hovertemplate="Sleep efficiency: %{x:.0f}%<br>Records: %{y}<extra></extra>",
        )
    )
    standard_deviation = values.std(ddof=0)
    if values.size > 1 and standard_deviation > 0:
        curve_x = np.linspace(max(0, values.min() - 10), min(100, values.max() + 10), 120)
        curve_y = (
            values.size
            * bin_size
            * (1 / (standard_deviation * np.sqrt(2 * np.pi)))
            * np.exp(-0.5 * ((curve_x - values.mean()) / standard_deviation) ** 2)
        )
        figure.add_trace(
            go.Scatter(
                x=curve_x,
                y=curve_y,
                name="Normal curve",
                mode="lines",
                line={"color": colors["mint"], "width": 3},
                hoverinfo="skip",
            )
        )
    figure.update_layout(barmode="overlay")
    figure.update_xaxes(title_text="Sleep efficiency (%)", range=[0, 100])
    figure.update_yaxes(title_text="Nights", rangemode="tozero")
    return figure


# ── Heart-rate chart ───────────────────────────────────────────────────────

def build_heartrate_chart(heartrate: pd.DataFrame, theme: str) -> go.Figure:
    """Build an hourly heart-rate area chart with resting, fat-burn, cardio, and peak zones."""
    if heartrate.empty:
        return _empty_chart("Heart Rate Analysis", theme, "No heart-rate records in this date range.", 390)
    colors = get_theme_colors(theme)
    hourly = (
        heartrate.dropna(subset=["Time", "Value"])
        .set_index("Time")["Value"]
        .resample("1h")
        .mean()
        .dropna()
        .reset_index()
    )
    if hourly.empty:
        return _empty_chart("Heart Rate Analysis", theme, "No heart-rate readings available.", 390)

    figure = _base_figure("Heart Rate Analysis", theme, 390)
    figure.add_hrect(y0=30, y1=60, fillcolor="rgba(77, 163, 255, 0.12)", line_width=0, annotation_text="Resting")
    figure.add_hrect(y0=60, y1=154, fillcolor="rgba(0, 245, 160, 0.08)", line_width=0, annotation_text="Fat burn")
    figure.add_hrect(y0=154, y1=187, fillcolor="rgba(255, 193, 7, 0.10)", line_width=0, annotation_text="Cardio")
    figure.add_hrect(y0=187, y1=220, fillcolor="rgba(255, 107, 107, 0.11)", line_width=0, annotation_text="Peak")
    figure.add_trace(
        go.Scatter(
            x=hourly["Time"],
            y=hourly["Value"],
            name="Average BPM",
            mode="lines",
            line={"color": colors["coral"], "width": 2.5},
            fill="tozeroy",
            fillcolor="rgba(255, 107, 107, 0.12)",
            hovertemplate="%{x|%b %d, %H:%M}<br><b>%{y:.0f} BPM</b><extra></extra>",
        )
    )
    figure.update_yaxes(title_text="BPM", range=[30, 220])
    return figure


# ── Weight and BMI charts ──────────────────────────────────────────────────

def build_weight_trend_chart(weight: pd.DataFrame, theme: str, goal_weight: float | None = None) -> go.Figure:
    """Build a weight trend line with an optional reference goal line."""
    if weight.empty:
        return _empty_chart("Weight Trend", theme, "No weight measurements in this date range.")
    colors = get_theme_colors(theme)
    date_column = "MeasurementDate" if "MeasurementDate" in weight else "Date"
    measurements = (
        weight.dropna(subset=[date_column, "WeightKg"])
        .groupby(date_column, as_index=False)["WeightKg"]
        .mean()
        .sort_values(date_column)
    )
    if measurements.empty:
        return _empty_chart("Weight Trend", theme, "No weight measurements available.")
    reference_goal = goal_weight if goal_weight is not None else float(measurements["WeightKg"].iloc[0] * 0.95)

    figure = _base_figure("Weight Trend", theme)
    figure.add_trace(
        go.Scatter(
            x=measurements[date_column],
            y=measurements["WeightKg"],
            name="Weight",
            mode="lines+markers",
            line={"color": colors["blue"], "width": 3},
            marker={"size": 7},
            hovertemplate="%{x|%b %d}<br><b>%{y:.1f} kg</b><extra></extra>",
        )
    )
    figure.add_hline(
        y=reference_goal,
        line_dash="dot",
        line_color=colors["mint"],
        annotation_text="Reference goal",
        annotation_font_color=colors["mint"],
    )
    figure.update_yaxes(title_text="Weight (kg)")
    return figure


def build_bmi_indicator(weight: pd.DataFrame, theme: str) -> go.Figure:
    """Build a BMI category gauge with standard underweight through obese zones."""
    if weight.empty or weight["BMI"].dropna().empty:
        return _empty_chart("BMI Category", theme, "No BMI measurements in this date range.")
    colors = get_theme_colors(theme)
    date_column = "MeasurementDate" if "MeasurementDate" in weight else "Date"
    measurements = (
        weight.dropna(subset=[date_column, "BMI"])
        .groupby(date_column, as_index=False)["BMI"]
        .mean()
        .sort_values(date_column)
    )
    latest_bmi = float(measurements["BMI"].iloc[-1])
    axis_max = max(40, np.ceil(latest_bmi / 5) * 5)
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=latest_bmi,
            number={"valueformat": ".1f", "font": {"size": 42, "color": colors["text"]}},
            title={"text": "Latest BMI", "font": {"size": 14, "color": colors["muted"]}},
            gauge={
                "axis": {"range": [0, axis_max], "tickcolor": colors["muted"]},
                "bar": {"color": colors["text"], "thickness": 0.3},
                "bgcolor": colors["grid"],
                "steps": [
                    {"range": [0, 18.5], "color": "#4DA3FF"},
                    {"range": [18.5, 25], "color": "#00F5A0"},
                    {"range": [25, 30], "color": "#FFC107"},
                    {"range": [30, axis_max], "color": "#FF6B6B"},
                ],
            },
        )
    )
    figure.update_layout(
        title={"text": "BMI Category", "font": {"size": 17, "family": "Space Grotesk, sans-serif"}},
        height=330,
        margin={"l": 20, "r": 20, "t": 55, "b": 18},
        paper_bgcolor=colors["card"],
        font={"family": "Inter, sans-serif", "color": colors["text"]},
        annotations=[
            {
                "text": "Underweight · Normal · Overweight · Obese",
                "showarrow": False,
                "x": 0.5,
                "y": -0.08,
                "xref": "paper",
                "yref": "paper",
                "font": {"size": 11, "color": colors["muted"]},
            }
        ],
    )
    return figure


# ── KPI card markup ────────────────────────────────────────────────────────

def _sparkline_points(values: list[float]) -> str:
    """Convert numeric values into SVG polyline points for a compact card sparkline."""
    clean_values = [float(value) for value in values if value is not None and not pd.isna(value)]
    if len(clean_values) < 2:
        return ""
    minimum, maximum = min(clean_values), max(clean_values)
    value_range = maximum - minimum or 1
    coordinates = []
    for index, value in enumerate(clean_values):
        x_position = index / (len(clean_values) - 1) * 100
        y_position = 26 - ((value - minimum) / value_range * 22)
        coordinates.append(f"{x_position:.1f},{y_position:.1f}")
    return " ".join(coordinates)


def build_kpi_card_html(kpi: dict[str, Any], theme: str) -> str:
    """Build a styled HTML KPI card with a mini sparkline and daily-goal progress ring."""
    colors = get_theme_colors(theme)
    delta_class = "positive" if kpi["trend_is_positive"] else "negative"
    if kpi["delta"] is None:
        delta_class = "neutral"
    sparkline_points = _sparkline_points(kpi["sparkline"])
    sparkline = (
        f"<svg class='fitpulse-sparkline' viewBox='0 0 100 30' preserveAspectRatio='none'>"
        f"<polyline points='{sparkline_points}' fill='none' stroke='{colors['mint']}' stroke-width='2.4' "
        "stroke-linecap='round' stroke-linejoin='round'></polyline></svg>"
        if sparkline_points
        else "<div class='fitpulse-no-sparkline'>Not enough trend data</div>"
    )
    ring = ""
    if kpi["key"] == "steps" and kpi["value"] is not None:
        progress = int(max(0, min(float(kpi["value"]) / 10_000 * 100, 100)))
        ring = (
            f"<div class='fitpulse-progress-ring' style='--progress:{progress}; --ring-color:{colors['mint']}'>"
            f"<span>{progress}%</span></div>"
        )
    return (
        "<div class='fitpulse-kpi-card'>"
        "<div class='fitpulse-kpi-top'>"
        f"<span class='fitpulse-kpi-icon'>{escape(str(kpi['icon']))}</span>"
        f"<span class='fitpulse-kpi-title'>{escape(str(kpi['title']))}</span>{ring}</div>"
        f"<div class='fitpulse-kpi-value'>{escape(str(kpi['value_label']))}</div>"
        f"<div class='fitpulse-kpi-detail'>{escape(str(kpi['detail']))}</div>"
        f"<div class='fitpulse-kpi-delta {delta_class}'>{escape(str(kpi['delta_label']))}</div>"
        f"{sparkline}"
        "</div>"
    )
