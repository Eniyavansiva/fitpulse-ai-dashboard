"""Standalone Streamlit dashboard for FitPulse AI fitness intelligence."""

from __future__ import annotations

import os
from html import escape
from typing import Any

import pandas as pd
import streamlit as st

from utils.ai_insights import generate_llm_insights
from utils.chart_builder import (
    build_activity_donut,
    build_bmi_indicator,
    build_segment_distribution_chart,
    build_daily_calories_chart,
    build_heartrate_chart,
    build_hourly_calories_heatmap,
    build_kpi_card_html,
    build_sleep_distribution_chart,
    build_sleep_duration_chart,
    build_sleep_efficiency_gauge,
    build_steps_trend,
    build_weight_trend_chart,
    get_theme_colors,
)
from utils.data_loader import DataLoadError, load_all_data
from utils.metrics import (
    build_user_performance_summary,
    build_wellness_daily,
    calculate_attention_flags,
    calculate_cohort_analysis,
    calculate_kpis,
    generate_insights,
    get_wellness_snapshot,
)


st.set_page_config(
    page_title="FitPulse AI",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ── Configuration ──────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
STYLE_PATH = os.path.join(PROJECT_ROOT, "assets", "style.css")
PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}
ALL_USERS_OPTION = "All users"
DATA_PERIOD_LABEL = "Mar 12, 2016 – May 12, 2016"


# ── Theme and session state ─────────────────────────────────────────────────

def _read_style_template() -> str:
    """Read the local CSS template used to style the active dashboard theme."""
    try:
        with open(STYLE_PATH, encoding="utf-8") as style_file:
            return style_file.read()
    except OSError as error:
        st.error(f"Could not load dashboard styles from assets/style.css: {error}")
        return ""


def _inject_theme(theme: str) -> None:
    """Replace CSS palette placeholders and inject the result into the Streamlit page."""
    colors = get_theme_colors(theme)
    stylesheet = _read_style_template()
    replacements = {
        "{{BACKGROUND}}": colors["background"],
        "{{CARD}}": colors["card"],
        "{{TEXT}}": colors["text"],
        "{{MUTED}}": colors["muted"],
        "{{MINT}}": colors["mint"],
        "{{CORAL}}": colors["coral"],
        "{{PURPLE}}": colors["purple"],
        "{{GRID}}": colors["grid"],
    }
    for placeholder, value in replacements.items():
        stylesheet = stylesheet.replace(placeholder, value)
    st.markdown(f"<style>{stylesheet}</style>", unsafe_allow_html=True)


def _all_user_ids(data: dict[str, pd.DataFrame]) -> list[str]:
    """Return the aggregate option followed by all unique anonymous Fitbit user names."""
    user_names = set()
    for dataframe in data.values():
        if "UserName" in dataframe:
            user_names.update(dataframe["UserName"].dropna().astype(str).tolist())
    return [ALL_USERS_OPTION, *sorted(user_names)]


def _initialise_session_state(data: dict[str, pd.DataFrame]) -> None:
    """Create stable session-state values for the aggregate view, theme, goals, and chat history."""
    user_ids = _all_user_ids(data)
    if len(user_ids) == 1:
        raise DataLoadError("No Fitbit user IDs were found in the supplied datasets.")
    defaults: dict[str, Any] = {
        "theme": "dark",
        "selected_user": ALL_USERS_OPTION,
        "chat_history": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ── Filter controls ────────────────────────────────────────────────────────

def _render_top_bar(data: dict[str, pd.DataFrame]) -> str:
    """Render the fixed-period header and return the active user scope."""
    user_ids = _all_user_ids(data)
    brand_column, user_column, theme_column = st.columns([1.6, 1.25, 0.62], vertical_alignment="bottom")

    with brand_column:
        st.markdown(
            "<div class='fitpulse-brand'><span class='fitpulse-brand-mark'>🏃</span>FitPulse AI</div>"
            f"<div class='fitpulse-brand-subtitle'>STATIC DATASET · {DATA_PERIOD_LABEL.upper()}</div>",
            unsafe_allow_html=True,
        )
    with user_column:
        selected_user = st.selectbox(
            "Athlete profile",
            user_ids,
            key="selected_user",
        )
    with theme_column:
        button_label = "☀️ Light" if st.session_state.theme == "dark" else "🌙 Dark"
        if st.button(button_label, width="stretch", help="Switch dashboard theme"):
            st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
            st.rerun()

    return selected_user


def _filter_dataset(
    dataframe: pd.DataFrame,
    user_id: str,
) -> pd.DataFrame:
    """Return a user-specific copy or the complete dataframe for the aggregate option."""
    if dataframe.empty:
        return dataframe.copy()
    if user_id == ALL_USERS_OPTION:
        return dataframe.copy()
    filtered = dataframe.loc[dataframe["UserName"].astype(str) == str(user_id)].copy()
    return filtered


def _apply_filters(
    data: dict[str, pd.DataFrame],
    user_id: str,
) -> dict[str, pd.DataFrame]:
    """Apply the active individual or all-users scope to every source dataframe."""
    return {
        key: _filter_dataset(dataframe, user_id)
        for key, dataframe in data.items()
    }


# ── Dashboard sections ─────────────────────────────────────────────────────

def _section_heading(title: str, description: str) -> None:
    """Render a consistent title and supporting line for one dashboard section."""
    st.markdown(f"<div class='fitpulse-section-title'>{title}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='fitpulse-section-copy'>{description}</div>", unsafe_allow_html=True)


def render_no_data_card(message: str) -> None:
    """Render a clean styled card when no data is available for the selected user."""
    st.markdown(
        f"""
        <div class="fitpulse-no-data-card">
            <div class="fitpulse-no-data-icon">📭</div>
            <div>{escape(message)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _format_metric_number(value: float | int | None, decimals: int = 0, suffix: str = "") -> str:
    """Format a metric value for custom dashboard cards."""
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):,.{decimals}f}{suffix}"


def _format_percent_delta(delta: float | None) -> str:
    """Format a weekly percentage delta in the same language as the KPI cards."""
    if delta is None or pd.isna(delta):
        return "No prior-week data"
    direction = "▲" if delta >= 0 else "▼"
    return f"{direction} {abs(delta):.1f}% vs prior week"


def _wellness_color_class(score: float | None) -> str:
    """Return the visual band used for the hero wellness score."""
    if score is None or pd.isna(score):
        return "muted"
    if score < 50:
        return "risk"
    if score < 70:
        return "watch"
    if score < 85:
        return "good"
    return "elite"


def _custom_kpi_payload(
    key: str,
    title: str,
    icon: str,
    value: float | None,
    value_label: str,
    detail: str,
    sparkline: list[float] | None = None,
    delta: float | None = None,
    positive_is_good: bool = True,
) -> dict[str, Any]:
    """Build a KPI payload compatible with the existing card renderer."""
    trend_is_positive = delta is not None and (delta >= 0 if positive_is_good else delta <= 0)
    return {
        "key": key,
        "title": title,
        "icon": icon,
        "value": value,
        "value_label": value_label,
        "delta": delta,
        "delta_label": _format_percent_delta(delta),
        "trend_is_positive": trend_is_positive,
        "sparkline": sparkline or [],
        "detail": detail,
    }


def _selected_summary_row(performance_summary: pd.DataFrame, selected_user: str) -> pd.Series | None:
    """Return the current user's row from the ranked performance summary."""
    if performance_summary.empty or selected_user == ALL_USERS_OPTION:
        return None
    matches = performance_summary.loc[performance_summary["UserName"].astype(str) == str(selected_user)]
    return matches.iloc[0] if not matches.empty else None


def _render_wellness_hero(
    filtered_data: dict[str, pd.DataFrame],
    selected_user: str,
    performance_summary: pd.DataFrame,
) -> dict[str, Any]:
    """Render the primary V2 wellness score hero and optional individual rank badge."""
    wellness_daily = build_wellness_daily(
        filtered_data["activity"],
        filtered_data["sleep"],
        filtered_data["heartrate"],
    )
    snapshot = get_wellness_snapshot(wellness_daily)
    score = snapshot["score"]
    score_label = _format_metric_number(score, 1)
    delta = snapshot["delta"]
    delta_label = "No prior-week comparison" if delta is None or pd.isna(delta) else f"{delta:+.1f} vs last week"
    scope_label = "All users cohort" if selected_user == ALL_USERS_OPTION else selected_user
    wellness_class = _wellness_color_class(score)

    rank_markup = ""
    selected_row = _selected_summary_row(performance_summary, selected_user)
    if selected_row is not None and pd.notna(selected_row.get("Rank")):
        rank = int(selected_row["Rank"])
        total_users = int(performance_summary["UserName"].nunique())
        rank_markup = (
            f"<div class='fitpulse-rank-badge'>Your ranking: "
            f"<strong>#{rank} of {total_users}</strong> users by Wellness Score</div>"
        )

    st.markdown(
        "<div class='fitpulse-wellness-hero'>"
        "<div>"
        f"<div class='fitpulse-wellness-eyebrow'>{escape(scope_label)}</div>"
        "<div class='fitpulse-wellness-title'>Wellness Score</div>"
        "<div class='fitpulse-wellness-subtitle'>Composite of steps, sleep, activity, and resting heart rate</div>"
        f"{rank_markup}"
        "</div>"
        "<div class='fitpulse-wellness-score-wrap'>"
        f"<div class='fitpulse-wellness-score {wellness_class}'>{score_label}</div>"
        f"<div class='fitpulse-wellness-delta'>{escape(delta_label)}</div>"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    return snapshot


def _build_streak_kpi(performance_summary: pd.DataFrame, selected_user: str) -> dict[str, Any]:
    """Build the current/best 7,500-step streak KPI card for individual or cohort scope."""
    if performance_summary.empty:
        return _custom_kpi_payload("streak", "Streak", "🔥", None, "—", "7,500-step daily streak")
    if selected_user == ALL_USERS_OPTION:
        current = float(performance_summary["CurrentStreak"].fillna(0).mean())
        best = float(performance_summary["BestStreak"].fillna(0).max())
        return _custom_kpi_payload(
            "streak",
            "Cohort Streak",
            "🔥",
            current,
            f"{current:.1f} days",
            f"Best active streak: {best:.0f} days",
            performance_summary["CurrentStreak"].fillna(0).astype(float).tolist(),
        )
    selected_row = _selected_summary_row(performance_summary, selected_user)
    if selected_row is None:
        return _custom_kpi_payload("streak", "Streak", "🔥", None, "—", "7,500-step daily streak")
    current = float(selected_row.get("CurrentStreak", 0) or 0)
    best = float(selected_row.get("BestStreak", 0) or 0)
    return _custom_kpi_payload(
        "streak",
        "Streak",
        "🔥",
        current,
        f"{current:.0f}-day streak",
        f"Best: {best:.0f} days",
        [current, best],
    )


def _build_wellness_kpi(wellness_snapshot: dict[str, Any]) -> dict[str, Any]:
    """Build the All Users average wellness KPI that replaces cohort BMI."""
    score = wellness_snapshot.get("score")
    return _custom_kpi_payload(
        "wellness",
        "Avg Wellness",
        "★",
        score,
        _format_metric_number(score, 1),
        "Steps, sleep, active time, resting HR",
        wellness_snapshot.get("sparkline", []),
        wellness_snapshot.get("delta"),
    )


def _render_kpi_cards(
    filtered_data: dict[str, pd.DataFrame],
    selected_user: str,
    performance_summary: pd.DataFrame,
    wellness_snapshot: dict[str, Any],
) -> None:
    """Render the seven KPI cards with weekly deltas, sparklines, and streak tracking."""
    _section_heading("Today at a glance", "Your selected profile and fixed dataset window, distilled into seven signals.")
    kpis = calculate_kpis(
        filtered_data["activity"],
        filtered_data["sleep"],
        filtered_data["heartrate"],
        filtered_data["weight"],
    )
    if selected_user == ALL_USERS_OPTION:
        kpis = [kpi for kpi in kpis if kpi["key"] != "bmi"]
        kpis.append(_build_wellness_kpi(wellness_snapshot))
    kpis.append(_build_streak_kpi(performance_summary, selected_user))
    st.markdown("<div class='fitpulse-kpi-scroll-hint'>Swipe horizontally on smaller screens to see all seven cards.</div>", unsafe_allow_html=True)
    for column, kpi in zip(st.columns(7, gap="small"), kpis):
        with column:
            st.markdown(build_kpi_card_html(kpi, st.session_state.theme), unsafe_allow_html=True)


def _format_cohort_value(value: float, suffix: str = "", decimals: int = 0) -> str:
    """Format a cohort aggregate safely for display in a Streamlit metric card."""
    if pd.isna(value):
        return "—"
    return f"{value:,.{decimals}f}{suffix}"


def _render_attention_flags(attention_flags: pd.DataFrame) -> None:
    """Render the collapsed all-users attention list near the top of the cohort view."""
    with st.expander(f"Show attention flags ({len(attention_flags)} users)", expanded=False):
        if attention_flags.empty:
            st.success("No users meet the current attention criteria.")
            return
        st.dataframe(
            attention_flags[["Name", "Issue", "Severity"]],
            width="stretch",
            hide_index=True,
        )


def _render_leaderboard(performance_summary: pd.DataFrame) -> None:
    """Render the top-five all-users leaderboard ranked by Wellness Score."""
    if performance_summary.empty:
        return
    _section_heading(
        "Top performers this period",
        "Top five users ranked by average Wellness Score across the fixed dataset period.",
    )
    medals = ["🥇", "🥈", "🥉", "4th", "5th"]
    rows = []
    for index, (_, user) in enumerate(performance_summary.head(5).iterrows()):
        rank_label = medals[index] if index < len(medals) else f"{index + 1}th"
        rows.append(
            "<tr>"
            f"<td>{escape(rank_label)}</td>"
            f"<td>{escape(str(user['UserName']))}</td>"
            f"<td>{_format_metric_number(user.get('WellnessScore'), 1)}</td>"
            f"<td>{_format_metric_number(user.get('AvgSteps'))}</td>"
            f"<td>{_format_metric_number(user.get('SleepScore'), 0, '%')}</td>"
            f"<td>{_format_metric_number(user.get('CurrentStreak'))} days</td>"
            "</tr>"
        )
    st.markdown(
        "<div class='fitpulse-leaderboard'>"
        "<table>"
        "<thead><tr><th>Rank</th><th>Name</th><th>Wellness Score</th><th>Avg Steps</th><th>Sleep Score</th><th>Streak</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
        "<div class='fitpulse-leaderboard-note'>Your individual ranking appears on your personal dashboard.</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def _render_cohort_section(filtered_data: dict[str, pd.DataFrame], wellness_snapshot: dict[str, Any]) -> dict[str, Any]:
    """Render all-users averages and activity/recovery segments for product-level analysis."""
    analysis = calculate_cohort_analysis(
        filtered_data["activity"],
        filtered_data["sleep"],
        filtered_data["heartrate"],
    )
    _section_heading(
        "Cohort intelligence",
        "Portfolio-level averages, wellness, and user segmentation across every available Fitbit profile.",
    )
    participants_column, steps_column, sleep_column, heart_column, wellness_column = st.columns(5, gap="small")
    participants_column.metric("Activity profiles", analysis["participant_count"])
    steps_column.metric("Avg daily steps", _format_cohort_value(analysis["average_daily_steps"]))
    sleep_column.metric("Avg sleep efficiency", _format_cohort_value(analysis["average_sleep_efficiency"], "%"))
    heart_column.metric("Avg heart rate", _format_cohort_value(analysis["average_heart_rate"], " bpm"))
    wellness_column.metric("Avg wellness", _format_metric_number(wellness_snapshot.get("score"), 1))
    st.caption(
        f"Sleep segmentation uses {analysis['sleep_participant_count']} profiles with sleep records; "
        "activity segmentation uses each profile’s average daily steps."
    )
    activity_segment_column, sleep_segment_column = st.columns(2, gap="medium")
    with activity_segment_column:
        st.plotly_chart(
            build_segment_distribution_chart(
                analysis["activity_segments"],
                "Activity Segmentation",
                st.session_state.theme,
            ),
            width="stretch",
            config=PLOTLY_CONFIG,
        )
    with sleep_segment_column:
        st.plotly_chart(
            build_segment_distribution_chart(
                analysis["sleep_segments"],
                "Recovery Segmentation",
                st.session_state.theme,
            ),
            width="stretch",
            config=PLOTLY_CONFIG,
        )
    return analysis


def _render_activity_section(filtered_data: dict[str, pd.DataFrame]) -> None:
    """Render the activity trend and activity-minute breakdown visualizations."""
    _section_heading("Activity overview", "Daily movement, consistency, and the balance of your active time.")
    steps_column, donut_column = st.columns([1.5, 1], gap="medium")
    with steps_column:
        st.plotly_chart(build_steps_trend(filtered_data["activity"], st.session_state.theme), width="stretch", config=PLOTLY_CONFIG)
    with donut_column:
        st.plotly_chart(build_activity_donut(filtered_data["activity"], st.session_state.theme), width="stretch", config=PLOTLY_CONFIG)


def _render_calorie_section(filtered_data: dict[str, pd.DataFrame]) -> None:
    """Render hourly calorie patterns alongside daily calorie totals."""
    _section_heading("Calories deep dive", "Spot the times and days where your energy expenditure peaks.")
    heatmap_column, calories_column = st.columns(2, gap="medium")
    with heatmap_column:
        st.plotly_chart(build_hourly_calories_heatmap(filtered_data["calories"], st.session_state.theme), width="stretch", config=PLOTLY_CONFIG)
    with calories_column:
        st.plotly_chart(build_daily_calories_chart(filtered_data["activity"], st.session_state.theme), width="stretch", config=PLOTLY_CONFIG)


def _render_sleep_section(filtered_data: dict[str, pd.DataFrame]) -> None:
    """Render sleep duration, average efficiency, and score-distribution charts."""
    _section_heading("Sleep analysis", "Recovery quality, bedtime efficiency, and the shape of your recent nights.")
    duration_column, gauge_column, distribution_column = st.columns([1.35, 1, 1], gap="medium")
    with duration_column:
        st.plotly_chart(build_sleep_duration_chart(filtered_data["sleep"], st.session_state.theme), width="stretch", config=PLOTLY_CONFIG)
    with gauge_column:
        st.plotly_chart(build_sleep_efficiency_gauge(filtered_data["sleep"], st.session_state.theme), width="stretch", config=PLOTLY_CONFIG)
    with distribution_column:
        st.plotly_chart(build_sleep_distribution_chart(filtered_data["sleep"], st.session_state.theme), width="stretch", config=PLOTLY_CONFIG)


def _render_heartrate_section(filtered_data: dict[str, pd.DataFrame]) -> None:
    """Render the full-width heart-rate timeline and training zones."""
    _section_heading("Heart rate analysis", "Minute-level readings cleaned into an hourly signal and mapped to intensity zones.")
    if filtered_data["heartrate"].empty:
        render_no_data_card("No heart rate data recorded for this user.")
        return
    st.plotly_chart(build_heartrate_chart(filtered_data["heartrate"], st.session_state.theme), width="stretch", config=PLOTLY_CONFIG)


def _render_weight_section(filtered_data: dict[str, pd.DataFrame]) -> None:
    """Render weight and BMI visualizations with a user-adjustable reference goal."""
    _section_heading("Weight & BMI trends", "Track your recorded measurements against a reference target.")
    weight_data = filtered_data["weight"]
    default_goal = 70.0
    if not weight_data.empty and weight_data["WeightKg"].notna().any():
        weight_date_column = "MeasurementDate" if "MeasurementDate" in weight_data else "Date"
        daily_weight = (
            weight_data.dropna(subset=[weight_date_column, "WeightKg"])
            .groupby(weight_date_column)["WeightKg"]
            .mean()
            .sort_index()
        )
        default_goal = round(float(daily_weight.iloc[0] * 0.95), 1)
    goal_key = f"weight_goal_{st.session_state.selected_user}"
    if goal_key not in st.session_state:
        st.session_state[goal_key] = default_goal
    weight_column, bmi_column = st.columns(2, gap="medium")
    with weight_column:
        if weight_data.empty or weight_data["WeightKg"].dropna().empty:
            render_no_data_card("No weight data recorded for this user.")
        else:
            goal_weight = st.number_input(
                "Reference goal (kg)",
                min_value=20.0,
                max_value=300.0,
                step=0.5,
                key=goal_key,
            )
            st.plotly_chart(build_weight_trend_chart(weight_data, st.session_state.theme, goal_weight), width="stretch", config=PLOTLY_CONFIG)
    with bmi_column:
        if weight_data.empty or weight_data["BMI"].dropna().empty:
            render_no_data_card("No BMI data recorded for this user.")
        else:
            st.plotly_chart(build_bmi_indicator(weight_data, st.session_state.theme), width="stretch", config=PLOTLY_CONFIG)


def _build_ai_insight_inputs(
    filtered_data: dict[str, pd.DataFrame],
    user_id: str,
    cohort_analysis: dict[str, Any] | None,
    performance_summary: pd.DataFrame,
    attention_flags: pd.DataFrame,
    wellness_snapshot: dict[str, Any],
) -> tuple[str, str, tuple[str, ...]]:
    """Prepare a compact factual prompt and deterministic fallback for five AI dashboard insights."""
    kpis = calculate_kpis(
        filtered_data["activity"],
        filtered_data["sleep"],
        filtered_data["heartrate"],
        filtered_data["weight"],
    )
    metric_summary = "; ".join(f"{kpi['title']}: {kpi['value_label']}" for kpi in kpis)
    wellness_context = f"Wellness score: {_format_metric_number(wellness_snapshot.get('score'), 1)}."
    deterministic_insights = generate_insights(
        filtered_data["activity"],
        filtered_data["sleep"],
        filtered_data["calories"],
        audience_label="Across all users, " if user_id == ALL_USERS_OPTION else "",
    )
    if user_id != ALL_USERS_OPTION or cohort_analysis is None:
        selected_row = _selected_summary_row(performance_summary, user_id)
        rank_context = ""
        if selected_row is not None and pd.notna(selected_row.get("Rank")):
            rank_context = (
                f" Ranked #{int(selected_row['Rank'])} of {performance_summary['UserName'].nunique()} users "
                f"with average Wellness Score {_format_metric_number(selected_row.get('WellnessScore'), 1)}."
            )
        fallback = (*deterministic_insights, "💡 Use the recent KPI pattern to set one achievable movement or recovery goal for the next week.")
        fact_context = (
            f"Selected profile: {user_id}. Fixed source period: {DATA_PERIOD_LABEL}. "
            f"Dashboard KPIs: {metric_summary}. {wellness_context}{rank_context} "
            f"Calculated observations: {' '.join(deterministic_insights)}"
        )
        return "individual", fact_context, fallback

    activity_segments = "; ".join(
        f"{segment}: {count} users" for segment, count in cohort_analysis["activity_segments"].items()
    )
    sleep_segments = "; ".join(
        f"{segment}: {count} users" for segment, count in cohort_analysis["sleep_segments"].items()
    )
    fallback = (
        *deterministic_insights,
        f"💡 The cohort average heart rate is {_format_cohort_value(cohort_analysis['average_heart_rate'], ' bpm')}, which should be interpreted as a descriptive population average rather than a clinical target.",
        f"💡 Activity segmentation shows {activity_segments}.",
        f"💡 Recovery segmentation shows {sleep_segments}.",
    )
    top_user = performance_summary.iloc[0] if not performance_summary.empty else None
    top_context = (
        f"Top performer: {top_user['UserName']} with Wellness Score {_format_metric_number(top_user.get('WellnessScore'), 1)}. "
        if top_user is not None
        else ""
    )
    fact_context = (
        f"Cohort scope: {cohort_analysis['participant_count']} activity profiles across {DATA_PERIOD_LABEL}. "
        f"Dashboard averages: {metric_summary}. Average heart rate: {_format_cohort_value(cohort_analysis['average_heart_rate'], ' bpm')}. "
        f"{wellness_context} {top_context}Users needing attention: {len(attention_flags)}. "
        f"Activity segments: {activity_segments}. Sleep segments: {sleep_segments}. "
        f"Calculated observations: {' '.join(deterministic_insights)}"
    )
    return "all_users", fact_context, fallback


def _render_insights_section(
    filtered_data: dict[str, pd.DataFrame],
    user_id: str,
    cohort_analysis: dict[str, Any] | None,
    performance_summary: pd.DataFrame,
    attention_flags: pd.DataFrame,
    wellness_snapshot: dict[str, Any],
) -> None:
    """Render five data-grounded LLM insights with deterministic fallbacks for unavailable providers."""
    scope, fact_context, fallback_insights = _build_ai_insight_inputs(
        filtered_data,
        user_id,
        cohort_analysis,
        performance_summary,
        attention_flags,
        wellness_snapshot,
    )
    section_title = "AI cohort insights" if scope == "all_users" else "AI fitness insights"
    section_description = (
        "Five data-grounded observations on cohort strengths, opportunities, and segmentation."
        if scope == "all_users"
        else "Five data-grounded observations on activity, recovery, and practical next steps."
    )
    _section_heading(section_title, section_description)
    with st.spinner("Generating data-grounded insights..."):
        insights = generate_llm_insights(scope, fact_context, tuple(fallback_insights))
    insight_markup = "".join(
        f"<div class='fitpulse-insight-item'><span class='fitpulse-insight-number'>{index}.</span>{insight}</div>"
        for index, insight in enumerate(insights, start=1)
    )
    st.markdown(f"<div class='fitpulse-insights'>{insight_markup}</div>", unsafe_allow_html=True)


def _render_footer() -> None:
    """Render the data provenance and application technology footer."""
    st.markdown(
        "<div class='fitpulse-footer'>Data source: Fitbit Fitness Tracker Data (Mobius, CC0) "
        "| Built with Streamlit + LangGraph | © FitPulse AI</div>",
        unsafe_allow_html=True,
    )


def _build_dashboard_context(
    filtered_data: dict[str, pd.DataFrame],
    user_id: str,
    cohort_analysis: dict[str, Any] | None,
    performance_summary: pd.DataFrame,
    attention_flags: pd.DataFrame,
    wellness_snapshot: dict[str, Any],
) -> dict[str, Any]:
    """Build a concise current-dashboard summary for the chat agent's initial context."""
    scope = "all tracked users combined" if user_id == ALL_USERS_OPTION else user_id
    kpis = calculate_kpis(
        filtered_data["activity"],
        filtered_data["sleep"],
        filtered_data["heartrate"],
        filtered_data["weight"],
    )
    kpi_by_key = {kpi["key"]: kpi for kpi in kpis}
    metric_summary = "; ".join(f"{kpi['title']}: {kpi['value_label']}" for kpi in kpis)
    insights = generate_insights(
        filtered_data["activity"],
        filtered_data["sleep"],
        filtered_data["calories"],
        audience_label="Across all users, " if user_id == ALL_USERS_OPTION else "",
    )
    cohort_context = ""
    if cohort_analysis is not None:
        activity_segments = "; ".join(
            f"{segment}: {count} users" for segment, count in cohort_analysis["activity_segments"].items()
        )
        sleep_segments = "; ".join(
            f"{segment}: {count} users" for segment, count in cohort_analysis["sleep_segments"].items()
        )
        cohort_context = (
            f" Cohort averages: {cohort_analysis['participant_count']} activity profiles; "
            f"average heart rate {_format_cohort_value(cohort_analysis['average_heart_rate'], ' bpm')}; "
            f"activity segments [{activity_segments}]; recovery segments [{sleep_segments}]."
        )
    if user_id == ALL_USERS_OPTION:
        top_user = performance_summary.iloc[0] if not performance_summary.empty else None
        chat_kpis = {
            "total_users": int(performance_summary["UserName"].nunique()) if not performance_summary.empty else 0,
            "avg_steps": cohort_analysis["average_daily_steps"] if cohort_analysis else None,
            "avg_sleep": cohort_analysis["average_sleep_efficiency"] if cohort_analysis else None,
            "avg_hr": cohort_analysis["average_heart_rate"] if cohort_analysis else None,
            "avg_wellness": wellness_snapshot.get("score"),
            "top_user": str(top_user["UserName"]) if top_user is not None else "Not available",
            "top_score": top_user.get("WellnessScore") if top_user is not None else None,
            "attention_count": len(attention_flags),
        }
    else:
        selected_row = _selected_summary_row(performance_summary, user_id)
        chat_kpis = {
            "wellness": wellness_snapshot.get("score"),
            "wellness_delta": wellness_snapshot.get("delta"),
            "steps": kpi_by_key.get("steps", {}).get("value"),
            "sleep": kpi_by_key.get("sleep", {}).get("value"),
            "active_mins": kpi_by_key.get("active_minutes", {}).get("value"),
            "resting_hr": kpi_by_key.get("resting_hr", {}).get("value"),
            "streak": selected_row.get("CurrentStreak") if selected_row is not None else None,
            "steps_delta": kpi_by_key.get("steps", {}).get("delta"),
            "rank": selected_row.get("Rank") if selected_row is not None else None,
            "total_users": int(performance_summary["UserName"].nunique()) if not performance_summary.empty else 0,
        }
    return {
        "user_id": user_id,
        "user_name": user_id,
        "is_all_users": user_id == ALL_USERS_OPTION,
        "chat_kpis": chat_kpis,
        "summary": (
            f"Dashboard scope: {scope}. Fixed source period: {DATA_PERIOD_LABEL}. "
            f"Visible KPIs: {metric_summary}.{cohort_context} Insights: {' '.join(insights)}"
        ),
    }


# ── Application entry point ────────────────────────────────────────────────

def render_dashboard() -> dict[str, Any]:
    """Load data, manage filters, and render the complete standalone FitPulse dashboard."""
    try:
        data = load_all_data()
        _initialise_session_state(data)
    except DataLoadError as error:
        st.error(str(error))
        st.stop()
    except Exception as error:
        st.error(f"FitPulse could not load the source data: {error}")
        st.stop()

    _inject_theme(st.session_state.theme)
    selected_user = _render_top_bar(data)
    performance_summary = build_user_performance_summary(
        data["activity"],
        data["sleep"],
        data["heartrate"],
    )
    attention_flags = calculate_attention_flags(
        data["activity"],
        data["sleep"],
        performance_summary,
    )
    filtered_data = _apply_filters(data, selected_user)
    scope_label = "all users combined" if selected_user == ALL_USERS_OPTION else f"athlete {selected_user}"
    st.caption(f"Viewing {scope_label} · fixed dataset period: {DATA_PERIOD_LABEL}")

    wellness_snapshot = _render_wellness_hero(filtered_data, selected_user, performance_summary)
    if selected_user == ALL_USERS_OPTION:
        _render_attention_flags(attention_flags)
    _render_kpi_cards(filtered_data, selected_user, performance_summary, wellness_snapshot)
    cohort_analysis = _render_cohort_section(filtered_data, wellness_snapshot) if selected_user == ALL_USERS_OPTION else None
    if selected_user == ALL_USERS_OPTION:
        _render_leaderboard(performance_summary)
    _render_activity_section(filtered_data)
    _render_calorie_section(filtered_data)
    _render_sleep_section(filtered_data)
    _render_heartrate_section(filtered_data)
    if selected_user != ALL_USERS_OPTION:
        _render_weight_section(filtered_data)
    _render_insights_section(
        filtered_data,
        selected_user,
        cohort_analysis,
        performance_summary,
        attention_flags,
        wellness_snapshot,
    )
    _render_footer()
    return _build_dashboard_context(
        filtered_data,
        selected_user,
        cohort_analysis,
        performance_summary,
        attention_flags,
        wellness_snapshot,
    )


if __name__ == "__main__":
    render_dashboard()
