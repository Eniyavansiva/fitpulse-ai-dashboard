"""KPI calculations and deterministic insight generation for FitPulse AI."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# ── Metric formatting ──────────────────────────────────────────────────────

BMI_CATEGORIES = (
    (18.5, "Underweight"),
    (25.0, "Normal"),
    (30.0, "Overweight"),
    (float("inf"), "Obese"),
)


def format_duration(minutes: float | int | None) -> str:
    """Format a duration in minutes as a compact hours-and-minutes label."""
    if minutes is None or pd.isna(minutes):
        return "—"
    total_minutes = max(int(round(float(minutes))), 0)
    hours, remaining_minutes = divmod(total_minutes, 60)
    return f"{hours}h {remaining_minutes:02d}m"


def get_bmi_category(bmi: float | int | None) -> str:
    """Return the standard BMI category for a numeric BMI measurement."""
    if bmi is None or pd.isna(bmi):
        return "No data"
    for upper_bound, category in BMI_CATEGORIES:
        if float(bmi) < upper_bound:
            return category
    return "No data"


def _format_number(value: float | int | None, decimals: int = 0) -> str:
    """Format a numeric value safely for display in a dashboard card."""
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):,.{decimals}f}"


def _format_delta(delta: float | None) -> str:
    """Format a percent change against the previous week for card display."""
    if delta is None or pd.isna(delta):
        return "No prior-week data"
    direction = "▲" if delta >= 0 else "▼"
    return f"{direction} {abs(delta):.1f}% vs prior week"


# ── Weekly comparisons ─────────────────────────────────────────────────────

def _daily_series(dataframe: pd.DataFrame, date_column: str, value_column: str) -> pd.Series:
    """Aggregate one metric into a sorted daily mean series."""
    if dataframe.empty or date_column not in dataframe or value_column not in dataframe:
        return pd.Series(dtype=float)
    series = (
        dataframe.dropna(subset=[date_column, value_column])
        .groupby(date_column)[value_column]
        .mean()
        .sort_index()
    )
    return series.astype(float)


def _weekly_delta(series: pd.Series) -> float | None:
    """Calculate the latest seven-day mean change relative to the preceding seven days."""
    if series.empty or len(series) < 2:
        return None

    latest_date = pd.Timestamp(series.index.max())
    current_window = series.loc[series.index >= latest_date - pd.Timedelta(days=6)]
    previous_window = series.loc[
        (series.index >= latest_date - pd.Timedelta(days=13))
        & (series.index < latest_date - pd.Timedelta(days=6))
    ]

    if current_window.empty or previous_window.empty:
        return None

    previous_mean = previous_window.mean()
    if pd.isna(previous_mean) or np.isclose(previous_mean, 0):
        return None
    return float((current_window.mean() - previous_mean) / previous_mean * 100)


def _card_payload(
    key: str,
    title: str,
    icon: str,
    value: float | None,
    value_label: str,
    series: pd.Series,
    detail: str,
    positive_is_good: bool = True,
) -> dict[str, Any]:
    """Create the consistent display payload consumed by one KPI card."""
    delta = _weekly_delta(series)
    trend_is_positive = delta is not None and (delta >= 0 if positive_is_good else delta <= 0)
    return {
        "key": key,
        "title": title,
        "icon": icon,
        "value": value,
        "value_label": value_label,
        "delta": delta,
        "delta_label": _format_delta(delta),
        "trend_is_positive": trend_is_positive,
        "sparkline": series.tail(14).tolist(),
        "detail": detail,
    }


# ── KPI calculations ───────────────────────────────────────────────────────

def _resting_hr_daily_series(heartrate: pd.DataFrame) -> pd.Series:
    """Estimate daily resting HR from the lowest quintile of minute-level readings."""
    if heartrate.empty or "Time" not in heartrate or "Value" not in heartrate:
        return pd.Series(dtype=float)
    prepared = heartrate.dropna(subset=["Time", "Value"]).copy()
    prepared["Date"] = pd.to_datetime(prepared["Time"]).dt.normalize()
    return prepared.groupby("Date")["Value"].apply(lambda values: values[values <= values.quantile(0.2)].mean())


def calculate_kpis(
    activity: pd.DataFrame,
    sleep: pd.DataFrame,
    heartrate: pd.DataFrame,
    weight: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Calculate the six primary dashboard KPIs for the active filters."""
    steps_series = _daily_series(activity, "ActivityDate", "TotalSteps")
    calories_series = _daily_series(activity, "ActivityDate", "Calories")
    active_series = _daily_series(
        activity.assign(
            ActiveMinutes=activity.get("VeryActiveMinutes", 0) + activity.get("FairlyActiveMinutes", 0)
        ),
        "ActivityDate",
        "ActiveMinutes",
    )
    sleep_series = _daily_series(sleep, "SleepDay", "SleepEfficiency")
    resting_hr_series = _resting_hr_daily_series(heartrate)
    bmi_series = _daily_series(weight, "MeasurementDate", "BMI")

    average_steps = steps_series.mean() if not steps_series.empty else None
    average_calories = calories_series.mean() if not calories_series.empty else None
    average_sleep_efficiency = sleep_series.mean() if not sleep_series.empty else None
    average_active_minutes = active_series.mean() if not active_series.empty else None
    average_resting_hr = resting_hr_series.mean() if not resting_hr_series.empty else None
    latest_bmi = bmi_series.iloc[-1] if not bmi_series.empty else None

    return [
        _card_payload(
            "steps",
            "Daily Steps",
            "👟",
            average_steps,
            _format_number(average_steps),
            steps_series,
            "Average daily steps · 10,000 goal",
        ),
        _card_payload(
            "calories",
            "Calories Burned",
            "🔥",
            average_calories,
            _format_number(average_calories),
            calories_series,
            "Average daily calories",
        ),
        _card_payload(
            "sleep",
            "Sleep Score",
            "😴",
            average_sleep_efficiency,
            f"{_format_number(average_sleep_efficiency, 0)}%",
            sleep_series,
            "Average sleep efficiency",
        ),
        _card_payload(
            "active_minutes",
            "Active Minutes",
            "⚡",
            average_active_minutes,
            format_duration(average_active_minutes),
            active_series,
            "Very + fairly active minutes",
        ),
        _card_payload(
            "resting_hr",
            "Resting HR",
            "❤️",
            average_resting_hr,
            f"{_format_number(average_resting_hr, 0)} bpm",
            resting_hr_series,
            "Lowest 20% of recorded BPM",
            positive_is_good=False,
        ),
        _card_payload(
            "bmi",
            "BMI",
            "⚖️",
            latest_bmi,
            _format_number(latest_bmi, 1),
            bmi_series,
            get_bmi_category(latest_bmi),
        ),
    ]


# ── Cohort analytics ───────────────────────────────────────────────────────

def _segment_counts(series: pd.Series, bins: list[float], labels: list[str]) -> dict[str, int]:
    """Categorize a numeric user-level series and return every requested segment count."""
    if series.empty:
        return {label: 0 for label in labels}
    segmented = pd.cut(series, bins=bins, labels=labels, right=False)
    counts = segmented.value_counts().reindex(labels, fill_value=0)
    return {label: int(counts[label]) for label in labels}


def calculate_cohort_analysis(
    activity: pd.DataFrame,
    sleep: pd.DataFrame,
    heartrate: pd.DataFrame,
) -> dict[str, Any]:
    """Calculate cohort averages and user segmentation counts for the all-users dashboard view."""
    activity_per_user = (
        activity.dropna(subset=["Id", "TotalSteps"])
        .groupby("Id")["TotalSteps"]
        .mean()
        .astype(float)
    )
    sleep_per_user = (
        sleep.dropna(subset=["Id", "SleepEfficiency"])
        .groupby("Id")["SleepEfficiency"]
        .mean()
        .astype(float)
    )
    activity_labels = [
        "Movement foundation (<7.5k)",
        "Building consistency (7.5k–10k)",
        "Goal achievers (10k+)",
    ]
    sleep_labels = [
        "Recovery needs attention (<75%)",
        "Moderate recovery (75%–85%)",
        "Restorative recovery (85%+)",
    ]
    average_heart_rate = heartrate["Value"].mean() if not heartrate.empty else np.nan
    return {
        "participant_count": int(activity_per_user.index.nunique()),
        "sleep_participant_count": int(sleep_per_user.index.nunique()),
        "average_daily_steps": float(activity_per_user.mean()) if not activity_per_user.empty else np.nan,
        "average_sleep_efficiency": float(sleep_per_user.mean()) if not sleep_per_user.empty else np.nan,
        "average_heart_rate": float(average_heart_rate) if not pd.isna(average_heart_rate) else np.nan,
        "activity_segments": _segment_counts(
            activity_per_user,
            [float("-inf"), 7_500, 10_000, float("inf")],
            activity_labels,
        ),
        "sleep_segments": _segment_counts(
            sleep_per_user,
            [float("-inf"), 75, 85, float("inf")],
            sleep_labels,
        ),
    }


# ── Wellness, streak, and operational analytics ────────────────────────────

def _user_column(dataframe: pd.DataFrame) -> str:
    """Return the preferred anonymous user display column available in a dataframe."""
    return "UserName" if "UserName" in dataframe.columns else "Id"


def compute_wellness_score(
    steps: float,
    sleep_efficiency: float,
    active_minutes: float,
    resting_hr: float,
) -> float:
    """Return the requested 0–100 composite wellness score from daily movement and recovery signals."""
    step_score = min(float(steps) / 10_000, 1.0) * 30
    sleep_score = min(float(sleep_efficiency) / 100, 1.0) * 30
    active_score = min(float(active_minutes) / 60, 1.0) * 20
    hr_score = max(0, (100 - float(resting_hr)) / 60) * 20
    return round(step_score + sleep_score + active_score + hr_score, 1)


def _resting_hr_by_user(heartrate: pd.DataFrame) -> pd.Series:
    """Estimate each user's resting heart rate from the lowest quintile of available BPM readings."""
    if heartrate.empty or "Value" not in heartrate:
        return pd.Series(dtype=float)
    user_column = _user_column(heartrate)
    prepared = heartrate.dropna(subset=[user_column, "Value"])
    return prepared.groupby(user_column)["Value"].apply(
        lambda values: values[values <= values.quantile(0.2)].mean()
    )


def build_wellness_daily(
    activity: pd.DataFrame,
    sleep: pd.DataFrame,
    heartrate: pd.DataFrame,
) -> pd.DataFrame:
    """Build one daily wellness score per user, using available recovery baselines for sparse sources."""
    if activity.empty:
        return pd.DataFrame(columns=["UserName", "ActivityDate", "WellnessScore"])
    user_column = _user_column(activity)
    daily = (
        activity.dropna(subset=[user_column, "ActivityDate", "TotalSteps"])
        .groupby([user_column, "ActivityDate"], as_index=False)[
            ["TotalSteps", "VeryActiveMinutes", "FairlyActiveMinutes"]
        ]
        .mean()
    )
    daily["ActiveMinutes"] = daily["VeryActiveMinutes"].fillna(0) + daily["FairlyActiveMinutes"].fillna(0)
    sleep_user_column = _user_column(sleep) if not sleep.empty else user_column
    sleep_by_user = (
        sleep.dropna(subset=[sleep_user_column, "SleepEfficiency"])
        .groupby(sleep_user_column)["SleepEfficiency"]
        .mean()
        if not sleep.empty
        else pd.Series(dtype=float)
    )
    resting_hr_by_user = _resting_hr_by_user(heartrate)
    fallback_sleep = float(sleep_by_user.median()) if not sleep_by_user.empty else 80.0
    fallback_hr = float(resting_hr_by_user.median()) if not resting_hr_by_user.empty else 70.0
    daily["SleepEfficiency"] = daily[user_column].map(sleep_by_user).fillna(fallback_sleep)
    daily["RestingHR"] = daily[user_column].map(resting_hr_by_user).fillna(fallback_hr)
    daily["WellnessScore"] = daily.apply(
        lambda row: compute_wellness_score(
            row["TotalSteps"],
            row["SleepEfficiency"],
            row["ActiveMinutes"],
            row["RestingHR"],
        ),
        axis=1,
    )
    if user_column != "UserName":
        daily = daily.rename(columns={user_column: "UserName"})
    return daily.sort_values(["UserName", "ActivityDate"]).reset_index(drop=True)


def _streak_lengths(daily_steps: pd.Series) -> tuple[int, int]:
    """Return current and best consecutive-day streaks for days meeting the 7,500-step threshold."""
    current_streak = 0
    best_streak = 0
    running_streak = 0
    previous_date: pd.Timestamp | None = None
    for activity_date, steps in daily_steps.sort_index().items():
        normalized_date = pd.Timestamp(activity_date).normalize()
        is_consecutive = previous_date is not None and normalized_date - previous_date == pd.Timedelta(days=1)
        if float(steps) >= 7_500:
            running_streak = running_streak + 1 if is_consecutive else 1
        else:
            running_streak = 0
        best_streak = max(best_streak, running_streak)
        current_streak = running_streak
        previous_date = normalized_date
    return current_streak, best_streak


def calculate_streaks(activity: pd.DataFrame) -> pd.DataFrame:
    """Calculate current and best 7,500-step streaks for every available user."""
    if activity.empty:
        return pd.DataFrame(columns=["UserName", "CurrentStreak", "BestStreak"])
    user_column = _user_column(activity)
    daily = (
        activity.dropna(subset=[user_column, "ActivityDate", "TotalSteps"])
        .groupby([user_column, "ActivityDate"])["TotalSteps"]
        .mean()
    )
    records = []
    for user_name, user_steps in daily.groupby(level=0):
        steps_by_date = user_steps.droplevel(0)
        current_streak, best_streak = _streak_lengths(steps_by_date)
        records.append(
            {
                "UserName": str(user_name),
                "CurrentStreak": current_streak,
                "BestStreak": best_streak,
            }
        )
    return pd.DataFrame(records)


def _weekly_delta_by_user(activity: pd.DataFrame) -> pd.Series:
    """Calculate each user's latest weekly percentage change in average daily steps."""
    if activity.empty:
        return pd.Series(dtype=float)
    user_column = _user_column(activity)
    deltas: dict[str, float | None] = {}
    for user_name, user_data in activity.groupby(user_column):
        series = _daily_series(user_data, "ActivityDate", "TotalSteps")
        deltas[str(user_name)] = _weekly_delta(series)
    return pd.Series(deltas, dtype=float)


def build_user_performance_summary(
    activity: pd.DataFrame,
    sleep: pd.DataFrame,
    heartrate: pd.DataFrame,
) -> pd.DataFrame:
    """Build a rankable user summary containing wellness, steps, sleep, HR, and streak indicators."""
    if activity.empty:
        return pd.DataFrame()
    user_column = _user_column(activity)
    wellness_daily = build_wellness_daily(activity, sleep, heartrate)
    summary = (
        activity.groupby(user_column)["TotalSteps"]
        .mean()
        .rename("AvgSteps")
        .reset_index()
        .rename(columns={user_column: "UserName"})
    )
    wellness_summary = wellness_daily.groupby("UserName")["WellnessScore"].mean().rename("WellnessScore")
    sleep_user_column = _user_column(sleep) if not sleep.empty else user_column
    sleep_summary = (
        sleep.groupby(sleep_user_column)["SleepEfficiency"].mean().rename("SleepScore")
        if not sleep.empty
        else pd.Series(dtype=float, name="SleepScore")
    )
    sleep_summary.index.name = "UserName"
    resting_summary = _resting_hr_by_user(heartrate).rename("RestingHR")
    resting_summary.index.name = "UserName"
    last_activity = activity.groupby(user_column)["ActivityDate"].max().rename("LastActivityDate")
    last_activity.index.name = "UserName"
    summary = summary.merge(wellness_summary, left_on="UserName", right_index=True, how="left")
    summary = summary.merge(sleep_summary, left_on="UserName", right_index=True, how="left")
    summary = summary.merge(resting_summary, left_on="UserName", right_index=True, how="left")
    summary = summary.merge(last_activity, left_on="UserName", right_index=True, how="left")
    summary = summary.merge(calculate_streaks(activity), on="UserName", how="left")
    summary["StepsDelta"] = summary["UserName"].map(_weekly_delta_by_user(activity))
    wellness_deltas: dict[str, float | None] = {}
    for user_name, user_data in wellness_daily.groupby("UserName"):
        wellness_deltas[str(user_name)] = _weekly_delta(
            user_data.set_index("ActivityDate")["WellnessScore"].sort_index()
        )
    summary["WellnessDelta"] = summary["UserName"].map(wellness_deltas)
    summary["Rank"] = summary["WellnessScore"].rank(method="min", ascending=False).astype("Int64")
    return summary.sort_values(["Rank", "UserName"]).reset_index(drop=True)


def calculate_attention_flags(
    activity: pd.DataFrame,
    sleep: pd.DataFrame,
    performance_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Return all users triggering weekly-step, sleep, or recent-activity attention criteria."""
    if performance_summary.empty or activity.empty:
        return pd.DataFrame(columns=["Name", "Issue", "Severity"])
    latest_activity_date = pd.to_datetime(activity["ActivityDate"]).max()
    no_activity_cutoff = latest_activity_date - pd.Timedelta(days=2)
    records = []
    for _, user in performance_summary.iterrows():
        issues = []
        if pd.notna(user["StepsDelta"]) and user["StepsDelta"] < -20:
            issues.append(f"Steps down {abs(user['StepsDelta']):.0f}% week-over-week")
        if pd.notna(user["SleepScore"]) and user["SleepScore"] < 75:
            issues.append(f"Sleep efficiency {user['SleepScore']:.0f}%")
        if pd.isna(user["LastActivityDate"]) or user["LastActivityDate"] < no_activity_cutoff:
            issues.append("No activity in the final 3 dataset days")
        if issues:
            severity = "🔴 High" if len(issues) >= 2 else "🟡 Watch"
            records.append(
                {
                    "Name": user["UserName"],
                    "Issue": " · ".join(issues),
                    "Severity": severity,
                }
            )
    return pd.DataFrame(records)


def get_wellness_snapshot(wellness_daily: pd.DataFrame) -> dict[str, float | None | list[float]]:
    """Summarize current wellness, weekly delta, and a recent trend series for hero display."""
    if wellness_daily.empty:
        return {"score": None, "delta": None, "sparkline": []}
    daily = wellness_daily.groupby("ActivityDate")["WellnessScore"].mean().sort_index()
    latest_date = pd.Timestamp(daily.index.max())
    current_score = float(daily.loc[daily.index >= latest_date - pd.Timedelta(days=6)].mean())
    return {
        "score": current_score,
        "delta": _weekly_delta(daily),
        "sparkline": daily.tail(14).tolist(),
    }


# ── Insight generation ─────────────────────────────────────────────────────

def _weekday_steps_insight(activity: pd.DataFrame) -> str | None:
    """Identify the weekday that differs most positively from overall step behavior."""
    if activity.empty or "TotalSteps" not in activity:
        return None
    prepared = activity.dropna(subset=["ActivityDate", "TotalSteps"]).copy()
    if prepared.empty:
        return None
    prepared["Weekday"] = pd.to_datetime(prepared["ActivityDate"]).dt.day_name()
    overall_average = prepared["TotalSteps"].mean()
    if pd.isna(overall_average) or np.isclose(overall_average, 0):
        return None
    weekday_average = prepared.groupby("Weekday")["TotalSteps"].mean()
    strongest_day = weekday_average.idxmax()
    difference = (weekday_average.max() - overall_average) / overall_average * 100
    if difference < 5:
        return None
    return f"💡 Daily steps are {difference:.0f}% higher on {strongest_day}s than the overall average."


def _sleep_activity_insight(activity: pd.DataFrame, sleep: pd.DataFrame) -> str | None:
    """Surface a sleep pattern following high-step days when the source overlap permits it."""
    if activity.empty or sleep.empty:
        return None
    activity_daily = (
        activity[["ActivityDate", "TotalSteps"]]
        .dropna()
        .groupby("ActivityDate", as_index=False)["TotalSteps"]
        .mean()
    )
    sleep_daily = (
        sleep[["SleepDay", "SleepEfficiency"]]
        .dropna()
        .groupby("SleepDay", as_index=False)["SleepEfficiency"]
        .mean()
    )
    activity_daily["SleepDay"] = pd.to_datetime(activity_daily["ActivityDate"]) + pd.Timedelta(days=1)
    paired = sleep_daily.merge(activity_daily[["SleepDay", "TotalSteps"]], on="SleepDay", how="inner")
    high_step_nights = paired.loc[paired["TotalSteps"] >= 15_000, "SleepEfficiency"]
    if high_step_nights.empty or high_step_nights.mean() >= 75:
        return None
    return f"💡 Sleep efficiency averages {high_step_nights.mean():.0f}% after 15,000+ step days."


def _calorie_timing_insight(calories: pd.DataFrame) -> str | None:
    """Describe the hour with the strongest average calorie burn."""
    if calories.empty or "Hour" not in calories or "Calories" not in calories:
        return None
    hourly_average = calories.dropna(subset=["Hour", "Calories"]).groupby("Hour")["Calories"].mean()
    if hourly_average.empty:
        return None
    peak_hour = int(hourly_average.idxmax())
    end_hour = (peak_hour + 1) % 24
    start_time = f"{peak_hour % 12 or 12} {'AM' if peak_hour < 12 else 'PM'}"
    end_time = f"{end_hour % 12 or 12} {'AM' if end_hour < 12 else 'PM'}"
    return f"💡 Peak calorie burn consistently occurs between {start_time}–{end_time}."


def _sedentary_trend_insight(activity: pd.DataFrame) -> str | None:
    """Compare sedentary minutes in the latest week with the preceding available week."""
    sedentary_series = _daily_series(activity, "ActivityDate", "SedentaryMinutes")
    delta = _weekly_delta(sedentary_series)
    if delta is None or abs(delta) < 5:
        return None
    direction = "increased" if delta > 0 else "decreased"
    emoji = "⚠️" if delta > 0 else "💡"
    return f"{emoji} Sedentary time {direction} {abs(delta):.0f}% compared with the prior week."


def _apply_audience_label(insight: str, audience_label: str) -> str:
    """Insert an audience label after an insight emoji while preserving natural sentence casing."""
    emoji, content = insight.split(" ", maxsplit=1)
    normalized_content = content[:1].lower() + content[1:]
    return f"{emoji} {audience_label}{normalized_content}"


def generate_insights(
    activity: pd.DataFrame,
    sleep: pd.DataFrame,
    calories: pd.DataFrame,
    audience_label: str = "",
) -> list[str]:
    """Generate concise calculation-based insights without calling an LLM."""
    candidates = [
        _weekday_steps_insight(activity),
        _sleep_activity_insight(activity, sleep),
        _calorie_timing_insight(calories),
        _sedentary_trend_insight(activity),
    ]
    insights = [insight for insight in candidates if insight]
    if not insights:
        insights = ["💡 Keep logging activity and sleep to unlock stronger personal patterns."]
    if not audience_label:
        return insights
    return [_apply_audience_label(insight, audience_label) for insight in insights]
