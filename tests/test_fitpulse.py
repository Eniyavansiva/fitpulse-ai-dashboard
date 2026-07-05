"""Basic regression tests for FitPulse AI data helpers and wellness metrics."""

from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.data_loader import (
    DATA_DIRECTORY,
    USER_NAME_MAP,
    get_user_filtered,
    load_daily_activity,
    load_sleep,
    load_weight,
)
from utils.metrics import compute_wellness_score


def _dataset_path(filename: str) -> str:
    """Return an absolute path to a CSV in the project DataSet folder."""
    return os.path.join(DATA_DIRECTORY, filename)


def test_daily_activity_loads() -> None:
    """Daily activity CSV must load with expected columns."""
    dataframe = pd.read_csv(_dataset_path("dailyActivity_cleaned.csv"))
    required_columns = {"Id", "TotalSteps", "Calories", "VeryActiveMinutes"}
    assert required_columns.issubset(dataframe.columns)
    assert not dataframe.empty


def test_sleep_loads() -> None:
    """Sleep CSV must load and include sleep-duration columns."""
    dataframe = pd.read_csv(_dataset_path("sleep_cleaned.csv"))
    assert "TotalMinutesAsleep" in dataframe.columns
    assert not dataframe.empty


def test_user_name_map_covers_all_activity_ids() -> None:
    """Every activity user ID must have a friendly name mapping."""
    dataframe = pd.read_csv(_dataset_path("dailyActivity_cleaned.csv"))
    missing_ids = set(dataframe["Id"].unique()) - set(USER_NAME_MAP.keys())
    assert not missing_ids, f"Missing USER_NAME_MAP entries: {missing_ids}"


def test_core_loaded_data_uses_user_names() -> None:
    """Core loaded datasets should include non-numeric user display names."""
    data = {
        "activity": load_daily_activity(),
        "sleep": load_sleep(),
        "weight": load_weight(),
    }
    for name, dataframe in data.items():
        assert "UserName" in dataframe.columns, name
        assert not dataframe["UserName"].astype(str).str.fullmatch(r"\d+").any(), name


def test_heartrate_file_loads() -> None:
    """Heart-rate file must load and include Id and Value columns."""
    path = _dataset_path("heartrate_seconds_merged.csv")
    if not os.path.exists(path):
        pytest.skip("Heart-rate file not present")
    dataframe = pd.read_csv(path, usecols=["Id", "Value"])
    assert {"Id", "Value"}.issubset(dataframe.columns)
    assert not dataframe.empty


def test_get_user_filtered_returns_data_when_date_too_narrow() -> None:
    """Date filtering should fall back to full user data if the date range removes all rows."""
    dataframe = pd.read_csv(_dataset_path("dailyActivity_cleaned.csv"))
    dataframe["ActivityDate"] = pd.to_datetime(dataframe["ActivityDate"])
    first_user_id = dataframe["Id"].iloc[0]
    result = get_user_filtered(
        dataframe,
        first_user_id,
        "ActivityDate",
        start_date=pd.Timestamp("2000-01-01"),
        end_date=pd.Timestamp("2000-01-02"),
    )
    assert result is not None
    assert not result.empty


def test_get_user_filtered_returns_none_for_missing_user() -> None:
    """Missing users should return None, not an empty dataframe."""
    dataframe = pd.read_csv(_dataset_path("dailyActivity_cleaned.csv"))
    dataframe["ActivityDate"] = pd.to_datetime(dataframe["ActivityDate"])
    result = get_user_filtered(dataframe, 9999999999, "ActivityDate")
    assert result is None


def test_wellness_score_range() -> None:
    """Wellness score must stay within 0 to 100."""
    score = compute_wellness_score(
        steps=8000,
        sleep_efficiency=85,
        active_minutes=45,
        resting_hr=65,
    )
    assert 0 <= score <= 100


def test_wellness_score_perfect() -> None:
    """Strong input values should produce a near-perfect score."""
    score = compute_wellness_score(
        steps=10000,
        sleep_efficiency=100,
        active_minutes=60,
        resting_hr=40,
    )
    assert score >= 95


def test_wellness_score_zero() -> None:
    """Zero movement and recovery values should produce zero."""
    score = compute_wellness_score(
        steps=0,
        sleep_efficiency=0,
        active_minutes=0,
        resting_hr=100,
    )
    assert score == 0
