"""Cached data loading and preprocessing utilities for FitPulse AI."""

from __future__ import annotations

import os
from collections.abc import Iterable

import pandas as pd
import streamlit as st


# ── Data paths and errors ──────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIRECTORY = os.path.join(PROJECT_ROOT, "DataSet")
USER_NAME_MAP = {
    1503960366: "Aria Chen",
    1624580081: "Blake Torres",
    1644430081: "Casey Morgan",
    1844505072: "Dana Kim",
    1927972279: "Eli Patel",
    2022484408: "Fiona Walsh",
    2026352035: "Grey Okafor",
    2320127002: "Harper Singh",
    2347167796: "Indira Novak",
    2873212765: "Juno Park",
    2891001357: "Mira Solis",
    3372868164: "Kai Fernandez",
    3977333714: "Lena Russo",
    4020332650: "Marcus Bell",
    4057192912: "Nadia Osei",
    4319703577: "Oscar Yuen",
    4388161847: "Priya Dahl",
    4445114986: "Quinn Adler",
    4558609924: "Noah Vance",
    4702921684: "Remy Santos",
    5553957443: "Sage Muller",
    5577150313: "Tara Jensen",
    6117666160: "Uma Reeves",
    6290855005: "Victor Lam",
    6391747486: "Rhea Kapoor",
    6775888955: "Wren Diaz",
    6962181067: "Xander Cole",
    7007744171: "Yuki Brennan",
    7086361926: "Zara Nkosi",
    8053475328: "Arlo Mistry",
    8253242879: "Theo Marin",
    8378563200: "Bea Larsson",
    8583815059: "Cruz Vance",
    8792009665: "Demi Huang",
    8877689391: "Willow Grant",
}


class DataLoadError(RuntimeError):
    """Represent a clear, user-facing failure while loading a source dataset."""


def get_data_path(filename: str) -> str:
    """Return the absolute path for a CSV file stored in the project dataset folder."""
    return os.path.join(DATA_DIRECTORY, filename)


def _read_csv(filename: str, **read_csv_options: object) -> pd.DataFrame:
    """Read one CSV file and raise a descriptive error when it is unavailable or invalid."""
    file_path = get_data_path(filename)

    if not os.path.exists(file_path):
        raise DataLoadError(
            f"Could not find '{filename}' in the DataSet folder. Expected path: {file_path}"
        )

    try:
        dataframe = pd.read_csv(file_path, **read_csv_options)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as error:
        raise DataLoadError(f"Could not read '{filename}': {error}") from error

    if dataframe.empty:
        raise DataLoadError(f"'{filename}' is empty, so FitPulse cannot render this dataset.")

    return dataframe


def _require_columns(dataframe: pd.DataFrame, columns: Iterable[str], filename: str) -> None:
    """Validate required columns before a dataset is processed further."""
    missing_columns = sorted(set(columns).difference(dataframe.columns))
    if missing_columns:
        raise DataLoadError(
            f"'{filename}' is missing required columns: {', '.join(missing_columns)}."
        )


def _normalize_user_ids(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with stable internal IDs and an anonymous human-readable display name."""
    normalized = dataframe.copy()
    numeric_ids = pd.to_numeric(normalized["Id"], errors="coerce").astype("Int64")
    normalized["Id"] = normalized["Id"].astype("string")
    normalized["UserName"] = numeric_ids.map(USER_NAME_MAP).fillna("Unassigned profile")
    return normalized


def _coerce_numeric_columns(dataframe: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Return a copy with the supplied columns safely converted to numeric values."""
    normalized = dataframe.copy()
    for column in columns:
        if column in normalized:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    return normalized


# ── Dataset loaders ────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_daily_activity() -> pd.DataFrame:
    """Load and normalize daily Fitbit activity records for all available users."""
    filename = "dailyActivity_cleaned.csv"
    dataframe = _read_csv(filename)
    _require_columns(
        dataframe,
        [
            "Id",
            "ActivityDate",
            "TotalSteps",
            "VeryActiveMinutes",
            "FairlyActiveMinutes",
            "LightlyActiveMinutes",
            "SedentaryMinutes",
            "Calories",
        ],
        filename,
    )
    dataframe = _normalize_user_ids(dataframe)
    dataframe["ActivityDate"] = pd.to_datetime(dataframe["ActivityDate"], errors="coerce").dt.normalize()
    dataframe = _coerce_numeric_columns(
        dataframe,
        [
            "TotalSteps",
            "TotalDistance",
            "VeryActiveMinutes",
            "FairlyActiveMinutes",
            "LightlyActiveMinutes",
            "SedentaryMinutes",
            "Calories",
        ],
    )
    return dataframe.dropna(subset=["Id", "ActivityDate"]).sort_values(["Id", "ActivityDate"])


@st.cache_data(show_spinner=False)
def load_sleep() -> pd.DataFrame:
    """Load sleep sessions and calculate nightly sleep efficiency as a percentage."""
    filename = "sleep_cleaned.csv"
    dataframe = _read_csv(filename)
    _require_columns(
        dataframe,
        ["Id", "SleepDay", "TotalSleepRecords", "TotalMinutesAsleep", "TotalTimeInBed"],
        filename,
    )
    dataframe = _normalize_user_ids(dataframe)
    dataframe["SleepDay"] = pd.to_datetime(dataframe["SleepDay"], errors="coerce").dt.normalize()
    dataframe = _coerce_numeric_columns(
        dataframe,
        ["TotalSleepRecords", "TotalMinutesAsleep", "TotalTimeInBed"],
    )
    dataframe["SleepEfficiency"] = (
        dataframe["TotalMinutesAsleep"].div(dataframe["TotalTimeInBed"].replace(0, pd.NA)).mul(100)
    )
    return dataframe.dropna(subset=["Id", "SleepDay"]).sort_values(["Id", "SleepDay"])


@st.cache_data(show_spinner=False)
def load_weight() -> pd.DataFrame:
    """Load weight measurements, preserving their timestamp and a normalized calendar date."""
    filename = "weight_cleaned.csv"
    dataframe = _read_csv(filename)
    _require_columns(dataframe, ["Id", "Date", "WeightKg", "BMI"], filename)
    dataframe = _normalize_user_ids(dataframe)
    dataframe["Date"] = pd.to_datetime(dataframe["Date"], errors="coerce")
    dataframe = _coerce_numeric_columns(dataframe, ["WeightKg", "BMI"])
    dataframe["MeasurementDate"] = dataframe["Date"].dt.normalize()
    return dataframe.dropna(subset=["Id", "Date"]).sort_values(["Id", "Date"])


@st.cache_data(show_spinner=False)
def load_hourly_calories() -> pd.DataFrame:
    """Load hourly calorie records with normalized date and hour fields for heatmap analysis."""
    filename = "hourlyCalories_cleaned.csv"
    dataframe = _read_csv(filename)
    _require_columns(dataframe, ["Id", "ActivityHour", "Calories"], filename)
    dataframe = _normalize_user_ids(dataframe)
    dataframe["ActivityHour"] = pd.to_datetime(dataframe["ActivityHour"], errors="coerce")
    dataframe = _coerce_numeric_columns(dataframe, ["Calories"])
    dataframe["Date"] = dataframe["ActivityHour"].dt.normalize()
    dataframe["Hour"] = dataframe["ActivityHour"].dt.hour
    return dataframe.dropna(subset=["Id", "ActivityHour"]).sort_values(["Id", "ActivityHour"])


# ── Heart-rate preprocessing ───────────────────────────────────────────────

def clean_heartrate(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate and implausible readings, then aggregate second-level BPM into minutes."""
    _require_columns(dataframe, ["Id", "Time", "Value"], "heartrate_seconds_merged.csv")
    cleaned = dataframe.drop_duplicates().copy()
    cleaned = _normalize_user_ids(cleaned)
    cleaned["Time"] = pd.to_datetime(cleaned["Time"], errors="coerce")
    cleaned["Value"] = pd.to_numeric(cleaned["Value"], errors="coerce")
    cleaned = cleaned.dropna(subset=["Id", "Time", "Value"])
    cleaned = cleaned.loc[cleaned["Value"].between(30, 220)]
    cleaned["Time"] = cleaned["Time"].dt.floor("min")
    cleaned = cleaned.groupby(["Id", "UserName", "Time"], as_index=False)["Value"].mean()
    return cleaned.sort_values(["Id", "Time"]).reset_index(drop=True)


@st.cache_data(show_spinner="Cleaning heart-rate data...")
def load_heartrate() -> pd.DataFrame:
    """Load and minute-resample Fitbit heart-rate records after quality filtering."""
    filename = "heartrate_seconds_merged.csv"
    dataframe = _read_csv(filename, usecols=["Id", "Time", "Value"])
    return clean_heartrate(dataframe)


@st.cache_data(show_spinner="Loading Fitbit datasets...")
def load_all_data() -> dict[str, pd.DataFrame]:
    """Load each dashboard dataset once and return it under a stable descriptive key."""
    return {
        "activity": load_daily_activity(),
        "sleep": load_sleep(),
        "weight": load_weight(),
        "calories": load_hourly_calories(),
        "heartrate": load_heartrate(),
    }
