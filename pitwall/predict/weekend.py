"""Weekend features from OpenF1 (2023+) and weather from Open-Meteo.

Nothing here may leak the future: practice data feeds the 'practice' moment,
qualifying data the 'quali' moment, and race data (RaceWet, Neutralised) is kept
only as an outcome, or as circuit history for LATER races.
"""
from __future__ import annotations

import itertools
import logging
from collections.abc import Callable

import numpy as np
import pandas as pd

from ..errors import DataNotAvailableError
from ..sources.base import ForecastSource, Record, WeekendSource
from ..sources.openf1 import FIRST_YEAR as OPENF1_FIRST_YEAR

log = logging.getLogger(__name__)

PRACTICE_SESSIONS = ("Practice 1", "Practice 2", "Practice 3")
WET_COMPOUNDS = {"INTERMEDIATE", "WET"}
DEFAULT_START = "13:00"  # UTC, if a race has no start time
RACE_HOURS = 2           # the weather window after the start
RACE_LEVEL = ["RaceId", "has_openf1", "wet_practice", "wet_quali", "track_temp_practice",
              "track_temp_quali", "RaceWet", "Neutralised"]
DRIVER_LEVEL = ["RaceId", "Number", "practice_pace_gap_pct", "has_practice_pace"]
FORECAST_COLUMNS = ["RaceId", "Lead", "air_temp_forecast", "rain_forecast_mm"]
OBSERVED_COLUMNS = ["RaceId", "RaceRainMm"]


# ---------- practice pace ----------
def long_run_pace(laps: pd.DataFrame, min_laps: int = 5, tolerance: float = 1.05) -> pd.Series:
    """Typical long-run lap time per car number in one session.

    A run is the laps between two pit exits. Laps over `tolerance` x the run's median
    (cool-down laps, traffic) are dropped; runs with fewer than `min_laps` laps left
    (qualifying simulations) are ignored. A driver's pace is his best run median.
    """
    if laps.empty:
        return pd.Series(dtype=float)
    laps = laps.sort_values(["Number", "LapNumber"])
    out_lap = laps["PitOutLap"].eq(True)
    laps = laps.assign(Run=out_lap.groupby(laps["Number"]).cumsum())
    laps = laps[~out_lap & laps["LapSeconds"].notna()].astype({"LapSeconds": float})
    typical = laps.groupby(["Number", "Run"])["LapSeconds"].transform("median")
    steady = laps[laps["LapSeconds"] <= typical * tolerance]
    runs = steady.groupby(["Number", "Run"])["LapSeconds"].agg(["median", "size"])
    runs = runs[runs["size"] >= min_laps]
    return runs["median"].groupby(level="Number").min()


def practice_pace_gaps(sessions_laps: list[list[Record]], min_laps: int = 5) -> dict[int, float]:
    """Long-run gap (%) to the session's best, averaged over practice sessions."""
    gaps = []
    for laps in sessions_laps:
        pace = long_run_pace(pd.DataFrame(laps), min_laps)
        if len(pace) >= 2:
            gaps.append((pace / pace.min() - 1) * 100)
    if not gaps:
        return {}
    mean = pd.concat(gaps).groupby(level=0).mean()
    return {int(number): float(gap) for number, gap in mean.items()}


# ---------- OpenF1 weekends ----------
def meeting_sessions(sessions: list[Record], race_date: pd.Timestamp) -> dict[str, Record]:
    """Sessions of the meeting whose race is within a day of race_date, by name."""
    for session in sessions:
        if (session.get("Name") == "Race" and not session.get("Cancelled")
                and _days_apart(session.get("DateStart"), race_date) <= 1):
            key = session.get("MeetingKey")
            return {s["Name"]: s for s in sessions
                    if s.get("MeetingKey") == key and not s.get("Cancelled")}
    return {}


def weekend_summary(source: WeekendSource,
                    sessions: dict[str, Record]) -> tuple[Record, dict[int, float]]:
    """Race-level weekend facts plus each car's practice long-run gap (~11 requests)."""
    practice = [sessions[name] for name in PRACTICE_SESSIONS if name in sessions]
    quali = sessions.get("Qualifying")
    race = sessions["Race"]
    practice_stints = list(itertools.chain.from_iterable(
        _fetch(source.get_stints, s) for s in practice))
    summary: Record = {
        "has_openf1": 1,
        "wet_practice": _wet(practice_stints),
        "wet_quali": _wet(_fetch(source.get_stints, quali)) if quali else None,
        "track_temp_practice": _latest_track_temp(source, practice),
        "track_temp_quali": (_mean_track_temp(_fetch(source.get_weather, quali))
                             if quali else None),
        "RaceWet": _wet(_fetch(source.get_stints, race)),
        "Neutralised": _neutralised(_fetch(source.get_race_control, race)),
    }
    laps = [_fetch(source.get_laps, s) for s in practice]
    return summary, practice_pace_gaps(laps)


def collect_weekends(races: pd.DataFrame,
                     source: WeekendSource) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(race-level table, driver-level table) for every 2023+ race in `races`."""
    sessions_by_year: dict[int, list[Record]] = {}
    race_rows: list[Record] = []
    driver_rows: list[Record] = []
    for race in races.itertuples(index=False):
        year = int(race.Year)
        if year < OPENF1_FIRST_YEAR:
            continue
        if year not in sessions_by_year:
            sessions_by_year[year] = source.get_sessions(year)
        sessions = meeting_sessions(sessions_by_year[year], race.Date)
        if "Race" not in sessions:
            log.warning("No OpenF1 sessions for race %s on %s", race.RaceId, race.Date.date())
            continue
        summary, gaps = weekend_summary(source, sessions)
        race_rows.append({"RaceId": race.RaceId, **summary})
        driver_rows += [{"RaceId": race.RaceId, "Number": number,
                         "practice_pace_gap_pct": gap, "has_practice_pace": 1}
                        for number, gap in gaps.items()]
    return (_typed(pd.DataFrame(race_rows, columns=RACE_LEVEL)),
            _typed(pd.DataFrame(driver_rows, columns=DRIVER_LEVEL)))


# ---------- Open-Meteo ----------
def race_window(date: pd.Timestamp, start: object) -> tuple[pd.Timestamp, pd.Timestamp]:
    """(start, end) of the race in UTC, from the date and an 'HH:MM' start time."""
    text = start if isinstance(start, str) and ":" in start else DEFAULT_START
    hours, minutes = (int(part) for part in text.split(":")[:2])
    begin = date.normalize() + pd.Timedelta(hours=hours, minutes=minutes)
    return begin, begin + pd.Timedelta(hours=RACE_HOURS)


def collect_forecasts(races: pd.DataFrame, source: ForecastSource) -> pd.DataFrame:
    """Per race and lead day: mean forecast air temp and total forecast rain in the window."""
    rows = []
    for race in races.itertuples(index=False):
        if pd.isna(race.Lat) or pd.isna(race.Long):
            continue
        forecast = source.get_race_forecast(float(race.Lat), float(race.Long),
                                            race.Date.strftime("%Y-%m-%d"))
        hourly = _in_window(forecast, race.Date, race.Time)
        if hourly.empty:
            continue
        for lead, group in hourly.groupby("LeadDays"):
            rows.append({"RaceId": race.RaceId, "Lead": int(lead),
                         "air_temp_forecast": pd.to_numeric(group["AirTemp"]).mean(),
                         "rain_forecast_mm": pd.to_numeric(group["RainMm"]).sum()})
    return _typed(pd.DataFrame(rows, columns=FORECAST_COLUMNS))


def collect_observed(races: pd.DataFrame, source: ForecastSource) -> pd.DataFrame:
    """Rain that actually fell during each race (a diagnostic, never a feature)."""
    rows = []
    for race in races.itertuples(index=False):
        if pd.isna(race.Lat) or pd.isna(race.Long):
            continue
        observed = source.get_observed(float(race.Lat), float(race.Long),
                                       race.Date.strftime("%Y-%m-%d"))
        hourly = _in_window(observed, race.Date, race.Time)
        if not hourly.empty:
            rows.append({"RaceId": race.RaceId,
                         "RaceRainMm": pd.to_numeric(hourly["RainMm"]).sum()})
    return _typed(pd.DataFrame(rows, columns=OBSERVED_COLUMNS))


def empty_frame(columns: list[str]) -> pd.DataFrame:
    """A typed empty table, used when a source is not given."""
    return _typed(pd.DataFrame(columns=columns))


# ---------- helpers ----------
def _fetch(method: Callable[[Record], list[Record]], session: Record) -> list[Record]:
    """Call a source method; a session without data is skipped, not fatal."""
    try:
        return method(session)
    except DataNotAvailableError as exc:
        log.warning("Skipping %s: %s", session.get("Name"), exc)
        return []


def _wet(stints: list[Record]) -> int | None:
    if not stints:
        return None
    return int(any(str(s.get("Compound")).upper() in WET_COMPOUNDS for s in stints))


def _latest_track_temp(source: WeekendSource, practice: list[Record]) -> float | None:
    """Mean track temperature of the latest practice session that has weather data."""
    for session in reversed(practice):
        temp = _mean_track_temp(_fetch(source.get_weather, session))
        if temp is not None:
            return temp
    return None


def _mean_track_temp(weather: list[Record]) -> float | None:
    temps = [w["TrackTemp"] for w in weather if w.get("TrackTemp") is not None]
    return float(np.mean(temps)) if temps else None


def _neutralised(control: list[Record]) -> int | None:
    """1 if a safety car or VSC was deployed or a red flag shown, else 0."""
    if not control:
        return None
    for row in control:
        message = str(row.get("Message") or "").upper()
        if row.get("Category") == "SafetyCar" and "DEPLOYED" in message:
            return 1
        if str(row.get("Flag") or "").upper() == "RED":
            return 1
    return 0


def _days_apart(text: object, date: pd.Timestamp) -> int:
    if not text:
        return 99
    return abs((pd.Timestamp(str(text)[:10]) - date.normalize()).days)


def _in_window(rows: list[Record], date: pd.Timestamp, start: object) -> pd.DataFrame:
    """Hourly rows inside the race window (times in UTC)."""
    hourly = pd.DataFrame(rows)
    if hourly.empty:
        return hourly
    begin, end = race_window(date, start)
    times = pd.to_datetime(hourly["Time"], utc=True).dt.tz_localize(None)
    return hourly[(times >= begin) & (times < end)]


def _typed(frame: pd.DataFrame) -> pd.DataFrame:
    """RaceId/Lead as int, Number as nullable int, the rest as float, so merges line up."""
    types: dict[str, str] = {}
    for column in frame.columns:
        if column in ("RaceId", "Lead"):
            types[column] = "int64"
        elif column == "Number":
            types[column] = "Int64"
        else:
            types[column] = "float64"
    return frame.astype(types)
