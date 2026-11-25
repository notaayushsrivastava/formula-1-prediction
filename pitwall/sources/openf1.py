"""Client for OpenF1 (api.openf1.org): free HISTORICAL session data from 2023 onwards.

OpenF1 treats data from 30 minutes before a session until 30 minutes after it as
live, and live data is a paid tier. pitwall never uses it: the client refuses any
session that ended less than an hour ago, before sending a single request.
Data licence: CC BY-NC-SA 4.0 (non-commercial use).
"""
from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from typing import Any

from ..cache import DiskCache
from ..errors import DataNotAvailableError
from .base import Record
from .http_json import JsonHttp
from .jolpica import RateLimiter

BASE_URL = "https://api.openf1.org/v1"
FIRST_YEAR = 2023
LIVE_MARGIN = dt.timedelta(hours=1)   # OpenF1's live window ends 30 min after a session
SETTLED_AFTER = dt.timedelta(days=3)  # after this, a session's data is cached forever
_LIMITER = RateLimiter(per_second=0.5, per_hour=1800)  # free tier: 3 req/s AND 30 req/min


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def parse_utc(text: str) -> dt.datetime:
    """'2024-03-02T15:00:00+00:00' (or ...Z, or no zone) -> an aware UTC datetime."""
    when = dt.datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    return when if when.tzinfo else when.replace(tzinfo=dt.timezone.utc)


class OpenF1Client:
    """Sessions, laps, stints, weather, race control and pit lane times.

    Follows the WeekendSource protocol; rate limited and cached like JolpicaClient.
    """

    def __init__(self, cache: DiskCache | None = None,
                 rate_limiter: RateLimiter | None = None, base_url: str = BASE_URL,
                 now: Callable[[], dt.datetime] | None = None) -> None:
        self.http = JsonHttp("OpenF1", rate_limiter or _LIMITER, cache)
        self.base_url = base_url.rstrip("/")
        self._now = now or _utcnow

    def get_sessions(self, year: int) -> list[Record]:
        """Every session of a season ([] before 2023, which OpenF1 doesn't cover)."""
        if year < FIRST_YEAR:
            return []
        max_age = 6 * 3600.0 if year >= self._now().year else None
        return [_session_record(r) for r in self._get("sessions", {"year": year}, max_age)]

    def get_laps(self, session: Record) -> list[Record]:
        """Every lap of one session."""
        return [_lap_record(r) for r in self._session_rows("laps", session)]

    def get_stints(self, session: Record) -> list[Record]:
        """Every stint (tyre set) of one session."""
        return [_stint_record(r) for r in self._session_rows("stints", session)]

    def get_weather(self, session: Record) -> list[Record]:
        """Track weather, about once a minute."""
        return [_weather_record(r) for r in self._session_rows("weather", session)]

    def get_race_control(self, session: Record) -> list[Record]:
        """Flags, safety cars and other race control messages."""
        return [_control_record(r) for r in self._session_rows("race_control", session)]

    def get_pit(self, session: Record) -> list[Record]:
        """Pit lane passages with lane and stationary times."""
        return [_pit_record(r) for r in self._session_rows("pit", session)]

    def get_drivers(self, session: Record) -> list[Record]:
        """The cars entered in one session: Number, Driver (3 letters), FullName, Team."""
        return [_driver_record(r) for r in self._session_rows("drivers", session)]

    def _session_rows(self, endpoint: str, session: Record) -> list[dict[str, Any]]:
        """Rows of one session, refusing anything still inside the (paid) live window."""
        ended = parse_utc(session["DateEnd"])
        now = self._now()
        if now < ended + LIVE_MARGIN:
            raise DataNotAvailableError(
                f"{session.get('Name')} on {str(session.get('DateStart'))[:10]} is live or "
                "just finished. OpenF1's free tier is historical only; try again about "
                "an hour after the session ends.")
        max_age = None if now - ended > SETTLED_AFTER else 6 * 3600.0
        return self._get(endpoint, {"session_key": session["SessionKey"]}, max_age)

    def _get(self, endpoint: str, params: dict[str, Any],
             max_age: float | None) -> list[dict[str, Any]]:
        data = self.http.get(f"{self.base_url}/{endpoint}", params, max_age)
        return data if isinstance(data, list) else []  # 404 or {"detail": ...} -> no rows


# ---------- flattening ----------
def _session_record(r: dict[str, Any]) -> Record:
    return {"SessionKey": r.get("session_key"), "MeetingKey": r.get("meeting_key"),
            "Name": r.get("session_name"), "Type": r.get("session_type"),
            "DateStart": r.get("date_start"), "DateEnd": r.get("date_end"),
            "Year": r.get("year"), "Cancelled": bool(r.get("is_cancelled"))}


def _lap_record(r: dict[str, Any]) -> Record:
    return {"Number": r.get("driver_number"), "LapNumber": r.get("lap_number"),
            "LapSeconds": r.get("lap_duration"), "PitOutLap": bool(r.get("is_pit_out_lap"))}


def _stint_record(r: dict[str, Any]) -> Record:
    return {"Number": r.get("driver_number"), "Stint": r.get("stint_number"),
            "Compound": str(r.get("compound") or "UNKNOWN").upper(),
            "LapStart": r.get("lap_start"), "LapEnd": r.get("lap_end"),
            "TyreAgeAtStart": r.get("tyre_age_at_start")}


def _weather_record(r: dict[str, Any]) -> Record:
    return {"Time": r.get("date"), "AirTemp": r.get("air_temperature"),
            "TrackTemp": r.get("track_temperature"), "Rainfall": r.get("rainfall")}


def _control_record(r: dict[str, Any]) -> Record:
    return {"Time": r.get("date"), "Lap": r.get("lap_number"), "Category": r.get("category"),
            "Flag": r.get("flag"), "Message": r.get("message")}


def _driver_record(r: dict[str, Any]) -> Record:
    return {"Number": r.get("driver_number"), "Driver": r.get("name_acronym"),
            "FullName": r.get("full_name"), "Team": r.get("team_name")}


def _pit_record(r: dict[str, Any]) -> Record:
    lane = r.get("lane_duration")
    if lane is None:
        lane = r.get("pit_duration")  # the older name, removed after the 2026 season
    return {"Number": r.get("driver_number"), "Lap": r.get("lap_number"),
            "LaneSeconds": lane, "StopSeconds": r.get("stop_duration")}
