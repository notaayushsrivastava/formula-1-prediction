"""Client for Open-Meteo (free, no API key, non-commercial use; data CC BY 4.0).

Two free archives:
- Previous Runs API: each day's forecast AS IT WAS ISSUED 1, 2 or 3 days earlier
  (archived from January 2024). A backtest using it only sees forecasts it could
  really have had, never the weather that actually happened.
- Historical Weather API: what actually happened. pitwall uses it only for
  diagnostics, never as a feature.
Rain *probability* is not archived for past forecasts, so pitwall uses the forecast
rain amount (mm) instead, in backtests and live predictions alike.
"""
from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Iterable
from typing import Any

from ..cache import DiskCache
from .base import Record
from .http_json import JsonHttp
from .jolpica import RateLimiter

PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
FORECAST_HORIZON_DAYS = 15  # the free forecast covers today + 15 days
FIRST_PREVIOUS_RUN = dt.date(2024, 1, 1)
ARCHIVE_DELAY_DAYS = 7
LEAD_DAYS = (1, 2, 3)
_LIMITER = RateLimiter(per_second=1.0, per_hour=4000)  # free: 600/min, 5,000/h, 10,000/day


class OpenMeteoClient:
    """Archived forecasts and observed weather at a circuit. Follows ForecastSource."""

    def __init__(self, cache: DiskCache | None = None,
                 rate_limiter: RateLimiter | None = None,
                 today: Callable[[], dt.date] | None = None) -> None:
        self.http = JsonHttp("Open-Meteo", rate_limiter or _LIMITER, cache)
        self._today = today or dt.date.today

    def get_race_forecast(self, lat: float, lon: float, date: str,
                          leads: Iterable[int] = LEAD_DAYS) -> list[Record]:
        """Hourly forecasts for one UTC day as issued `leads` days before ([] before 2024)."""
        day = dt.date.fromisoformat(date[:10])
        if day < FIRST_PREVIOUS_RUN:
            return []
        lead_days = tuple(leads)
        names = [f"{var}_previous_day{n}" for n in lead_days
                 for var in ("temperature_2m", "precipitation")]
        data = self.http.get(PREVIOUS_RUNS_URL, _params(lat, lon, day, names),
                             self._max_age(day))
        return forecast_records(data or {}, lead_days)

    def get_observed(self, lat: float, lon: float, date: str) -> list[Record]:
        """Observed hourly weather (reanalysis); [] until it reaches the archive."""
        day = dt.date.fromisoformat(date[:10])
        if day > self._today() - dt.timedelta(days=ARCHIVE_DELAY_DAYS):
            return []
        params = _params(lat, lon, day, ["temperature_2m", "precipitation"])
        hourly = (self.http.get(ARCHIVE_URL, params, None) or {}).get("hourly", {})
        return [{"Time": t, "AirTemp": a, "RainMm": r}
                for t, a, r in zip(hourly.get("time", []), hourly.get("temperature_2m", []),
                                   hourly.get("precipitation", []))]

    def get_live_forecast(self, lat: float, lon: float, date: str) -> list[Record]:
        """Today's forecast for one UTC day (same fields as the archived ones, no LeadDays).

        [] if the day is in the past or beyond the forecast horizon. Re-checked hourly.
        """
        day = dt.date.fromisoformat(date[:10])
        today = self._today()
        if day < today or day > today + dt.timedelta(days=FORECAST_HORIZON_DAYS):
            return []
        params = _params(lat, lon, day, ["temperature_2m", "precipitation"])
        hourly = (self.http.get(FORECAST_URL, params, 3600.0) or {}).get("hourly", {})
        return [{"Time": t, "AirTemp": a, "RainMm": r}
                for t, a, r in zip(hourly.get("time", []), hourly.get("temperature_2m", []),
                                   hourly.get("precipitation", []))
                if a is not None or r is not None]

    def _max_age(self, day: dt.date) -> float | None:
        """Old forecasts never change; recent ones are re-checked every 3 hours."""
        return None if day < self._today() - dt.timedelta(days=7) else 3 * 3600.0


def forecast_records(data: dict[str, Any], leads: Iterable[int]) -> list[Record]:
    """Previous Runs JSON -> one record per (hour, lead day) that has any value."""
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    out: list[Record] = []
    for n in leads:
        temps = hourly.get(f"temperature_2m_previous_day{n}") or [None] * len(times)
        rain = hourly.get(f"precipitation_previous_day{n}") or [None] * len(times)
        for t, air, mm in zip(times, temps, rain):
            if air is not None or mm is not None:
                out.append({"Time": t, "LeadDays": n, "AirTemp": air, "RainMm": mm})
    return out


def _params(lat: float, lon: float, day: dt.date, hourly: list[str]) -> dict[str, Any]:
    return {"latitude": round(lat, 4), "longitude": round(lon, 4), "hourly": ",".join(hourly),
            "start_date": day.isoformat(), "end_date": day.isoformat(), "timezone": "GMT"}
