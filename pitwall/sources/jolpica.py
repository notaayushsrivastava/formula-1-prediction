"""Client for the Jolpica-F1 API (the successor to Ergast)."""
from __future__ import annotations

import datetime as dt
import logging
import time
from collections import deque
from typing import Any
from urllib.parse import urlencode

import requests

from ..cache import DiskCache
from ..errors import RateLimitError, SourceRequestError
from .base import Record, parse_time_to_seconds

log = logging.getLogger(__name__)

BASE_URL = "https://api.jolpi.ca/ergast/f1"
PAGE_LIMIT = 100                   # Jolpica's maximum page size
CURRENT_SEASON_MAX_AGE = 6 * 3600  # seconds; past seasons are cached forever


class RateLimiter:
    """Client-side limiter: at most `per_second` calls/s and `per_hour` calls/h."""

    def __init__(self, per_second: float = 3.0, per_hour: int = 500) -> None:
        self.min_interval = 1.0 / per_second
        self.per_hour = per_hour
        self._stamps: deque[float] = deque()  # times of recent requests

    def wait(self) -> None:
        """Sleep until the next request is allowed, or raise if the hour's budget is spent."""
        now = time.monotonic()
        while self._stamps and now - self._stamps[0] > 3600:
            self._stamps.popleft()
        if len(self._stamps) >= self.per_hour:
            minutes = (3600 - (now - self._stamps[0])) / 60
            raise RateLimitError(
                f"Used {self.per_hour} Jolpica requests this hour; retry in ~{minutes:.0f} min. "
                "Anything already cached still works."
            )
        if self._stamps:
            gap = now - self._stamps[-1]
            if gap < self.min_interval:
                time.sleep(self.min_interval - gap)
        self._stamps.append(time.monotonic())


_SHARED_LIMITER = RateLimiter()  # shared by all clients in this Python process


class JolpicaClient:
    """Fetches, caches and flattens Jolpica data. Follows the ResultsSource protocol."""

    def __init__(
        self,
        cache: DiskCache | None = None,
        base_url: str = BASE_URL,
        timeout: float = 30.0,
        max_retries: int = 4,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self.cache = cache or DiskCache()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.limiter = rate_limiter or _SHARED_LIMITER
        self.http = requests.Session()
        self.http.headers["User-Agent"] = "pitwall-f1/1.0 (educational F1 analysis library)"

    # ---------- one race weekend ----------
    def get_schedule(self, year: int) -> list[Record]:
        return [_schedule_record(r) for r in self._races(f"{year}/races", year)]

    def get_results(self, year: int, rnd: int) -> list[Record]:
        return self._result_rows(f"{year}/{rnd}/results", year, "Results")

    def get_sprint_results(self, year: int, rnd: int) -> list[Record]:
        """Sprint results (sprints exist from 2021; other weekends return [])."""
        return self._result_rows(f"{year}/{rnd}/sprint", year, "SprintResults")

    def get_qualifying(self, year: int, rnd: int) -> list[Record]:
        """Qualifying. Before the 2006 knockout format only Q1 is filled; old seasons may have gaps."""
        races = self._races(f"{year}/{rnd}/qualifying", year)
        return [_qualifying_record(race, row)
                for race in races for row in race.get("QualifyingResults", [])]

    def get_laps(self, year: int, rnd: int) -> list[Record]:
        races = self._races(f"{year}/{rnd}/laps", year)
        return [
            {
                "DriverId": t["driverId"],
                "LapNumber": int(lap["number"]),
                "Position": _to_int(t.get("position")),
                "LapTimeSeconds": parse_time_to_seconds(t.get("time")),
            }
            for race in races
            for lap in race.get("Laps", [])
            for t in lap.get("Timings", [])
        ]

    def get_pitstops(self, year: int, rnd: int) -> list[Record]:
        """Pit stops (Jolpica has these from 2011; earlier years return [])."""
        races = self._races(f"{year}/{rnd}/pitstops", year)
        return [
            {
                "DriverId": p["driverId"],
                "LapNumber": int(p["lap"]),
                "Stop": _to_int(p.get("stop")),
                "DurationSeconds": parse_time_to_seconds(p.get("duration")),
            }
            for race in races
            for p in race.get("PitStops", [])
        ]

    # ---------- whole season ----------
    def get_season_results(self, year: int, sprint: bool = False) -> list[Record]:
        """Every completed round in one paginated call (~5 requests for a full season)."""
        if sprint:
            return self._result_rows(f"{year}/sprint", year, "SprintResults")
        return self._result_rows(f"{year}/results", year, "Results")

    def get_driver_standings(self, year: int, rnd: int | None = None) -> list[Record]:
        lists = self._standings(_path(year, rnd, "driverstandings"), year)
        return [_driver_standing_record(lst, row)
                for lst in lists for row in lst.get("DriverStandings", [])]

    def get_constructor_standings(self, year: int, rnd: int | None = None) -> list[Record]:
        lists = self._standings(_path(year, rnd, "constructorstandings"), year)
        return [_constructor_standing_record(lst, row)
                for lst in lists for row in lst.get("ConstructorStandings", [])]

    # ---------- pagination ----------
    def _result_rows(self, path: str, year: int, key: str) -> list[Record]:
        """Flatten race or sprint results; key is 'Results' or 'SprintResults'."""
        return [_result_record(race, row)
                for race in self._races(path, year) for row in race.get(key, [])]

    def _races(self, path: str, year: int) -> list[dict[str, Any]]:
        """Every 'Race' object across all pages (one race can be split over pages)."""
        return [race for page in self._pages(path, year) for race in page["RaceTable"]["Races"]]

    def _standings(self, path: str, year: int) -> list[dict[str, Any]]:
        """Every 'StandingsList' object across all pages."""
        return [lst for page in self._pages(path, year)
                for lst in page["StandingsTable"]["StandingsLists"]]

    def _pages(self, path: str, year: int) -> list[dict[str, Any]]:
        """Download every page of an endpoint; returns the list of MRData dicts."""
        pages: list[dict[str, Any]] = []
        offset = 0
        while True:
            params = {"limit": PAGE_LIMIT, "offset": offset}
            mrdata = self._get_json(path, params, year)["MRData"]
            pages.append(mrdata)
            offset += max(1, int(mrdata.get("limit", PAGE_LIMIT)))
            if offset >= int(mrdata.get("total", 0)):
                return pages

    # ---------- HTTP + cache ----------
    def _get_json(self, path: str, params: dict[str, int], year: int) -> dict[str, Any]:
        url = f"{self.base_url}/{path}.json"
        key = f"{url}?{urlencode(sorted(params.items()))}"
        max_age = CURRENT_SEASON_MAX_AGE if year >= dt.date.today().year else None
        cached = self.cache.get(key, max_age=max_age)
        if cached is not None:
            return cached
        data = self._download(url, params)
        if int(data["MRData"].get("total", 0)) > 0:  # never cache "no data yet"
            self.cache.set(key, data)
        return data

    def _download(self, url: str, params: dict[str, int]) -> dict[str, Any]:
        """GET with rate limiting and exponential backoff on 429/5xx/network errors."""
        attempts = self.max_retries + 1
        delay = 2.0
        for attempt in range(1, attempts + 1):
            self.limiter.wait()
            log.info("Downloading %s %s", url, params)
            try:
                resp = self.http.get(url, params=params, timeout=self.timeout)
            except (requests.ConnectionError, requests.Timeout) as exc:
                problem = str(exc)
            else:
                if resp.ok:
                    return resp.json()
                if resp.status_code != 429 and resp.status_code < 500:
                    raise SourceRequestError(f"HTTP {resp.status_code} for {resp.url}")
                problem = f"HTTP {resp.status_code}"
                delay = max(delay, _retry_after(resp))
            if attempt == attempts:
                break
            log.warning("Request failed (%s); retry %d in %.0f s", problem, attempt, delay)
            time.sleep(delay)
            delay *= 2
        raise SourceRequestError(f"Giving up on {url} after {attempts} attempts")


# ---------- flattening helpers ----------
def _path(year: int, rnd: int | None, endpoint: str) -> str:
    """'2024/driverstandings' (latest) or '2024/5/driverstandings' (after round 5)."""
    return f"{year}/{endpoint}" if rnd is None else f"{year}/{rnd}/{endpoint}"


def _driver_fields(drv: dict[str, Any]) -> Record:
    family = drv.get("familyName", "???")
    return {
        "DriverId": drv.get("driverId"),
        "Driver": drv.get("code") or family[:3].upper(),  # old drivers lack codes
        "FullName": f"{drv.get('givenName', '')} {family}".strip(),
    }


def _team_fields(team: dict[str, Any]) -> Record:
    return {"TeamId": team.get("constructorId"), "Team": team.get("name")}


def _schedule_record(race: dict[str, Any]) -> Record:
    circuit = race.get("Circuit", {})
    loc = circuit.get("Location", {})
    return {
        "Round": int(race["round"]),
        "EventName": race.get("raceName", ""),
        "Date": race.get("date"),
        "CircuitId": circuit.get("circuitId", ""),
        "Circuit": circuit.get("circuitName", ""),
        "Locality": loc.get("locality", ""),
        "Country": loc.get("country", ""),
        "Lat": _to_float(loc.get("lat")),
        "Long": _to_float(loc.get("long")),
        "HasSprint": "Sprint" in race,  # sprint weekends carry a 'Sprint' session block
    }


def _result_record(race: dict[str, Any], row: dict[str, Any]) -> Record:
    fastest = row.get("FastestLap", {}).get("Time", {}).get("time")
    return {
        "Round": int(race["round"]),
        **_driver_fields(row.get("Driver", {})),
        **_team_fields(row.get("Constructor", {})),
        "Number": row.get("number"),
        "GridPosition": _to_int(row.get("grid")),  # 0 = pit-lane start
        "Position": _to_int(row.get("position")),
        "ClassifiedPosition": row.get("positionText"),  # 'R' retired, 'D' DSQ, ...
        "Status": row.get("status"),
        "Points": _to_float(row.get("points")) or 0.0,
        "LapsCompleted": _to_int(row.get("laps")),
        "FastestLapSeconds": parse_time_to_seconds(fastest),
    }


def _qualifying_record(race: dict[str, Any], row: dict[str, Any]) -> Record:
    return {
        "Round": int(race["round"]),
        **_driver_fields(row.get("Driver", {})),
        **_team_fields(row.get("Constructor", {})),
        "Number": row.get("number"),
        "Position": _to_int(row.get("position")),
        "Q1Seconds": parse_time_to_seconds(row.get("Q1")),
        "Q2Seconds": parse_time_to_seconds(row.get("Q2")),
        "Q3Seconds": parse_time_to_seconds(row.get("Q3")),
    }


def _driver_standing_record(lst: dict[str, Any], row: dict[str, Any]) -> Record:
    teams = row.get("Constructors") or [{}]
    return {
        "Round": int(lst["round"]),
        **_driver_fields(row.get("Driver", {})),
        **_team_fields(teams[-1]),  # a mid-season team switch lists several; keep the last
        "Position": _to_int(row.get("position")),
        "Points": _to_float(row.get("points")) or 0.0,
        "Wins": _to_int(row.get("wins")) or 0,
    }


def _constructor_standing_record(lst: dict[str, Any], row: dict[str, Any]) -> Record:
    return {
        "Round": int(lst["round"]),
        **_team_fields(row.get("Constructor", {})),
        "Position": _to_int(row.get("position")),
        "Points": _to_float(row.get("points")) or 0.0,
        "Wins": _to_int(row.get("wins")) or 0,
    }


def _retry_after(resp: requests.Response) -> float:
    try:
        return float(resp.headers.get("Retry-After", 0))
    except ValueError:
        return 0.0


def _to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
