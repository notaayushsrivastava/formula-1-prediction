"""Client for F1's Live Timing archive (post-session data), using YOUR F1 TV token.

Every request carries your own token. pitwall does nothing else: no shared tokens,
no mirrors, and no retrying around 401/403.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
from typing import Any

import requests

from ..auth import auth_header, get_token
from ..cache import DiskCache
from ..errors import (
    AccessBlockedError, AuthenticationError, DataNotAvailableError, SourceRequestError,
)
from .base import Record, parse_time_to_seconds
from .jolpica import RateLimiter
from .telemetry_feeds import parse_z_stream

log = logging.getLogger(__name__)

STATIC_URL = "https://livetiming.formula1.com/static"
SESSION_NAMES = {"R": "Race", "Q": "Qualifying", "S": "Sprint"}
TRACK_STATUS = {"1": "Green", "2": "Yellow", "4": "SafetyCar", "5": "RedFlag",
                "6": "VSC", "7": "VSCEnding"}

_LIMITER = RateLimiter(per_second=1.0, per_hour=600)  # no published limits: be gentle

StreamEntry = tuple[float, dict[str, Any]]  # (session time in seconds, message)


class LiveTimingClient:
    """Downloads and caches archive files. Every request carries your own token."""

    def __init__(self, cache: DiskCache | None = None, token: str | None = None,
                 timeout: float = 30.0, rate_limiter: RateLimiter | None = None) -> None:
        self.cache = cache or DiskCache()
        self._token = token  # None: read PITWALL_F1TV_TOKEN at request time
        self.timeout = timeout
        self.limiter = rate_limiter or _LIMITER
        self.http = requests.Session()
        self.http.headers["User-Agent"] = "pitwall-f1/1.0 (personal F1 analysis)"
        self._complete: set[str] = set()

    def __repr__(self) -> str:
        return "LiveTimingClient(token=<hidden>)"

    # ---------- public ----------
    def season_index(self, year: int) -> dict[str, Any]:
        """Meetings and sessions of a season (static/{year}/Index.json)."""
        max_age = 6 * 3600 if year >= dt.date.today().year else None
        return self._json(f"{year}/Index.json", max_age=max_age)

    def session_path(self, year: int, event_date: str, kind: str) -> str:
        """Archive folder of a session, e.g. '2024/2024-09-01_Italian_Grand_Prix/2024-09-01_Race/'."""
        return find_session_path(self.season_index(year), event_date, SESSION_NAMES[kind])

    def keyframe(self, path: str, topic: str) -> dict[str, Any]:
        """A topic's final state, e.g. keyframe(path, 'TimingAppData')."""
        self._require_complete(path)
        return self._json(f"{path}{topic}.json")

    def stream(self, path: str, topic: str) -> list[StreamEntry]:
        """Every change of a topic, e.g. stream(path, 'TrackStatus')."""
        self._require_complete(path)
        return parse_json_stream(self._text(f"{path}{topic}.jsonStream"))

    def z_stream(self, path: str, topic: str) -> list[StreamEntry]:
        """A compressed topic ('CarData.z' or 'Position.z'), decoded."""
        self._require_complete(path)
        return parse_z_stream(self._text(f"{path}{topic}.jsonStream"))

    # ---------- HTTP + cache ----------
    def _require_complete(self, path: str) -> None:
        if path in self._complete:
            return
        status = self._json(f"{path}ArchiveStatus.json", use_cache=False).get("Status")
        if status != "Complete":
            raise DataNotAvailableError(
                f"The archive for {path} is not complete yet (status: {status}). "
                "F1 publishes it some time after the session ends; try again later.")
        self._complete.add(path)

    def _json(self, path: str, use_cache: bool = True, max_age: float | None = None) -> Any:
        return json.loads(self._text(path, use_cache, max_age))

    def _text(self, path: str, use_cache: bool = True, max_age: float | None = None) -> str:
        url = f"{STATIC_URL}/{path}"
        if use_cache:
            hit = self.cache.get(url, max_age=max_age)  # key = URL only, never the token
            if hit is not None:
                return hit
        text = self._download(url)
        if use_cache:
            self.cache.set(url, text)
        return text

    def _download(self, url: str) -> str:
        """One GET with your token. 401/403 are reported, never worked around."""
        token = self._token or get_token()
        self.limiter.wait()
        log.info("Downloading %s", url)  # the URL only, never the headers
        try:
            resp = self.http.get(url, headers=auth_header(token), timeout=self.timeout)
        except (requests.ConnectionError, requests.Timeout) as exc:
            raise SourceRequestError(f"Could not reach F1 Live Timing ({type(exc).__name__}).") from None
        if resp.status_code == 401:
            raise AuthenticationError(
                "F1 rejected your token (HTTP 401). It may have expired, or the account may "
                "not have F1 TV Access or higher. Sign in again and update PITWALL_F1TV_TOKEN.")
        if resp.status_code == 403:
            raise AccessBlockedError(
                "F1 refused the request (HTTP 403), as happens from some cloud or CI networks. "
                "pitwall won't try to get around this; run it from your own connection.")
        if resp.status_code == 404:
            raise DataNotAvailableError(f"Not in the archive: {url}")
        if not resp.ok:
            raise SourceRequestError(f"HTTP {resp.status_code} for {url}")
        return resp.content.decode("utf-8-sig")  # archive files start with a BOM


# ---------- parsing (pure functions, easy to test) ----------
def parse_json_stream(text: str) -> list[StreamEntry]:
    """'00:04:10.000{"Status": "4"}' lines -> [(250.0, {"Status": "4"}), ...]."""
    entries: list[StreamEntry] = []
    for line in text.lstrip("\ufeff").splitlines():
        brace = line.find("{")
        seconds = parse_time_to_seconds(line[:brace]) if brace > 0 else None
        if seconds is not None:
            entries.append((seconds, json.loads(line[brace:])))
    return entries


def find_session_path(index: dict[str, Any], event_date: str, session_name: str) -> str:
    """Pick the meeting whose race is within a day of event_date, then the named session."""
    target = dt.date.fromisoformat(event_date[:10])
    for meeting in index.get("Meetings", []):
        sessions = meeting.get("Sessions", [])
        if not any(_is_race_near(s, target) for s in sessions):
            continue
        for session in sessions:
            if session.get("Name") == session_name:
                path = session.get("Path") or _rebuild_path(index, meeting, session)
                return path if path.endswith("/") else f"{path}/"
        raise DataNotAvailableError(f"{meeting.get('Name')} has no '{session_name}' session.")
    raise DataNotAvailableError(f"No meeting in the archive has a race on or near {event_date}.")


def parse_stints(app_data: dict[str, Any]) -> list[Record]:
    """TimingAppData keyframe -> one record per driver stint.

    Keys: Number, Stint (1-based), Compound, New, StartLaps (tyre age when fitted),
    TotalLaps (tyre age at the end), StintLaps (laps driven in this stint).
    """
    records: list[Record] = []
    for number, line in app_data.get("Lines", {}).items():
        for i, stint in enumerate(_as_list(line.get("Stints", [])), start=1):
            start, total = _int(stint.get("StartLaps")), _int(stint.get("TotalLaps"))
            records.append({
                "Number": str(number), "Stint": i,
                "Compound": str(stint.get("Compound") or "UNKNOWN").upper(),
                "New": str(stint.get("New", "")).lower() == "true",
                "StartLaps": start, "TotalLaps": total, "StintLaps": max(0, total - start),
            })
    return records


def parse_lap_starts(lap_count: list[StreamEntry]) -> dict[int, float]:
    """LapCount stream -> {lap number: session time the leader started that lap}."""
    starts: dict[int, float] = {}
    for seconds, message in lap_count:
        lap = message.get("CurrentLap")
        if lap is not None and int(lap) not in starts:
            starts[int(lap)] = seconds
    return starts


def parse_track_status(entries: list[StreamEntry]) -> list[tuple[float, str]]:
    """TrackStatus stream -> [(session seconds, status code), ...]."""
    return [(t, str(msg["Status"])) for t, msg in entries if "Status" in msg]


def track_status_by_lap(lap_starts: dict[int, float],
                        changes: list[tuple[float, str]]) -> dict[int, str]:
    """Every status in force during each leader lap, e.g. {3: '14'} (green, then SC).

    A lap runs from its start to the next lap's start: the status at the start
    counts, plus every change during the lap (FastF1 uses the same idea).
    """
    laps = sorted(lap_starts)
    by_lap: dict[int, str] = {}
    for i, lap in enumerate(laps):
        begin = lap_starts[lap]
        end = lap_starts[laps[i + 1]] if i + 1 < len(laps) else float("inf")
        before = [code for t, code in changes if t <= begin]
        during = [code for t, code in changes if begin < t < end]
        codes = ([before[-1]] if before else ["1"]) + during
        by_lap[lap] = "".join(dict.fromkeys(codes))  # keep order, drop repeats
    return by_lap


def _is_race_near(session: dict[str, Any], target: dt.date) -> bool:
    if session.get("Name") != "Race" or not session.get("StartDate"):
        return False
    start = dt.date.fromisoformat(session["StartDate"][:10])
    return abs((start - target).days) <= 1


def _rebuild_path(index: dict[str, Any], meeting: dict[str, Any], session: dict[str, Any]) -> str:
    """Rebuild a missing Path the way the archive names folders (an assumption)."""
    race = next(s for s in meeting["Sessions"] if s.get("Name") == "Race")
    meeting_dir = f"{race['StartDate'][:10]}_{meeting['Name'].replace(' ', '_')}"
    session_dir = f"{session['StartDate'][:10]}_{session['Name'].replace(' ', '_')}"
    return f"{index['Year']}/{meeting_dir}/{session_dir}/"


def _as_list(stints: Any) -> list[dict[str, Any]]:
    """Stints arrive as a list, or as {'0': {...}, '1': {...}} after updates."""
    if isinstance(stints, dict):
        return [stints[k] for k in sorted(stints, key=int)]
    return list(stints)


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
