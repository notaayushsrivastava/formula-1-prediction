"""Phase 12: predicting a race that has not happened yet.

f1db is released a day or two after each race, so the newest races may be missing.
`ToppedUpHistory` adds them from Jolpica, converting Jolpica's ids to f1db's
('max_verstappen' -> 'max-verstappen'). The race we want to predict gets placeholder
rows (the entry list, no result), so the SAME dataset code that built the training
rows builds its features: what the models see live is computed exactly like what
they learned from.

Every missing piece of information becomes an availability flag of 0 plus a warning,
never a guess. OpenF1 is only read an hour or more after a session ends (its live
tier is paid); `build_live_rows` refuses earlier.
"""
from __future__ import annotations

import csv
import datetime as dt
import logging
import math
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import pandas as pd

from ..errors import DataNotAvailableError, PitwallError
from ..sources.base import ForecastSource, HistorySource, Record, ResultsSource, WeekendSource
from ..sources.openf1 import LIVE_MARGIN, parse_utc
from .dataset import LABELS, LEAD_DAYS, MOMENTS, OBSERVATIONS, build_dataset
from .weekend import PRACTICE_SESSIONS, meeting_sessions

log = logging.getLogger(__name__)

LIVE_ID_BASE = 900_000           # RaceIds for races f1db does not list yet
CIRCUIT_MATCH_KM = 10.0          # a Jolpica circuit within this distance is the same circuit
PENALTY_NOTE = ("Grid = qualifying order. FIA grid penalties announced after qualifying "
                "are NOT parsed; check the official starting grid.")
POSITION_TEXT = {"R": "DNF", "D": "DSQ", "W": "DNS", "N": "NC", "E": "EX", "F": "DNQ"}
FINISHED = re.compile(r"^(finished|\+\d+ laps?|lapped)$", re.IGNORECASE)
Clock = Callable[[], dt.datetime]


class IdMappingError(PitwallError):
    """Jolpica ids that could not be matched to f1db ids."""


class LiveWindowError(DataNotAvailableError):
    """A session ended less than an hour ago: OpenF1 would still be in its (paid) live tier."""


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


# ---------- id mapping ----------
def load_overrides(path: Path | str | None = None) -> dict[tuple[str, str], str]:
    """(kind, jolpica_id) -> f1db_id from id_overrides.csv ('#' lines are comments)."""
    if path is None:
        text = resources.files("pitwall.predict").joinpath("id_overrides.csv").read_text("utf-8")
    else:
        text = Path(path).read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if line.strip() and not line.startswith("#")]
    out: dict[tuple[str, str], str] = {}
    for row in csv.DictReader(lines):
        out[(row["kind"].strip(), row["jolpica_id"].strip())] = row["f1db_id"].strip()
    return out


def slug(text: str) -> str:
    """'Haas F1 Team' -> 'haas-f1-team'; 'max_verstappen' -> 'max-verstappen'."""
    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")


@dataclass
class IdMapper:
    """Converts Jolpica driver, team and circuit ids to f1db ids.

    Order of attempts: the override file; the same id with '-' for '_'; then a fallback
    (drivers and teams: the car number in f1db's latest season; teams also by name;
    circuits: the nearest f1db circuit by coordinates). Unmatched ids are collected and
    raised together by `check()`, never guessed.
    """

    drivers: set[str]
    teams: set[str]
    circuits: pd.DataFrame                         # CircuitId, Lat, Long
    number_driver: dict[int, str] = field(default_factory=dict)
    number_team: dict[int, str] = field(default_factory=dict)
    overrides: dict[tuple[str, str], str] = field(default_factory=dict)
    unmatched: list[str] = field(default_factory=list)

    @classmethod
    def from_history(cls, source: HistorySource, min_year: int,
                     overrides: dict[tuple[str, str], str] | None = None) -> IdMapper:
        """Known ids from the f1db history (and the latest season's car numbers)."""
        results = pd.DataFrame(source.get_race_results(min_year))
        races = pd.DataFrame(source.get_races(min_year))
        drivers = {str(r["DriverId"]) for r in source.get_drivers()}
        drivers |= set(results.get("DriverId", pd.Series(dtype=str)).astype(str))
        teams = set(results.get("TeamId", pd.Series(dtype=str)).astype(str))
        number_driver: dict[int, str] = {}
        number_team: dict[int, str] = {}
        if not results.empty:
            latest = results.merge(races[["RaceId", "Date"]], on="RaceId").sort_values("Date")
            latest = latest[pd.to_datetime(latest["Date"]).dt.year
                            == pd.to_datetime(latest["Date"]).dt.year.max()]
            for row in latest.itertuples(index=False):
                number = _int(row.Number)
                if number is not None:
                    number_driver[number] = str(row.DriverId)
                    number_team[number] = str(row.TeamId)
        circuits = (races[["CircuitId", "Lat", "Long"]].dropna().drop_duplicates("CircuitId")
                    if not races.empty else pd.DataFrame(columns=["CircuitId", "Lat", "Long"]))
        return cls(drivers, teams, circuits.reset_index(drop=True), number_driver, number_team,
                   load_overrides() if overrides is None else overrides)

    def driver(self, jolpica_id: str, number: Any = None) -> str:
        """f1db driver id."""
        found = self._direct("driver", jolpica_id, self.drivers)
        if found is None and _int(number) in self.number_driver:
            found = self.number_driver[_int(number)]  # type: ignore[index]
        return self._done("driver", jolpica_id, found)

    def team(self, jolpica_id: str, name: str = "", numbers: Iterable[Any] = ()) -> str:
        """f1db constructor id (by id, then by its cars' numbers, then by name)."""
        found = self._direct("team", jolpica_id, self.teams)
        if found is None:
            votes = [self.number_team[n] for n in (_int(x) for x in numbers)
                     if n in self.number_team]
            if votes:
                found = max(set(votes), key=votes.count)
        if found is None and name:
            wanted = slug(name)
            matches = [t for t in self.teams if wanted.startswith(t) or t.startswith(wanted)]
            found = max(matches, key=len) if matches else None
        return self._done("team", jolpica_id, found)

    def circuit(self, jolpica_id: str, lat: Any = None, lon: Any = None) -> str:
        """f1db circuit id (by id, then the nearest circuit within CIRCUIT_MATCH_KM)."""
        known = set(self.circuits["CircuitId"].astype(str))
        found = self._direct("circuit", jolpica_id, known)
        if found is None and lat is not None and lon is not None and not self.circuits.empty:
            km = [_km(float(lat), float(lon), float(a), float(b))
                  for a, b in zip(self.circuits["Lat"], self.circuits["Long"])]
            best = min(range(len(km)), key=km.__getitem__)
            if km[best] <= CIRCUIT_MATCH_KM:
                found = str(self.circuits["CircuitId"].iloc[best])
        return self._done("circuit", jolpica_id, found)

    def check(self) -> None:
        """Raise IdMappingError listing every id that could not be matched."""
        if self.unmatched:
            shown = ", ".join(sorted(set(self.unmatched)))
            raise IdMappingError(
                f"Could not match these Jolpica ids to f1db: {shown}. Add a line "
                "'kind,jolpica_id,f1db_id' to pitwall/predict/id_overrides.csv.")

    def _direct(self, kind: str, jolpica_id: str, known: set[str]) -> str | None:
        if (kind, jolpica_id) in self.overrides:
            return self.overrides[(kind, jolpica_id)]
        for candidate in (jolpica_id, str(jolpica_id).replace("_", "-")):
            if candidate in known:
                return candidate
        return None

    def _done(self, kind: str, jolpica_id: str, found: str | None) -> str:
        if found is None:
            self.unmatched.append(f"{kind}:{jolpica_id}")
            return f"?{jolpica_id}"
        return found


# ---------- history with the newest races from Jolpica ----------
@dataclass
class Placeholder:
    """The upcoming race: its entry list and (once run) qualifying."""

    year: int
    round: int
    entries: list[Record]            # DriverId, TeamId, Number, GridPosition (or None)
    qualifying: list[Record] = field(default_factory=list)


class ToppedUpHistory:
    """f1db history plus races Jolpica has and f1db not yet. Follows HistorySource."""

    def __init__(self, base: HistorySource, jolpica: ResultsSource, years: Iterable[int],
                 mapper: IdMapper | None = None, now: Clock | None = None,
                 min_year: int = 2000) -> None:
        self.base = base
        self.jolpica = jolpica
        self.years = sorted({int(y) for y in years})
        self.now = now or _utcnow
        self.mapper = mapper or IdMapper.from_history(base, max(min_year, self.years[0] - 3))
        self.placeholder: Placeholder | None = None
        self._added: dict[str, list[Record]] | None = None

    # ----- HistorySource -----
    def get_races(self, min_year: int) -> list[Record]:
        return self.base.get_races(min_year) + [r for r in self._extra()["races"]
                                                if int(r["Year"]) >= min_year]

    def get_race_results(self, min_year: int) -> list[Record]:
        return self.base.get_race_results(min_year) + self._extra()["results"] + self._entries()

    def get_qualifying(self, min_year: int) -> list[Record]:
        extra = self._extra()["qualifying"]
        if self.placeholder is not None:
            extra = extra + self.placeholder.qualifying
        return self.base.get_qualifying(min_year) + extra

    def get_pitstops(self, min_year: int) -> list[Record]:
        return self.base.get_pitstops(min_year) + self._extra()["pitstops"]

    def get_driver_standings(self, min_year: int) -> list[Record]:
        return self.base.get_driver_standings(min_year) + self._extra()["standings"]

    def get_drivers(self) -> list[Record]:
        return self.base.get_drivers() + self._extra()["drivers"]

    # ----- what was added -----
    @property
    def added_races(self) -> list[Record]:
        """Races that came from Jolpica (with results), for messages and tests."""
        done = {r["RaceId"] for r in self._extra()["results"]}
        return [r for r in self.get_races(self.years[0]) if r["RaceId"] in done]

    def race_table(self) -> pd.DataFrame:
        """Every race (f1db and Jolpica-only), with HasResults, oldest first."""
        races = pd.DataFrame(self.get_races(self.years[0]))
        with_results = {int(r["RaceId"]) for r in self.get_race_results(self.years[0])
                        if r not in self._entries()}
        races["HasResults"] = races["RaceId"].astype(int).isin(with_results)
        races["Date"] = pd.to_datetime(races["Date"])
        return races.sort_values("Date").reset_index(drop=True)

    def race_id(self, year: int, rnd: int) -> int:
        """The RaceId of a (year, round): f1db's if it lists it, else a stable live id."""
        for race in self.base.get_races(year):
            if int(race["Year"]) == year and int(race["Round"]) == rnd:
                return int(race["RaceId"])
        return LIVE_ID_BASE + year * 100 + rnd

    def _entries(self) -> list[Record]:
        if self.placeholder is None:
            return []
        rid = self.race_id(self.placeholder.year, self.placeholder.round)
        return [{"RaceId": rid, "DriverId": e["DriverId"], "TeamId": e["TeamId"],
                 "Number": e.get("Number"), "Position": None, "PositionText": "TBD",
                 "DisplayOrder": i + 1, "GridPosition": e.get("GridPosition"),
                 "GridText": None, "Laps": None, "TimeMillis": None, "ReasonRetired": None,
                 "Points": 0.0}
                for i, e in enumerate(self.placeholder.entries)]

    def _extra(self) -> dict[str, list[Record]]:
        """Jolpica races after f1db's newest result, converted to f1db's format."""
        if self._added is not None:
            return self._added
        added: dict[str, list[Record]] = {k: [] for k in
                                          ("races", "results", "qualifying", "pitstops",
                                           "standings", "drivers")}
        base_races = pd.DataFrame(self.base.get_races(self.years[0]))
        base_results = {int(r["RaceId"]) for r in self.base.get_race_results(self.years[0])}
        known = {(int(r.Year), int(r.Round)): int(r.RaceId)
                 for r in base_races.itertuples(index=False)} if not base_races.empty else {}
        today = self.now().date()
        for year in self.years:
            for race in self.jolpica.get_schedule(year):
                rnd = int(race["Round"])
                rid = known.get((year, rnd), LIVE_ID_BASE + year * 100 + rnd)
                if (year, rnd) not in known:
                    added["races"].append(self._race_record(rid, year, race))
                race_day = dt.date.fromisoformat(str(race["Date"])[:10])
                if rid in base_results or race_day > today:
                    continue
                results = self.jolpica.get_results(year, rnd)
                if not results:
                    continue  # raced but Jolpica has no results yet: stays a future race
                self._add_race(added, rid, year, rnd, results)
        self.mapper.check()
        self._added = added
        return added

    def _race_record(self, rid: int, year: int, race: Record) -> Record:
        circuit = self.mapper.circuit(str(race.get("CircuitId")), race.get("Lat"),
                                      race.get("Long"))
        same = [r for r in self.base.get_races(max(self.years[0] - 5, 1950))
                if r.get("CircuitId") == circuit]
        latest = same[-1] if same else {}
        return {"RaceId": rid, "Year": year, "Round": int(race["Round"]),
                "Date": str(race["Date"])[:10], "Time": race.get("Time"),
                "EventName": race.get("EventName"), "CircuitId": circuit,
                "CircuitType": latest.get("CircuitType", "RACE"),
                "DistanceKm": latest.get("DistanceKm"), "Lat": race.get("Lat"),
                "Long": race.get("Long"), "HasSprint": bool(race.get("HasSprint"))}

    def _add_race(self, added: dict[str, list[Record]], rid: int, year: int, rnd: int,
                  results: list[Record]) -> None:
        log.info("Topping up %s round %s from Jolpica", year, rnd)
        numbers: dict[str, list[Any]] = {}
        for row in results:
            numbers.setdefault(str(row["TeamId"]), []).append(row.get("Number"))
        teams = {t: self.mapper.team(t, next((str(r.get("Team")) for r in results
                                              if str(r["TeamId"]) == t), ""), n)
                 for t, n in numbers.items()}
        drivers = {str(r["DriverId"]): self.mapper.driver(str(r["DriverId"]), r.get("Number"))
                   for r in results}
        for i, row in enumerate(sorted(results, key=lambda r: _int(r.get("Position")) or 99)):
            added["results"].append(convert_result(row, rid, drivers, teams, i + 1))
            added["drivers"].append({"DriverId": drivers[str(row["DriverId"])],
                                     "Driver": row.get("Driver"),
                                     "FullName": row.get("FullName"), "DateOfBirth": None})
        for row in self.jolpica.get_qualifying(year, rnd):
            if str(row["DriverId"]) in drivers:
                added["qualifying"].append(convert_qualifying(row, rid, drivers))
        team_of = {drivers[str(r["DriverId"])]: teams[str(r["TeamId"])] for r in results}
        for stop in self.jolpica.get_pitstops(year, rnd):
            driver = drivers.get(str(stop["DriverId"]))
            if driver is not None:
                added["pitstops"].append({"RaceId": rid, "DriverId": driver,
                                          "TeamId": team_of[driver], "Stop": stop.get("Stop"),
                                          "Lap": stop.get("LapNumber"),
                                          "DurationSeconds": stop.get("DurationSeconds")})
        for row in self.jolpica.get_driver_standings(year, rnd):
            driver = drivers.get(str(row["DriverId"]))
            if driver is None:  # a driver who did not race this round keeps his points
                driver = self.mapper.driver(str(row["DriverId"]))
            added["standings"].append({"RaceId": rid, "DriverId": driver,
                                       "Points": row.get("Points")})


def convert_result(row: Record, rid: int, drivers: dict[str, str], teams: dict[str, str],
                   order: int) -> Record:
    """One Jolpica result row in f1db's HistorySource format."""
    text = str(row.get("ClassifiedPosition") or "")
    classified = text.isdigit()
    status = str(row.get("Status") or "")
    grid = _int(row.get("GridPosition"))
    return {"RaceId": rid, "DriverId": drivers[str(row["DriverId"])],
            "TeamId": teams[str(row["TeamId"])], "Number": _int(row.get("Number")),
            "Position": int(text) if classified else None,
            "PositionText": text if classified else POSITION_TEXT.get(text, "DNF"),
            "DisplayOrder": order, "GridPosition": grid if grid else None,
            "GridText": None if grid else "PL", "Laps": row.get("LapsCompleted"),
            "TimeMillis": None,
            "ReasonRetired": None if (classified or FINISHED.match(status)) else status,
            "Points": row.get("Points") or 0.0}


def convert_qualifying(row: Record, rid: int, drivers: dict[str, str]) -> Record:
    """One Jolpica qualifying row in f1db's format."""
    return {"RaceId": rid, "DriverId": drivers[str(row["DriverId"])],
            "QualiPosition": row.get("Position"), "Q1Seconds": row.get("Q1Seconds"),
            "Q2Seconds": row.get("Q2Seconds"), "Q3Seconds": row.get("Q3Seconds")}


# ---------- weekend and weather sources that never touch live data ----------
class SettledWeekend:
    """Wraps a WeekendSource so live (paid) OpenF1 data can never be requested.

    The session list (a timetable, not data) passes through. Asking for the data of a
    session that has not ended at least LIVE_MARGIN ago raises LiveWindowError before
    any request; the dataset code treats that as "not available" (flag 0, warning).
    """

    DATA_METHODS = ("get_laps", "get_stints", "get_weather", "get_race_control", "get_pit",
                    "get_drivers")

    def __init__(self, source: WeekendSource, now: Clock | None = None) -> None:
        self.source = source
        self.now = now or _utcnow

    def get_sessions(self, year: int) -> list[Record]:
        return self.source.get_sessions(year)

    def settled(self, session: Record) -> bool:
        """True once the session ended at least LIVE_MARGIN ago."""
        end = session.get("DateEnd")
        return bool(end) and parse_utc(str(end)) + LIVE_MARGIN <= self.now()

    def __getattr__(self, name: str) -> Any:
        method = getattr(self.source, name)
        if name not in self.DATA_METHODS:
            return method

        def guarded(session: Record) -> list[Record]:
            if not self.settled(session):
                raise LiveWindowError(f"{session.get('Name')} has not finished an hour ago "
                                      "yet; its data is not used.")
            return method(session)
        return guarded


class LiveForecast:
    """Archived forecasts for past races; the normal forecast for the target race.

    The target's forecast is labelled with the moment's lead day, so the dataset code
    picks it up exactly like an archived forecast issued that many days ahead.
    """

    def __init__(self, archive: ForecastSource, target_date: str, lead_days: int,
                 fetch_live: Callable[[float, float, str], list[Record]] | None) -> None:
        self.archive = archive
        self.target_date = target_date[:10]
        self.lead_days = lead_days
        self.fetch_live = fetch_live

    def get_race_forecast(self, lat: float, lon: float, date: str) -> list[Record]:
        if date[:10] != self.target_date:
            return self.archive.get_race_forecast(lat, lon, date)
        if self.fetch_live is None:
            return []
        return [{**r, "LeadDays": self.lead_days} for r in self.fetch_live(lat, lon, date)]

    def get_observed(self, lat: float, lon: float, date: str) -> list[Record]:
        if date[:10] == self.target_date:
            return []
        return self.archive.get_observed(lat, lon, date)


# ---------- the upcoming race ----------
@dataclass
class LiveRows:
    """The dataset including the upcoming race, plus what was (not) available."""

    data: pd.DataFrame            # every row (training rows + the race), one moment
    race_id: int
    year: int
    round: int
    event: str
    date: pd.Timestamp
    moment: str
    flags: dict[str, int]
    warnings: list[str]
    notes: list[str]
    circuit_id: str = ""
    lat: float | None = None
    lon: float | None = None

    @property
    def race(self) -> pd.DataFrame:
        """Only the upcoming race's rows (no labels)."""
        return self.data[self.data["RaceId"] == self.race_id]


def find_event(schedule: list[Record], event: str | int) -> Record:
    """A race from the schedule by round number or (part of) its name/place."""
    text = str(event).strip().lower()
    if text.isdigit():
        hits = [r for r in schedule if int(r["Round"]) == int(text)]
    else:
        hits = [r for r in schedule
                if any(text in str(r.get(k, "")).lower()
                       for k in ("EventName", "Locality", "Country", "Circuit", "CircuitId"))]
    if not hits:
        names = ", ".join(f"{r['Round']} {r.get('EventName')}" for r in schedule)
        raise DataNotAvailableError(f"No race matches {event!r}. This season: {names}.")
    if len(hits) > 1:
        names = ", ".join(str(r.get("EventName")) for r in hits)
        raise ValueError(f"{event!r} matches several races ({names}); use the round number.")
    return hits[0]


def next_race(schedule: list[Record], today: dt.date) -> Record | None:
    """The first race of the schedule on or after today (None if the season is over)."""
    upcoming = [r for r in schedule if dt.date.fromisoformat(str(r["Date"])[:10]) >= today]
    return min(upcoming, key=lambda r: str(r["Date"])) if upcoming else None


def session_state(source: WeekendSource | None, year: int, race_date: pd.Timestamp,
                  now: dt.datetime) -> dict[str, str]:
    """Each session of the meeting: 'settled' (>= 1 h after its end), 'live' or 'future'."""
    if source is None:
        return {}
    try:
        sessions = meeting_sessions(source.get_sessions(year), race_date)
    except DataNotAvailableError:
        return {}
    out = {}
    for name, s in sessions.items():
        start, end = parse_utc(str(s["DateStart"])), parse_utc(str(s["DateEnd"]))
        out[name] = ("settled" if now >= end + LIVE_MARGIN else
                     "future" if now < start else "live")
    return out


def check_moment(moment: str, states: dict[str, str], quali_rows: int) -> list[str]:
    """Why the moment's data is NOT available ([] = ready). Raises LiveWindowError when
    a needed session is live or ended less than an hour ago."""
    if moment not in MOMENTS:
        raise ValueError(f"Unknown moment {moment!r}. Use one of: {', '.join(MOMENTS)}.")
    needed = {"practice": PRACTICE_SESSIONS, "quali": ("Qualifying",)}.get(moment, ())
    live = [n for n in needed if states.get(n) == "live"]
    if live:
        raise LiveWindowError(
            f"{', '.join(live)} ended less than an hour ago (or is running). OpenF1's free "
            "tier is historical only; try again an hour after the session ends.")
    if moment == "practice" and not any(states.get(n) == "settled" for n in PRACTICE_SESSIONS):
        return ["no practice session has finished (an hour ago or more) yet"]
    if moment == "quali" and quali_rows == 0:
        return ["qualifying results are not published yet"]
    return []


def build_live_rows(year: int, event: str | int, moment: str, history: HistorySource,
                    jolpica: ResultsSource, weekend: WeekendSource | None = None,
                    weather: ForecastSource | None = None,
                    live_forecast: Callable[[float, float, str], list[Record]] | None = None,
                    first_year: int = 2018, history_from: int = 2000,
                    now: Clock | None = None) -> LiveRows:
    """Dataset rows (same columns as build_dataset) for an UPCOMING race at a moment.

    Raises DataNotAvailableError if the moment's data does not exist yet (e.g. no
    qualifying for 'quali'), LiveWindowError inside OpenF1's live window, and refuses
    races that already have results (a prediction then would not be a forecast).
    """
    clock = now or _utcnow
    when = clock()
    schedule = jolpica.get_schedule(year)
    race = find_event(schedule, event)
    rnd, date = int(race["Round"]), pd.Timestamp(str(race["Date"])[:10])
    topped = ToppedUpHistory(history, jolpica, range(max(first_year, year - 1), year + 1),
                             now=clock, min_year=history_from)
    race_id = topped.race_id(year, rnd)
    if any(int(r["RaceId"]) == race_id for r in topped.get_race_results(year)):
        raise PitwallError(f"{year} round {rnd} already has results: a prediction now "
                           "would not be a forecast. Use the backtest for past races.")
    states = session_state(weekend, year, date, when)
    quali = jolpica.get_qualifying(year, rnd) if moment == "quali" else []
    missing = check_moment(moment, states, len(quali))
    if missing:
        raise DataNotAvailableError(f"'{moment}' data for {race.get('EventName')} is not "
                                    f"available: {'; '.join(missing)}.")
    warnings: list[str] = []
    notes: list[str] = []
    entries = _entry_list(topped, quali, weekend, states, year, date, warnings)
    if moment == "quali":
        notes.append(PENALTY_NOTE)
    topped.placeholder = Placeholder(year, rnd, entries, _placeholder_quali(topped, quali,
                                                                            race_id))
    safe_weekend = SettledWeekend(weekend, clock) if weekend is not None else None
    forecast = None
    if weather is not None or live_forecast is not None:
        forecast = LiveForecast(weather or _NoArchive(), str(race["Date"]), LEAD_DAYS[moment],
                                live_forecast)
    data = build_dataset(topped, safe_weekend, forecast, first_year=first_year,
                         last_year=year, history_from=history_from)
    data = data[data["Moment"] == moment].reset_index(drop=True)
    target = data["RaceId"] == race_id
    if not target.any():
        raise DataNotAvailableError(f"Could not build rows for {race.get('EventName')}.")
    data.loc[target, LABELS + OBSERVATIONS] = float("nan")
    rows = data[target]
    flags = {"has_openf1": _flag(rows["has_openf1"]),
             "has_practice_pace": _flag(rows["has_practice_pace"]),
             "has_forecast": _flag(rows["has_forecast"]),
             "has_track_temp": _flag(rows["has_track_temp"]),
             "has_qualifying": int(bool(quali)),
             "topped_up_races": len(topped.added_races)}
    if moment == "pre":
        flags.update(has_openf1=0, has_practice_pace=0, has_track_temp=0)
    for name, label in (("has_forecast", "weather forecast"),
                        ("has_practice_pace", "practice long-run pace"),
                        ("has_track_temp", "track temperature")):
        wanted = moment != "pre" or name == "has_forecast"
        if wanted and not flags[name]:
            warnings.append(f"No {label}: its features are filled with defaults "
                            "(availability flag = 0).")
    if topped.added_races:
        notes.append("Races added from Jolpica (newer than the f1db release): "
                     + ", ".join(str(r["EventName"]) for r in topped.added_races) + ".")
    for text in warnings:
        log.warning(text)
    return LiveRows(data, race_id, year, rnd, str(race.get("EventName")), date, moment,
                    flags, warnings, notes, str(rows["CircuitId"].iloc[0]),
                    _float(race.get("Lat")), _float(race.get("Long")))


def _entry_list(topped: ToppedUpHistory, quali: list[Record], weekend: WeekendSource | None,
                states: dict[str, str], year: int, date: pd.Timestamp,
                warnings: list[str]) -> list[Record]:
    """Qualifying's drivers if it ran, else the latest race's starters (checked against
    OpenF1's driver list once a session of this meeting has settled)."""
    mapper = topped.mapper
    if quali:
        numbers: dict[str, list[Any]] = {}
        for row in quali:
            numbers.setdefault(str(row["TeamId"]), []).append(row.get("Number"))
        entries = [{"DriverId": mapper.driver(str(r["DriverId"]), r.get("Number")),
                    "TeamId": mapper.team(str(r["TeamId"]), str(r.get("Team") or ""),
                                          numbers[str(r["TeamId"])]),
                    "Number": _int(r.get("Number")), "GridPosition": _int(r.get("Position"))}
                   for r in sorted(quali, key=lambda r: _int(r.get("Position")) or 99)]
        mapper.check()
        return entries
    results = pd.DataFrame(topped.get_race_results(year - 1))
    races = pd.DataFrame(topped.get_races(year - 1))
    done = results.merge(races[["RaceId", "Date"]], on="RaceId")
    done = done[~done["PositionText"].astype(str).isin({"DNQ", "DNPQ", "DNP", "EX"})]
    if done.empty:
        raise DataNotAvailableError("No previous race to take an entry list from.")
    last = done[done["Date"] == done["Date"].max()].sort_values("DisplayOrder")
    entries = [{"DriverId": r.DriverId, "TeamId": r.TeamId, "Number": _int(r.Number),
                "GridPosition": None} for r in last.itertuples(index=False)]
    settled = [n for n, s in states.items() if s == "settled"]
    if weekend is not None and settled and hasattr(weekend, "get_drivers"):
        sessions = meeting_sessions(weekend.get_sessions(year), date)
        session = sessions.get(settled[-1])
        cars = ({_int(d.get("Number")) for d in getattr(weekend, "get_drivers")(session)}
                if session else set())
        known = {e["Number"] for e in entries}
        if cars and cars != known:
            warnings.append(
                f"OpenF1 lists cars {sorted(c for c in cars - known if c is not None)} not in "
                f"the last race, and not cars {sorted(c for c in known - cars if c is not None)}"
                ". Using the last race's entry list until qualifying is published.")
    else:
        warnings.append("Entry list = the last race's starters (no session of this "
                        "meeting has finished yet).")
    return entries


def _placeholder_quali(topped: ToppedUpHistory, quali: list[Record],
                       race_id: int) -> list[Record]:
    drivers = {str(r["DriverId"]): topped.mapper.driver(str(r["DriverId"]), r.get("Number"))
               for r in quali}
    return [convert_qualifying(r, race_id, drivers) for r in quali]


class _NoArchive:
    """A ForecastSource with nothing archived (when only the live forecast is used)."""

    def get_race_forecast(self, lat: float, lon: float, date: str) -> list[Record]:
        return []

    def get_observed(self, lat: float, lon: float, date: str) -> list[Record]:
        return []


def _flag(values: pd.Series) -> int:
    """1 if any row has the information (NaN = hidden at this moment = 0)."""
    return int(pd.to_numeric(values, errors="coerce").fillna(0).max() > 0)


def _int(value: Any) -> int | None:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(min(1.0, a)))
