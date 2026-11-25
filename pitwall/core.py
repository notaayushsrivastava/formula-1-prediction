"""Core containers: Session (one session of a race weekend) and Laps (a smarter DataFrame)."""
from __future__ import annotations

import numbers
from collections.abc import Iterable
from typing import Any

import pandas as pd

from .errors import DataNotAvailableError
from .sources.base import Record, ResultsSource
from .sources.jolpica import JolpicaClient

SESSION_KINDS = {"R": "Race", "Q": "Qualifying", "S": "Sprint"}
LAP_COLUMNS = [
    "Driver", "DriverId", "Team", "TeamId", "LapNumber", "Position",
    "LapTime", "LapTimeSeconds", "Stint", "PitInLap", "PitOutLap",
]
INT_RESULT_COLUMNS = ["Round", "GridPosition", "Position", "LapsCompleted"]
Q_COLUMNS = ["Q1Seconds", "Q2Seconds", "Q3Seconds"]


class Laps(pd.DataFrame):
    """One row per driver per lap, with FastF1-style helper methods."""

    @property
    def _constructor(self) -> type[Laps]:
        return Laps

    def pick_drivers(self, *drivers: str | Iterable[str]) -> Laps:
        """Keep laps of the given drivers, by code ('LEC') or id ('leclerc')."""
        wanted = {str(d).upper() for d in _flatten(drivers)}
        by_code = self["Driver"].str.upper().isin(wanted)
        by_id = self["DriverId"].str.upper().isin(wanted)
        return self[by_code | by_id]

    def pick_teams(self, *teams: str | Iterable[str]) -> Laps:
        """Keep laps of the given teams, by id ('red_bull') or name ('Red Bull')."""
        wanted = {str(t).lower() for t in _flatten(teams)}
        by_id = self["TeamId"].str.lower().isin(wanted)
        by_name = self["Team"].str.lower().isin(wanted)
        return self[by_id | by_name]

    def pick_laps(self, *laps: int | Iterable[int]) -> Laps:
        """Keep lap numbers: pick_laps(1), pick_laps(range(10, 21)), pick_laps(1, [5, 6])."""
        wanted = {int(n) for n in _flatten(laps)}
        return self[self["LapNumber"].isin(wanted)]

    def pick_fastest(self) -> pd.Series | None:
        """The single fastest timed lap, or None if no lap has a time."""
        timed = self.dropna(subset=["LapTimeSeconds"])
        if timed.empty:
            return None
        return timed.loc[timed["LapTimeSeconds"].idxmin()]

    def pick_quicklaps(self, threshold: float = 1.07) -> Laps:
        """Laps faster than threshold x the fastest lap in THIS selection.

        Removes most safety car, VSC, red-flag, pit and lap-1 outliers.
        """
        fastest = self.pick_fastest()
        if fastest is None:
            return self.iloc[0:0]
        return self[self["LapTimeSeconds"] < fastest["LapTimeSeconds"] * threshold]

    def pick_wo_box(self) -> Laps:
        """Drop in-laps and out-laps (laps distorted by a pit stop)."""
        return self[~(self["PitInLap"] | self["PitOutLap"])]

    def pick_compounds(self, *compounds: str | Iterable[str]) -> Laps:
        """Keep laps on the given compounds ('SOFT', 'MEDIUM', ...). Needs Live Timing."""
        self._require("Compound")
        wanted = {str(c).upper() for c in _flatten(compounds)}
        return self[self["Compound"].isin(wanted)]

    def pick_track_status(self, status: str, how: str = "equals") -> Laps:
        """Filter by track status: '1' green, '2' yellow, '4' SC, '5' red, '6'/'7' VSC.

        how='equals' (exactly this all lap), 'contains' (at some point), 'excludes' (never).
        """
        self._require("TrackStatus")
        codes = self["TrackStatus"].astype(str)
        if how == "equals":
            mask = codes == status
        elif how == "contains":
            mask = codes.str.contains(status, regex=False)
        elif how == "excludes":
            mask = ~codes.str.contains(status, regex=False)
        else:
            raise ValueError("how must be 'equals', 'contains' or 'excludes'.")
        return self[mask]

    def pick_green(self) -> Laps:
        """Laps run entirely under green-flag conditions. Needs Live Timing."""
        return self.pick_track_status("1", how="equals")

    def _require(self, column: str) -> None:
        if column not in self.columns:
            raise ValueError(f"No '{column}' column: call pitwall.load_live_timing(session) first.")

    def stints(self) -> pd.DataFrame:
        """One row per driver stint: first lap, last lap, number of laps."""
        if self.empty:
            return pd.DataFrame(columns=["Driver", "Stint", "StartLap", "EndLap", "Laps"])
        summary = self.groupby(["Driver", "Stint"], as_index=False).agg(
            StartLap=("LapNumber", "min"),
            EndLap=("LapNumber", "max"),
            Laps=("LapNumber", "count"),
        )
        return pd.DataFrame(summary)


class Session:
    """One session of a race weekend: 'R' race, 'Q' qualifying or 'S' sprint.

    Jolpica only has lap-by-lap data for races, so `laps` stays empty for Q and S.
    """

    def __init__(self, year: int, rnd: int, source: ResultsSource, kind: str = "R") -> None:
        self.year = year
        self.round = rnd
        self.source = source
        self.kind = kind
        self.event: pd.Series | None = None
        self.results = pd.DataFrame()
        self.pit_stops = pd.DataFrame()
        self.laps = Laps(columns=LAP_COLUMNS)
        self.track_status = pd.DataFrame()  # filled by pitwall.load_live_timing()
        self.timing_laps = pd.DataFrame()   # filled by pitwall.load_telemetry()
        self.car_data = pd.DataFrame()
        self.positions = pd.DataFrame()

    @property
    def name(self) -> str:
        """Readable name such as '2024 Italian Grand Prix Qualifying'."""
        label = "" if self.kind == "R" else f" {SESSION_KINDS[self.kind]}"
        if self.event is None:
            return f"{self.year} round {self.round}{label}"
        return f"{self.year} {self.event['EventName']}{label}"

    def load(self) -> Session:
        """Download (or read from cache) everything; returns self so you can chain."""
        self.event = _event_row(self.source.get_schedule(self.year), self.year, self.round)
        if self.kind == "Q":
            rows = self.source.get_qualifying(self.year, self.round)
            self.results = _build_qualifying(rows, self.name)
        elif self.kind == "S":
            self._load_sprint()
        else:
            self._load_race()
        return self

    def _load_sprint(self) -> None:
        assert self.event is not None  # load() sets it before calling this
        if not self.event.get("HasSprint", True):
            raise DataNotAvailableError(f"{self.name}: this weekend has no sprint.")
        rows = self.source.get_sprint_results(self.year, self.round)
        self.results = _build_results(rows, self.name)

    def _load_race(self) -> None:
        self.results = _build_results(self.source.get_results(self.year, self.round), self.name)
        stops = self.source.get_pitstops(self.year, self.round)
        self.pit_stops = pd.DataFrame(stops)
        laps = self.source.get_laps(self.year, self.round)
        self.laps = _build_laps(laps, self.results, stops, self.name)

    def get_driver(self, identifier: str) -> pd.Series:
        """Results row for a driver code ('VER') or id ('max_verstappen')."""
        ident = identifier.upper()
        res = self.results
        hits = res[(res["Driver"].str.upper() == ident) | (res["DriverId"].str.upper() == ident)]
        if hits.empty:
            raise ValueError(f"Unknown driver {identifier!r}. Known: {', '.join(res['Driver'])}")
        return hits.iloc[0]


def get_session(year: int, event: int | str, kind: str = "R",
                source: ResultsSource | None = None) -> Session:
    """Find a session by round number or name, e.g. get_session(2024, 'Monza', 'Q').

    kind: 'R' race (default), 'Q' qualifying, 'S' sprint.
    """
    kind = kind.upper()
    if kind not in SESSION_KINDS:
        raise ValueError(f"Unknown session {kind!r}. Use one of: {', '.join(SESSION_KINDS)}.")
    source = source or JolpicaClient()
    schedule = pd.DataFrame(source.get_schedule(year))
    if schedule.empty:
        raise DataNotAvailableError(f"No schedule found for {year}.")
    return Session(year, _resolve_round(schedule, event, year), source, kind)


# ---------- builders (small, testable functions) ----------
def _build_results(rows: list[Record], name: str) -> pd.DataFrame:
    if not rows:
        raise DataNotAvailableError(f"No results for {name} (has it happened yet?)")
    df = pd.DataFrame(rows)
    for col in INT_RESULT_COLUMNS:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    # Classified = has a numeric finishing position (retired drivers can still be classified)
    df["Classified"] = df["ClassifiedPosition"].astype(str).str.isdigit()
    return df.sort_values("Position", na_position="last").reset_index(drop=True)


def _build_qualifying(rows: list[Record], name: str) -> pd.DataFrame:
    """Official qualifying order, plus each driver's best time from any part (Q1/Q2/Q3)."""
    if not rows:
        raise DataNotAvailableError(f"No qualifying results for {name} (has it happened yet?)")
    df = pd.DataFrame(rows)
    df["Position"] = pd.to_numeric(df["Position"], errors="coerce").astype("Int64")
    times = df[Q_COLUMNS].apply(pd.to_numeric, errors="coerce")  # None -> NaN
    df[Q_COLUMNS] = times
    df["BestSeconds"] = times.min(axis=1)  # min() skips NaN
    df["DeltaToFastestSeconds"] = df["BestSeconds"] - df["BestSeconds"].min()
    return df.sort_values("Position", na_position="last").reset_index(drop=True)


def _build_laps(rows: list[Record], results: pd.DataFrame,
                stops: list[Record], name: str) -> Laps:
    if not rows:
        raise DataNotAvailableError(
            f"No lap-by-lap data for {name}. Jolpica has laps from 1996 onwards."
        )
    laps = pd.DataFrame(rows).sort_values(["DriverId", "LapNumber"]).reset_index(drop=True)
    laps["LapTimeSeconds"] = pd.to_numeric(laps["LapTimeSeconds"], errors="coerce")
    laps["LapTime"] = pd.to_timedelta(laps["LapTimeSeconds"], unit="s")
    laps = _attach_driver_info(laps, results)
    laps = _mark_pit_laps(laps, stops)
    return Laps(laps[LAP_COLUMNS])


def _attach_driver_info(laps: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """Add Driver code and Team to every lap (a left merge on DriverId)."""
    info = results[["DriverId", "Driver", "Team", "TeamId"]]
    merged = laps.merge(info, on="DriverId", how="left")
    merged["Driver"] = merged["Driver"].fillna(merged["DriverId"])
    merged[["Team", "TeamId"]] = merged[["Team", "TeamId"]].fillna("unknown")
    return merged


def _mark_pit_laps(laps: pd.DataFrame, stops: list[Record]) -> pd.DataFrame:
    """Flag in-laps/out-laps and number the stints (stint = stops so far + 1)."""
    stop_keys = {(s["DriverId"], s["LapNumber"]) for s in stops}
    keys = list(zip(laps["DriverId"], laps["LapNumber"]))
    laps["PitInLap"] = [k in stop_keys for k in keys]
    laps["PitOutLap"] = [(d, n - 1) in stop_keys for d, n in keys]
    laps["Stint"] = laps.groupby("DriverId")["PitOutLap"].cumsum().astype(int) + 1
    return laps


def _event_row(schedule: list[Record], year: int, rnd: int) -> pd.Series:
    for race in schedule:
        if race["Round"] == rnd:
            return pd.Series(race)
    raise DataNotAvailableError(f"Round {rnd} not found in the {year} schedule.")


def _resolve_round(schedule: pd.DataFrame, event: int | str, year: int) -> int:
    if isinstance(event, int):
        if event not in set(schedule["Round"]):
            raise ValueError(f"{year} has no round {event} (it has {len(schedule)} rounds).")
        return event
    needle = event.strip().lower()
    text = schedule[["EventName", "Circuit", "Locality", "Country"]].fillna("")
    matches = text.apply(lambda col: col.str.lower().str.contains(needle, regex=False))
    hits = schedule[matches.any(axis=1)]
    if len(hits) == 1:
        return int(hits["Round"].iloc[0])
    shown = hits if len(hits) else schedule
    options = ", ".join(f"{r.Round}: {r.EventName}" for r in shown.itertuples())
    problem = "matches several races" if len(hits) else "matches no race"
    raise ValueError(f"'{event}' {problem} in {year}. Options: {options}")


def _flatten(items: Iterable[Any]) -> list[Any]:
    """('VER', ['LEC', 'SAI']) -> ['VER', 'LEC', 'SAI'];  (1, range(5, 7)) -> [1, 5, 6]."""
    out: list[Any] = []
    for item in items:
        if isinstance(item, (str, numbers.Integral)):
            out.append(item)
        else:
            out.extend(item)
    return out
