"""Season: every round of one championship year, plus the standings."""
from __future__ import annotations

import pandas as pd

from .core import Session, get_session
from .errors import DataNotAvailableError
from .sources.base import Record, ResultsSource
from .sources.jolpica import JolpicaClient

INT_COLUMNS = ["Round", "GridPosition", "Position", "LapsCompleted", "Wins"]
RESULT_KEYS = ["Round", "DriverId", "Driver", "TeamId", "Team", "Position", "Points"]
DRIVER_STANDING_KEYS = ["Round", "DriverId", "Driver", "FullName", "TeamId", "Team",
                        "Position", "Points", "Wins"]
CONSTRUCTOR_STANDING_KEYS = ["Round", "TeamId", "Team", "Position", "Points", "Wins"]


class Season:
    """One championship year: schedule, race and sprint results, and standings.

    after_round=N rewinds the season to just after round N (handy for checking
    the title-fight calculator against history).
    """

    def __init__(self, year: int, source: ResultsSource | None = None,
                 after_round: int | None = None) -> None:
        self.year = year
        self.source = source or JolpicaClient()
        self.after_round = after_round
        self.schedule = pd.DataFrame()
        self.results = pd.DataFrame()
        self.sprint_results = pd.DataFrame()
        self.driver_standings = pd.DataFrame()
        self.constructor_standings = pd.DataFrame()

    def load(self) -> Season:
        """Download (or read from cache) the season: about 10 requests the first time."""
        src, y, rnd = self.source, self.year, self.after_round
        self.schedule = pd.DataFrame(src.get_schedule(y))
        if self.schedule.empty:
            raise DataNotAvailableError(f"No schedule found for {y}.")
        self.results = _upto(_frame(src.get_season_results(y), RESULT_KEYS), rnd)
        self.sprint_results = _upto(
            _frame(src.get_season_results(y, sprint=True), RESULT_KEYS), rnd)
        self.driver_standings = _frame(src.get_driver_standings(y, rnd), DRIVER_STANDING_KEYS)
        self.constructor_standings = _frame(
            src.get_constructor_standings(y, rnd), CONSTRUCTOR_STANDING_KEYS)
        return self

    @property
    def completed_rounds(self) -> list[int]:
        """Rounds that already have race results, in order."""
        return sorted(int(r) for r in self.results["Round"].unique())

    @property
    def remaining_events(self) -> pd.DataFrame:
        """Schedule rows for rounds without race results yet."""
        done = set(self.completed_rounds)
        return self.schedule[~self.schedule["Round"].isin(done)].reset_index(drop=True)

    def points_by_round(self, include_sprints: bool = True) -> pd.DataFrame:
        """Drivers x rounds table of points scored, plus a 'Total' column, best first."""
        parts = [self.results, self.sprint_results] if include_sprints else [self.results]
        parts = [p for p in parts if not p.empty]  # concat warns about empty frames
        if not parts:
            return pd.DataFrame(columns=["Total"])
        scored = pd.concat(parts, ignore_index=True)
        table = scored.pivot_table(index="Driver", columns="Round", values="Points",
                                   aggfunc="sum", fill_value=0.0)
        table["Total"] = table.sum(axis=1)
        return table.sort_values("Total", ascending=False)

    def session(self, event: int | str, kind: str = "R") -> Session:
        """A Session from this season that reuses the same data source."""
        return get_session(self.year, event, kind, source=self.source)


def _frame(rows: list[Record], columns: list[str]) -> pd.DataFrame:
    """DataFrame from records; the key columns exist even when there are no rows."""
    if not rows:
        return pd.DataFrame(columns=columns)
    df = pd.DataFrame(rows)
    for col in INT_COLUMNS:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df.sort_values(["Round", "Position"], na_position="last").reset_index(drop=True)


def _upto(df: pd.DataFrame, rnd: int | None) -> pd.DataFrame:
    """Rows up to and including round rnd (all rows if rnd is None)."""
    if rnd is None or df.empty:
        return df
    return df[df["Round"] <= rnd].reset_index(drop=True)
