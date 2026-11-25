"""Who can still win the championship? A best-case check (FastF1 gallery idea).

A rival is still in it if, by winning every remaining race and sprint while the
leader scores nothing, they could reach the leader's CURRENT points.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from .errors import DataNotAvailableError
from .points import points_system
from .season import Season

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PointsLeft:
    """What is still to be raced, and the most one entrant can score from it."""

    races: int
    sprints: int
    max_points: float


def remaining_sessions(season: Season) -> list[tuple[int, str]]:
    """(round, 'sprint' or 'race') for every points-paying session still to run, in order.

    A sprint already run on a half-finished weekend is not included.
    """
    sprints_done = {int(r) for r in season.sprint_results["Round"].dropna()}
    sessions: list[tuple[int, str]] = []
    for event in season.remaining_events.itertuples():
        rnd = int(event.Round)
        if getattr(event, "HasSprint", False) and rnd not in sprints_done:
            sessions.append((rnd, "sprint"))
        sessions.append((rnd, "race"))
    return sessions


def points_still_available(season: Season, cars: int = 1) -> PointsLeft:
    """Best case for one entrant (cars=1: a driver, cars=2: a constructor)."""
    system = points_system(season.year)
    last_round = int(season.schedule["Round"].max())
    sessions = remaining_sessions(season)
    total = sum(system.sprint_max(cars) if kind == "sprint"
                else system.race_max(cars, last_round=(rnd == last_round))
                for rnd, kind in sessions)
    races = sum(kind == "race" for _, kind in sessions)
    return PointsLeft(races, len(sessions) - races, float(total))


def who_can_still_win(season: Season, constructors: bool = False) -> pd.DataFrame:
    """Each contender's best case against the leader's current points.

    Status: 'Champion' (title decided), 'Leader', 'Can win', 'Tie only'
    (best case equals the leader; settled on countback, most wins first) or 'Out'.
    """
    standings = _standings(season, constructors)
    left = points_still_available(season, cars=2 if constructors else 1)
    names = ["Team"] if constructors else ["Driver", "Team"]
    table = standings[["Position", *names, "Points", "Wins"]].copy()
    table["MaxPoints"] = table["Points"] + left.max_points
    table["GapToLeader"] = table["Points"].iloc[0] - table["Points"]
    table["Status"] = _statuses(table["GapToLeader"], left)
    return table


def title_fight_summary(season: Season, constructors: bool = False) -> str:
    """A one-line answer, e.g. '2 races and 1 sprint left (up to 58 points). ...'."""
    table = who_can_still_win(season, constructors)
    left = points_still_available(season, cars=2 if constructors else 1)
    name = "Team" if constructors else "Driver"
    leader = table.iloc[0]
    if leader["Status"] == "Champion":
        return f"{leader[name]} has clinched the {season.year} title with {leader['Points']:g} points."
    alive = table.loc[table["Status"].isin(["Can win", "Tie only"]), name].tolist()
    return (f"{_plural(left.races, 'race')} and {_plural(left.sprints, 'sprint')} left "
            f"(up to {left.max_points:g} points). {leader[name]} leads on "
            f"{leader['Points']:g}; still in it: {', '.join(alive)}.")


def _statuses(gaps: pd.Series, left: PointsLeft) -> list[str]:
    """One status per row; row 0 is the leader."""
    season_over = left.races == 0
    rivals = [_rival_status(g, left.max_points, season_over) for g in gaps.iloc[1:]]
    clinched = all(s == "Out" for s in rivals)
    return ["Champion" if clinched else "Leader", *rivals]


def _rival_status(gap: float, max_points: float, season_over: bool) -> str:
    if season_over or gap > max_points:
        return "Out"
    return "Tie only" if gap == max_points else "Can win"


def _standings(season: Season, constructors: bool) -> pd.DataFrame:
    table = season.constructor_standings if constructors else season.driver_standings
    if table.empty:
        raise DataNotAvailableError(
            f"No standings for {season.year} yet. Did you call season.load()?")
    _warn_if_stale(season, table)
    return table.sort_values("Position", na_position="last").reset_index(drop=True)


def _warn_if_stale(season: Season, standings: pd.DataFrame) -> None:
    """Jolpica updates results and standings separately; say so if they disagree."""
    results_round = max(season.completed_rounds, default=0)
    standings_round = int(standings["Round"].max())
    if standings_round != results_round:
        log.warning("Standings are after round %d but results go to round %d; "
                    "the calculator may be off until both update.",
                    standings_round, results_round)


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"
