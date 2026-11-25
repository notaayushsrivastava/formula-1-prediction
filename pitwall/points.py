"""F1 points systems (2010 onwards) as data, so calculators never hard-code 25."""
from __future__ import annotations

from dataclasses import dataclass

RACE_POINTS = (25, 18, 15, 12, 10, 8, 6, 4, 2, 1)  # 2010 onwards
SPRINT_POINTS_2021 = (3, 2, 1)                     # 2021 "sprint qualifying"
SPRINT_POINTS = (8, 7, 6, 5, 4, 3, 2, 1)           # 2022 onwards


@dataclass(frozen=True)
class PointsSystem:
    """One season's points. Tuples are indexed by finishing position - 1."""

    year: int
    race: tuple[int, ...]
    sprint: tuple[int, ...]
    fastest_lap: int = 0                    # bonus for fastest lap (if finishing top 10)
    double_points_last_round: bool = False  # 2014 Abu Dhabi

    def race_max(self, cars: int = 1, last_round: bool = False) -> int:
        """Most points `cars` cars of one entrant can score in one Grand Prix."""
        best = sum(self.race[:cars]) + self.fastest_lap  # only one fastest lap per race
        return 2 * best if last_round and self.double_points_last_round else best

    def sprint_max(self, cars: int = 1) -> int:
        """Most points `cars` cars of one entrant can score in one sprint."""
        return sum(self.sprint[:cars])


def points_system(year: int) -> PointsSystem:
    """The points system used in `year`. Future years assume the latest known rules."""
    if year < 2010:
        raise ValueError("pitwall knows points systems from 2010 onwards (25-18-15...).")
    if year < 2021:
        sprint: tuple[int, ...] = ()
    elif year == 2021:
        sprint = SPRINT_POINTS_2021
    else:
        sprint = SPRINT_POINTS
    return PointsSystem(
        year=year,
        race=RACE_POINTS,
        sprint=sprint,
        fastest_lap=1 if 2019 <= year <= 2024 else 0,
        double_points_last_round=(year == 2014),
    )
