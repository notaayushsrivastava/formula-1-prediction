"""The contracts data sources follow (Protocols), plus shared helpers.

Sources return flat records (lists of dicts), so the rest of pitwall never sees raw
JSON or CSV. Any object with the right methods is a valid source; tests use fakes.
"""
from __future__ import annotations

from typing import Any, Protocol

Record = dict[str, Any]


class ResultsSource(Protocol):
    """Anything that can provide F1 data as flat records (lists of dicts)."""

    # ----- one race weekend -----
    def get_schedule(self, year: int) -> list[Record]:
        """One record per race: Round, EventName, Date, CircuitId, Circuit,
        Locality, Country, Lat, Long, HasSprint."""
        ...

    def get_results(self, year: int, rnd: int) -> list[Record]:
        """One record per driver: Round, DriverId, Driver (3-letter code), Number,
        FullName, TeamId, Team, GridPosition, Position, ClassifiedPosition,
        Status, Points, LapsCompleted, FastestLapSeconds."""
        ...

    def get_sprint_results(self, year: int, rnd: int) -> list[Record]:
        """Same keys as get_results, for the sprint ([] if the weekend had none)."""
        ...

    def get_qualifying(self, year: int, rnd: int) -> list[Record]:
        """One record per driver: Round, DriverId, Driver, Number, FullName, TeamId,
        Team, Position, Q1Seconds, Q2Seconds, Q3Seconds (None if not reached)."""
        ...

    def get_laps(self, year: int, rnd: int) -> list[Record]:
        """One record per driver per lap: DriverId, LapNumber, Position, LapTimeSeconds."""
        ...

    def get_pitstops(self, year: int, rnd: int) -> list[Record]:
        """One record per stop: DriverId, LapNumber, Stop, DurationSeconds."""
        ...

    # ----- whole season -----
    def get_season_results(self, year: int, sprint: bool = False) -> list[Record]:
        """get_results (or sprint results) for every completed round, in one call."""
        ...

    def get_driver_standings(self, year: int, rnd: int | None = None) -> list[Record]:
        """Championship after round rnd (latest if None): Round, DriverId, Driver,
        FullName, TeamId, Team, Position, Points, Wins."""
        ...

    def get_constructor_standings(self, year: int, rnd: int | None = None) -> list[Record]:
        """Constructors' championship: Round, TeamId, Team, Position, Points, Wins."""
        ...


class HistorySource(Protocol):
    """Bulk race history, e.g. the f1db CSV release.

    Every method returns all rows from `min_year` onwards. Ids are the source's own
    (f1db writes 'max-verstappen', Jolpica 'max_verstappen').
    """

    def get_races(self, min_year: int) -> list[Record]:
        """One record per race: RaceId, Year, Round, Date ('YYYY-MM-DD', UTC), Time
        ('HH:MM' UTC or None), EventName, CircuitId, CircuitType ('RACE', 'ROAD' or
        'STREET'), DistanceKm, Lat, Long, HasSprint."""
        ...

    def get_race_results(self, min_year: int) -> list[Record]:
        """One record per car: RaceId, DriverId, TeamId, Number, Position (None if not
        classified), PositionText ('1', 'DNF', 'DSQ', 'DNS', ...), DisplayOrder,
        GridPosition (None = pit-lane start), GridText, Laps, TimeMillis,
        ReasonRetired, Points."""
        ...

    def get_qualifying(self, min_year: int) -> list[Record]:
        """One record per driver: RaceId, DriverId, QualiPosition, Q1Seconds,
        Q2Seconds, Q3Seconds (None if not reached)."""
        ...

    def get_pitstops(self, min_year: int) -> list[Record]:
        """One record per stop: RaceId, DriverId, TeamId, Stop, Lap, DurationSeconds
        (time spent in the pit lane)."""
        ...

    def get_driver_standings(self, min_year: int) -> list[Record]:
        """Championship after each race: RaceId, DriverId, Points (sprints included)."""
        ...

    def get_drivers(self) -> list[Record]:
        """One record per driver: DriverId, Driver (3-letter code), FullName, DateOfBirth."""
        ...


class WeekendSource(Protocol):
    """Session data of 2023+ race weekends, e.g. OpenF1.

    Session records come from get_sessions; every other method takes one of them.
    """

    def get_sessions(self, year: int) -> list[Record]:
        """SessionKey, MeetingKey, Name ('Practice 1', 'Qualifying', 'Race', ...),
        Type, DateStart, DateEnd (ISO 8601, UTC), Year, Cancelled."""
        ...

    def get_laps(self, session: Record) -> list[Record]:
        """Number (car), LapNumber, LapSeconds (None if untimed), PitOutLap."""
        ...

    def get_stints(self, session: Record) -> list[Record]:
        """Number, Stint, Compound ('SOFT' ... 'INTERMEDIATE', 'WET'), LapStart,
        LapEnd, TyreAgeAtStart."""
        ...

    def get_weather(self, session: Record) -> list[Record]:
        """About one per minute: Time, AirTemp, TrackTemp (deg C), Rainfall (0 or 1)."""
        ...

    def get_race_control(self, session: Record) -> list[Record]:
        """Time, Lap, Category ('Flag', 'SafetyCar', ...), Flag, Message."""
        ...

    def get_pit(self, session: Record) -> list[Record]:
        """Number, Lap, LaneSeconds (time in the pit lane), StopSeconds (stationary)."""
        ...


class ForecastSource(Protocol):
    """Weather at a circuit, e.g. Open-Meteo."""

    def get_race_forecast(self, lat: float, lon: float, date: str) -> list[Record]:
        """Hourly forecasts for one UTC day AS ISSUED 1-3 days earlier: Time
        ('YYYY-MM-DDTHH:MM', UTC), LeadDays, AirTemp (deg C), RainMm.
        [] when no archived forecast exists."""
        ...

    def get_observed(self, lat: float, lon: float, date: str) -> list[Record]:
        """What actually happened, hourly: Time, AirTemp, RainMm ([] if not archived yet)."""
        ...


def parse_time_to_seconds(text: str | None) -> float | None:
    """Convert '1:23.456', '23.456' or '1:01:23.4' to seconds. Bad input -> None."""
    if not text:
        return None
    try:
        parts = text.strip().split(":")
        seconds = float(parts[-1])
        for power, part in enumerate(reversed(parts[:-1]), start=1):
            seconds += int(part) * 60**power
        return seconds
    except ValueError:
        return None
