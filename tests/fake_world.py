"""Offline fake sources for Phase 12/13 tests: an f1db-like history, a Jolpica-like
client (Jolpica ids, one renamed team, other circuit ids), an OpenF1-like weekend
source and an Open-Meteo-like forecast source. Nothing touches the network.

Seasons 2024-2026, six rounds each. In 2026, f1db has rounds 1-4 with results; round 5
has results only in Jolpica (the top-up); round 6 is the upcoming race.
"""
from __future__ import annotations

import datetime as dt

import numpy as np

YEARS = (2024, 2025, 2026)
ROUNDS = 6
DRIVERS = ["alpha-one", "bravo-two", "charlie-three", "delta-four", "echo-five", "foxtrot-six",
           "golf-seven", "hotel-eight", "india-nine", "juliet-ten"]
CODES = ["ALP", "BRA", "CHA", "DEL", "ECH", "FOX", "GOL", "HOT", "IND", "JUL"]
TEAMS = ["red-team", "blue-team", "green-team", "white-team", "black-team"]
SKILL = np.linspace(1.5, -1.5, len(DRIVERS))
CIRCUITS = [("north-ring", 50.0, 6.0), ("south-park", -30.0, 25.0), ("east-bay", 25.3, 51.5),
            ("west-hill", 40.0, -3.0), ("city-streets", 1.3, 103.8), ("desert-loop", 24.5, 54.6)]
UTC = dt.timezone.utc


def race_date(year: int, rnd: int) -> dt.date:
    return dt.date(year, 3, 1) + dt.timedelta(weeks=3 * (rnd - 1))


def race_id(year: int, rnd: int) -> int:
    return year * 100 + rnd


def team_of(i: int) -> str:
    return TEAMS[i // 2]


def finishing_order(year: int, rnd: int) -> tuple[list[int], int]:
    """(order of driver indices, index of the one DNF)."""
    rng = np.random.default_rng(race_id(year, rnd))
    pace = SKILL + rng.normal(0, 1.0, len(DRIVERS))
    order = list(np.argsort(-pace))
    return order, int(order[-1])


POINTS = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]


def f1db_has_results(year: int, rnd: int) -> bool:
    return year < 2026 or rnd <= 4


class FakeF1db:
    """HistorySource in f1db's format."""

    def get_races(self, min_year: int) -> list[dict]:
        out = []
        for year in YEARS:
            for rnd in range(1, ROUNDS + 1):
                if year < min_year:
                    continue
                cid, lat, lon = CIRCUITS[rnd - 1]
                out.append({"RaceId": race_id(year, rnd), "Year": year, "Round": rnd,
                            "Date": race_date(year, rnd).isoformat(), "Time": "13:00",
                            "EventName": f"{cid.title()} Grand Prix", "CircuitId": cid,
                            "CircuitType": "STREET" if cid == "city-streets" else "RACE",
                            "DistanceKm": 305.0, "Lat": lat, "Long": lon, "HasSprint": False})
        return out

    def get_race_results(self, min_year: int) -> list[dict]:
        out = []
        for year in YEARS:
            for rnd in range(1, ROUNDS + 1):
                if year < min_year or not f1db_has_results(year, rnd):
                    continue
                order, dnf = finishing_order(year, rnd)
                place = 0
                for display, i in enumerate(order, start=1):
                    finished = i != dnf
                    place += finished
                    out.append({"RaceId": race_id(year, rnd), "DriverId": DRIVERS[i],
                                "TeamId": team_of(i), "Number": i + 1,
                                "Position": place if finished else None,
                                "PositionText": str(place) if finished else "DNF",
                                "DisplayOrder": display, "GridPosition": _grid(year, rnd, i),
                                "GridText": None, "Laps": 57,
                                "TimeMillis": 5_400_000 + display if display == 1 else None,
                                "ReasonRetired": None if finished else "Engine",
                                "Points": POINTS[place - 1] if finished and place <= 10 else 0})
        return out

    def get_qualifying(self, min_year: int) -> list[dict]:
        out = []
        for year in YEARS:
            for rnd in range(1, ROUNDS + 1):
                if year < min_year or not f1db_has_results(year, rnd):
                    continue
                for i in range(len(DRIVERS)):
                    grid = _grid(year, rnd, i)
                    out.append({"RaceId": race_id(year, rnd), "DriverId": DRIVERS[i],
                                "QualiPosition": grid, "Q1Seconds": 90 + 0.1 * grid,
                                "Q2Seconds": None, "Q3Seconds": None})
        return out

    def get_pitstops(self, min_year: int) -> list[dict]:
        return [{"RaceId": race_id(y, r), "DriverId": DRIVERS[i], "TeamId": team_of(i),
                 "Stop": 1, "Lap": 20, "DurationSeconds": 22.0 + 0.1 * i}
                for y in YEARS for r in range(1, ROUNDS + 1) for i in range(len(DRIVERS))
                if y >= min_year and f1db_has_results(y, r)]

    def get_driver_standings(self, min_year: int) -> list[dict]:
        out = []
        for year in YEARS:
            totals = dict.fromkeys(DRIVERS, 0.0)
            for rnd in range(1, ROUNDS + 1):
                if not f1db_has_results(year, rnd):
                    continue
                for row in self.get_race_results(year):
                    if row["RaceId"] == race_id(year, rnd):
                        totals[row["DriverId"]] += row["Points"]
                if year >= min_year:
                    out += [{"RaceId": race_id(year, rnd), "DriverId": d, "Points": p}
                            for d, p in totals.items()]
        return out

    def get_drivers(self) -> list[dict]:
        return [{"DriverId": d, "Driver": c, "FullName": d.replace("-", " ").title(),
                 "DateOfBirth": "2000-01-01"} for d, c in zip(DRIVERS, CODES)]


def _grid(year: int, rnd: int, i: int) -> int:
    rng = np.random.default_rng(race_id(year, rnd) + 7)
    pace = SKILL + rng.normal(0, 0.7, len(DRIVERS))
    return int(list(np.argsort(-pace)).index(i)) + 1


def jolpica_team(team: str) -> tuple[str, str]:
    """Jolpica's id and name; 'black-team' is renamed in Jolpica ('shadow_racing')."""
    if team == "black-team":
        return "shadow_racing", "Shadow Racing F1 Team"
    return team.replace("-", "_"), team.replace("-", " ").title()


class FakeJolpica:
    """ResultsSource with Jolpica ids. `results_upto` controls which 2026 rounds have run."""

    def __init__(self, results_upto: int = 5, quali_upto: int = 5) -> None:
        self.results_upto = results_upto
        self.quali_upto = quali_upto
        self.calls: list[tuple] = []

    def get_schedule(self, year: int) -> list[dict]:
        out = []
        for rnd in range(1, ROUNDS + 1):
            cid, lat, lon = CIRCUITS[rnd - 1]
            out.append({"Round": rnd, "EventName": f"{cid.title()} Grand Prix",
                        "Date": race_date(year, rnd).isoformat(),
                        "CircuitId": "j_" + cid.replace("-", "_"), "Circuit": cid,
                        "Locality": cid.split("-")[0].title(), "Country": "Testland",
                        "Lat": lat + 0.01, "Long": lon - 0.01, "HasSprint": False})
        return out

    def _ran(self, year: int, rnd: int) -> bool:
        return year < 2026 or rnd <= self.results_upto

    def get_results(self, year: int, rnd: int) -> list[dict]:
        self.calls.append(("results", year, rnd))
        return self._results(year, rnd)

    def _results(self, year: int, rnd: int) -> list[dict]:
        if not self._ran(year, rnd):
            return []
        order, dnf = finishing_order(year, rnd)
        out, place = [], 0
        for display, i in enumerate(order, start=1):
            finished = i != dnf
            place += finished
            tid, tname = jolpica_team(team_of(i))
            out.append({"Round": rnd, "DriverId": DRIVERS[i].replace("-", "_"),
                        "Driver": CODES[i], "FullName": DRIVERS[i], "TeamId": tid,
                        "Team": tname, "Number": str(i + 1),
                        "GridPosition": _grid(year, rnd, i), "Position": display,
                        "ClassifiedPosition": str(place) if finished else "R",
                        "Status": "Finished" if finished else "Engine",
                        "Points": float(POINTS[place - 1]) if finished and place <= 10 else 0.0,
                        "LapsCompleted": 57, "FastestLapSeconds": 80.0})
        return out

    def get_qualifying(self, year: int, rnd: int) -> list[dict]:
        if year == 2026 and rnd > self.quali_upto:
            return []
        out = []
        for i in range(len(DRIVERS)):
            grid = _grid(year, rnd, i)
            tid, tname = jolpica_team(team_of(i))
            out.append({"Round": rnd, "DriverId": DRIVERS[i].replace("-", "_"),
                        "Driver": CODES[i], "Number": str(i + 1), "FullName": DRIVERS[i],
                        "TeamId": tid, "Team": tname, "Position": grid,
                        "Q1Seconds": 90 + 0.1 * grid, "Q2Seconds": None, "Q3Seconds": None})
        return out

    def get_pitstops(self, year: int, rnd: int) -> list[dict]:
        if not self._ran(year, rnd):
            return []
        return [{"DriverId": d.replace("-", "_"), "LapNumber": 20, "Stop": 1,
                 "DurationSeconds": 22.5} for d in DRIVERS]

    def get_driver_standings(self, year: int, rnd: int | None = None) -> list[dict]:
        totals = dict.fromkeys(DRIVERS, 0.0)
        for r in range(1, (rnd or ROUNDS) + 1):
            for row in self._results(year, r):
                totals[row["DriverId"].replace("_", "-")] += row["Points"]
        return [{"Round": rnd, "DriverId": d.replace("-", "_"), "Points": p}
                for d, p in totals.items()]


def session_times(year: int, rnd: int) -> dict[str, tuple[dt.datetime, dt.datetime]]:
    day = race_date(year, rnd)
    base = dt.datetime(day.year, day.month, day.day, tzinfo=UTC)
    return {"Practice 1": (base - dt.timedelta(days=2, hours=-11),
                           base - dt.timedelta(days=2, hours=-12)),
            "Practice 2": (base - dt.timedelta(days=2, hours=-15),
                           base - dt.timedelta(days=2, hours=-16)),
            "Practice 3": (base - dt.timedelta(days=1, hours=-11),
                           base - dt.timedelta(days=1, hours=-12)),
            "Qualifying": (base - dt.timedelta(days=1, hours=-15),
                           base - dt.timedelta(days=1, hours=-16)),
            "Race": (base + dt.timedelta(hours=13), base + dt.timedelta(hours=15))}


class FakeOpenF1:
    """WeekendSource. Records every session whose data was requested."""

    def __init__(self) -> None:
        self.requested: list[str] = []

    def get_sessions(self, year: int) -> list[dict]:
        out = []
        for rnd in range(1, ROUNDS + 1):
            for k, (name, (start, end)) in enumerate(session_times(year, rnd).items()):
                out.append({"SessionKey": race_id(year, rnd) * 10 + k,
                            "MeetingKey": race_id(year, rnd), "Name": name,
                            "Type": name.split()[0], "DateStart": start.isoformat(),
                            "DateEnd": end.isoformat(), "Year": year, "Cancelled": False})
        return out

    def _laps(self, session: dict) -> list[dict]:
        rng = np.random.default_rng(int(session["SessionKey"]))
        laps = []
        for i in range(len(DRIVERS)):
            base = 92.0 - 0.3 * SKILL[i]
            laps.append({"Number": i + 1, "LapNumber": 1, "LapSeconds": None, "PitOutLap": True})
            for n in range(2, 10):
                laps.append({"Number": i + 1, "LapNumber": n, "PitOutLap": False,
                             "LapSeconds": base + rng.normal(0, 0.1)})
        return laps

    def get_laps(self, session: dict) -> list[dict]:
        self.requested.append(f"laps:{session['SessionKey']}")
        return self._laps(session)

    def get_stints(self, session: dict) -> list[dict]:
        self.requested.append(f"stints:{session['SessionKey']}")
        return [{"Number": i + 1, "Stint": 1, "Compound": "MEDIUM", "LapStart": 1,
                 "LapEnd": 30, "TyreAgeAtStart": 0} for i in range(len(DRIVERS))]

    def get_weather(self, session: dict) -> list[dict]:
        return [{"Time": session["DateStart"], "AirTemp": 25.0, "TrackTemp": 38.0,
                 "Rainfall": 0}]

    def get_race_control(self, session: dict) -> list[dict]:
        sc = int(session["SessionKey"]) % 3 == 0
        return [{"Time": session["DateStart"], "Lap": 5, "Category": "SafetyCar",
                 "Flag": None, "Message": "SAFETY CAR DEPLOYED" if sc else "CLEAR"}]

    def get_pit(self, session: dict) -> list[dict]:
        return []

    def get_drivers(self, session: dict) -> list[dict]:
        return [{"Number": i + 1, "Driver": c, "FullName": d, "Team": team_of(i)}
                for i, (d, c) in enumerate(zip(DRIVERS, CODES))]


class FakeMeteo:
    """ForecastSource with a live forecast."""

    def get_race_forecast(self, lat: float, lon: float, date: str) -> list[dict]:
        return [{"Time": f"{date[:10]}T{h:02d}:00", "LeadDays": lead, "AirTemp": 24.0,
                 "RainMm": 0.0} for lead in (1, 2, 3) for h in range(12, 16)]

    def get_observed(self, lat: float, lon: float, date: str) -> list[dict]:
        return [{"Time": f"{date[:10]}T{h:02d}:00", "AirTemp": 24.0, "RainMm": 0.0}
                for h in range(12, 16)]

    def get_live_forecast(self, lat: float, lon: float, date: str) -> list[dict]:
        return [{"Time": f"{date[:10]}T{h:02d}:00", "AirTemp": 31.0, "RainMm": 0.2}
                for h in range(12, 16)]


def at(year: int, rnd: int, session: str, after: dt.timedelta) -> dt.datetime:
    """A moment relative to a session's end (negative = before it ended)."""
    return session_times(year, rnd)[session][1] + after


def sources(results_upto: int = 5, quali_upto: int = 5, weather: bool = True,
            openf1: bool = True):
    from pitwall.predict.service import Sources
    meteo = FakeMeteo() if weather else None
    return Sources(FakeF1db(), FakeJolpica(results_upto, quali_upto),
                   FakeOpenF1() if openf1 else None, meteo,
                   meteo.get_live_forecast if meteo else None)
