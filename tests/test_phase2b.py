"""Offline tests for Phase 2b plots (no network, no windows)."""
import matplotlib

matplotlib.use("Agg")  # draw off-screen; must come before pyplot is imported

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from pitwall import Season, get_session  # noqa: E402
from pitwall.plotting import (  # noqa: E402
    championship_progress, laptime_distribution, points_heatmap, position_changes,
    qualifying_overview, style_for, team_pace_boxplot,
)

LAP_TIMES = {
    "ALP": [92.0, 90.0, 90.2, 90.4, 90.1, 90.3],
    "BRA": [92.5, 90.9, 91.1, 105.0, 92.0, 91.0],  # pits at the end of lap 4
    "ARC": [93.0, 90.6, 90.8, 91.0, 90.7, 90.9],
    "BOX": [94.0, 91.5, 91.7, 91.9, 91.6, 91.8],
}
RUNNING_ORDER = ["ALP", "BRA", "ARC", "BOX"]


def _result(rnd, code, team, pos, grid, points):
    return {"Round": rnd, "DriverId": code.lower(), "Driver": code, "FullName": code,
            "TeamId": team, "Team": team.title(), "Number": "0", "GridPosition": grid,
            "Position": pos, "ClassifiedPosition": str(pos), "Status": "Finished",
            "Points": points, "LapsCompleted": 6, "FastestLapSeconds": None}


def _quali(code, team, pos, q1):
    return {"Round": 1, "DriverId": code.lower(), "Driver": code, "FullName": code,
            "TeamId": team, "Team": team.title(), "Number": "0", "Position": pos,
            "Q1Seconds": q1, "Q2Seconds": None, "Q3Seconds": None}


class FakeRaceSource:
    """Team red (ALP, ARC) and blue (BRA, BOX). Rounds 1-2 done, round 3 to come.

    Round 1: ARC starts from the pit lane (grid 0) and has no qualifying time.
    """

    def get_schedule(self, year):
        names = ["Alpha", "Bravo", "Charlie"]
        return [{"Round": i, "EventName": f"{n} Grand Prix", "Date": "2024-01-01",
                 "CircuitId": n.lower(), "Circuit": f"{n} Ring", "Locality": "",
                 "Country": "", "Lat": 0.0, "Long": 0.0, "HasSprint": False}
                for i, n in enumerate(names, start=1)]

    def get_season_results(self, year, sprint=False):
        if sprint:
            return []
        return [_result(1, "ALP", "red", 1, 2, 25.0), _result(1, "BRA", "blue", 2, 1, 18.0),
                _result(1, "ARC", "red", 3, 0, 15.0), _result(1, "BOX", "blue", 4, 3, 12.0),
                _result(2, "BRA", "blue", 1, 1, 25.0), _result(2, "BOX", "blue", 2, 2, 18.0),
                _result(2, "ALP", "red", 3, 3, 15.0), _result(2, "ARC", "red", 4, 4, 12.0)]

    def get_results(self, year, rnd):
        return [r for r in self.get_season_results(year) if r["Round"] == rnd]

    def get_laps(self, year, rnd):
        return [{"DriverId": code.lower(), "LapNumber": lap,
                 "Position": RUNNING_ORDER.index(code) + 1, "LapTimeSeconds": t}
                for code, times in LAP_TIMES.items()
                for lap, t in enumerate(times, start=1)]

    def get_pitstops(self, year, rnd):
        return [{"DriverId": "bra", "LapNumber": 4, "Stop": 1, "DurationSeconds": 22.0}]

    def get_qualifying(self, year, rnd):
        return [_quali("BRA", "blue", 1, 80.0), _quali("ALP", "red", 2, 80.2),
                _quali("BOX", "blue", 3, 80.5), _quali("ARC", "red", 4, None)]

    def get_driver_standings(self, year, rnd=None):
        return []

    def get_constructor_standings(self, year, rnd=None):
        return []


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def source():
    return FakeRaceSource()


@pytest.fixture
def race(source):
    return get_session(2024, 1, source=source).load()


@pytest.fixture
def quali(source):
    return get_session(2024, 1, "Q", source=source).load()


@pytest.fixture
def season(source):
    return Season(2024, source=source).load()


def _labels(ticks):
    return [t.get_text() for t in ticks]


def test_style_for_second_teammate_dashed(race):
    assert style_for(race.results, "ALP")["linestyle"] == "-"
    assert style_for(race.results, "ARC")["linestyle"] == "--"


def test_laptime_distribution_in_finishing_order(race):
    ax = laptime_distribution(race)
    assert _labels(ax.get_xticklabels()) == ["ALP", "BRA", "ARC", "BOX"]


def test_position_changes_start_from_grid(race):
    ax = position_changes(race)
    lines = {line.get_label(): line for line in ax.get_lines()}
    assert set(lines) == {"ALP", "BRA", "ARC", "BOX"}
    arc = lines["ARC"]
    assert arc.get_xdata()[0] == 0 and arc.get_ydata()[0] == 4  # pit-lane start -> back


def test_team_pace_fastest_team_first(race):
    ax = team_pace_boxplot(race)
    assert _labels(ax.get_xticklabels()) == ["Red", "Blue"]


def test_qualifying_overview_skips_drivers_without_time(quali):
    ax = qualifying_overview(quali)
    assert _labels(ax.get_yticklabels()) == ["BRA", "ALP", "BOX"]
    assert len(ax.patches) == 3


def test_wrong_session_types_raise(race, quali):
    with pytest.raises(ValueError):
        qualifying_overview(race)
    with pytest.raises(ValueError):
        position_changes(quali)


def test_points_heatmap_grid(season):
    ax = points_heatmap(season)
    assert ax.images[0].get_array().shape == (4, 2)  # 4 drivers x 2 rounds
    assert _labels(ax.get_xticklabels()) == ["Alpha", "Bravo"]
    assert _labels(ax.get_yticklabels())[0] == "BRA (43)"


def test_championship_progress_ends_at_totals(season):
    ax = championship_progress(season, top_n=2)
    finals = {line.get_label().split()[0]: line.get_ydata()[-1] for line in ax.get_lines()}
    assert finals == {"BRA": 43.0, "ALP": 40.0}
