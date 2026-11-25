"""Offline tests for Phase 2c: points systems and the title-fight calculator."""
import matplotlib

matplotlib.use("Agg")  # draw off-screen; must come before pyplot is imported

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from pitwall import (  # noqa: E402
    Season, points_still_available, points_system, title_fight_summary, who_can_still_win,
)
from pitwall.plotting import title_fight_chart  # noqa: E402

DRIVERS = [("AAA", "red"), ("BBB", "blue"), ("CCC", "red"), ("DDD", "blue")]


def _result(rnd, code, team, pos):
    return {"Round": rnd, "DriverId": code.lower(), "Driver": code, "FullName": code,
            "TeamId": team, "Team": team.title(), "Number": "0", "GridPosition": pos,
            "Position": pos, "ClassifiedPosition": str(pos), "Status": "Finished",
            "Points": 0.0, "LapsCompleted": 50, "FastestLapSeconds": None}


class FakeTitleSource:
    """2025 rules (no fastest-lap point), 4 rounds; rounds 3 and 4 are sprint weekends.

    Round 3's sprint has been run, but its race hasn't (as on a Saturday night).
    """

    def __init__(self, points=(100.0, 60.0, 42.0, 30.0), done_rounds=(1, 2)):
        self.points = points
        self.done_rounds = done_rounds
        self.standings_round_asked = "never"

    def get_schedule(self, year):
        return [{"Round": r, "EventName": f"Round {r} Grand Prix", "Date": "2025-01-01",
                 "CircuitId": f"c{r}", "Circuit": f"C{r}", "Locality": "", "Country": "",
                 "Lat": 0.0, "Long": 0.0, "HasSprint": r >= 3} for r in (1, 2, 3, 4)]

    def get_season_results(self, year, sprint=False):
        rounds = [3] if sprint else self.done_rounds
        return [_result(r, code, team, pos)
                for r in rounds for pos, (code, team) in enumerate(DRIVERS, start=1)]

    def get_driver_standings(self, year, rnd=None):
        self.standings_round_asked = rnd
        at = rnd or max(self.done_rounds)
        return [{"Round": at, "DriverId": code.lower(), "Driver": code, "FullName": code,
                 "TeamId": team, "Team": team.title(), "Position": pos, "Points": pts,
                 "Wins": 0}
                for pos, ((code, team), pts) in enumerate(zip(DRIVERS, self.points), start=1)]

    def get_constructor_standings(self, year, rnd=None):
        at = rnd or max(self.done_rounds)
        red = self.points[0] + self.points[2]
        blue = self.points[1] + self.points[3]
        return [{"Round": at, "TeamId": "red", "Team": "Red", "Position": 1, "Points": red, "Wins": 0},
                {"Round": at, "TeamId": "blue", "Team": "Blue", "Position": 2, "Points": blue, "Wins": 0}]


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _season(**kwargs):
    return Season(2025, source=FakeTitleSource(**kwargs)).load()


def test_points_systems():
    assert points_system(2024).race_max() == 26          # fastest-lap point era
    assert points_system(2025).race_max() == 25          # bonus scrapped
    assert points_system(2025).race_max(cars=2) == 43    # constructors: 25 + 18
    assert points_system(2025).sprint_max(cars=2) == 15  # 8 + 7
    assert points_system(2021).sprint_max() == 3         # 2021 sprints paid 3-2-1
    assert points_system(2014).race_max(last_round=True) == 50
    with pytest.raises(ValueError):
        points_system(2009)


def test_points_left_skips_sprint_already_run():
    left = points_still_available(_season())
    # races in rounds 3 and 4 (2 x 25) + round 4 sprint (8); round 3 sprint is done
    assert (left.races, left.sprints, left.max_points) == (2, 1, 58.0)


def test_every_status_once():
    table = who_can_still_win(_season())
    # gaps to 100: 40 < 58 can win, 58 == 58 tie only, 70 > 58 out
    assert table["Status"].tolist() == ["Leader", "Can win", "Tie only", "Out"]
    assert table["MaxPoints"].tolist() == [158.0, 118.0, 100.0, 88.0]


def test_clinched_when_every_rival_is_out():
    table = who_can_still_win(_season(points=(100.0, 40.0, 30.0, 20.0)))  # gap 60 > 58
    assert table["Status"].tolist() == ["Champion", "Out", "Out", "Out"]


def test_season_over():
    season = _season(done_rounds=(1, 2, 3, 4))
    assert points_still_available(season).races == 0
    assert who_can_still_win(season)["Status"].iloc[0] == "Champion"


def test_constructors_use_two_cars():
    season = _season()
    assert points_still_available(season, cars=2).max_points == 2 * 43 + 15
    table = who_can_still_win(season, constructors=True)
    assert table["Team"].tolist() == ["Red", "Blue"]
    assert table["Status"].tolist() == ["Leader", "Can win"]  # gap 52 < 101


def test_after_round_rewinds_the_season():
    source = FakeTitleSource()
    season = Season(2025, source=source, after_round=1).load()
    assert source.standings_round_asked == 1
    assert season.completed_rounds == [1]
    left = points_still_available(season)
    assert (left.races, left.sprints) == (3, 2)  # rounds 2-4; both sprints not yet run


def test_summary_text():
    text = title_fight_summary(_season())
    assert text.startswith("2 races and 1 sprint left (up to 58 points)")
    assert "still in it: BBB, CCC" in text


def test_title_fight_chart():
    ax = title_fight_chart(_season())
    assert len(ax.patches) == 8  # a solid and a faded bar per driver
    assert ax.get_yticklabels()[0].get_text() == "AAA · Leader"
