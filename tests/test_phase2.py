"""Offline tests for Phase 2a (no network)."""
import pytest

from pitwall import DataNotAvailableError, Laps, Season, get_session
from pitwall.sources.jolpica import _driver_standing_record, _qualifying_record


def _result(rnd, code, team, pos, points):
    return {"Round": rnd, "DriverId": code.lower(), "Driver": code, "FullName": code,
            "TeamId": team, "Team": team.title(), "Number": "0", "GridPosition": pos,
            "Position": pos, "ClassifiedPosition": str(pos), "Status": "Finished",
            "Points": points, "LapsCompleted": 50, "FastestLapSeconds": None}


def _quali(code, team, pos, q1, q2=None, q3=None):
    return {"Round": 1, "DriverId": code.lower(), "Driver": code, "FullName": code,
            "TeamId": team, "Team": team.title(), "Number": "0", "Position": pos,
            "Q1Seconds": q1, "Q2Seconds": q2, "Q3Seconds": q3}


class FakeSeasonSource:
    """3 rounds: 1 and 2 are done (2 had a sprint), 3 is still to come."""

    def get_schedule(self, year):
        return [{"Round": r, "EventName": f"Round {r} GP", "Date": "2024-01-01",
                 "CircuitId": f"c{r}", "Circuit": f"Circuit {r}", "Locality": "",
                 "Country": "", "Lat": 0.0, "Long": 0.0, "HasSprint": r == 2}
                for r in (1, 2, 3)]

    def get_season_results(self, year, sprint=False):
        if sprint:
            return [_result(2, "BRA", "blue", 1, 8.0), _result(2, "ALP", "red", 2, 7.0)]
        return [_result(1, "ALP", "red", 1, 25.0), _result(1, "BRA", "blue", 2, 18.0),
                _result(2, "BRA", "blue", 1, 25.0), _result(2, "ALP", "red", 2, 18.0)]

    def get_results(self, year, rnd):
        return [r for r in self.get_season_results(year) if r["Round"] == rnd]

    def get_sprint_results(self, year, rnd):
        return [r for r in self.get_season_results(year, sprint=True) if r["Round"] == rnd]

    def get_qualifying(self, year, rnd):
        # BRA's Q2 lap (80.3) beat ALP's pole lap in Q3 (80.5), but grid order uses Q3.
        return [_quali("ALP", "red", 1, 81.0, 80.6, 80.5),
                _quali("BRA", "blue", 2, 81.2, 80.3, 80.9),
                _quali("CHA", "green", 16, 81.5)]  # knocked out in Q1

    def get_driver_standings(self, year, rnd=None):
        base = {"Round": 2, "FullName": "", "Wins": 1}
        return [{**base, "DriverId": "bra", "Driver": "BRA", "TeamId": "blue",
                 "Team": "Blue", "Position": 1, "Points": 51.0},
                {**base, "DriverId": "alp", "Driver": "ALP", "TeamId": "red",
                 "Team": "Red", "Position": 2, "Points": 50.0}]

    def get_constructor_standings(self, year, rnd=None):
        return [{"Round": 2, "TeamId": "blue", "Team": "Blue", "Position": 1, "Points": 51.0, "Wins": 1},
                {"Round": 2, "TeamId": "red", "Team": "Red", "Position": 2, "Points": 50.0, "Wins": 1}]


@pytest.fixture
def season():
    return Season(2024, source=FakeSeasonSource()).load()


def test_completed_and_remaining_rounds(season):
    assert season.completed_rounds == [1, 2]
    assert season.remaining_events["Round"].tolist() == [3]


def test_points_by_round_includes_sprints(season):
    table = season.points_by_round()
    assert table.index.tolist() == ["BRA", "ALP"]
    assert table.loc["ALP", 2] == 25.0  # 18 race + 7 sprint
    assert season.points_by_round(include_sprints=False).loc["ALP", "Total"] == 43.0


def test_standings_agree_with_points_table(season):
    table = season.points_by_round()
    for row in season.driver_standings.itertuples():
        assert table.loc[row.Driver, "Total"] == row.Points


def test_qualifying_keeps_official_order():
    q = get_session(2024, 1, "Q", source=FakeSeasonSource()).load()
    assert q.name == "2024 Round 1 GP Qualifying"
    assert q.results["Driver"].tolist() == ["ALP", "BRA", "CHA"]
    assert q.get_driver("BRA")["BestSeconds"] == pytest.approx(80.3)
    assert q.get_driver("BRA")["DeltaToFastestSeconds"] == pytest.approx(0.0)
    assert q.get_driver("ALP")["DeltaToFastestSeconds"] == pytest.approx(0.2)
    assert q.laps.empty


def test_sprint_session():
    s = get_session(2024, 2, "S", source=FakeSeasonSource()).load()
    assert s.results["Driver"].iloc[0] == "BRA"
    with pytest.raises(DataNotAvailableError):
        get_session(2024, 1, "S", source=FakeSeasonSource()).load()


def test_bad_session_kind():
    with pytest.raises(ValueError):
        get_session(2024, 1, "FP1", source=FakeSeasonSource())


def test_pick_laps_and_teams():
    laps = Laps({"LapNumber": [1, 2, 3, 4, 5],
                 "Team": ["Red Bull"] * 2 + ["Ferrari"] * 3,
                 "TeamId": ["red_bull"] * 2 + ["ferrari"] * 3})
    assert laps.pick_laps(1)["LapNumber"].tolist() == [1]
    assert laps.pick_laps(range(2, 4), [5])["LapNumber"].tolist() == [2, 3, 5]
    assert len(laps.pick_teams("red bull")) == 2
    assert len(laps.pick_teams("FERRARI", "red_bull")) == 5


DRIVER_JSON = {"driverId": "leclerc", "code": "LEC",
               "givenName": "Charles", "familyName": "Leclerc"}


def test_qualifying_record_flattening():
    row = {"number": "16", "position": "1", "Q1": "1:20.000", "Q2": "1:19.500",
           "Driver": DRIVER_JSON,
           "Constructor": {"constructorId": "ferrari", "name": "Ferrari"}}
    rec = _qualifying_record({"round": "16"}, row)
    assert rec["Round"] == 16 and rec["Driver"] == "LEC" and rec["TeamId"] == "ferrari"
    assert rec["Q2Seconds"] == pytest.approx(79.5) and rec["Q3Seconds"] is None


def test_driver_standing_uses_last_team():
    row = {"position": "3", "points": "40", "wins": "1", "Driver": DRIVER_JSON,
           "Constructors": [{"constructorId": "a", "name": "A"},
                            {"constructorId": "b", "name": "B"}]}
    rec = _driver_standing_record({"round": "5"}, row)
    assert rec["TeamId"] == "b" and rec["Points"] == 40.0 and rec["Wins"] == 1
