"""Offline tests for Phase 7: history, features, weekend data and the dataset (no network)."""
import numpy as np
import pandas as pd
import pytest

from pitwall.predict import FEATURES, build_dataset, feature_columns, load_dataset, save_dataset
from pitwall.predict._optional import require
from pitwall.predict.circuits import circuit_cluster
from pitwall.predict.history import load_history
from pitwall.predict.weekend import long_run_pace

# RaceId, Year, Round, Date, CircuitId, CircuitType, HasSprint
RACES = [
    (1, 2023, 1, "2023-03-05", "alpha", "RACE", False),
    (2, 2023, 2, "2023-03-19", "beta", "STREET", False),
    (3, 2024, 1, "2024-03-03", "alpha", "RACE", False),
    (4, 2024, 2, "2024-03-17", "beta", "STREET", True),
]
TEAMS = {"aaa": "red", "bbb": "red", "ccc": "blue", "ddd": "blue"}
NUMBERS = {"aaa": 1, "bbb": 2, "ccc": 3, "ddd": 4}
# (driver, finish or None, grid or None for a pit-lane start, reason retired)
RESULTS = {
    1: [("aaa", 1, 2, None), ("bbb", 2, 1, None), ("ccc", 3, 3, None), ("ddd", None, 4, "Engine")],
    2: [("bbb", 1, 1, None), ("aaa", 2, 2, None), ("ddd", 3, 4, None), ("ccc", None, 3, "Collision")],
    3: [("aaa", 1, 1, None), ("ccc", 2, 2, None), ("bbb", 3, 3, None), ("ddd", 4, None, None)],
    4: [("ccc", 1, 1, None), ("aaa", 2, 2, None), ("bbb", 3, 3, None), ("ddd", 4, 4, None)],
}
POINTS = {1: 25.0, 2: 18.0, 3: 15.0, 4: 12.0}
WINNER_MS = {"alpha": 4_860_000, "beta": 6_000_000}  # 300 km -> 222.2 and 180 km/h
RACE_3_QUALI = {"aaa": (80.6, 80.5, 80.0), "ccc": (80.7, 80.6, 80.2),
                "bbb": (80.9, 81.0, None), "ddd": (81.5, None, None)}


class FakeHistory:
    """f1db-like history of 4 races; `upto` hides later races (for the time-travel test)."""

    def __init__(self, upto=4):
        self.upto = upto

    def _races(self):
        return [r for r in RACES if r[0] <= self.upto]

    def get_races(self, min_year):
        return [{"RaceId": rid, "Year": year, "Round": rnd, "Date": date, "Time": "14:00",
                 "EventName": f"{circuit.title()} GP", "CircuitId": circuit,
                 "CircuitType": ctype, "DistanceKm": 300.0, "Lat": 10.0, "Long": 20.0,
                 "HasSprint": sprint}
                for rid, year, rnd, date, circuit, ctype, sprint in self._races()
                if year >= min_year]

    def get_race_results(self, min_year):
        rows = []
        for rid, _, _, _, circuit, _, _ in self._races():
            for order, (drv, pos, grid, reason) in enumerate(RESULTS[rid], start=1):
                rows.append({"RaceId": rid, "DriverId": drv, "TeamId": TEAMS[drv],
                             "Number": NUMBERS[drv], "Position": pos,
                             "PositionText": str(pos) if pos else "DNF",
                             "DisplayOrder": order, "GridPosition": grid,
                             "GridText": str(grid) if grid else "PL",
                             "Laps": 50 if pos else 10,
                             "TimeMillis": WINNER_MS[circuit] if pos == 1 else None,
                             "ReasonRetired": reason, "Points": POINTS.get(pos, 0.0)})
        return rows

    def get_qualifying(self, min_year):
        rows = []
        for rid, *_ in self._races():
            for drv, _, grid, _ in RESULTS[rid]:
                q = RACE_3_QUALI[drv] if rid == 3 else (80.0 + 0.1 * (grid or 5), None, None)
                rows.append({"RaceId": rid, "DriverId": drv, "QualiPosition": grid,
                             "Q1Seconds": q[0], "Q2Seconds": q[1], "Q3Seconds": q[2]})
        return rows

    def get_pitstops(self, min_year):
        return [{"RaceId": rid, "DriverId": drv, "TeamId": TEAMS[drv], "Stop": 1, "Lap": 20,
                 "DurationSeconds": 22.0 if TEAMS[drv] == "red" else 23.0}
                for rid, *_ in self._races() for drv, *_ in RESULTS[rid]]

    def get_driver_standings(self, min_year):
        totals, rows = {}, []
        for rid, year, *_ in self._races():
            for drv, pos, _, _ in RESULTS[rid]:
                totals[(year, drv)] = totals.get((year, drv), 0.0) + POINTS.get(pos, 0.0)
                rows.append({"RaceId": rid, "DriverId": drv, "Points": totals[(year, drv)]})
        return rows

    def get_drivers(self):
        return [{"DriverId": d, "Driver": d.upper(), "FullName": d, "DateOfBirth": "2000-01-01"}
                for d in TEAMS]


class FakeWeekend:
    """OpenF1-like: long runs only in race 3's practice, inters in race 3's qualifying,
    and a safety car only in race 1. Session keys: race id x 10 + 1 (FP1), 5 (Q), 9 (R)."""

    def get_sessions(self, year):
        out = []
        for rid, y, _, date, *_ in RACES:
            if y != year:
                continue
            day = pd.Timestamp(date)
            for offset, name, n in ((-2, "Practice 1", 1), (-1, "Qualifying", 5), (0, "Race", 9)):
                start = (day + pd.Timedelta(days=offset, hours=14)).isoformat()
                out.append({"SessionKey": rid * 10 + n, "MeetingKey": rid, "Name": name,
                            "Type": name, "DateStart": start, "DateEnd": start, "Year": y,
                            "Cancelled": False})
        return out

    def get_laps(self, session):
        if session["SessionKey"] != 31:
            return []
        rows = []
        for number, times in ((1, [90.0] * 6), (3, [90.9] * 6), (2, [89.0] * 2)):
            rows.append({"Number": number, "LapNumber": 1, "LapSeconds": None, "PitOutLap": True})
            rows += [{"Number": number, "LapNumber": i, "LapSeconds": t, "PitOutLap": False}
                     for i, t in enumerate(times, start=2)]
        return rows

    def get_stints(self, session):
        compound = "INTERMEDIATE" if session["SessionKey"] == 35 else "SOFT"
        return [{"Number": 1, "Stint": 1, "Compound": compound, "LapStart": 1, "LapEnd": 10,
                 "TyreAgeAtStart": 0}]

    def get_weather(self, session):
        temp = 40.0 if session["Name"].startswith("Practice") else 35.0
        return [{"Time": session["DateStart"], "AirTemp": 25.0, "TrackTemp": temp, "Rainfall": 0}]

    def get_race_control(self, session):
        if session["SessionKey"] == 19:
            return [{"Time": "", "Lap": 3, "Category": "SafetyCar", "Flag": None,
                     "Message": "SAFETY CAR DEPLOYED"}]
        return [{"Time": "", "Lap": 1, "Category": "Flag", "Flag": "GREEN",
                 "Message": "GREEN LIGHT - PIT EXIT OPEN"}]

    def get_pit(self, session):
        return []


class FakeWeather:
    """Forecasts only from 2024 (like Open-Meteo). The 13:00 row is before the start."""

    def get_race_forecast(self, lat, lon, date):
        if date < "2024-01-01":
            return []
        rows = []
        for lead in (1, 2, 3):
            rows.append({"Time": f"{date}T13:00", "LeadDays": lead, "AirTemp": 0.0, "RainMm": 99.0})
            rows += [{"Time": f"{date}T{h}:00", "LeadDays": lead, "AirTemp": 20.0 + lead,
                      "RainMm": 0.5 * lead} for h in (14, 15)]
        return rows

    def get_observed(self, lat, lon, date):
        return [{"Time": f"{date}T{h}:00", "AirTemp": 20.0, "RainMm": rain}
                for h, rain in ((14, 0.2), (15, 0.2), (16, 5.0))]


@pytest.fixture(scope="module")
def data():
    return build_dataset(FakeHistory(), FakeWeekend(), FakeWeather(),
                         first_year=2023, history_from=2023)


def _row(data, race, driver, moment="quali"):
    hit = data[(data["RaceId"] == race) & (data["Driver"] == driver) & (data["Moment"] == moment)]
    assert len(hit) == 1
    return hit.iloc[0]


def test_labels_and_mechanical_dnfs():
    res = load_history(FakeHistory(), 2023).results.set_index(["RaceId", "DriverId"])
    assert res.loc[(1, "ddd"), "DNF"] == 1 and res.loc[(1, "ddd"), "MechDNF"] == 1  # engine
    assert res.loc[(2, "ccc"), "DNF"] == 1 and res.loc[(2, "ccc"), "MechDNF"] == 0  # collision
    assert res.loc[(1, "ddd"), "FinishForAverage"] == 4.0  # a DNF counts as last
    assert res.loc[(1, "aaa"), ["Win", "Podium", "PointsFinish"]].tolist() == [1, 1, 1]
    assert res.loc[(3, "ddd"), "Grid"] == 4 and res.loc[(3, "ddd"), "PitLaneStart"] == 1


def test_form_features_only_use_earlier_races(data):
    first = _row(data, 1, "AAA")
    assert first["driver_avg_finish_5"] == 15.0 and first["experience"] == 0  # rookie fill
    assert _row(data, 2, "AAA")["driver_avg_finish_5"] == 1.0
    assert _row(data, 2, "AAA")["driver_points_5"] == 25.0
    assert _row(data, 2, "DDD")["driver_dnf_rate_10"] == 1.0
    assert _row(data, 2, "CCC")["team_mech_dnf_rate_10"] == 0.5
    assert _row(data, 2, "BBB")["team_avg_points_5"] == 43.0
    assert _row(data, 2, "AAA")["team_pit_delta_10"] == pytest.approx(-0.5)


def test_championship_points_reset_each_season(data):
    assert _row(data, 2, "AAA")["champ_points_before"] == 25.0
    assert _row(data, 3, "AAA")["champ_points_before"] == 0.0
    assert _row(data, 4, "AAA")["champ_points_before"] == 25.0
    assert _row(data, 3, "AAA")["experience"] == 2


def test_circuit_features(data):
    r1, r2, r3 = (_row(data, race, "AAA") for race in (1, 2, 3))
    assert (r1["circuit_high_speed"], r3["circuit_high_speed"]) == (0, 1)  # needs a past race
    assert r2["circuit_street"] == 1
    assert r3["circuit_places_moved"] == pytest.approx(2 / 3)
    assert r2["circuit_places_moved"] == pytest.approx(2 / 3)  # new circuit: field-wide past
    assert circuit_cluster("RACE", float("nan")) == "mixed"
    assert circuit_cluster("RACE", 190.0) == "high_downforce"


def test_safety_car_rate_is_shrunk(data):
    first = _row(data, 1, "AAA")
    assert (first["circuit_sc_rate"], first["has_sc_history"]) == (0.0, 0)
    assert _row(data, 2, "AAA")["circuit_sc_rate"] == pytest.approx(1.0)
    assert _row(data, 3, "AAA")["circuit_sc_rate"] == pytest.approx(2 / 3)  # (1 + 2*0.5) / 3


def test_moments_hide_future_information(data):
    pre, practice, quali = (_row(data, 3, "DDD", m) for m in ("pre", "practice", "quali"))
    assert np.isnan(pre["grid"]) and np.isnan(pre["practice_pace_gap_pct"])
    assert np.isnan(practice["grid"]) and practice["track_temp"] == 40.0
    assert practice["wet_practice"] == 0
    assert quali["grid"] == 4 and quali["pit_lane_start"] == 1
    assert quali["track_temp"] == 35.0 and quali["wet_quali"] == 1
    assert set(FEATURES["pre"]) < set(FEATURES["practice"]) < set(FEATURES["quali"])


def test_practice_long_runs(data):
    assert _row(data, 3, "AAA", "practice")["practice_pace_gap_pct"] == pytest.approx(0.0)
    assert _row(data, 3, "CCC", "practice")["practice_pace_gap_pct"] == pytest.approx(1.0)
    bbb = _row(data, 3, "BBB", "practice")  # only a 2-lap run: no long-run pace
    assert bbb["has_practice_pace"] == 0
    assert bbb["practice_pace_gap_pct"] == pytest.approx(0.5)  # the field median
    assert long_run_pace(pd.DataFrame()).empty


def test_forecast_lead_matches_the_moment(data):
    assert _row(data, 3, "AAA", "pre")["rain_forecast_mm"] == pytest.approx(3.0)  # 2 h x 1.5
    assert _row(data, 3, "AAA", "quali")["air_temp_forecast"] == pytest.approx(21.0)
    assert _row(data, 1, "AAA", "pre")["has_forecast"] == 0  # nothing archived before 2024
    assert _row(data, 3, "AAA")["RaceRainMm"] == pytest.approx(0.4)


def test_quali_gaps(data):
    assert _row(data, 3, "AAA")["teammate_quali_gap_pct"] == pytest.approx(
        (80.5 / 81.0 - 1) * 100)  # compared in Q2: BBB has no Q3 time
    # Phase 14: measured inside BBB's deepest session (Q2: 81.0 vs the Q2 best 80.5)
    assert _row(data, 3, "BBB")["quali_gap_pct"] == pytest.approx((81.0 / 80.5 - 1) * 100)
    assert _row(data, 3, "AAA")["quali_gap_pct"] == pytest.approx(0.0)


def test_adding_a_later_race_changes_nothing_earlier():
    kwargs = {"first_year": 2023, "history_from": 2023}
    short = build_dataset(FakeHistory(upto=3), FakeWeekend(), FakeWeather(), **kwargs)
    full = build_dataset(FakeHistory(upto=4), FakeWeekend(), FakeWeather(), **kwargs)
    columns = ["Moment", "RaceId", "DriverId", *FEATURES["quali"]]

    def tidy(df):
        rows = df.loc[df["RaceId"] <= 3, columns]
        return rows.sort_values(["Moment", "RaceId", "DriverId"]).reset_index(drop=True)

    pd.testing.assert_frame_equal(tidy(short), tidy(full), check_dtype=False)


def test_history_only_dataset_flags_missing_sources():
    data = build_dataset(FakeHistory(), first_year=2023, history_from=2023)
    quali = data[data["Moment"] == "quali"]
    assert (quali["has_openf1"] == 0).all() and (quali["has_forecast"] == 0).all()
    assert (quali["practice_pace_gap_pct"] == 0).all() and quali["RaceRainMm"].isna().all()


def test_parquet_roundtrip(data, tmp_path):
    pytest.importorskip("pyarrow")
    path = save_dataset(data, tmp_path / "dataset.parquet")
    assert load_dataset(path).shape == data.shape


def test_helpful_errors():
    with pytest.raises(ImportError, match=r"pitwall-f1\[ml\]"):
        require("surely_not_an_installed_package", "Testing")
    with pytest.raises(ValueError, match="moment"):
        feature_columns("lunch")
