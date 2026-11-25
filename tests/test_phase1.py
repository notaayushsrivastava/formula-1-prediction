"""Offline tests for Phase 1 (no network)."""
import pytest

from pitwall import Laps, get_session
from pitwall.cache import DiskCache
from pitwall.sources.base import parse_time_to_seconds


class FakeSource:
    """2 drivers, 5 laps. ALP pits on lap 3; BRA has a safety-car lap 3."""

    def get_schedule(self, year):
        return [{"Round": 1, "EventName": "Test Grand Prix", "Date": "2024-01-01",
                 "CircuitId": "test", "Circuit": "Test Ring", "Locality": "Nowhere",
                 "Country": "Testland", "Lat": 0.0, "Long": 0.0}]

    def get_results(self, year, rnd):
        base = {"Number": "1", "Status": "Finished", "Points": 0.0, "LapsCompleted": 5}
        return [
            {**base, "DriverId": "alpha", "Driver": "ALP", "FullName": "A Alpha",
             "TeamId": "red", "Team": "Red", "GridPosition": 2, "Position": 1,
             "ClassifiedPosition": "1"},
            {**base, "DriverId": "bravo", "Driver": "BRA", "FullName": "B Bravo",
             "TeamId": "blue", "Team": "Blue", "GridPosition": 1, "Position": None,
             "ClassifiedPosition": "R", "Status": "Engine"},
        ]

    def get_laps(self, year, rnd):
        times = {"alpha": [95.0, 90.0, 110.0, 112.0, 90.5],
                 "bravo": [96.0, 91.0, 130.0, 91.2, 91.4]}
        return [{"DriverId": d, "LapNumber": i + 1, "Position": 1, "LapTimeSeconds": t}
                for d, ts in times.items() for i, t in enumerate(ts)]

    def get_pitstops(self, year, rnd):
        return [{"DriverId": "alpha", "LapNumber": 3, "Stop": 1, "DurationSeconds": 22.0}]


@pytest.fixture
def session():
    return get_session(2024, "test ring", source=FakeSource()).load()


def test_parse_time_to_seconds():
    assert parse_time_to_seconds("1:23.456") == pytest.approx(83.456)
    assert parse_time_to_seconds("23.4") == pytest.approx(23.4)
    assert parse_time_to_seconds("bad") is None
    assert parse_time_to_seconds(None) is None


def test_laps_type_and_chaining(session):
    picked = session.laps.pick_drivers("alp")
    assert isinstance(picked, Laps)
    assert len(picked) == 5


def test_stints_from_pitstops(session):
    stints = session.laps.pick_drivers("ALP").stints()
    assert stints[["StartLap", "EndLap"]].values.tolist() == [[1, 3], [4, 5]]


def test_pick_wo_box_and_quicklaps(session):
    assert len(session.laps.pick_drivers("ALP").pick_wo_box()) == 3
    assert len(session.laps.pick_quicklaps()) == 7  # drops 110, 112 and the SC lap 130


def test_pick_fastest(session):
    fastest = session.laps.pick_fastest()
    assert fastest["Driver"] == "ALP" and fastest["LapNumber"] == 2


def test_retired_driver_not_classified(session):
    bra = session.get_driver("BRA")
    assert not bra["Classified"] and bra["Status"] == "Engine"


def test_cache_roundtrip(tmp_path):
    cache = DiskCache(tmp_path)
    cache.set("k", {"a": 1})
    assert cache.get("k") == {"a": 1}
    assert cache.get("k", max_age=-1) is None  # expired
    assert cache.get("missing") is None
