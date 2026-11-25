"""Real-API smoke tests. Run with: pytest --run-network"""
import os

import pytest

import pitwall
from pitwall.auth import TOKEN_ENV_VAR
from pitwall.cache import DiskCache
from pitwall.sources import F1dbLoader, OpenF1Client, OpenMeteoClient

pytestmark = pytest.mark.network


def test_jolpica_real_race():
    session = pitwall.get_session(2023, 1).load()  # 2023 Bahrain Grand Prix
    assert session.results["Driver"].iloc[0] == "VER"
    assert len(session.laps) > 1000


def test_live_timing_real_archive():
    if not os.environ.get(TOKEN_ENV_VAR):
        pytest.skip(f"set {TOKEN_ENV_VAR} to your own F1 TV token to run this")
    session = pitwall.load_live_timing(pitwall.get_session(2024, "Monza").load())
    assert (session.laps["Compound"] != "UNKNOWN").mean() > 0.9


def test_f1db_latest_release(tmp_path):
    races = F1dbLoader(directory=tmp_path).get_races(min_year=2024)
    assert any(r["Year"] == 2024 for r in races)


def test_openf1_historical_sessions(tmp_path):
    sessions = OpenF1Client(cache=DiskCache(tmp_path)).get_sessions(2024)
    assert any(s["Name"] == "Race" for s in sessions)


def test_open_meteo_archived_forecast(tmp_path):
    rows = OpenMeteoClient(cache=DiskCache(tmp_path)).get_race_forecast(26.03, 50.51, "2024-03-02")
    assert rows and {r["LeadDays"] for r in rows} == {1, 2, 3}
