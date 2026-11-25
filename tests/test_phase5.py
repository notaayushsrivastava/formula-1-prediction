"""Offline tests for Phase 5: telemetry feeds, lap slicing, distance, delta and plots."""
import matplotlib

matplotlib.use("Agg")  # draw off-screen; must come before pyplot is imported

import base64  # noqa: E402
import datetime as dt  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import zlib  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

from pitwall import (  # noqa: E402
    Telemetry, delta_time, get_session, lap_telemetry, load_telemetry, timing_lap,
)
from pitwall.plotting import gear_map, speed_trace  # noqa: E402
from pitwall.sources.telemetry_feeds import parse_z_stream  # noqa: E402

EPOCH = dt.datetime(2024, 9, 1, 14, 0, tzinfo=dt.timezone.utc)
LAP_METRES = 1000.0
SPEEDS = {"16": [40.0, 50.0, 45.0], "44": [35.0, 40.0, 38.0]}  # m/s for laps 1-3
STEP = 0.25  # seconds between car samples


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


# ---------- fake feed builders ----------
def _clock(seconds):
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{secs:06.3f}"


def _utc(t):
    return (EPOCH + dt.timedelta(seconds=t)).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def _z(payload):
    """Compress like F1: raw DEFLATE, then base64, then a JSON string."""
    packer = zlib.compressobj(wbits=-zlib.MAX_WBITS)
    raw = packer.compress(json.dumps(payload).encode()) + packer.flush()
    return json.dumps(base64.b64encode(raw).decode())


def _state(number, t):
    """(lap index, speed m/s, metres into the lap) at session time t."""
    start = 0.0
    speeds = SPEEDS[number]
    for i, v in enumerate(speeds):
        end = start + LAP_METRES / v
        if t < end or i == len(speeds) - 1:
            return i, v, (t - start) * v
        start = end


def _lap_ends(number):
    ends, t = [], 0.0
    for v in SPEEDS[number]:
        t += LAP_METRES / v
        ends.append(t)
    return ends


def _car_and_position_text():
    car_lines, pos_lines = [], []
    for second in range(1, 81):  # one line per second, 4 samples each
        samples = [second - 1 + STEP * k for k in range(1, 5)]
        car_entries, pos_entries = [], []
        for t in samples:
            cars, spots = {}, {}
            for number in SPEEDS:
                _, v, metres = _state(number, t)
                cars[number] = {"Channels": {"0": 11000, "2": v * 3.6,
                                             "3": 8 if v >= 45 else 7, "4": 100, "5": 0}}
                angle = 2 * math.pi * metres / LAP_METRES
                radius = LAP_METRES / (2 * math.pi) * 10  # decimetres
                spots[number] = {"Status": "OnTrack", "X": radius * math.cos(angle),
                                 "Y": radius * math.sin(angle), "Z": 0}
            car_entries.append({"Utc": _utc(t), "Cars": cars})
            pos_entries.append({"Timestamp": _utc(t), "Entries": spots})
        car_lines.append(_clock(second) + _z({"Entries": car_entries}))
        pos_lines.append(_clock(second) + _z({"Position": pos_entries}))
    return "\ufeff" + "\r\n".join(car_lines), "\ufeff" + "\r\n".join(pos_lines)


def _timing_entries():
    entries = []
    for number in SPEEDS:
        for lap, end in enumerate(_lap_ends(number), start=1):
            value = "" if lap == 1 else f"{LAP_METRES / SPEEDS[number][lap - 1]:.3f}"  # lap 1 = out-lap
            entries.append((round(end, 3), {"Lines": {number: {
                "NumberOfLaps": lap, "LastLapTime": {"Value": value}}}}))
    return sorted(entries, key=lambda e: e[0])


CAR_TEXT, POSITION_TEXT = _car_and_position_text()


class FakeLiveClient:
    def session_path(self, year, event_date, kind):
        return "2024/fake/quali/"

    def stream(self, path, topic):
        assert topic == "TimingData"
        return _timing_entries()

    def z_stream(self, path, topic):
        return parse_z_stream({"CarData.z": CAR_TEXT, "Position.z": POSITION_TEXT}[topic])


class FakeQualiSource:
    def get_schedule(self, year):
        return [{"Round": 1, "EventName": "Test Grand Prix", "Date": "2024-09-01",
                 "CircuitId": "t", "Circuit": "Test Ring", "Locality": "", "Country": "",
                 "Lat": 0.0, "Long": 0.0, "HasSprint": False}]

    def get_qualifying(self, year, rnd):
        def row(code, number, pos, best):
            return {"Round": 1, "DriverId": code.lower(), "Driver": code, "FullName": code,
                    "TeamId": "red" if code == "ALP" else "blue", "Team": "T", "Number": number,
                    "Position": pos, "Q1Seconds": best, "Q2Seconds": None, "Q3Seconds": None}
        return [row("ALP", "16", 1, 20.0), row("BRA", "44", 2, 25.0)]


@pytest.fixture(scope="module")
def quali():
    session = get_session(2024, 1, "Q", source=FakeQualiSource()).load()
    return load_telemetry(session, client=FakeLiveClient())


# ---------- tests ----------
def test_z_stream_roundtrip():
    line = _clock(62.345) + _z({"hello": [1, 2]})
    assert parse_z_stream("\ufeff" + line) == [(62.345, {"hello": [1, 2]})]


def test_timing_laps_and_out_lap(quali):
    alp = quali.timing_laps[quali.timing_laps["Driver"] == "ALP"]
    assert alp["LapNumber"].tolist() == [1, 2, 3]
    assert pd.isna(alp["LapTimeSeconds"].iloc[0])  # out-lap has no time
    assert alp["LapStartSeconds"].tolist()[1:] == pytest.approx([25.0, 45.0])


def test_fastest_lap_matches_official_time(quali):
    assert timing_lap(quali, "ALP")["LapNumber"] == 2
    assert timing_lap(quali, "BRA", lap=3)["LapTimeSeconds"] == pytest.approx(26.316)


def test_lap_telemetry_speed_distance_and_position(quali):
    tel = lap_telemetry(quali, "ALP")
    assert isinstance(tel, Telemetry)
    assert tel["Speed"].median() == pytest.approx(180.0)  # 50 m/s
    assert tel["Distance"].iloc[-1] == pytest.approx(LAP_METRES, rel=0.01)
    assert tel[["X", "Y"]].notna().all().all()
    assert tel["Brake"].dtype == bool and "DRS" not in tel.columns


def test_delta_time_reaches_lap_time_gap(quali):
    delta = delta_time(lap_telemetry(quali, "ALP"), lap_telemetry(quali, "BRA"))
    assert delta["Delta"].iloc[-1] == pytest.approx(5.0, abs=0.1)  # 25 s vs 20 s


def test_telemetry_needs_loading_first():
    session = get_session(2024, 1, "Q", source=FakeQualiSource()).load()
    with pytest.raises(ValueError, match="load_telemetry"):
        lap_telemetry(session, "ALP")


def test_plots(quali):
    ax = speed_trace(quali, ["ALP", "BRA"])
    assert [line.get_label() for line in ax.get_lines()] == ["ALP lap 2", "BRA lap 2"]
    ax = gear_map(quali, "ALP")
    assert isinstance(ax.collections[0], LineCollection)
