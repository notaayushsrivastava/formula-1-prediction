"""Offline tests for Phase 4: auth, the Live Timing client and data merging."""
import matplotlib

matplotlib.use("Agg")  # draw off-screen; must come before pyplot is imported

import base64  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from pitwall import (  # noqa: E402
    AccessBlockedError, AuthenticationError, DataNotAvailableError, Laps, get_session,
    load_live_timing,
)
from pitwall.auth import TOKEN_ENV_VAR, get_token, token_status  # noqa: E402
from pitwall.cache import DiskCache  # noqa: E402
from pitwall.plotting import driver_laptimes_scatter  # noqa: E402
from pitwall.sources.jolpica import RateLimiter  # noqa: E402
from pitwall.sources.livetiming import (  # noqa: E402
    LiveTimingClient, find_session_path, parse_json_stream, parse_stints, track_status_by_lap,
)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


# ---------- auth ----------
def _jwt(exp):
    def enc(data):
        return base64.urlsafe_b64encode(json.dumps(data).encode()).decode().rstrip("=")
    return f"{enc({'alg': 'none'})}.{enc({'exp': exp})}.signature"


def test_missing_token(monkeypatch):
    monkeypatch.delenv(TOKEN_ENV_VAR, raising=False)
    with pytest.raises(AuthenticationError):
        get_token()


def test_expired_token(monkeypatch):
    monkeypatch.setenv(TOKEN_ENV_VAR, _jwt(time.time() - 60))
    with pytest.raises(AuthenticationError, match="expired"):
        get_token()


def test_valid_token_bearer_prefix_and_status_hides_it(monkeypatch):
    token = _jwt(time.time() + 3600)
    monkeypatch.setenv(TOKEN_ENV_VAR, f"Bearer {token}")
    assert get_token() == token
    assert token not in token_status()


# ---------- HTTP client ----------
TOKEN = "secret-test-token"


class FakeResponse:
    def __init__(self, status, text=""):
        self.status_code = status
        self.ok = status < 400
        self.content = ("\ufeff" + text).encode("utf-8")
        self.url = "https://example.invalid"


def _client(tmp_path, responses):
    client = LiveTimingClient(cache=DiskCache(tmp_path), token=TOKEN,
                              rate_limiter=RateLimiter(per_second=1000))
    calls = []

    def fake_get(url, headers=None, timeout=None):
        calls.append((url, headers))
        return responses[url.rsplit("/", 1)[-1]]

    client.http.get = fake_get
    return client, calls


def test_token_is_sent_but_never_cached_or_shown(tmp_path):
    client, calls = _client(tmp_path, {
        "ArchiveStatus.json": FakeResponse(200, '{"Status": "Complete"}'),
        "TimingAppData.json": FakeResponse(200, '{"Lines": {}}'),
    })
    assert client.keyframe("2024/a/b/", "TimingAppData") == {"Lines": {}}
    assert all(headers["Authorization"] == f"Bearer {TOKEN}" for _, headers in calls)
    assert all(TOKEN not in f.read_text() for f in tmp_path.glob("*.json"))
    assert TOKEN not in repr(client)
    client.keyframe("2024/a/b/", "TimingAppData")
    assert len(calls) == 2  # second call: status remembered, file read from cache


@pytest.mark.parametrize("status, error", [(401, AuthenticationError), (403, AccessBlockedError)])
def test_refusals_are_reported_not_worked_around(tmp_path, status, error):
    client, calls = _client(tmp_path, {"ArchiveStatus.json": FakeResponse(status)})
    with pytest.raises(error) as info:
        client.keyframe("2024/a/b/", "TimingAppData")
    assert len(calls) == 1  # exactly one attempt
    assert TOKEN not in str(info.value)


def test_unfinished_archive_is_not_cached(tmp_path):
    client, _ = _client(tmp_path, {
        "ArchiveStatus.json": FakeResponse(200, '{"Status": "Generating"}')})
    with pytest.raises(DataNotAvailableError):
        client.keyframe("2024/a/b/", "TimingAppData")
    assert not list(tmp_path.glob("*.json"))


# ---------- parsing ----------
def _clock(seconds):
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{secs:06.3f}"


def _stream(*entries):
    return "\ufeff" + "\r\n".join(f"{_clock(t)}{json.dumps(msg)}" for t, msg in entries)


def test_parse_json_stream_handles_bom_and_times():
    entries = parse_json_stream(_stream((0, {"Status": "1"}), (250, {"Status": "4"})))
    assert entries == [(0.0, {"Status": "1"}), (250.0, {"Status": "4"})]


INDEX = {"Year": 2024, "Meetings": [{"Name": "Italian Grand Prix", "Sessions": [
    {"Name": "Qualifying", "StartDate": "2024-08-31T16:00:00",
     "Path": "2024/2024-09-01_Italian_Grand_Prix/2024-08-31_Qualifying/"},
    {"Name": "Race", "StartDate": "2024-09-01T15:00:00"},  # no Path: gets rebuilt
]}]}


def test_find_session_path():
    assert find_session_path(INDEX, "2024-09-01", "Qualifying").endswith("_Qualifying/")
    assert (find_session_path(INDEX, "2024-09-02", "Race")  # one day off still matches
            == "2024/2024-09-01_Italian_Grand_Prix/2024-09-01_Race/")
    with pytest.raises(DataNotAvailableError):
        find_session_path(INDEX, "2024-09-10", "Race")


APP_DATA = {"Lines": {
    "16": {"Stints": [{"Compound": "MEDIUM", "New": "true", "StartLaps": 0, "TotalLaps": 3},
                      {"Compound": "HARD", "New": "true", "StartLaps": 0, "TotalLaps": 3}]},
    # BRA: red-flag tyre change after lap 4, no pit stop; second set already used for 2 laps
    "44": {"Stints": {"0": {"Compound": "SOFT", "New": "true", "StartLaps": 0, "TotalLaps": 4},
                      "1": {"Compound": "MEDIUM", "New": "false", "StartLaps": 2,
                            "TotalLaps": 4}}},
}}


def test_parse_stints_list_and_dict_forms():
    stints = {(s["Number"], s["Stint"]): s for s in parse_stints(APP_DATA)}
    assert stints[("16", 2)]["Compound"] == "HARD"
    assert stints[("44", 2)]["StintLaps"] == 2 and not stints[("44", 2)]["New"]


def test_track_status_by_lap():
    starts = {1: 0.0, 2: 100.0, 3: 200.0}
    changes = [(0.0, "1"), (250.0, "4")]
    assert track_status_by_lap(starts, changes) == {1: "1", 2: "1", 3: "14"}


# ---------- end to end ----------
TRACK_STATUS_TEXT = _stream((0, {"Status": "1"}), (250, {"Status": "4"}),
                            (320, {"Status": "1"}), (395, {"Status": "5"}),
                            (405, {"Status": "1"}))
LAP_COUNT_TEXT = _stream(*[(100 * (lap - 1), {"CurrentLap": lap}) for lap in range(1, 7)])


class FakeLiveClient:
    def session_path(self, year, event_date, kind):
        return "2024/fake/race/"

    def keyframe(self, path, topic):
        return APP_DATA

    def stream(self, path, topic):
        return parse_json_stream({"TrackStatus": TRACK_STATUS_TEXT,
                                  "LapCount": LAP_COUNT_TEXT}[topic])


class FakeRaceSource:
    """ALP (#16) pits on lap 3. BRA (#44) never pits."""

    def get_schedule(self, year):
        return [{"Round": 1, "EventName": "Test Grand Prix", "Date": "2024-01-01",
                 "CircuitId": "t", "Circuit": "Test Ring", "Locality": "", "Country": "",
                 "Lat": 0.0, "Long": 0.0, "HasSprint": False}]

    def get_results(self, year, rnd):
        def row(code, number, pos):
            return {"Round": 1, "DriverId": code.lower(), "Driver": code, "FullName": code,
                    "Number": number, "TeamId": "t", "Team": "T", "GridPosition": pos,
                    "Position": pos, "ClassifiedPosition": str(pos), "Status": "Finished",
                    "Points": 0.0, "LapsCompleted": 6, "FastestLapSeconds": None}
        return [row("ALP", "16", 1), row("BRA", "44", 2)]

    def get_laps(self, year, rnd):
        return [{"DriverId": d, "LapNumber": lap, "Position": 1, "LapTimeSeconds": 90 + lap / 10}
                for d in ("alp", "bra") for lap in range(1, 7)]

    def get_pitstops(self, year, rnd):
        return [{"DriverId": "alp", "LapNumber": 3, "Stop": 1, "DurationSeconds": 22.0}]


@pytest.fixture
def session():
    race = get_session(2024, 1, source=FakeRaceSource()).load()
    return load_live_timing(race, client=FakeLiveClient())


def test_real_compounds_and_tyre_life(session):
    alp = session.laps.pick_drivers("ALP")
    assert alp["Compound"].tolist() == ["MEDIUM"] * 3 + ["HARD"] * 3
    assert alp["TyreLife"].tolist() == [1, 2, 3, 1, 2, 3]


def test_red_flag_tyre_change_is_a_new_stint(session):
    bra = session.laps.pick_drivers("BRA")
    assert bra["Stint"].tolist() == [1, 1, 1, 1, 2, 2]  # Jolpica alone would say all 1
    assert bra["TyreLife"].tolist() == [1, 2, 3, 4, 3, 4]  # used set starts at 3
    assert bra["FreshTyre"].tolist() == [True] * 4 + [False] * 2


def test_track_status_per_lap_and_filters(session):
    alp = session.laps.pick_drivers("ALP")
    assert alp["TrackStatus"].tolist() == ["1", "1", "14", "415", "51", "1"]
    assert len(session.laps.pick_green()) == 6  # laps 1, 2, 6 of both drivers
    assert len(session.laps.pick_track_status("4", how="contains")) == 4
    assert len(session.laps.pick_compounds("hard")) == 3
    assert "SafetyCar" in session.track_status["Meaning"].tolist()


def test_filters_need_live_timing():
    with pytest.raises(ValueError, match="load_live_timing"):
        Laps({"LapNumber": [1]}).pick_green()


def test_scatter_uses_real_compounds(session):
    ax = driver_laptimes_scatter(session, "ALP")
    assert ax.get_legend_handles_labels()[1] == ["Hard", "Medium"]
