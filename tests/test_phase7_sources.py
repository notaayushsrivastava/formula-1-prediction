"""Offline tests for Phase 7 sources: HTTP helper, f1db, OpenF1, Open-Meteo (no network)."""
import datetime as dt
import io
import zipfile

import pytest

from pitwall.cache import DiskCache
from pitwall.errors import AccessBlockedError, DataNotAvailableError, SourceRequestError
from pitwall.sources import f1db, http_json
from pitwall.sources.f1db import F1dbLoader
from pitwall.sources.http_json import JsonHttp
from pitwall.sources.openf1 import OpenF1Client
from pitwall.sources.openmeteo import OpenMeteoClient


class NoLimit:
    """A rate limiter that never waits (tests must not sleep)."""

    def wait(self):
        pass


class FakeResponse:
    def __init__(self, status=200, payload=None, content=b""):
        self.status_code = status
        self.ok = status < 400
        self.url = "https://example.invalid"
        self.content = content
        self._payload = payload

    def json(self):
        return self._payload


def _wire(api, reply):
    """Replace a JsonHttp's requests session; reply(url, params) -> FakeResponse."""
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append((url, dict(params or {})))
        return reply(url, params)

    api.http.get = fake_get
    return calls


# ---------- shared HTTP helper ----------
def test_json_http_caches_but_never_caches_empty(tmp_path):
    api = JsonHttp("Test", NoLimit(), DiskCache(tmp_path))
    answers = iter([FakeResponse(payload=[]), FakeResponse(payload=[{"a": 1}])])
    calls = _wire(api, lambda url, params: next(answers))
    assert api.get("https://x.invalid/a", {"k": 1}) == []
    assert api.get("https://x.invalid/a", {"k": 1}) == [{"a": 1}]
    assert api.get("https://x.invalid/a", {"k": 1}) == [{"a": 1}]  # from the cache
    assert len(calls) == 2


def test_json_http_retries_temporary_errors(tmp_path, monkeypatch):
    slept = []
    monkeypatch.setattr(http_json.time, "sleep", slept.append)
    answers = iter([FakeResponse(503), FakeResponse(429), FakeResponse(payload={"ok": True})])
    api = JsonHttp("Test", NoLimit(), DiskCache(tmp_path))
    _wire(api, lambda url, params: next(answers))
    assert api.get("https://x.invalid/b", {}) == {"ok": True}
    assert slept == [2.0, 4.0]


@pytest.mark.parametrize("status, error", [(403, AccessBlockedError), (400, SourceRequestError)])
def test_json_http_refusals_fail_fast(tmp_path, status, error):
    api = JsonHttp("Test", NoLimit(), DiskCache(tmp_path))
    calls = _wire(api, lambda url, params: FakeResponse(status))
    with pytest.raises(error):
        api.get("https://x.invalid/c", {})
    assert len(calls) == 1  # never retried, never worked around


def test_json_http_404_means_no_data(tmp_path):
    api = JsonHttp("Test", NoLimit(), DiskCache(tmp_path))
    calls = _wire(api, lambda url, params: FakeResponse(404, {"detail": "No results found."}))
    assert api.get("https://x.invalid/d", {}) is None
    api.get("https://x.invalid/d", {})
    assert len(calls) == 2  # "no data" is not cached


# ---------- f1db ----------
F1DB_TABLES = {
    "races": ("id,year,round,date,time,grandPrixId,circuitId,circuitType,distance,sprintRaceDate\n"
              "1,2009,1,2009-03-29,06:00,australia,melbourne,STREET,307.6,\n"
              "2,2024,6,2024-05-05,20:00,miami,miami,STREET,308.3,2024-05-04\n"),
    "grands-prix": "id,name\naustralia,Australian Grand Prix\nmiami,Miami Grand Prix\n",
    "circuits": "id,latitude,longitude\nmelbourne,-37.8,144.9\nmiami,25.9,-80.2\n",
    "races-race-results": (
        "raceId,positionDisplayOrder,positionNumber,positionText,driverNumber,driverId,"
        "constructorId,laps,timeMillis,reasonRetired,points,gridPositionNumber,gridPositionText\n"
        "2,1,1,1,4,lando-norris,mclaren,57,5491000,,25,5,5\n"
        "2,2,,DNF,2,logan-sargeant,williams,27,,Collision,0,,PL\n"
        "1,1,1,1,22,jenson-button,brawn,58,5460000,,10,1,1\n"),
    "races-qualifying-results": ("raceId,positionNumber,driverId,timeMillis,q1Millis,q2Millis,q3Millis\n"
                                 "2,5,lando-norris,87000,88000,87500,87000\n"),
    "races-pit-stops": "raceId,driverId,constructorId,stop,lap,timeMillis\n2,lando-norris,mclaren,1,21,22100\n",
    "races-driver-standings": "raceId,driverId,points\n2,lando-norris,58\n",
    "drivers": "id,abbreviation,fullName,dateOfBirth\nlando-norris,NOR,Lando Norris,1999-11-13\n",
}


def _f1db_zip():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, text in F1DB_TABLES.items():
            archive.writestr(f"f1db-{name}.csv", text)
    return buffer.getvalue()


@pytest.fixture
def loader(tmp_path):
    (tmp_path / "f1db-csv-test.zip").write_bytes(_f1db_zip())
    return F1dbLoader(version="test", directory=tmp_path)


def test_f1db_records(loader):
    races = loader.get_races(min_year=2010)
    assert [r["RaceId"] for r in races] == [2]  # 2009 filtered out
    assert races[0]["EventName"] == "Miami Grand Prix" and races[0]["HasSprint"]
    results = {r["DriverId"]: r for r in loader.get_race_results(2010)}
    assert set(results) == {"lando-norris", "logan-sargeant"}
    assert results["logan-sargeant"]["Position"] is None
    assert results["logan-sargeant"]["GridPosition"] is None  # pit-lane start
    assert results["lando-norris"]["Points"] == 25
    assert loader.get_qualifying(2010)[0]["Q3Seconds"] == pytest.approx(87.0)
    assert loader.get_pitstops(2010)[0]["DurationSeconds"] == pytest.approx(22.1)
    assert loader.get_driver_standings(2010)[0]["Points"] == 58
    assert loader.get_drivers()[0]["Driver"] == "NOR"


def test_f1db_missing_table_is_explained(loader):
    with pytest.raises(DataNotAvailableError, match="f1db-drivers.csv"):
        loader.table("nope")


def test_f1db_downloads_a_pinned_release_once(tmp_path, monkeypatch):
    calls = []

    def fake_get(url, timeout=None, headers=None):
        calls.append(url)
        return FakeResponse(content=_f1db_zip())

    monkeypatch.setattr(f1db.requests, "get", fake_get)
    first = F1dbLoader(version="v2026.20.0", directory=tmp_path)
    assert first.url.endswith("/download/v2026.20.0/f1db-csv.zip")
    assert len(first.get_drivers()) == 1
    F1dbLoader(version="2026.20.0", directory=tmp_path).get_drivers()
    assert len(calls) == 1


# ---------- OpenF1 ----------
NOW = dt.datetime(2026, 11, 29, 18, 0, tzinfo=dt.timezone.utc)
OLD_SESSION = {"SessionKey": 9839, "Name": "Race",
               "DateStart": "2025-12-07T13:00:00+00:00", "DateEnd": "2025-12-07T15:00:00+00:00"}
LIVE_SESSION = {"SessionKey": 9999, "Name": "Race",
                "DateStart": "2026-11-29T16:00:00+00:00", "DateEnd": "2026-11-29T17:30:00+00:00"}


def _openf1(tmp_path, reply):
    client = OpenF1Client(cache=DiskCache(tmp_path), rate_limiter=NoLimit(), now=lambda: NOW)
    return client, _wire(client.http, reply)


def test_openf1_refuses_live_sessions_without_asking(tmp_path):
    client, calls = _openf1(tmp_path, lambda url, params: FakeResponse(payload=[]))
    with pytest.raises(DataNotAvailableError, match="historical"):
        client.get_laps(LIVE_SESSION)
    assert calls == []


def test_openf1_has_nothing_before_2023(tmp_path):
    client, calls = _openf1(tmp_path, lambda url, params: FakeResponse(payload=[]))
    assert client.get_sessions(2022) == [] and calls == []


def test_openf1_records(tmp_path):
    payloads = {
        "sessions": [{"session_key": 1, "meeting_key": 2, "session_name": "Practice 1",
                      "session_type": "Practice", "date_start": "2025-03-14T01:30:00+00:00",
                      "date_end": "2025-03-14T02:30:00+00:00", "year": 2025,
                      "is_cancelled": False}],
        "laps": [{"driver_number": 16, "lap_number": 3, "lap_duration": 91.7,
                  "is_pit_out_lap": None}],
        "pit": [{"driver_number": 16, "lap_number": 31, "pit_duration": 22.2,
                 "stop_duration": None}],
        "stints": [{"driver_number": 16, "stint_number": 1, "compound": "intermediate",
                    "lap_start": 1, "lap_end": 9, "tyre_age_at_start": 0}],
    }
    client, calls = _openf1(
        tmp_path, lambda url, params: FakeResponse(payload=payloads[url.rsplit("/", 1)[-1]]))
    assert client.get_sessions(2025)[0]["Name"] == "Practice 1"
    assert client.get_laps(OLD_SESSION) == [
        {"Number": 16, "LapNumber": 3, "LapSeconds": 91.7, "PitOutLap": False}]
    assert calls[1][1] == {"session_key": 9839}
    assert client.get_pit(OLD_SESSION)[0]["LaneSeconds"] == 22.2  # older field name works
    assert client.get_stints(OLD_SESSION)[0]["Compound"] == "INTERMEDIATE"


# ---------- Open-Meteo ----------
def _meteo(tmp_path, payload):
    client = OpenMeteoClient(cache=DiskCache(tmp_path), rate_limiter=NoLimit(),
                             today=lambda: dt.date(2026, 11, 24))
    return client, _wire(client.http, lambda url, params: FakeResponse(payload=payload))


def test_open_meteo_forecast_by_lead_day(tmp_path):
    payload = {"hourly": {"time": ["2024-03-03T14:00", "2024-03-03T15:00"],
                          "temperature_2m_previous_day1": [20.0, 21.0],
                          "precipitation_previous_day1": [0.0, 0.4],
                          "temperature_2m_previous_day3": [19.0, None],
                          "precipitation_previous_day3": [None, None]}}
    client, calls = _meteo(tmp_path, payload)
    rows = client.get_race_forecast(26.0, 50.5, "2024-03-03", leads=(1, 3))
    assert [(r["LeadDays"], r["AirTemp"], r["RainMm"]) for r in rows] == [
        (1, 20.0, 0.0), (1, 21.0, 0.4), (3, 19.0, None)]
    assert "precipitation_previous_day3" in calls[0][1]["hourly"]
    assert client.get_race_forecast(26.0, 50.5, "2023-03-05") == []  # not archived
    assert len(calls) == 1


def test_open_meteo_observed_waits_for_the_archive(tmp_path):
    payload = {"hourly": {"time": ["2024-03-03T14:00"], "temperature_2m": [21.0],
                          "precipitation": [0.3]}}
    client, calls = _meteo(tmp_path, payload)
    assert client.get_observed(10.0, 20.0, "2026-11-22") == [] and calls == []
    assert client.get_observed(10.0, 20.0, "2024-03-03") == [
        {"Time": "2024-03-03T14:00", "AirTemp": 21.0, "RainMm": 0.3}]
