"""Offline tests for infrastructure: cache, rate limiter, Jolpica client, lookups, CLI."""
import pytest

from pitwall import get_session
from pitwall.auth import TOKEN_ENV_VAR
from pitwall.cache import DiskCache
from pitwall.cli import main
from pitwall.errors import RateLimitError, SourceRequestError
from pitwall.plotting import TEAM_COLOURS, format_laptime, team_colour
from pitwall.sources import jolpica
from pitwall.sources.jolpica import JolpicaClient, RateLimiter, _result_record


# ---------- cache ----------
def test_corrupt_cache_file_is_a_miss(tmp_path):
    cache = DiskCache(tmp_path)
    cache.set("key", {"a": 1})
    next(tmp_path.glob("*.json")).write_text("{not json")
    assert cache.get("key") is None


def test_cache_clear_counts_files(tmp_path):
    cache = DiskCache(tmp_path)
    cache.set("a", 1)
    cache.set("b", 2)
    assert cache.clear() == 2 and cache.get("a") is None


# ---------- rate limiter ----------
class FakeClock:
    def __init__(self):
        self.now = 1000.0
        self.slept = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


def test_rate_limiter_spacing_and_hourly_budget(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(jolpica.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(jolpica.time, "sleep", clock.sleep)
    limiter = RateLimiter(per_second=2.0, per_hour=3)
    limiter.wait()
    limiter.wait()
    assert clock.slept == [pytest.approx(0.5)]  # 2 per second -> 0.5 s apart
    limiter.wait()
    with pytest.raises(RateLimitError):
        limiter.wait()  # 4th request within the hour
    clock.now += 3601
    limiter.wait()  # an hour later the budget is back


# ---------- Jolpica client ----------
class NoLimit:
    def wait(self):
        pass


class FakeResponse:
    def __init__(self, status=200, payload=None):
        self.status_code = status
        self.ok = status < 400
        self.headers = {}
        self.url = "https://api.jolpi.ca/test"
        self._payload = payload

    def json(self):
        return self._payload


def _pit_page(offset, total=250, limit=100):
    count = max(0, min(limit, total - offset))
    stops = [{"driverId": f"d{offset + i}", "lap": "10", "stop": "1", "duration": "22.5"}
             for i in range(count)]
    return {"MRData": {"limit": str(limit), "offset": str(offset), "total": str(total),
                       "RaceTable": {"Races": [{"round": "1", "PitStops": stops}]}}}


def _client(tmp_path, responder):
    client = JolpicaClient(cache=DiskCache(tmp_path), rate_limiter=NoLimit())
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append(params)
        return responder(params)

    client.http.get = fake_get
    return client, calls


def test_pagination_and_cache(tmp_path):
    client, calls = _client(tmp_path, lambda p: FakeResponse(payload=_pit_page(p["offset"])))
    stops = client.get_pitstops(2020, 1)
    assert len(stops) == 250 and stops[0]["DurationSeconds"] == 22.5
    assert [c["offset"] for c in calls] == [0, 100, 200]
    client.get_pitstops(2020, 1)
    assert len(calls) == 3  # second time everything comes from the cache


def test_server_errors_are_retried_with_backoff(tmp_path, monkeypatch):
    slept = []
    monkeypatch.setattr(jolpica.time, "sleep", slept.append)
    replies = iter([FakeResponse(503), FakeResponse(payload=_pit_page(0, total=1))])
    client, _ = _client(tmp_path, lambda p: next(replies))
    assert len(client.get_pitstops(2020, 1)) == 1
    assert slept == [2.0]


def test_client_errors_fail_fast(tmp_path):
    client, calls = _client(tmp_path, lambda p: FakeResponse(404))
    with pytest.raises(SourceRequestError):
        client.get_pitstops(2020, 1)
    assert len(calls) == 1  # 4xx (except 429) is never retried


def test_empty_answers_are_not_cached(tmp_path):
    client, calls = _client(tmp_path, lambda p: FakeResponse(payload=_pit_page(0, total=0)))
    assert client.get_pitstops(2020, 1) == []
    client.get_pitstops(2020, 1)
    assert len(calls) == 2  # asked again: "no data yet" might change


def test_old_drivers_get_a_code():
    row = {"Driver": {"driverId": "fangio", "givenName": "Juan", "familyName": "Fangio"},
           "Constructor": {}, "position": "1", "positionText": "1", "points": "8"}
    record = _result_record({"round": "1"}, row)
    assert record["Driver"] == "FAN" and record["Points"] == 8.0


# ---------- event lookup ----------
class ScheduleOnly:
    def get_schedule(self, year):
        return [{"Round": r, "EventName": name, "Date": "2024-01-01", "CircuitId": "",
                 "Circuit": "", "Locality": "", "Country": "", "Lat": 0.0, "Long": 0.0}
                for r, name in ((1, "Bahrain Grand Prix"), (2, "Saudi Arabian Grand Prix"))]


def test_event_lookup_errors_are_helpful():
    with pytest.raises(ValueError, match="matches no race"):
        get_session(2024, "Atlantis", source=ScheduleOnly())
    with pytest.raises(ValueError, match="several"):
        get_session(2024, "Grand Prix", source=ScheduleOnly())
    with pytest.raises(ValueError, match="no round 9"):
        get_session(2024, 9, source=ScheduleOnly())
    assert get_session(2024, "saudi", source=ScheduleOnly()).round == 2


# ---------- plotting helpers ----------
def test_team_colours_are_stable():
    assert team_colour("ferrari") == TEAM_COLOURS["ferrari"]
    assert team_colour("brand_new_team") == team_colour("brand_new_team")
    assert team_colour("brand_new_team").startswith("#")
    assert team_colour(None) == team_colour("unknown")


def test_format_laptime():
    assert format_laptime(83.456) == "1:23.456"
    assert format_laptime(59.9) == "0:59.900"
    assert format_laptime(float("nan")) == ""


# ---------- CLI ----------
def test_cli_token_never_prints_the_token(monkeypatch, capsys):
    monkeypatch.setenv(TOKEN_ENV_VAR, "abc.def.not-a-real-token")
    assert main(["token"]) == 0
    assert "abc.def" not in capsys.readouterr().out


def test_cli_rejects_unknown_commands():
    with pytest.raises(SystemExit):
        main(["warp-speed"])
