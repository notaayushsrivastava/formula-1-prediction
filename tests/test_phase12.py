"""Phase 12: live top-up, live rows, CLI, prediction records, track record, final 2026,
workflow files. Everything offline with the fake sources in fake_world.py."""
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import fake_world as fw
from pitwall import cli
from pitwall.errors import DataNotAvailableError, PitwallError
from pitwall.predict.backtest import LABELS
from pitwall.predict.final import (
    RERUN_LABEL, FinalRunError, check_season_finished, read_lock, run_final_2026,
)
from pitwall.predict.live import (
    IdMapper, IdMappingError, LiveWindowError, SettledWeekend, ToppedUpHistory,
    build_live_rows, convert_result, find_event, load_overrides, next_race,
)
from pitwall.predict.pipeline import MomentChoice, PipelineConfig, default_config
from pitwall.predict.records import (
    RecordExistsError, RecordMeta, load_records, make_record, save_record, score_record,
    write_docs, write_track_record,
)
from pitwall.predict.service import predict_race, results_table, score_all
from synthetic_f1 import make_dataset

ROOT = Path(__file__).resolve().parents[1]
H = dt.timedelta(hours=1)


def clock(when):
    return lambda: when


def live(moment, when, **kw):
    s = fw.sources(**kw)
    return build_live_rows(2026, 6, moment, s.history, s.jolpica, s.weekend, s.weather,
                           s.live_forecast, first_year=2024, history_from=2024,
                           now=clock(when))


# ---------- id mapping ----------
def test_ids_map_by_underscore_number_and_coordinates():
    mapper = IdMapper.from_history(fw.FakeF1db(), 2024, overrides={})
    assert mapper.driver("max_like_id_missing", number=3) == "charlie-three"  # by car number
    assert mapper.driver("alpha_one") == "alpha-one"
    assert mapper.team("shadow_racing", "Shadow Racing F1 Team", ["9", "10"]) == "black-team"
    assert mapper.circuit("j_east_bay", 25.31, 51.49) == "east-bay"
    mapper.check()  # nothing unmatched


def test_overrides_win_and_unmatched_ids_are_listed():
    mapper = IdMapper.from_history(fw.FakeF1db(), 2024,
                                   overrides={("team", "shadow_racing"): "white-team"})
    assert mapper.team("shadow_racing", "Shadow", ["9"]) == "white-team"
    mapper.driver("nobody_known", number=99)
    mapper.circuit("j_moon", 0.0, 0.0)
    with pytest.raises(IdMappingError, match="driver:nobody_known.*circuit:j_moon|circuit:j_moon"):
        mapper.check()


def test_override_file_parses_and_ignores_comments(tmp_path):
    path = tmp_path / "o.csv"
    path.write_text("# comment\nkind,jolpica_id,f1db_id\nteam,sauber,kick-sauber\n")
    assert load_overrides(path) == {("team", "sauber"): "kick-sauber"}
    assert isinstance(load_overrides(), dict)  # the packaged file is valid


def test_jolpica_result_conversion():
    row = {"DriverId": "a_b", "TeamId": "t", "Number": "4", "ClassifiedPosition": "R",
           "Status": "Engine", "GridPosition": 0, "Points": 0.0, "LapsCompleted": 10}
    out = convert_result(row, 7, {"a_b": "a-b"}, {"t": "team"}, 20)
    assert out["Position"] is None and out["PositionText"] == "DNF"
    assert out["GridPosition"] is None and out["ReasonRetired"] == "Engine"
    finished = convert_result({**row, "ClassifiedPosition": "3", "Status": "+1 Lap",
                               "GridPosition": 5}, 7, {"a_b": "a-b"}, {"t": "team"}, 3)
    assert finished["Position"] == 3 and finished["ReasonRetired"] is None


# ---------- top-up ----------
def test_top_up_adds_the_missing_race_without_duplicates():
    s = fw.sources()
    topped = ToppedUpHistory(s.history, s.jolpica, [2025, 2026],
                             now=clock(fw.at(2026, 6, "Practice 1", -H)))
    results = pd.DataFrame(topped.get_race_results(2024))
    per_race = results.groupby("RaceId").size()
    assert (per_race == 10).all()                         # no race twice
    assert fw.race_id(2026, 5) in per_race.index          # the topped-up race
    assert fw.race_id(2026, 6) not in per_race.index      # the future race has no results
    assert [r["Round"] for r in topped.added_races] == [5]
    assert set(results["DriverId"]) == set(fw.DRIVERS)    # all ids are f1db ids
    assert set(results["TeamId"]) == set(fw.TEAMS)
    standings = pd.DataFrame(topped.get_driver_standings(2026))
    assert fw.race_id(2026, 5) in set(standings["RaceId"])
    # f1db races are not re-fetched from Jolpica
    fetched = {c[1:] for c in s.jolpica.calls if c[0] == "results"}
    assert (2026, 1) not in fetched and (2025, 3) not in fetched


def test_schedule_helpers():
    schedule = fw.FakeJolpica().get_schedule(2026)
    assert find_event(schedule, "east")["Round"] == 3
    assert find_event(schedule, "4")["Round"] == 4
    with pytest.raises(DataNotAvailableError):
        find_event(schedule, "Monaco")
    assert next_race(schedule, fw.race_date(2026, 5) + dt.timedelta(days=1))["Round"] == 6
    assert next_race(schedule, dt.date(2027, 1, 1)) is None


# ---------- live rows ----------
def test_live_rows_have_dataset_columns_and_no_labels():
    rows = live("practice", fw.at(2026, 6, "Practice 2", 2 * H))
    assert len(rows.race) == 10 and rows.race_id == fw.race_id(2026, 6)
    assert rows.race[LABELS].isna().all().all()
    assert (rows.data["Moment"] == "practice").all()
    assert rows.flags["has_practice_pace"] == 1 and rows.flags["has_forecast"] == 1
    assert rows.race["air_temp_forecast"].eq(31.0).all()  # the LIVE forecast, not archived
    assert rows.flags["topped_up_races"] == 1


def test_missing_pieces_are_flagged_not_guessed():
    rows = live("pre", fw.at(2026, 6, "Practice 1", -5 * H), weather=False, openf1=False)
    assert rows.flags["has_forecast"] == 0
    assert rows.race["has_forecast"].eq(0).all()
    assert any("weather forecast" in w for w in rows.warnings)
    assert rows.race["grid"].isna().all()  # no grid before qualifying


def test_live_window_is_refused():
    with pytest.raises(LiveWindowError):
        live("quali", fw.at(2026, 6, "Qualifying", 0.5 * H), quali_upto=6)
    with pytest.raises(LiveWindowError):
        live("practice", fw.at(2026, 6, "Practice 1", 0.5 * H))


def test_settled_weekend_never_requests_unsettled_sessions():
    source = fw.FakeOpenF1()
    safe = SettledWeekend(source, clock(fw.at(2026, 6, "Practice 1", 2 * H)))
    sessions = {s["Name"]: s for s in safe.get_sessions(2026)
                if s["MeetingKey"] == fw.race_id(2026, 6)}
    assert safe.get_laps(sessions["Practice 1"])
    with pytest.raises(LiveWindowError):
        safe.get_laps(sessions["Practice 2"])
    assert source.requested == [f"laps:{sessions['Practice 1']['SessionKey']}"]


def test_quali_needs_published_qualifying_and_uses_its_grid():
    with pytest.raises(DataNotAvailableError, match="qualifying"):
        live("quali", fw.at(2026, 6, "Qualifying", 3 * H), quali_upto=5)
    rows = live("quali", fw.at(2026, 6, "Qualifying", 3 * H), quali_upto=6)
    expected = {fw.DRIVERS[i]: fw._grid(2026, 6, i) for i in range(10)}
    got = dict(zip(rows.race["DriverId"], rows.race["grid"]))
    assert got == expected
    assert any("penalties" in n for n in rows.notes)


def test_races_with_results_are_refused():
    s = fw.sources()
    with pytest.raises(PitwallError, match="already has results"):
        build_live_rows(2026, 4, "pre", s.history, s.jolpica, now=clock(
            fw.at(2026, 6, "Practice 1", -H)), first_year=2024, history_from=2024)


# ---------- records ----------
def _pred(drivers=("a", "b", "c")):
    n = len(drivers)
    rng = np.random.default_rng(0)
    m = rng.random((n, n)) + 0.1
    for _ in range(200):
        m /= m.sum(1, keepdims=True)
        m /= m.sum(0, keepdims=True)
    from pitwall.predict.models.base import prediction_frame
    race = pd.DataFrame({"RaceId": 1, "DriverId": list(drivers),
                         "Driver": [d.upper() for d in drivers]})
    return prediction_frame(race, m, np.full(n, 0.1))


def _meta(moment="pre"):
    return RecordMeta(2026, 6, 1, "Test GP", "2026-06-14", moment, {"m": 1}, {"f": 1}, [], [])


def test_record_schema_and_never_overwrite(tmp_path):
    record = make_record(_pred(), _meta(), baseline=_pred())
    for key in ("created_utc", "git", "config_hash", "version", "availability", "drivers",
                "baseline", "race_id", "moment"):
        assert key in record
    first = record["drivers"][0]
    assert {"P_win", "P_podium", "P_points", "P_dnf", "ExpectedPosition", "P_pos"} <= set(first)
    assert sum(first["P_pos"]) == pytest.approx(1, abs=1e-5)
    path = save_record(record, tmp_path)
    assert path.name == "2026-06-pre.json"
    before = path.read_text()
    with pytest.raises(RecordExistsError):
        save_record({**record, "drivers": []}, tmp_path)
    assert path.read_text() == before
    save_record(make_record(_pred(), _meta("quali")), tmp_path)  # new moment = new file
    assert [r["moment"] for r in load_records(tmp_path)] == ["pre", "quali"]


def test_scoring_maths_by_hand():
    pred = _pred()
    record = make_record(pred, _meta(), baseline=pred)
    results = pd.DataFrame({"DriverId": ["a", "b", "c"], "Driver": ["A", "B", "C"],
                            "Finish": [2.0, 1.0, np.nan], "Win": [0, 1, 0],
                            "Podium": [1, 1, 0], "PointsFinish": [1, 1, 0], "DNF": [0, 0, 1]})
    got = score_record(record, results)
    p = pred.set_index("DriverId")["P_win"]
    expected = -np.mean([np.log(1 - p["a"]), np.log(p["b"]), np.log(1 - p["c"])])
    assert got["win_logloss"] == pytest.approx(expected, abs=1e-5)
    assert got["b0_win_logloss"] == pytest.approx(expected, abs=1e-5)
    assert got["Winner"] == "B"
    assert got["WinnerHit"] == int(p.idxmax() == "b")
    assert score_record(record, results.iloc[0:0]) is None


def test_track_record_and_docs(tmp_path):
    scores = pd.DataFrame([{"Stem": "2026-06-pre", "Year": 2026, "Round": 6, "Event": "T",
                            "Moment": "pre", "Created": "2026-06-11T12:00:00+00:00",
                            "Drivers": 3, "win_logloss": 0.3, "podium_logloss": 0.5,
                            "points_logloss": 0.1, "dnf_logloss": 0.2,
                            "b0_win_logloss": 0.4, "b0_podium_logloss": 0.6,
                            "Favourite": "A", "Winner": "B", "WinnerHit": 0,
                            "PodiumOverlap": 2 / 3, "P_win_of_winner": 0.3}])
    path = write_track_record(scores, pd.DataFrame(), tmp_path / "track.md", ["X"])
    text = path.read_text()
    assert "0.075" in text and "0.191" in text and "0.3000" in text and "Waiting" in text
    save_record(make_record(_pred(), _meta()), tmp_path / "pred")
    index = write_docs(load_records(tmp_path / "pred"), tmp_path / "docs", path)
    assert "2026 Test GP" in index.read_text()
    assert (tmp_path / "docs" / "predictions" / "2026-06-pre.md").exists()
    assert (tmp_path / "docs" / "track-record.md").exists()


# ---------- the whole flow, and the CLI ----------
@pytest.fixture
def world(tmp_path, monkeypatch):
    """CLI runs in a temp folder with fake sources and a fake clock."""
    monkeypatch.chdir(tmp_path)
    state = {"now": fw.at(2026, 6, "Practice 2", 2 * H), "results_upto": 5, "quali_upto": 5}
    monkeypatch.setattr(cli, "make_sources", lambda openf1=True, weather=True: fw.sources(
        state["results_upto"], state["quali_upto"], weather, openf1))
    monkeypatch.setattr(cli, "utcnow", lambda: state["now"])
    monkeypatch.setattr("pitwall.predict.service.HISTORY_PREDICTIONS", tmp_path / "none.parquet")
    import pitwall.predict.live as live_module
    real = live_module.build_live_rows

    def short_history(*args, **kw):
        kw.update(first_year=2024, history_from=2024)
        return real(*args, **kw)
    monkeypatch.setattr("pitwall.predict.service.build_live_rows", short_history)
    return state


def test_cli_predict_saves_once_and_scores(world, capsys):
    assert cli.main(["predict", "2026", "Desert", "--moment", "practice"]) == cli.EXIT_OK
    record = json.loads(Path("predictions/2026-06-practice.json").read_text())
    assert record["availability"]["has_practice_pace"] == 1
    assert len(record["drivers"]) == 10 and "baseline" in record
    assert Path("docs/index.md").exists() and Path("docs/img/2026-06-practice-bars.png").exists()
    assert cli.main(["predict", "2026", "6", "--moment", "practice"]) == cli.EXIT_ERROR
    assert cli.main(["predict", "2026", "6", "--moment", "practice",
                     "--skip-if-exists"]) == cli.EXIT_OK
    # quali before qualifying is published: exit 3, or 0 with --skip-if-unavailable
    world["now"] = fw.at(2026, 6, "Qualifying", 2 * H)
    assert cli.main(["predict", "next", "--moment", "quali"]) == cli.EXIT_UNAVAILABLE
    assert cli.main(["predict", "next", "--moment", "quali",
                     "--skip-if-unavailable"]) == cli.EXIT_OK
    world["quali_upto"] = 6
    assert cli.main(["predict", "next", "--moment", "quali", "--json", "q.json"]) == 0
    assert json.loads(Path("q.json").read_text())["moment"] == "quali"
    capsys.readouterr()
    assert cli.main(["next"]) == 0
    assert "Latest prediction (quali" in capsys.readouterr().out
    # before the race: nothing to score
    assert cli.main(["score"]) == 0
    assert "Waiting for results" in Path("reports/track_record.md").read_text()
    # the race runs; Monday's score job grades both records
    world["results_upto"] = 6
    world["now"] = fw.at(2026, 6, "Race", 40 * H)
    assert cli.main(["score"]) == 0
    text = Path("reports/track_record.md").read_text()
    assert "practice" in text and "quali" in text and "Waiting" not in text
    assert Path("docs/track-record.md").exists()


def test_results_table_from_topped_up_history():
    s = fw.sources(results_upto=6)
    table = results_table(s, [2026], now=clock(fw.at(2026, 6, "Race", 40 * H)))
    race6 = table[table["Round"] == 6]
    assert len(race6) == 10 and race6["Win"].sum() == 1 and race6["DNF"].sum() == 1


def test_predict_service_respects_existing_record(tmp_path, monkeypatch):
    monkeypatch.setattr("pitwall.predict.service.HISTORY_PREDICTIONS", tmp_path / "x.parquet")
    s = fw.sources()
    when = clock(fw.at(2026, 6, "Practice 1", -5 * H))
    out = predict_race(2026, "6", "pre", s, records_dir=tmp_path, first_year=2024, now=when,
                       config=default_config())
    assert out.path.exists() and out.record["pipeline"]["note"].startswith("default")
    with pytest.raises(RecordExistsError):
        predict_race(2026, "6", "pre", s, records_dir=tmp_path, first_year=2024, now=when)
    assert score_all(s, tmp_path, tmp_path / "t.md", None, now=when).empty


def test_cli_dataset_build_and_backtest_fast(world, tmp_path, capsys):
    world["now"] = fw.at(2026, 6, "Practice 1", -5 * H)
    assert cli.main(["dataset", "build", "--from", "2024", "--out", "d.parquet"]) == 0
    data = pd.read_parquet("d.parquet")
    assert set(data["Year"]) == {2024, 2025, 2026}
    make_dataset(seasons=(2023, 2024, 2025)).to_parquet("s.parquet")
    code = cli.main(["backtest", "--season", "2025", "--fast", "--dataset", "s.parquet",
                     "--moments", "quali", "--save-predictions", "p.parquet",
                     "--write-config-if-missing"])
    assert code == 0
    assert Path("reports/backtest.md").exists() and Path("reports/final_config.json").exists()
    assert set(pd.read_parquet("p.parquet")["Model"]) == {"B0 grid slot", "B1 logistic",
                                                          "M1 strength"}
    assert cli.main(["backtest", "--season", "2026", "--fast",
                     "--dataset", "s.parquet"]) == cli.EXIT_ERROR  # the lock


def test_seed_files_are_averaged(tmp_path):
    base = pd.DataFrame({"Model": "M2-full", "Moment": "pre", "RaceId": 1,
                         "DriverId": ["a", "b"], "P_win": [0.6, 0.4], "P_pos_1": [0.6, 0.4],
                         "ExpectedPosition": [1.4, 1.6], "Win": [1, 0]})
    other = base.assign(P_win=[0.2, 0.8], P_pos_1=[0.2, 0.8], ExpectedPosition=[1.8, 1.2])
    base.to_parquet(tmp_path / "s0.parquet")
    other.to_parquet(tmp_path / "s1.parquet")
    merged = cli._merge_seed_files([str(tmp_path / "s0.parquet"), str(tmp_path / "s1.parquet")])
    assert merged["P_win"].tolist() == pytest.approx([0.4, 0.6])
    assert merged["Win"].tolist() == [1, 0]


# ---------- the final 2026 run ----------
def _races(finished=True):
    rows = [{"Year": 2026, "Round": r, "EventName": f"GP{r}", "HasResults": finished or r < 6}
            for r in range(1, 7)]
    return pd.DataFrame(rows)


def test_final_refuses_before_the_season_ends(tmp_path):
    with pytest.raises(FinalRunError, match="not finished"):
        check_season_finished(_races(finished=False))
    with pytest.raises(FinalRunError, match="i-understand"):
        run_final_2026(pd.DataFrame(), _races(), default_config(), lock_path=tmp_path / "L")


def test_final_writes_the_lock_and_labels_reruns(tmp_path):
    data = make_dataset(seasons=(2024, 2025, 2026), races_per_season=4, n_drivers=8)
    config = PipelineConfig({"quali": MomentChoice(["B1 logistic", "B0 grid slot"], "equal")},
                            n_seeds=1)
    kw = dict(confirmed=True, report_path=tmp_path / "final.md", lock_path=tmp_path / "LOCK.json",
              moments=["quali"])
    outcome = run_final_2026(data, _races(), config, **kw)
    text = outcome.report.read_text()
    assert "final evaluation: 2026" in text and RERUN_LABEL not in text
    assert "Final pipeline without B1 logistic" in text and "0.075" in text
    lock = read_lock(tmp_path / "LOCK.json")
    assert lock["season"] == 2026 and not lock["rerun"] and lock["config_hash"]
    with pytest.raises(FinalRunError, match="already happened"):
        run_final_2026(data, _races(), config, **kw)
    again = run_final_2026(data, _races(), config, force=True, plots=False, **kw)
    assert again.rerun and RERUN_LABEL in again.report.read_text()
    assert read_lock(tmp_path / "LOCK.json")["previous"]["timestamp"] == lock["timestamp"]


def test_cli_final_needs_confirmation(world):
    assert cli.main(["final-2026"]) == cli.EXIT_ERROR
    assert cli.main(["final-2026", "--i-understand-this-is-final"]) == cli.EXIT_ERROR  # unfinished


# ---------- workflows ----------
WORKFLOWS = ROOT / ".github" / "workflows"
FORBIDDEN = ["PITWALL_F1TV_TOKEN", "livetiming.formula1.com", "odds-api", "the-odds-api",
             "betfair", "pinnacle", "runs-on: macos", "openf1.org/v1/live"]


@pytest.mark.parametrize("name", ["tests.yml", "predict.yml", "train.yml", "score.yml",
                                  "race-report.yml"])
def test_workflows_parse_and_stay_free(name):
    yaml = pytest.importorskip("yaml")
    text = (WORKFLOWS / name).read_text()
    data = yaml.safe_load(text)
    triggers = data.get("on", data.get(True))  # PyYAML reads the key 'on' as True
    assert triggers and data["jobs"]
    for job in data["jobs"].values():
        assert job["runs-on"] == "ubuntu-latest" or "matrix" in str(job["runs-on"])
    lowered = text.lower()
    for word in FORBIDDEN:
        assert word.lower() not in lowered, f"{name} mentions {word}"
    secrets = [part.split("}}")[0] for part in text.split("secrets.")[1:]]
    assert all(s.strip() == "GITHUB_TOKEN" for s in secrets)
    if name != "tests.yml":
        assert "workflow_dispatch" in triggers and "concurrency" in data
        writers = [j for j in data["jobs"].values()
                   if j.get("permissions", {}).get("contents") == "write"]
        assert data.get("permissions", {}).get("contents") != "write"
        for job in writers:
            assert "timeout-minutes" in job


def test_gitignore_keeps_secrets_data_and_models_out():
    text = (ROOT / ".gitignore").read_text()
    for pattern in (".env", "*.token", "/data/", "/models/", "*.parquet"):
        assert pattern in text
