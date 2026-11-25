"""Surprise and data checks in the Monday grading."""
import numpy as np
import pandas as pd

from pitwall.predict.models.base import prediction_frame
from pitwall.predict.records import (
    RecordMeta, check_record, check_records, expected_surprises, make_record,
    write_track_record,
)


def _record(matrix, drivers=("fav", "mid", "rookie")):
    race = pd.DataFrame({"RaceId": 1, "DriverId": list(drivers),
                         "Driver": [d.upper()[:3] for d in drivers]})
    pred = prediction_frame(race, np.asarray(matrix, float), np.full(len(drivers), 0.05))
    meta = RecordMeta(1984, 6, 1, "Monaco GP", "1984-06-03", "quali", {}, {}, [], [])
    return {**make_record(pred, meta), "_stem": "1984-06-quali"}


# rows = drivers, columns = finishing positions; the rookie is almost sure to be last
MATRIX = [[0.900, 0.098, 0.002],
          [0.098, 0.899, 0.003],
          [0.002, 0.003, 0.995]]


def _results(order, drivers=("fav", "mid", "rookie")):
    return pd.DataFrame({"Year": 1984, "Round": 6, "DriverId": list(drivers),
                         "Driver": [d.upper()[:3] for d in drivers],
                         "Finish": [float(order[d]) for d in drivers],
                         "Win": [int(order[d] == 1) for d in drivers]})


def test_rookie_on_the_podium_is_flagged_as_a_surprise():
    checks = check_record(_record(MATRIX), _results({"fav": 1, "mid": 3, "rookie": 2}))
    rookie = [c for c in checks if c["Driver"] == "ROO"]
    assert len(rookie) == 1 and rookie[0]["Kind"] == "surprise"
    assert rookie[0]["Chance"] < 0.011
    assert "P2" in rookie[0]["Text"]


def test_unexpected_winner_is_an_upset_and_a_surprise():
    checks = check_record(_record(MATRIX), _results({"fav": 2, "mid": 3, "rookie": 1}))
    kinds = sorted(c["Kind"] for c in checks if c["Driver"] == "ROO")
    assert kinds == ["surprise", "upset"]


def test_expected_results_raise_nothing():
    assert check_record(_record(MATRIX), _results({"fav": 1, "mid": 2, "rookie": 3})) == []


def test_entry_list_mismatches_are_data_problems():
    results = _results({"fav": 1, "mid": 2, "senna": 3}, drivers=("fav", "mid", "senna"))
    checks = check_record(_record(MATRIX), results)
    data = sorted(c["Text"][:10] for c in checks if c["Kind"] == "data")
    assert len(data) == 2
    assert any("not in the prediction" in c["Text"] for c in checks)
    assert any("did not start" in c["Text"] for c in checks)


def test_track_record_compares_surprises_with_chance(tmp_path):
    record = _record(MATRIX)
    results = _results({"fav": 1, "mid": 3, "rookie": 2})
    checks = check_records([record], results)
    graded = results.assign(Moment="quali")
    assert expected_surprises(graded) == 0.03
    scores = pd.DataFrame([{"Stem": "1984-06-quali", "Year": 1984, "Round": 6,
                            "Event": "Monaco GP", "Moment": "quali", "Created": "x" * 16,
                            "Drivers": 3, "win_logloss": 0.1, "podium_logloss": 0.1,
                            "points_logloss": 0.1, "dnf_logloss": 0.1,
                            "b0_win_logloss": np.nan, "b0_podium_logloss": np.nan,
                            "Favourite": "FAV", "Winner": "FAV", "WinnerHit": 1,
                            "PodiumOverlap": 1.0, "P_win_of_winner": 0.9}])
    text = write_track_record(scores, graded, tmp_path / "t.md", checks=checks).read_text()
    assert "Surprises and data checks" in text and "observed: 1" in text and "ROO" in text
