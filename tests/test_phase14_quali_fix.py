"""Phase 14 step 0: qualifying gaps measured inside one session, capped, and model
inputs that can never be infinite or wildly out of range."""
import numpy as np
import pandas as pd
import pytest

from pitwall.predict import features
from pitwall.predict.models import base
from pitwall.predict.models.base import design_matrix, standardise


def _quali(rows):
    df = pd.DataFrame(rows, columns=["RaceId", "DriverId", "QualiPosition", "Q1Seconds",
                                     "Q2Seconds", "Q3Seconds"])
    df["TeamId"] = ["t1", "t1", "t2", "t2"][: len(df)]
    df["Grid"] = df["QualiPosition"]
    df["PitLaneStart"] = 0
    return df


def test_dry_q1_lap_does_not_beat_wet_q3_pole():
    # Q1 dry (fast), Q3 wet (slow): the Q1 knock-out must not look faster than pole.
    df = _quali([[1, "pole", 1, 82.0, 85.0, 95.0], [1, "p2", 2, 82.2, 85.3, 96.0],
                 [1, "out_q2", 3, 81.9, 86.0, None], [1, "out_q1", 4, 83.0, None, None]])
    out = features.add_quali_features(df).set_index("DriverId")
    assert out.loc["pole", "quali_gap_pct"] == 0.0
    assert out.loc["out_q2", "quali_gap_pct"] >= out.loc["p2", "quali_gap_pct"]
    assert out.loc["out_q1", "quali_gap_pct"] >= out.loc["out_q2", "quali_gap_pct"]
    old = features._phase7_gaps(df).set_axis(df["DriverId"])
    assert old["out_q2"] == 0.0  # the Phase 7 bug: a Q1 lap counted as the fastest


def test_gaps_are_capped_and_zero_times_ignored():
    df = _quali([[1, "a", 1, 80.0, None, None], [1, "b", 2, 99.0, None, None],
                 [1, "c", 3, 0.0, None, None], [1, "d", 4, None, None, None]])
    out = features.add_quali_features(df).set_index("DriverId")
    assert out["quali_gap_pct"].max() == features.QUALI_GAP_CAP
    assert np.isfinite(out["quali_gap_pct"]).all()
    assert out.loc["c", "QualiGapObs"] != out.loc["c", "QualiGapObs"]  # NaN, not -100%


def test_design_matrix_never_returns_huge_numbers():
    rows = pd.DataFrame({"x": [1.0, np.inf, -np.inf, np.nan]})
    assert design_matrix(rows, ["x"]).tolist() == [[1.0], [0.0], [0.0], [0.0]]


def test_standardise_clips_extreme_inputs(monkeypatch):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(np.array([[0.0], [1.0], [2.0]]))
    z = standardise(scaler, np.array([[1000.0]]))
    assert z[0, 0] == base.Z_CLIP
    monkeypatch.setattr(base, "Z_CLIP", None)
    assert standardise(scaler, np.array([[1000.0]]))[0, 0] > 100
    with pytest.raises(AssertionError):
        assert standardise(scaler, np.array([[1000.0]]))[0, 0] == 5.0
