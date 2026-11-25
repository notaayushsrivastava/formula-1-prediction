"""Offline tests for Phase 8: probability maths, B0/B1/B2, backtest guards, metrics, report."""
import matplotlib

matplotlib.use("Agg")  # draw off-screen; must come before pyplot is imported

import math  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from synthetic_f1 import make_dataset, rank_desc  # noqa: E402

from pitwall.plotting import calibration_plot  # noqa: E402
from pitwall.predict import (  # noqa: E402
    LABELS, GridSlotBaseline, LeakageError, LockedSeasonError, run_backtest, walk_forward,
    write_report,
)
from pitwall.predict.backtest import check_no_future  # noqa: E402
from pitwall.predict.metrics import (  # noqa: E402
    METRIC_NAMES, binary_log_loss, brier, calibration_table, expected_calibration_error,
    mean_spearman, paired_bootstrap, podium_overlap, ranked_probability_score, summarise,
    winner_hit_rate, winner_log_loss,
)
from pitwall.predict.models.base import (  # noqa: E402
    check_consistency, fit_scale, pl_loglik_and_grad, position_matrix, predict_from_scores,
    simulate_positions, sinkhorn,
)
from pitwall.predict.models.grid import slots  # noqa: E402

FAST = 2000  # fewer simulated races in tests


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def data():
    return make_dataset()


def _moment(data, moment, before=None):
    rows = data[data["Moment"] == moment]
    return rows if before is None else rows[rows["Year"] < before]


def _race(data, moment, race_id):
    return data[(data["Moment"] == moment) & (data["RaceId"] == race_id)]


# ---------- probability maths ----------
def test_monte_carlo_matrix_is_doubly_stochastic_and_ordered():
    rng = np.random.default_rng(0)
    matrix = simulate_positions(np.array([2.0, 0.0, -2.0]), np.zeros(3), 5000, rng)
    assert matrix.sum(axis=1) == pytest.approx(np.ones(3))
    assert matrix.sum(axis=0) == pytest.approx(np.ones(3))
    assert matrix[0, 0] > matrix[1, 0] > matrix[2, 0] > 0  # never exactly 0 (smoothing)


def test_predictions_are_consistent_and_seeded(data):
    race = _race(data, "quali", 1)
    scores, dnf = -race["grid"].to_numpy(float), np.full(len(race), 0.1)
    first = predict_from_scores(race, scores, dnf, n_sims=FAST, seed=3)
    check_consistency(first)
    pd.testing.assert_frame_equal(first, predict_from_scores(race, scores, dnf,
                                                             n_sims=FAST, seed=3))
    broken = first.copy()
    broken.loc[0, "P_win"] += 0.1
    with pytest.raises(ValueError, match="P_win"):
        check_consistency(broken)


def test_sinkhorn_balances_rows_and_columns():
    out = sinkhorn(np.array([[5.0, 1.0, 1.0], [1.0, 2.0, 1.0], [3.0, 1.0, 4.0]]))
    assert out.sum(axis=1) == pytest.approx(np.ones(3))
    assert out.sum(axis=0) == pytest.approx(np.ones(3))


def test_plackett_luce_gradient_matches_finite_differences():
    rng = np.random.default_rng(1)
    u = rng.normal(size=(2, 4))
    mask = np.array([[True] * 4, [True, True, True, False]])
    _, grad = pl_loglik_and_grad(u, mask)
    step = 1e-6
    for r, k in [(0, 0), (0, 3), (1, 1)]:
        bumped = u.copy()
        bumped[r, k] += step
        numeric = (pl_loglik_and_grad(bumped, mask)[0] - pl_loglik_and_grad(u, mask)[0]) / step
        assert grad[r, k] == pytest.approx(numeric, abs=1e-4)


def _plackett_luce_races(n_races, n_drivers, weights, rng):
    """Random races whose orders really follow Plackett-Luce(features @ weights)."""
    features = rng.normal(size=(n_races * n_drivers, len(weights)))
    races = np.repeat(np.arange(n_races), n_drivers)
    finish = np.empty(len(races))
    for r in range(n_races):
        rows = slice(r * n_drivers, (r + 1) * n_drivers)
        keys = features[rows] @ weights + rng.gumbel(size=n_drivers)
        finish[rows] = rank_desc(keys)
    return features, races, finish


def test_fit_scale_recovers_a_known_temperature():
    features, races, finish = _plackett_luce_races(300, 8, np.array([2.0]),
                                                   np.random.default_rng(2))
    assert fit_scale(features[:, 0], races, finish) == pytest.approx(2.0, abs=0.3)


def test_linear_plackett_luce_recovers_known_weights():
    from pitwall.predict.models.logistic import fit_linear_plackett_luce
    truth = np.array([1.5, -0.5])
    features, races, finish = _plackett_luce_races(300, 8, truth, np.random.default_rng(3))
    weights = fit_linear_plackett_luce(features, races, finish, l2=1e-3)
    assert weights == pytest.approx(truth, abs=0.25)


# ---------- B0 ----------
def _tiny_quali_history():
    """4 two-car races: the pole sitter (grid 1) wins 3 of them."""
    rows = []
    for race, pole_wins in enumerate([True, True, True, False], start=1):
        for driver, grid in (("a", 1), ("b", 2)):
            won = (grid == 1) == pole_wins
            rows.append({"RaceId": race, "DriverId": driver, "Driver": driver.upper(),
                         "Moment": "quali", "grid": float(grid), "FinishOrder": 1 if won else 2,
                         "DNF": 0})
    return pd.DataFrame(rows)


def test_b0_learns_known_slot_probabilities():
    model = GridSlotBaseline().fit(_tiny_quali_history())
    assert model.slot_table_[0, 0] == pytest.approx((3 + 1) / (4 + 2))  # Laplace smoothing
    assert model.slot_table_[1, 0] == pytest.approx((1 + 1) / (4 + 2))
    race = _tiny_quali_history().query("RaceId == 1")
    pred = model.predict(race).set_index("Driver")
    assert pred.loc["A", "P_win"] == pytest.approx(2 / 3)
    assert pred["P_dnf"].tolist() == [0.0, 0.0]
    check_consistency(pred.reset_index())


def test_b0_slots_per_moment(data):
    race = _race(data, "practice", 1).copy()
    race.loc[race.index[0], "has_practice_pace"] = 0.0  # fastest driver has no long run
    assert slots(race, "practice")[0] == len(race)  # ...so he goes after those who have one
    pre = _race(data, "pre", 1)
    assert sorted(slots(pre, "pre")) == list(range(1, len(pre) + 1))


def test_b0_on_a_boring_season_trusts_the_grid():
    # 30 training races: P(win | pole) = (30 + 1) / (30 + 12) = 0.74 after smoothing
    boring = make_dataset(seasons=(2019, 2020, 2021, 2022, 2023, 2024), deterministic=True)
    model = GridSlotBaseline().fit(_moment(boring, "quali", before=2024))
    pred = model.predict(_race(boring, "quali", 31))  # first race of 2024
    assert pred.loc[pred["P_win"].idxmax(), "DriverId"] == "d00"
    assert pred["P_win"].max() == pytest.approx(31 / 42)


def test_b0_refuses_another_moment(data):
    model = GridSlotBaseline().fit(_moment(data, "quali", before=2025))
    with pytest.raises(ValueError, match="quali"):
        model.predict(_race(data, "pre", 1))


# ---------- B1 ----------
def test_b1_is_consistent_sensible_and_deterministic(data):
    pytest.importorskip("sklearn")
    from pitwall.predict import LogisticModel
    train, race = _moment(data, "quali", before=2026), _race(data, "quali", 19)
    first = LogisticModel(n_sims=FAST).fit(train).predict(race)
    check_consistency(first)
    p_win = first.set_index("DriverId")["P_win"]
    assert p_win["d00"] > p_win["d11"]
    again = LogisticModel(n_sims=FAST).fit(train).predict(race)
    pd.testing.assert_frame_equal(first, again)
    assert LogisticModel().fit(train).coefficients().index[0] in train.columns


# ---------- B2 ----------
def test_b2_is_consistent_monotone_and_deterministic(data):
    pytest.importorskip("lightgbm")
    from pitwall.predict import LightGBMRanker
    train, race = _moment(data, "quali", before=2026), _race(data, "quali", 19)
    model = LightGBMRanker(n_estimators=40, min_child_samples=5, n_sims=FAST).fit(train)
    pred = model.predict(race)
    check_consistency(pred)
    assert model.scale_ > 0
    features = race[model.features_].to_numpy(float)
    worse = features.copy()
    worse[:, model.features_.index("grid")] += 5  # everyone starts 5 places further back
    assert (model.ranker_.predict(worse) <= model.ranker_.predict(features) + 1e-9).all()
    again = LightGBMRanker(n_estimators=40, min_child_samples=5, n_sims=FAST).fit(train)
    pd.testing.assert_frame_equal(pred, again.predict(race))
    assert len(model.feature_importance()) == len(model.features_)


# ---------- backtest ----------
class SpyModel:
    """Wraps B0 and records what it was trained on and what it was shown."""

    name = "spy"

    def __init__(self):
        self.fits = 0
        self.seen_columns = set()

    def params(self):
        return {}

    def fit(self, train):
        self.fits += 1
        self.trained_until = train["Date"].max()
        self.inner = GridSlotBaseline().fit(train)
        return self

    def predict(self, race):
        assert race["Date"].iloc[0] > self.trained_until  # no future training data
        self.seen_columns |= set(race.columns)
        return self.inner.predict(race)


def test_walk_forward_never_trains_on_the_future(data):
    spy = SpyModel()
    preds = walk_forward([spy], data, "quali", seasons=[2025], retrain_every=2)
    assert preds["RaceId"].nunique() == 6 and spy.fits == 3
    assert not set(LABELS) & spy.seen_columns  # predict() never sees the outcome
    assert {"Win", "FinishOrder", "Model"} <= set(preds.columns)


def test_leakage_guard_raises(data):
    rows = _moment(data, "quali")
    race_date = rows["Date"].sort_values().iloc[-1]
    with pytest.raises(LeakageError):
        check_no_future(rows, race_date, "quali")  # includes the race itself
    with pytest.raises(LeakageError):
        check_no_future(data[data["Date"] < race_date], race_date, "quali")  # mixed moments


def test_2026_is_locked_until_final(data):
    with pytest.raises(LockedSeasonError):
        walk_forward([GridSlotBaseline()], data, "quali", seasons=[2025, 2026])
    preds = walk_forward([GridSlotBaseline()], data, "quali", seasons=[2026], final=True)
    assert set(preds["Year"]) == {2026}


class TwinB0(GridSlotBaseline):
    name = "B0 twin"


def test_run_backtest_logs_compares_and_reports(data, tmp_path):
    log = tmp_path / "experiments.csv"
    result = run_backtest([GridSlotBaseline(), TwinB0()], data, seasons=[2025],
                          moments=["pre", "quali"], log_path=log)
    assert set(result.summary["Moment"]) == {"pre", "quali"}
    twin = result.comparisons
    assert ((twin["Low"] <= 0) & (twin["High"] >= 0)).all()  # identical models: CI has 0
    logged = pd.read_csv(log)
    assert len(logged) == 4 and {"git", "params", "win_logloss"} <= set(logged.columns)
    text = write_report(result, tmp_path / "backtest.md").read_text()
    assert "## Moment: quali" in text and "Target: best open model" in text
    assert "(baseline)" in text and "the 0.075 target" in text


def test_backtest_is_deterministic(data):
    first = walk_forward([GridSlotBaseline()], data, "practice", seasons=[2025])
    second = walk_forward([GridSlotBaseline()], data, "practice", seasons=[2025])
    pd.testing.assert_frame_equal(first, second)


# ---------- metrics (hand-computed) ----------
def _two_car_race():
    return pd.DataFrame({
        "RaceId": [1, 1], "DriverId": ["a", "b"], "P_win": [0.8, 0.2],
        "P_podium": [1.0, 1.0], "P_points": [1.0, 1.0], "P_dnf": [0.1, 0.1],
        "ExpectedPosition": [1.2, 1.8], "P_pos_1": [0.8, 0.2], "P_pos_2": [0.2, 0.8],
        "Win": [1, 0], "Podium": [1, 1], "PointsFinish": [1, 1], "DNF": [0, 0],
        "FinishOrder": [1, 2]})


def test_binary_metrics_by_hand():
    assert binary_log_loss([0.5, 0.5], [1, 0]) == pytest.approx(math.log(2))
    assert binary_log_loss([0.0], [1]) == pytest.approx(-math.log(1e-6))  # clipped
    assert brier([0.2, 0.9], [0, 1]) == pytest.approx(0.025)


def test_race_metrics_by_hand():
    race = _two_car_race()
    other = race.assign(RaceId=2, P_win=[0.25, 0.75], P_pos_1=[0.25, 0.75],
                        P_pos_2=[0.75, 0.25])
    both = pd.concat([race, other], ignore_index=True)
    assert winner_log_loss(both) == pytest.approx((math.log(1 / 0.8) + math.log(4)) / 2)
    assert ranked_probability_score(race) == pytest.approx(0.04)
    four_cars = pd.concat([race, race.assign(DriverId=["c", "d"])])
    assert mean_spearman(four_cars) == pytest.approx(1.0)
    assert winner_hit_rate(both) == 0.5
    assert podium_overlap(race) == 1.0
    assert position_matrix(race).shape == (2, 2)
    assert set(summarise(race)) == set(METRIC_NAMES)


def test_calibration_by_hand():
    p, y = [0.1, 0.1, 0.9, 0.9], [0, 1, 1, 1]
    table = calibration_table(p, y)
    assert table["Count"].tolist() == [2, 2]
    assert table["Observed"].tolist() == [0.5, 1.0]
    assert expected_calibration_error(p, y) == pytest.approx(0.25)


def test_paired_bootstrap():
    a = pd.Series([0.1, 0.2, 0.3, 0.4], index=[1, 2, 3, 4])
    same = paired_bootstrap(a, a)
    assert (same.mean, same.low, same.high) == (0.0, 0.0, 0.0) and not same.significant
    worse = paired_bootstrap(a + 0.1, a)
    assert worse.mean == pytest.approx(0.1) and worse.significant
    with pytest.raises(ValueError):
        paired_bootstrap(a, pd.Series([0.1], index=[99]))


# ---------- plot ----------
def test_calibration_plot_draws_each_model_and_the_diagonal(data):
    preds = walk_forward([GridSlotBaseline(), TwinB0()], data, "quali", seasons=[2025])
    ax = calibration_plot(preds)
    assert len(ax.get_lines()) == 3
    assert ax.get_legend_handles_labels()[1][-1] == "Perfect"
