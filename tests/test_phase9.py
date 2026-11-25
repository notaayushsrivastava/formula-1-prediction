"""Offline tests for Phase 9: Kalman strength filter, M1, observations, gate and plots."""
import matplotlib

matplotlib.use("Agg")  # draw off-screen; must come before pyplot is imported

import math  # noqa: E402
from dataclasses import asdict  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from synthetic_f1 import WORLD_PARAMS, make_strength_world  # noqa: E402

from pitwall.plotting import (  # noqa: E402
    car_strength_plot, driver_skill_plot, strength_vs_result,
)
from pitwall.predict import (  # noqa: E402
    LABELS, OBSERVATIONS, BacktestResult, StrengthFilter, StrengthModel, StrengthParams,
    gate, observation_table, walk_forward, write_report,
)
from pitwall.predict.backtest import (  # noqa: E402
    COMPARE_COLUMNS, strip_outcomes, summarise_by_model,
)
from pitwall.predict.models.base import (  # noqa: E402
    SMOOTHING, check_consistency, simulate_positions,
)
from pitwall.predict.strength import (  # noqa: E402
    fit_params, fitted_params, hyper_subset, kalman_update,
)

FAST = 2000


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def world():
    return make_strength_world()


@pytest.fixture(scope="module")
def obs(world):
    data, _ = world
    return observation_table(data[data["Moment"] == "quali"])


@pytest.fixture(scope="module")
def filtered(obs):
    return StrengthFilter(WORLD_PARAMS).run(obs, keep_history=True)


# ---------- the maths ----------
def test_kalman_update_matches_a_hand_computed_example():
    # prior N(0, 1), observe y = 2 with noise variance 1: gain 1/2
    mean, cov, loglik = kalman_update(np.array([0.0]), np.array([[1.0]]), np.array([[1.0]]),
                                      np.array([2.0]), np.array([1.0]))
    assert mean == pytest.approx([1.0])
    assert cov.ravel() == pytest.approx([0.5])
    assert loglik == pytest.approx(-0.5 * (4 / 2 + math.log(2) + math.log(2 * math.pi)))


def test_params_round_trip():
    params = StrengthParams(q_car=0.02, mu_rookie=-0.3)
    back = StrengthParams.from_vector(params.to_vector())
    assert asdict(back) == pytest.approx(asdict(params))


# ---------- recovering a known world ----------
def test_filter_recovers_driver_skills(world, filtered):
    _, truth = world
    table = filtered.strengths()
    estimate = table[table["Kind"] == "skill"].set_index("Id")["Mean"]
    true = pd.Series(truth["skills"])
    assert np.corrcoef(estimate[true.index], true)[0, 1] > 0.9


def test_filter_recovers_cars(world, filtered):
    _, truth = world
    table = filtered.strengths()
    estimate = table[table["Kind"] == "car"].set_index("Id")["Mean"]
    last = truth["cars"][truth["cars"]["RaceId"] == truth["cars"]["RaceId"].max()]
    true = last.set_index("TeamId")["Car"]
    assert np.corrcoef(estimate[true.index], true)[0, 1] > 0.9


def test_car_uncertainty_jumps_at_the_new_era(world, filtered):
    data, _ = world
    races = data.drop_duplicates("RaceId").set_index("RaceId")["Year"]
    last_2021 = races[races == 2021].index.max()
    first_2022 = races[races == 2022].index.min()
    pred = filtered.predictions.set_index(["RaceId", "DriverId"])["car_sd"]
    assert pred.loc[(first_2022, "d04")] > 2 * pred.loc[(last_2021, "d04")]


def test_team_swap_separates_car_from_driver(world, filtered):
    _, truth = world
    table = filtered.strengths().set_index("Id")["Mean"]
    true_skill_gap = truth["skills"]["d02"] - truth["skills"]["d00"]
    assert table["d02"] - table["d00"] == pytest.approx(true_skill_gap, abs=0.2)
    last = truth["cars"][truth["cars"]["RaceId"] == truth["cars"]["RaceId"].max()]
    true_car = last.set_index("TeamId")["Car"]
    assert table["t0"] - table["t1"] == pytest.approx(true_car["t0"] - true_car["t1"], abs=0.2)


def test_adding_later_races_changes_nothing_earlier(obs):
    short = StrengthFilter(WORLD_PARAMS).run(obs[obs["RaceId"] <= 20]).predictions
    full = StrengthFilter(WORLD_PARAMS).run(obs).predictions
    pd.testing.assert_frame_equal(short, full[full["RaceId"] <= 20].reset_index(drop=True))


def test_smoother_is_more_certain_than_the_forecast(filtered):
    smoothed = filtered.smooth()
    first = smoothed[(smoothed["RaceId"] == 1) & (smoothed["Kind"] == "car")]
    forecast = filtered.predictions[filtered.predictions["RaceId"] == 1]
    assert first["Sd"].max() < forecast["car_sd"].min()  # hindsight knows more
    with pytest.raises(ValueError, match="keep_history"):
        StrengthFilter(WORLD_PARAMS).smooth()


def test_likelihood_fit_recovers_q_car_roughly(obs):
    fitted = fit_params(obs, start=StrengthParams(q_car=0.1), maxiter=80)
    assert 0.002 < fitted.q_car < 0.05  # truth: 0.01 (start was 10x too big)


def test_fitted_params_are_cached_and_use_complete_seasons(obs):
    subset = hyper_subset(obs)
    assert subset["Year"].max() < obs["Year"].max()
    first = fitted_params(obs[obs["RaceId"] <= 12], maxiter=3)
    assert fitted_params(obs[obs["RaceId"] <= 12], maxiter=3) is first


# ---------- Monte Carlo with score uncertainty ----------
def _phase8_positions(scores, p_dnf, n_sims, rng):
    """The Phase 8 simulate_positions, copied verbatim, to prove nothing changed."""
    n = len(scores)
    keys = scores[None, :] + rng.gumbel(size=(n_sims, n))
    retired = rng.random((n_sims, n)) < p_dnf[None, :]
    keys = np.where(retired, keys - 1e9, keys)
    order = np.argsort(-keys, axis=1)
    place = np.empty_like(order)
    np.put_along_axis(place, order, np.arange(n)[None, :], axis=1)
    cells = np.arange(n)[None, :] * n + place
    counts = np.bincount(cells.ravel(), minlength=n * n).reshape(n, n)
    return (counts + SMOOTHING) / (n_sims + SMOOTHING * n)


def test_score_sd_none_reproduces_phase8_exactly():
    scores, dnf = np.array([1.0, 0.3, -0.5, -1.0]), np.full(4, 0.1)
    old = _phase8_positions(scores, dnf, FAST, np.random.default_rng(7))
    new = simulate_positions(scores, dnf, FAST, np.random.default_rng(7))
    assert np.array_equal(old, new)
    zero = simulate_positions(scores, dnf, FAST, np.random.default_rng(7), np.zeros(4))
    assert np.array_equal(old, zero)
    unsure = simulate_positions(scores, dnf, FAST, np.random.default_rng(7), np.full(4, 3.0))
    assert unsure[0, 0] < new[0, 0]  # less sure of his pace -> less sure he wins


# ---------- M1 ----------
def _train_and_race(world, moment):
    data, _ = world
    rows = data[data["Moment"] == moment]
    return rows[rows["Year"] < 2023], rows[rows["RaceId"] == 31]  # first race of 2023


@pytest.mark.parametrize("pure", [False, True])
def test_m1_is_consistent_and_deterministic(world, pure):
    pytest.importorskip("sklearn")
    train, race = _train_and_race(world, "pre")
    first = StrengthModel(pure=pure, params=WORLD_PARAMS, n_sims=FAST).fit(train)
    pred = first.predict(strip_outcomes(race))
    check_consistency(pred)
    again = StrengthModel(pure=pure, params=WORLD_PARAMS, n_sims=FAST).fit(train)
    pd.testing.assert_frame_equal(pred, again.predict(strip_outcomes(race)))
    assert len(first.coefficients()) >= 1
    assert set(first.strengths()["Kind"]) == {"car", "skill"}
    assert "strength" in first.params()


def test_m1_pure_favours_faster_packages(world):
    pytest.importorskip("sklearn")
    train, race = _train_and_race(world, "pre")
    model = StrengthModel(pure=True, params=WORLD_PARAMS, n_sims=FAST).fit(train)
    p_win = model.predict(strip_outcomes(race)).set_index("DriverId")["P_win"]
    pace = model.strength_features(strip_outcomes(race)).set_index("DriverId")["pace_pred"]
    assert p_win[pace.idxmin()] > p_win[pace.idxmax()]  # fastest beats slowest


class ObservationSpy(StrengthModel):
    """M1 that records what predict() is shown and checks the filter's last race."""

    def __init__(self):
        super().__init__(params=WORLD_PARAMS, n_sims=FAST)
        self.seen = set()

    def predict(self, race):
        assert self.filter_.last_date < race["Date"].iloc[0]  # filter never saw this race
        self.seen |= set(race.columns)
        return super().predict(race)


def test_predict_never_sees_observations_or_labels(world):
    pytest.importorskip("sklearn")
    data, _ = world
    spy = ObservationSpy()
    preds = walk_forward([spy], data, "pre", seasons=[2023])
    assert preds["RaceId"].nunique() == 10
    assert not (set(OBSERVATIONS) | set(LABELS)) & spy.seen


# ---------- gate ----------
def _gate_result():
    """20 identical 4-car races: 'good' is clearly better than 'bad' at win AND podium."""
    rows = []
    for race in range(1, 21):
        for model, p_win, p_podium in (("good", [0.7, 0.1, 0.1, 0.1], [0.95, 0.8, 0.8, 0.45]),
                                       ("bad", [0.25] * 4, [0.75] * 4),
                                       ("twin", [0.25] * 4, [0.75] * 4)):
            for k in range(4):
                rows.append({"Model": model, "Moment": "quali", "RaceId": race,
                             "DriverId": f"d{k}", "P_win": p_win[k], "P_podium": p_podium[k],
                             "P_points": 1.0, "P_dnf": 0.1, "ExpectedPosition": k + 1.0,
                             "Win": int(k == 0), "Podium": int(k < 3), "PointsFinish": 1,
                             "DNF": 0, "FinishOrder": k + 1})
    preds = pd.DataFrame(rows)
    return BacktestResult(preds, summarise_by_model(preds),
                          pd.DataFrame(columns=COMPARE_COLUMNS), (2025,), 2, "bad", "test")


def test_gate_keeps_only_clear_improvements(tmp_path):
    result = _gate_result()
    kept = gate(result, "good", ["bad"])
    assert kept.kept and kept.passed_moments == ("quali",) and "KEPT" in kept.verdict()
    same = gate(result, "twin", ["bad"])
    assert not same.kept and "NOT kept" in same.verdict()
    text = write_report(result, tmp_path / "r.md", gates=[kept, same]).read_text()
    assert "## Gate: good" in text and "Both clearly better?" in text


# ---------- plots ----------
def test_strength_plots(world, filtered):
    data, _ = world
    ax = car_strength_plot(filtered.smooth())
    assert len(ax.get_lines()) == 5 + 1  # 5 teams + the 2022 era line
    ax = driver_skill_plot(filtered.strengths())
    assert len(ax.get_yticklabels()) == 10
    race = data[(data["Moment"] == "quali") & (data["RaceId"] == 40)]
    flt = StrengthFilter(WORLD_PARAMS).run(observation_table(
        data[(data["Moment"] == "quali") & (data["RaceId"] < 40)]))
    table = flt.forecast(race).merge(race[["DriverId", "Driver", "FinishOrder"]], on="DriverId")
    assert "Spearman" in strength_vs_result(table).get_title()
