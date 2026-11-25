"""Offline tests for Phase 11: score sources, M3, online stacking, calibration, pipeline."""
import matplotlib

matplotlib.use("Agg")  # draw off-screen; must come before pyplot is imported

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from scipy import stats  # noqa: E402
from synthetic_f1 import WORLD_PARAMS, make_dataset, make_strength_world  # noqa: E402

from pitwall.plotting import start_weight_plot, weights_plot  # noqa: E402
from pitwall.predict import (  # noqa: E402
    GridSlotBaseline, LeakageError, LockedSeasonError, LogisticModel, MomentChoice,
    PipelineConfig, FinalPipeline, RaceSimulator, SimParams, StrengthModel, as_score_source,
    build_stacks, calibrate_forward, fit_temperature, fit_weights, pool, stack_forward,
    walk_forward,
)
from pitwall.predict.backtest import strip_outcomes  # noqa: E402
from pitwall.predict.calibration import apply_platt, fit_platt  # noqa: E402
from pitwall.predict.ensemble import check_past_only  # noqa: E402
from pitwall.predict.models.base import (  # noqa: E402
    check_consistency, design_matrix, predict_from_scores, simulate_positions, sinkhorn,
)
from pitwall.predict.models.simulator import fit_wet_model, simulate_race  # noqa: E402

FAST = 2000
SIM = SimParams(c0=0.0, c1=1.5, g=1.0, kappa_sc=3.0, kappa_wet=2.0)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


# ---------- score sources ----------
def test_score_sources_reproduce_model_predictions():
    pytest.importorskip("sklearn")
    world, _ = make_strength_world()
    rows = world[world["Moment"] == "pre"]
    model = StrengthModel(pure=True, params=WORLD_PARAMS, n_sims=FAST).fit(rows[rows["Year"] < 2023])
    race = strip_outcomes(rows[rows["RaceId"] == 31])
    score, spread = as_score_source(model).scores(race)
    z = model.scaler_.transform(design_matrix(model.strength_features(race), model.features_))
    p_dnf = model.dnf_.predict_proba(z)[:, 1]
    rebuilt = predict_from_scores(race, score, p_dnf, 1.0, FAST, model.seed, spread)
    pd.testing.assert_frame_equal(rebuilt, model.predict(race))

    data = make_dataset()
    quali = data[data["Moment"] == "quali"]
    b1 = LogisticModel(n_sims=FAST).fit(quali[quali["Year"] < 2026])
    race = strip_outcomes(quali[quali["RaceId"] == 19])
    score, spread = as_score_source(b1).scores(race)
    p_dnf = b1.dnf_.predict_proba(b1.scaler_.transform(design_matrix(race, b1.features_)))[:, 1]
    rebuilt = predict_from_scores(race, score, p_dnf, 1.0, FAST, b1.seed, spread)
    pd.testing.assert_frame_equal(rebuilt, b1.predict(race))


# ---------- M3 ----------
class FakeSource:
    """A 'model' whose scores are a column of the rows (so tests control the pace)."""

    name = "fake"

    def fit(self, train):
        return self

    def params(self):
        return {}

    def scores(self, race):
        return race["pace"].to_numpy(float), np.zeros(len(race))

    def training_scores(self, train):
        return self.scores(train)


def _hand_race(race_id, moved):
    """6 cars: d0 fastest but starting last (grid reversed to pace)."""
    rows = []
    for i in range(6):
        rows.append({"RaceId": race_id, "DriverId": f"d{i}", "Driver": f"D{i}",
                     "TeamId": f"t{i // 2}", "Moment": "quali",
                     "Date": pd.Timestamp(2025, 3, 1) + pd.Timedelta(days=7 * race_id),
                     "Year": 2025, "pace": 2.5 - i, "grid": 6.0 - i,
                     "circuit_places_moved": moved, "circuit_sc_rate": 0.0,
                     "has_sc_history": 1.0, "rain_forecast_mm": 0.0,
                     "team_mech_dnf_rate_10": 0.0, "driver_dnf_rate_10": 0.0,
                     "Finish": float(i + 1), "FinishOrder": i + 1, "DNF": 0, "MechDNF": 0,
                     "RaceWet": 0.0, "Neutralised": 0.0})
    return pd.DataFrame(rows)


def test_more_overtaking_means_pace_matters_more():
    sim = RaceSimulator(source_model=FakeSource(), params=SIM, n_sims=4000)
    sim.fit(_hand_race(1, 2.0))
    pace_rank = np.arange(1, 7)
    easy = sim.predict(_hand_race(2, 4.5))["ExpectedPosition"]
    hard = sim.predict(_hand_race(3, 0.2))["ExpectedPosition"]
    assert stats.spearmanr(easy, pace_rank)[0] > stats.spearmanr(hard, pace_rank)[0]
    assert sim.start_curve()["a_c"].is_monotonic_increasing


def _entropy(matrix):
    return float(-np.sum(matrix * np.log(matrix)) / len(matrix))


def test_chaos_raises_outcome_entropy():
    pace, zeros = np.array([2.0, 1.0, 0.0, -1.0]), np.zeros(4)
    teams = np.array(["a", "a", "b", "b"])
    calm = simulate_race(pace, zeros, zeros, 1.0, 0.0, 0.0, zeros, zeros, teams, SIM, 4000,
                         np.random.default_rng(0))
    wild = simulate_race(pace, zeros, zeros, 1.0, 1.0, 0.0, zeros, zeros, teams, SIM, 4000,
                         np.random.default_rng(0))
    assert _entropy(wild.positions) > _entropy(calm.positions)


def test_teammates_mechanical_dnfs_are_correlated():
    zeros, risk = np.zeros(4), np.full(4, 0.3)
    draws = simulate_race(zeros, zeros, zeros, 1.0, 0.0, 0.0, risk, zeros,
                          np.array(["a", "a", "b", "b"]), SIM, 20000, np.random.default_rng(1))
    dnf = draws.dnf.astype(float)
    assert np.corrcoef(dnf[:, 0], dnf[:, 1])[0, 1] > 0.15  # same team: shared shock
    assert abs(np.corrcoef(dnf[:, 0], dnf[:, 2])[0, 1]) < 0.05  # different teams


def test_wet_model_learns_only_from_given_past_races():
    rows = pd.DataFrame({"RaceId": range(20), "rain_forecast_mm": [0.0, 5.0] * 10,
                         "RaceWet": [0.0, 1.0] * 10})
    unknown = pd.DataFrame({"RaceId": range(20, 30), "rain_forecast_mm": [5.0] * 10,
                            "RaceWet": [np.nan] * 10})
    wet = fit_wet_model(pd.concat([rows, unknown]))
    assert wet.prob(5.0) > 0.8 and wet.prob(0.0) < 0.2
    sim = RaceSimulator(source_model=FakeSource(), params=SIM, n_sims=FAST)
    sim.fit(_hand_race(1, 2.0))
    race = strip_outcomes(_hand_race(2, 2.0))
    assert "RaceWet" not in race.columns
    check_consistency(sim.predict(race))


def test_m3_fits_and_is_consistent_and_deterministic():
    pytest.importorskip("sklearn")
    world, _ = make_strength_world()
    rows = world[world["Moment"] == "quali"]
    train, race = rows[rows["Year"] < 2023], strip_outcomes(rows[rows["RaceId"] == 31])

    def make():
        source = StrengthModel(pure=True, params=WORLD_PARAMS, n_sims=FAST)
        return RaceSimulator(source_model=source, n_sims=FAST, maxiter=15).fit(train)

    first = make().predict(race)
    check_consistency(first)
    pd.testing.assert_frame_equal(first, make().predict(race))
    assert "sim" in make().params()


# ---------- ensemble ----------
def _fake_preds(n_races=20, n=4, seed=0):
    """'perfect' puts 0.9 on the true position; 'random' is uniform."""
    rng = np.random.default_rng(seed)
    rows = []
    for race in range(1, n_races + 1):
        truth = rng.permutation(n)
        perfect = 0.1 / n + 0.9 * np.eye(n)[truth]
        for model, matrix in (("perfect", perfect), ("random", np.full((n, n), 1.0 / n))):
            for i in range(n):
                row = {"Model": model, "Moment": "quali", "RaceId": race,
                       "RaceKey": f"r{race}", "EventName": "E",
                       "Date": pd.Timestamp(2024, 3, 1) + pd.Timedelta(days=14 * race),
                       "Year": 2024 if race <= n_races // 2 else 2025,
                       "DriverId": f"d{i}", "Driver": f"D{i}", "P_win": matrix[i, 0],
                       "P_podium": matrix[i, :3].sum(), "P_points": 1.0, "P_dnf": 0.1,
                       "ExpectedPosition": float(matrix[i] @ np.arange(1, n + 1)),
                       "Finish": float(truth[i] + 1), "FinishOrder": int(truth[i] + 1),
                       "Win": int(truth[i] == 0), "Podium": int(truth[i] < 3),
                       "PointsFinish": 1, "DNF": 0, "MechDNF": 0, "Points": 0.0}
                row.update({f"P_pos_{k + 1}": matrix[i, k] for k in range(n)})
                rows.append(row)
    return pd.DataFrame(rows)


def test_linear_pool_trusts_the_perfect_model():
    stacks = build_stacks(_fake_preds(), "quali")
    weights = dict(zip(stacks[0].models, fit_weights(stacks, "linear", l2=0.0)))
    assert weights["perfect"] > 0.9


def test_pools_keep_consistency():
    stack = build_stacks(_fake_preds(), "quali")[0]
    for method in ("linear", "log"):
        matrix = pool(stack.matrices, np.array([0.5, 0.5]), method)
        assert matrix.sum(axis=0) == pytest.approx(np.ones(4))
        assert matrix.sum(axis=1) == pytest.approx(np.ones(4))


def test_stack_forward_uses_only_earlier_races():
    preds = _fake_preds()
    ens, history = stack_forward(preds, "quali", eval_seasons=[2025], warmup_seasons=[2024],
                                 method="linear", min_train_races=3)
    check_consistency(ens)
    races = preds.drop_duplicates("RaceId").set_index("RaceId")["Date"]
    for row in history.itertuples():
        assert row.TrainRaces == int((races < row.Date).sum())
    assert set(ens.loc[ens["Eval"], "Year"]) == {2025}
    stacks = build_stacks(preds, "quali")
    with pytest.raises(LeakageError):
        check_past_only(stacks[:3], stacks[2].date)  # includes the race itself
    with pytest.raises(LockedSeasonError):
        stack_forward(preds, "quali", eval_seasons=[2026], warmup_seasons=[2025])


# ---------- calibration ----------
def test_temperature_undoes_squared_probabilities():
    rng = np.random.default_rng(3)
    matrices, orders = [], []
    for _ in range(300):
        scores = rng.normal(size=6)
        truth = simulate_positions(scores, np.zeros(6), 4000, rng)
        matrices.append(sinkhorn(truth**2))  # over-confident by a power of 2
        keys = scores + rng.gumbel(size=6)
        orders.append(np.argsort(np.argsort(-keys)))
    assert fit_temperature(matrices, orders) == pytest.approx(0.5, abs=0.15)


def test_platt_scaling_fixes_a_biased_dnf_rate():
    p, y = np.full(1000, 0.5), (np.arange(1000) < 100).astype(float)  # real rate 10%
    a, b = fit_platt(p, y)
    assert apply_platt(np.array([0.5]), a, b)[0] == pytest.approx(0.1, abs=0.02)


def test_calibrate_forward_online():
    ens, _ = stack_forward(_fake_preds(), "quali", [2025], [2024], "linear", min_train_races=3)
    calibrated, history = calibrate_forward(ens, min_train_races=3)
    check_consistency(calibrated)
    assert calibrated["Model"].iloc[0] == "ENS-linear+cal"
    assert (history["Temperature"] > 0).all()


# ---------- final pipeline ----------
def test_pipeline_config_round_trip(tmp_path):
    config = PipelineConfig({"quali": MomentChoice(["B0 grid slot"])}, n_seeds=2)
    back = PipelineConfig.load(config.save(tmp_path / "c.json"))
    assert back.moments["quali"].models == ["B0 grid slot"] and back.n_seeds == 2


def test_pipeline_respects_the_2026_lock():
    data = make_dataset()
    pipeline = FinalPipeline(PipelineConfig({"quali": MomentChoice(["B0 grid slot"])}))
    with pytest.raises(LockedSeasonError):
        pipeline.predict(data, 19, "quali")  # first race of 2026
    check_consistency(pipeline.predict(data, 19, "quali", final=True))
    check_consistency(pipeline.predict(data, 13, "quali"))


def test_pipeline_combines_models_with_past_weights():
    pytest.importorskip("sklearn")
    data = make_dataset()
    history = walk_forward([GridSlotBaseline(), LogisticModel(n_sims=FAST)], data, "quali",
                           seasons=[2025])
    choice = MomentChoice(["B0 grid slot", "B1 logistic"], "linear", calibrate=True)
    pipeline = FinalPipeline(PipelineConfig({"quali": choice}), history=history)
    check_consistency(pipeline.predict(data, 18, "quali"))


# ---------- plots ----------
def test_phase11_plots():
    _, history = stack_forward(_fake_preds(), "quali", [2025], [2024], "linear",
                               min_train_races=3)
    ax = weights_plot(history)
    assert ax.get_ylim() == (0, 1)
    sim = RaceSimulator(source_model=FakeSource(), params=SIM, n_sims=FAST)
    sim.fit(_hand_race(1, 2.0))
    assert len(start_weight_plot(sim.start_curve()).get_lines()) == 1
