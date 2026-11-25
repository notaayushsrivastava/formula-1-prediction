"""Offline tests for Phase 10: M2 neural ranker, strength refactor, ablation report, plots."""
import matplotlib

matplotlib.use("Agg")  # draw off-screen; must come before pyplot is imported

import sys  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from synthetic_f1 import WORLD_PARAMS, make_dataset, make_strength_world  # noqa: E402

from pitwall.plotting import attribution_plot, embedding_plot  # noqa: E402
from pitwall.predict import (  # noqa: E402
    LABELS, OBSERVATIONS, BacktestResult, LockedSeasonError, NeuralRanker,
    StrengthFeatures, StrengthParams, walk_forward, write_report,
)
from pitwall.predict.backtest import (  # noqa: E402
    COMPARE_COLUMNS, strip_outcomes, summarise_by_model,
)
from pitwall.predict.models.base import check_consistency, pl_loglik_and_grad  # noqa: E402
from pitwall.predict.models.neural import (  # noqa: E402
    MONOTONE_FEATURES, pl_loglik_torch, validation_races,
)

TINY = {"n_seeds": 2, "max_epochs": 30, "patience": 10, "n_sims": 2000, "lr": 0.03}
FLAT = StrengthParams(q_car=1e-8, q_skill=1e-8, q_winter=1e-8, q_era=1e-8, r_quali=0.0225,
                      r_practice=0.09, mu_rookie=0.0, p_skill=0.1)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def data():
    return make_dataset()


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


def _quali(data, before=2026):
    rows = data[data["Moment"] == "quali"]
    return rows[rows["Year"] < before], rows[rows["RaceId"] == 19]  # first race of 2026


@pytest.fixture(scope="module")
def fitted(data):
    pytest.importorskip("torch")
    train, _ = _quali(data)
    return NeuralRanker(use_strength=False, **TINY).fit(train)


# ---------- loss ----------
def test_torch_plackett_luce_matches_numpy(torch):
    rng = np.random.default_rng(0)
    u = rng.normal(size=(3, 5))
    mask = np.array([[True] * 5, [True] * 3 + [False] * 2, [True] * 2 + [False] * 3])
    loglik, grad = pl_loglik_and_grad(u, mask)
    tensor = torch.tensor(u, dtype=torch.float64, requires_grad=True)
    value = pl_loglik_torch(torch, tensor, torch.tensor(mask))
    value.backward()
    assert float(value) == pytest.approx(loglik, abs=1e-9)
    assert np.allclose(tensor.grad.numpy(), grad, atol=1e-9)


# ---------- monotonicity ----------
def test_worse_grid_or_quali_gap_never_raises_a_score(data, fitted):
    _, race = _quali(data)
    base = fitted.seed_scores(race)
    for column, step in (("grid", 5.0), ("quali_gap_pct", 1.0),
                         ("teammate_quali_gap_pct", 0.5), ("practice_pace_gap_pct", 0.5)):
        worse = race.copy()
        worse[column] = worse[column] + step
        assert (fitted.seed_scores(worse) <= base + 1e-6).all(), column
    weights = fitted.coefficients()
    assert (weights[[f for f in weights.index if f in MONOTONE_FEATURES]] <= 0).all()


# ---------- what it learns ----------
def test_driver_contribution_tracks_true_skill(torch):
    world, truth = make_strength_world(params=FLAT, cars=[0.0] * 5)  # only skill matters
    pre = world[world["Moment"] == "pre"]
    model = NeuralRanker(use_strength=False, use_mlp=False, n_seeds=2, max_epochs=300,
                         patience=300, lr=0.05, weight_decay=1e-4, n_sims=2000).fit(pre)
    contribution = model.driver_contributions().drop("<unknown>")
    skill = pd.Series(truth["skills"])
    assert np.corrcoef(contribution[skill.index], -skill)[0, 1] > 0.7  # lower gap = faster


def test_unseen_driver_gets_the_unknown_embedding(data, fitted):
    _, race = _quali(data)
    rookie = race.copy()
    rookie.loc[rookie.index[0], "DriverId"] = "brand-new-rookie"
    ids = fitted.encode_drivers(rookie)
    assert ids[0] == 0 and (ids[1:] > 0).all()
    check_consistency(fitted.predict(strip_outcomes(rookie)))
    assert "<unknown>" in fitted.driver_contributions().index


def test_validation_split_is_the_newest_races(data, fitted):
    train, _ = _quali(data)
    held = validation_races(train)
    newest = train.drop_duplicates("RaceId").nlargest(len(held), "Date")["RaceId"]
    assert len(held) == 3 and set(held) == set(newest)
    older = train[~train["RaceId"].isin(held)]
    assert older["Date"].max() < train[train["RaceId"].isin(held)]["Date"].min()
    assert set(fitted.validation_races_) == set(held)


# ---------- consistency, determinism, explanations ----------
def test_consistent_deterministic_and_logged(data, fitted):
    train, race = _quali(data)
    pred = fitted.predict(strip_outcomes(race))
    check_consistency(pred)
    again = NeuralRanker(use_strength=False, **TINY).fit(train)
    pd.testing.assert_frame_equal(pred, again.predict(strip_outcomes(race)))
    params = fitted.params()
    assert len(params["best_epochs"]) == 2 and params["use_strength"] is False


def test_integrated_gradients_add_up_for_a_linear_model(data, torch):
    train, race = _quali(data)
    model = NeuralRanker(use_strength=False, use_mlp=False, **TINY).fit(train)
    attributions = model.integrated_gradients(race)
    average = race.copy()
    for i, feature in enumerate(model.features_):
        average[feature] = model.mean_[i]  # the baseline: training means
    gain = model.seed_scores(race).mean(axis=0) - model.seed_scores(average).mean(axis=0)
    assert np.allclose(attributions.sum(axis=1).to_numpy(), gain, atol=1e-4)


def test_strength_inputs_work_and_are_monotone(torch):
    world, _ = make_strength_world()
    rows = world[world["Moment"] == "quali"]
    model = NeuralRanker(use_strength=True, strength_params=WORLD_PARAMS, **TINY)
    model.fit(rows[rows["Year"] < 2023])
    check_consistency(model.predict(strip_outcomes(rows[rows["RaceId"] == 31])))
    weights = model.coefficients()
    assert weights["car_mean"] <= 0 and weights["skill_mean"] <= 0
    assert model.name == "M2-full"


def test_strength_features_refactor_adds_one_step_ahead_columns():
    world, _ = make_strength_world()
    rows = world[world["Moment"] == "pre"]
    features = StrengthFeatures(WORLD_PARAMS)
    train = features.fit(rows[rows["Year"] < 2023])
    assert {"car_mean", "skill_mean", "pace_sd"} <= set(train.columns)
    race = features.for_race(strip_outcomes(rows[rows["RaceId"] == 31]))
    assert race["pace_sd"].notna().all() and len(race) == 10


# ---------- safety ----------
class NeuralSpy(NeuralRanker):
    def __init__(self):
        super().__init__(use_strength=False, **TINY)
        self.seen = set()

    def predict(self, race):
        self.seen |= set(race.columns)
        return super().predict(race)


def test_predict_never_sees_labels_or_observations(data, torch):
    spy = NeuralSpy()
    preds = walk_forward([spy], data, "quali", seasons=[2025])
    assert preds["RaceId"].nunique() == 6
    assert not (set(LABELS) | set(OBSERVATIONS)) & spy.seen


def test_2026_lock_is_respected(data):
    with pytest.raises(LockedSeasonError):
        walk_forward([NeuralRanker(**TINY)], data, "quali", seasons=[2026])


def test_helpful_error_without_torch(data, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)  # pretend torch is not installed
    train, _ = _quali(data)
    with pytest.raises(ImportError, match=r"pitwall-f1\[nn\]") as info:
        NeuralRanker(**TINY).fit(train)
    assert "download.pytorch.org/whl/cpu" in str(info.value)


def test_variant_names():
    assert NeuralRanker().name == "M2-full"
    assert NeuralRanker(use_strength=False).name == "M2-full (no strength)"
    assert NeuralRanker(use_embeddings=False, use_mlp=False).name == "M2-linear"
    assert NeuralRanker(use_mlp=False).name == "M2-no-MLP"
    assert NeuralRanker(use_embeddings=False).name == "M2-no-emb"


# ---------- report and plots ----------
def test_ablation_section(tmp_path):
    rows = []
    for race in range(1, 21):
        for model, p_win in (("M2-full", [0.7, 0.1, 0.1, 0.1]), ("M2-linear", [0.25] * 4)):
            for k in range(4):
                rows.append({"Model": model, "Moment": "quali", "RaceId": race,
                             "DriverId": f"d{k}", "P_win": p_win[k], "P_podium": 0.75,
                             "P_points": 1.0, "P_dnf": 0.1, "ExpectedPosition": k + 1.0,
                             "Win": int(k == 0), "Podium": int(k < 3), "PointsFinish": 1,
                             "DNF": 0, "FinishOrder": k + 1})
    preds = pd.DataFrame(rows)
    result = BacktestResult(preds, summarise_by_model(preds),
                            pd.DataFrame(columns=COMPARE_COLUMNS), (2025,), 2, "M2-full", "t")
    text = write_report(result, tmp_path / "r.md", ablation=("M2-full", ["M2-linear"])).read_text()
    assert "## Ablation vs M2-full" in text and "(worse)" in text


def test_explanation_plots(data, fitted):
    _, race = _quali(data)
    ax = embedding_plot(fitted)
    assert len(ax.collections[0].get_offsets()) == 12
    ax = attribution_plot(fitted.integrated_gradients(race), top=5)
    assert len(ax.images) == 1 and len(ax.get_xticklabels()) == 5
