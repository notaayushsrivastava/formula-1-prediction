"""Online stacking: combine models' predictions with weights fitted on EARLIER races only.

Methods: 'equal' (the hard-to-beat baseline), 'linear' (weighted average of the
position matrices), 'log' (weighted geometric mean, then Sinkhorn). Weights live on the
simplex (softmax), fitted by position log loss with a small pull toward equal weights.

Phase 15: `pool_forward` names a walk-forward pool ('POOL-log B0+B1'), and `PooledModel`
is the same pool as a RaceModel for CPCV, where "earlier races" do not exist: its
weights are fitted on inner out-of-fold predictions of the TRAINING races only.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.special import logsumexp, softmax

from .backtest import (
    BacktestResult, LeakageError, _with_labels, check_lock, compare_to_baseline, git_hash,
    strip_outcomes, summarise_by_model,
)
from .dataset import LABELS
from .metrics import BENCHMARKS
from .models.base import RaceModel, moment_of, position_matrix, prediction_frame, sinkhorn
from .report import markdown_table

METHODS = ("equal", "linear", "log")
L2_TO_EQUAL = 0.05     # pull toward equal weights (few races = noisy weights)
MIN_TRAIN_RACES = 5
EPS = 1e-9
CONTEXT = ["RaceKey", "Year", "Date", "EventName", "Moment"]


@dataclass
class RaceStack:
    """Every model's prediction for one race, aligned by driver."""

    race_id: int
    date: pd.Timestamp
    year: int
    models: list[str]
    frame: pd.DataFrame      # ids, context and labels (drivers sorted by DriverId)
    matrices: np.ndarray     # (models, n, n) position probabilities
    dnf: np.ndarray          # (models, n)
    order: np.ndarray        # actual position - 1, per driver


def common_models(rows: pd.DataFrame) -> list[str]:
    """Models that predicted every race in `rows` (order of first appearance)."""
    total = rows["RaceId"].nunique()
    counts = rows.groupby("Model")["RaceId"].nunique()
    return [m for m in rows["Model"].unique() if counts[m] == total]


def build_stacks(preds: pd.DataFrame, moment: str,
                 models: Sequence[str] | None = None) -> list[RaceStack]:
    """One RaceStack per race of a moment (ensemble rows are ignored), oldest first."""
    rows = preds[(preds["Moment"] == moment) & ~preds["Model"].str.startswith("ENS")]
    names = list(models) if models is not None else common_models(rows)
    stacks = []
    for race_id, race in rows[rows["Model"].isin(names)].groupby("RaceId"):
        parts = {m: g.sort_values("DriverId") for m, g in race.groupby("Model")}
        if set(parts) != set(names):
            continue
        first = parts[names[0]]
        drivers = first["DriverId"].tolist()
        if any(p["DriverId"].tolist() != drivers for p in parts.values()):
            continue
        n = len(first)
        stacks.append(RaceStack(
            int(race_id), pd.Timestamp(first["Date"].iloc[0]), int(first["Year"].iloc[0]),
            names, first[["RaceId", "DriverId", "Driver", *CONTEXT, *LABELS]]
            .reset_index(drop=True),
            np.stack([position_matrix(parts[m])[:, :n] for m in names]),
            np.stack([parts[m]["P_dnf"].to_numpy(float) for m in names]),
            first["FinishOrder"].to_numpy(int) - 1))
    return sorted(stacks, key=lambda s: (s.date, s.race_id))


def pool(matrices: np.ndarray, weights: np.ndarray, method: str) -> np.ndarray:
    """Combine (models, n, n) matrices; Sinkhorn makes every row and column sum to 1."""
    if method == "log":
        mixed = np.exp(np.tensordot(weights, np.log(np.clip(matrices, EPS, 1.0)), axes=1))
        return sinkhorn(mixed)
    return sinkhorn(np.tensordot(weights, matrices, axes=1))


def fit_weights(stacks: Sequence[RaceStack], method: str,
                l2: float = L2_TO_EQUAL) -> np.ndarray:
    """Simplex weights minimising mean position log loss (+ l2 pull toward equal)."""
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}. Use one of {METHODS}.")
    if not stacks:
        raise ValueError("No earlier races to fit weights on.")
    count = stacks[0].matrices.shape[0]
    equal = np.full(count, 1.0 / count)
    if method == "equal" or count == 1:
        return equal

    def loss(theta: np.ndarray) -> float:
        w = softmax(theta)
        races = np.mean([_race_loss(s, w, method) for s in stacks])
        return float(races + l2 * np.sum((w - equal) ** 2))

    result = optimize.minimize(loss, np.zeros(count), method="L-BFGS-B")
    return softmax(result.x)


def check_past_only(past: Sequence[RaceStack], date: pd.Timestamp) -> None:
    """Raise LeakageError unless every training race is dated strictly before `date`."""
    if any(s.date >= date for s in past):
        raise LeakageError(f"Ensemble weights for {date.date()} would use that race or later.")


def stack_forward(preds: pd.DataFrame, moment: str, eval_seasons: Iterable[int],
                  warmup_seasons: Iterable[int], method: str = "linear",
                  l2: float = L2_TO_EQUAL, min_train_races: int = MIN_TRAIN_RACES,
                  models: Sequence[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict every race (warm-up and eval seasons) that has >= min_train_races earlier
    races, with weights fitted on those earlier races only. Eval=True marks eval seasons.

    Returns (predictions with labels, weights history).
    """
    evals = {int(s) for s in eval_seasons}
    seasons = evals | {int(s) for s in warmup_seasons}
    check_lock(seasons, final=False)
    stacks = [s for s in build_stacks(preds, moment, models) if s.year in seasons]
    name = f"ENS-{method}"
    out, history = [], []
    for stack in stacks:
        past = [s for s in stacks if s.date < stack.date]
        check_past_only(past, stack.date)
        if len(past) < min_train_races:
            continue
        weights = fit_weights(past, method, l2)
        frame = prediction_frame(stack.frame, pool(stack.matrices, weights, method),
                                 weights @ stack.dnf)
        out.append(_with_truth(frame, stack.frame, name, stack.year in evals))
        history.append({"RaceId": stack.race_id, "Date": stack.date, "Year": stack.year,
                        "Moment": moment, "Method": method, "TrainRaces": len(past),
                        **dict(zip(stack.models, weights))})
    table = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    return table, pd.DataFrame(history)


def result_from_predictions(preds: pd.DataFrame, baseline: str, seasons: Iterable[int],
                            retrain_every: int = 2) -> BacktestResult:
    """A BacktestResult (summary + comparisons) for the given seasons of a predictions table."""
    chosen = preds[preds["Year"].isin(list(seasons))].reset_index(drop=True)
    return BacktestResult(chosen, summarise_by_model(chosen), compare_to_baseline(chosen, baseline),
                          tuple(int(s) for s in seasons), retrain_every, baseline, git_hash())


def choose_pipelines(summary: pd.DataFrame) -> dict[str, str]:
    """Per moment: the pipeline with the lowest win log loss (ties: podium log loss)."""
    choices = {}
    for moment, rows in summary.groupby("Moment", sort=False):
        best = rows.sort_values(["win_logloss", "podium_logloss"]).iloc[0]
        choices[str(moment)] = str(best["Model"])
    return choices


def ensemble_section(summary: pd.DataFrame, names: Sequence[str]) -> list[str]:
    """Markdown: every ensemble pipeline per moment."""
    rows = []
    for row in summary[summary["Model"].isin(names)].to_dict("records"):
        rows.append([row["Moment"], row["Model"], f"{row['win_logloss']:.4f}",
                     f"{row['podium_logloss']:.4f}", f"{row['points_logloss']:.4f}",
                     f"{row['dnf_logloss']:.4f}", f"{row['win_ece']:.4f}"])
    header = ["Moment", "Pipeline", "Win LL", "Podium LL", "Points LL", "DNF LL", "Win ECE"]
    return ["## Ensembles (online stacking, 2025)", "",
            "Weights fitted only on earlier races (warm-up: 2024). '+cal' = online temperature "
            "and Platt calibration.", "", markdown_table(header, rows), ""]


def choice_section(choices: dict[str, str], summary: pd.DataFrame) -> list[str]:
    """Markdown: the chosen pipeline per moment and plain verdicts vs the targets."""
    lines = ["## Final pipeline per moment", ""]
    for moment, name in choices.items():
        row = summary[(summary["Moment"] == moment) & (summary["Model"] == name)].iloc[0]
        parts = []
        for key, label in (("win_logloss", "win"), ("podium_logloss", "podium")):
            word = "beats" if row[key] < BENCHMARKS[key] else "does NOT beat"
            parts.append(f"{label} LL {row[key]:.4f} {word} the {BENCHMARKS[key]} target")
        lines.append(f"- **{moment}**: {name} — " + "; ".join(parts) + ".")
    return lines + [""]


# ---------- Phase 15: named pools ----------
INNER_FOLDS = 3
INNER_EMBARGO = 1


def short_name(model: str) -> str:
    """'B1 logistic' -> 'B1', 'M1-pure strength' -> 'M1-pure'."""
    return model.split()[0]


def pool_name(members: Sequence[str], method: str, l2: float = L2_TO_EQUAL) -> str:
    """'POOL-log B0+B1' (the l2 is added when it is not the default)."""
    name = f"POOL-{method} " + "+".join(short_name(m) for m in members)
    return name if l2 == L2_TO_EQUAL else f"{name} (l2={l2:g})"


def pool_params(members: Sequence[str], method: str, l2: float) -> dict[str, Any]:
    """The configuration of a pool, identical for walk-forward and CPCV (one trial)."""
    return {"members": list(members), "method": method, "l2": float(l2)}


def pool_forward(preds: pd.DataFrame, moment: str, members: Sequence[str], method: str,
                 eval_seasons: Iterable[int], warmup_seasons: Iterable[int],
                 l2: float = L2_TO_EQUAL,
                 min_train_races: int = MIN_TRAIN_RACES) -> tuple[pd.DataFrame, pd.DataFrame]:
    """stack_forward for a chosen set of members, renamed to pool_name(...)."""
    table, history = stack_forward(preds, moment, eval_seasons, warmup_seasons, method, l2,
                                   min_train_races, members)
    if not table.empty:
        table["Model"] = pool_name(members, method, l2)
    return table, history


class PooledModel:
    """A pool of member models as one RaceModel (for CPCV).

    fit(train): the training races are cut into `inner_folds` consecutive folds; for each
    fold the members are fitted on the other training races (minus `inner_embargo` races
    after the fold) and predict the fold. The pool weights are fitted on those inner
    out-of-fold predictions, then the members are refitted on all training races.
    Nothing outside `train` is ever used.
    """

    def __init__(self, factories: Sequence[Callable[[], RaceModel]], method: str = "linear",
                 l2: float = L2_TO_EQUAL, inner_folds: int = INNER_FOLDS,
                 inner_embargo: int = INNER_EMBARGO,
                 cache: dict[tuple[str, str, int], RaceModel] | None = None) -> None:
        if method not in ("linear", "log", "equal"):
            raise ValueError(f"Unknown method {method!r}.")
        self.factories = list(factories)
        self.members = [f().name for f in self.factories]
        self.method, self.l2 = method, l2
        self.inner_folds, self.inner_embargo = inner_folds, inner_embargo
        self.cache = cache  # shared between pools: same member + same rows = fitted once
        self.name = pool_name(self.members, method, l2)

    def params(self) -> dict[str, Any]:
        """Same configuration dict as the walk-forward pool (counted as one trial)."""
        return pool_params(self.members, self.method, self.l2)

    def fit(self, train: pd.DataFrame) -> PooledModel:
        """Inner out-of-fold weights, then members refitted on all of `train`."""
        moment = moment_of(train)
        races = (train.drop_duplicates("RaceId").sort_values(["Date", "RaceId"])["RaceId"]
                 .to_numpy())
        if len(races) < 2 * self.inner_folds:
            raise ValueError("Too few training races for the inner folds.")
        oof = []
        position = {r: i for i, r in enumerate(races)}
        for fold in np.array_split(races, self.inner_folds):
            last = max(position[r] for r in fold)
            blocked = set(fold) | {r for r in races
                                   if last < position[r] <= last + self.inner_embargo}
            fit_rows = train[~train["RaceId"].isin(blocked)]
            members = [self._fitted(f, fit_rows) for f in self.factories]
            for race_id in fold:
                race = train[train["RaceId"] == race_id]
                for member in members:
                    oof.append(_with_labels(member.predict(strip_outcomes(race)), race,
                                            member.name))
        stacks = build_stacks(pd.concat(oof, ignore_index=True), moment, self.members)
        self.weights_ = fit_weights(stacks, self.method, self.l2)
        self.members_ = [self._fitted(f, train) for f in self.factories]
        return self

    def _fitted(self, factory: Callable[[], RaceModel], rows: pd.DataFrame) -> RaceModel:
        """A member fitted on `rows`, reused from the shared cache when possible."""
        if self.cache is None:
            return factory().fit(rows)
        model = factory()
        key = (model.name, moment_of(rows), hash(tuple(sorted(rows["RaceId"].unique()))))
        if key not in self.cache:
            self.cache[key] = model.fit(rows)
        return self.cache[key]

    def predict(self, race: pd.DataFrame) -> pd.DataFrame:
        """Every member's prediction, pooled with the fitted weights."""
        n = len(race)
        preds = [m.predict(race) for m in self.members_]
        matrices = np.stack([position_matrix(p)[:, :n] for p in preds])
        dnf = np.stack([p["P_dnf"].to_numpy(float) for p in preds])
        return prediction_frame(race, pool(matrices, self.weights_, self.method),
                                self.weights_ @ dnf)


def _race_loss(stack: RaceStack, w: np.ndarray, method: str) -> float:
    """Mean -log P(actual position) for one race under the pooled matrix.

    The log pool is row-normalised here (fast); predictions use full Sinkhorn.
    """
    idx = np.arange(len(stack.order))
    if method == "log":
        mixed = np.tensordot(w, np.log(np.clip(stack.matrices, EPS, 1.0)), axes=1)
        rows = mixed - logsumexp(mixed, axis=1, keepdims=True)
        return -float(np.mean(rows[idx, stack.order]))
    picked = np.tensordot(w, stack.matrices[:, idx, stack.order], axes=1)
    return -float(np.mean(np.log(np.clip(picked, EPS, 1.0))))


def _with_truth(frame: pd.DataFrame, truth: pd.DataFrame, name: str,
                is_eval: bool) -> pd.DataFrame:
    out = frame.merge(truth[["RaceId", "DriverId", *CONTEXT, *LABELS]],
                      on=["RaceId", "DriverId"], how="left")
    out.insert(0, "Model", name)
    out["Eval"] = is_eval
    return out
