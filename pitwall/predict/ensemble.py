"""Online stacking: combine models' predictions with weights fitted on EARLIER races only.

Methods: 'equal' (the hard-to-beat baseline), 'linear' (weighted average of the
position matrices), 'log' (weighted geometric mean, then Sinkhorn). Weights live on the
simplex (softmax), fitted by position log loss with a small pull toward equal weights.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.special import logsumexp, softmax

from .backtest import (
    BacktestResult, LeakageError, check_lock, compare_to_baseline, git_hash, summarise_by_model,
)
from .dataset import LABELS
from .metrics import BENCHMARKS
from .models.base import position_matrix, prediction_frame, sinkhorn
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
