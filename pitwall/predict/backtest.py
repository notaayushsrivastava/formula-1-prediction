"""Walk-forward backtest: predict every test race using ONLY races dated before it.

For each test race (oldest first) the models are retrained every `retrain_every`
races on rows of the same moment with Date < that race's Date. This is checked in
code: any violation raises LeakageError. The race itself is passed to predict()
WITHOUT its labels or observations, so a model cannot peek even by accident.
Seasons in LOCKED_SEASONS (2026) are only run with final=True: tune on 2025, report
2026 once.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..errors import DataNotAvailableError, PitwallError
from .dataset import DIAGNOSTICS, LABELS, MOMENTS, OBSERVATIONS
from .metrics import METRIC_NAMES, paired_bootstrap, per_race_log_loss, summarise
from .models.base import RaceModel, check_consistency

LOCKED_SEASONS = frozenset({2026})
REPORTS_DIR = Path("reports")
EXPERIMENTS_PATH = REPORTS_DIR / "experiments.csv"
LABEL_CONTEXT = ["RaceKey", "Year", "Date", "EventName", "Moment"]
LOG_COLUMNS = ["timestamp", "git", "model", "params", "moment", "seasons", "retrain_every",
               *METRIC_NAMES]
COMPARE_COLUMNS = ["Moment", "Model", "Reference", "Metric", "Mean", "Low", "High", "Races"]


class LockedSeasonError(PitwallError):
    """A locked (final test) season was requested without final=True."""


class LeakageError(PitwallError):
    """Training data would include information from the race being predicted (or later)."""


@dataclass(frozen=True)
class BacktestResult:
    """Everything a backtest produced."""

    predictions: pd.DataFrame   # one row per model x race x driver, labels attached
    summary: pd.DataFrame       # one row per (Moment, Model): every metric
    comparisons: pd.DataFrame   # each model vs the baseline: paired-bootstrap CIs
    seasons: tuple[int, ...]
    retrain_every: int
    baseline: str
    git: str


@dataclass(frozen=True)
class GateResult:
    """Does a new model beat the best previous model? (both win AND podium log loss)"""

    candidate: str
    comparisons: pd.DataFrame        # candidate minus best previous, per moment and metric
    passed_moments: tuple[str, ...]  # moments where BOTH intervals are entirely below 0

    @property
    def kept(self) -> bool:
        """Kept if it clearly wins at one moment or more."""
        return bool(self.passed_moments)

    def verdict(self) -> str:
        """The decision in plain words."""
        if self.kept:
            return (f"{self.candidate} is KEPT: clearly better than the best previous model "
                    f"on both win and podium log loss at: {', '.join(self.passed_moments)}.")
        return (f"{self.candidate} is NOT kept as a model: at no moment are both its win and "
                "podium log loss clearly better than the best previous model. It stays "
                "available as a feature source only.")


# ---------- guards ----------
def check_lock(seasons: Iterable[int], final: bool) -> None:
    """Raise unless locked seasons are asked for deliberately with final=True."""
    locked = sorted(LOCKED_SEASONS & {int(s) for s in seasons})
    if locked and not final:
        raise LockedSeasonError(
            f"Season(s) {locked} are the final test. Tune and evaluate on 2025; run "
            f"{locked} ONCE at the very end with final=True.")


def check_no_future(train: pd.DataFrame, race_date: pd.Timestamp, moment: str) -> None:
    """Raise LeakageError if any training row is not strictly earlier, or another moment."""
    if train.empty:
        raise DataNotAvailableError(f"No training rows before {race_date.date()}.")
    if (train["Date"] >= race_date).any():
        raise LeakageError(f"Training rows dated on/after the test race ({race_date.date()}).")
    if (train["Moment"] != moment).any():
        raise LeakageError(f"Training rows from another moment than '{moment}'.")


def strip_outcomes(rows: pd.DataFrame) -> pd.DataFrame:
    """The rows a model may see at prediction time: no labels, observations or diagnostics."""
    hidden = [*LABELS, *OBSERVATIONS, *DIAGNOSTICS]
    return rows.drop(columns=[c for c in hidden if c in rows.columns])


# ---------- the walk ----------
def walk_forward(models: Sequence[RaceModel], data: pd.DataFrame, moment: str,
                 seasons: Iterable[int], retrain_every: int = 2,
                 first_train_year: int = 2018, final: bool = False) -> pd.DataFrame:
    """Predictions (with labels) of every model for every race of `seasons` at `moment`."""
    seasons = tuple(int(s) for s in seasons)
    check_lock(seasons, final)
    if retrain_every < 1:
        raise ValueError("retrain_every must be at least 1.")
    rows = data[data["Moment"] == moment]
    tests = rows[rows["Year"].isin(seasons)].drop_duplicates("RaceId").sort_values("Date")
    if tests.empty:
        raise DataNotAvailableError(f"No '{moment}' rows for seasons {seasons}.")
    out: list[pd.DataFrame] = []
    cutoff = pd.Timestamp.min
    for i, test in enumerate(tests.itertuples(index=False)):
        if i % retrain_every == 0:
            train = rows[(rows["Date"] < test.Date) & (rows["Year"] >= first_train_year)]
            check_no_future(train, test.Date, moment)
            for model in models:
                model.fit(train)
            cutoff = train["Date"].max()
        if cutoff >= test.Date:  # never true by construction; kept as a guard
            raise LeakageError(f"Models trained up to {cutoff.date()} for {test.Date.date()}.")
        race = rows[rows["RaceId"] == test.RaceId]
        for model in models:
            prediction = model.predict(strip_outcomes(race))
            check_consistency(prediction)
            out.append(_with_labels(prediction, race, model.name))
    return pd.concat(out, ignore_index=True)


def run_backtest(models: Sequence[RaceModel], data: pd.DataFrame,
                 seasons: Iterable[int] = (2025,), moments: Iterable[str] = MOMENTS,
                 retrain_every: int = 2, first_train_year: int = 2018, final: bool = False,
                 log_path: Path | str | None = EXPERIMENTS_PATH,
                 baseline: str | None = None) -> BacktestResult:
    """Walk forward at every moment, score every model, compare each to the baseline
    (default: the first model, normally B0) and log the run."""
    seasons = tuple(int(s) for s in seasons)
    baseline = baseline or models[0].name
    predictions = pd.concat(
        [walk_forward(models, data, m, seasons, retrain_every, first_train_year, final)
         for m in moments], ignore_index=True)
    summary = summarise_by_model(predictions)
    result = BacktestResult(predictions, summary, compare_to_baseline(predictions, baseline),
                            seasons, retrain_every, baseline, git_hash())
    if log_path is not None:
        log_experiments(Path(log_path), result, models)
    return result


# ---------- scoring and comparing ----------
def summarise_by_model(predictions: pd.DataFrame) -> pd.DataFrame:
    """One row of metrics per (Moment, Model)."""
    rows = [{"Moment": moment, "Model": model, **summarise(group)}
            for (moment, model), group in predictions.groupby(["Moment", "Model"], sort=False)]
    return pd.DataFrame(rows)


def compare_models(predictions: pd.DataFrame, model: str, reference: str,
                   kinds: Iterable[str] = ("win", "podium")) -> pd.DataFrame:
    """Paired bootstrap per moment of per-race log loss: model minus reference."""
    rows = []
    for moment, group in predictions.groupby("Moment", sort=False):
        mine, theirs = group[group["Model"] == model], group[group["Model"] == reference]
        if mine.empty or theirs.empty:
            continue
        for kind in kinds:
            res = paired_bootstrap(per_race_log_loss(mine, kind),
                                   per_race_log_loss(theirs, kind))
            rows.append({"Moment": moment, "Model": model, "Reference": reference,
                         "Metric": f"{kind}_logloss", "Mean": res.mean, "Low": res.low,
                         "High": res.high, "Races": res.n_races})
    return pd.DataFrame(rows, columns=COMPARE_COLUMNS)


def compare_to_baseline(predictions: pd.DataFrame, baseline: str) -> pd.DataFrame:
    """Every other model compared with the baseline (negative Mean = better)."""
    others = [m for m in predictions["Model"].unique() if m != baseline]
    parts = [compare_models(predictions, model, baseline) for model in others]
    parts = [p for p in parts if not p.empty]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=COMPARE_COLUMNS)


def best_previous(summary: pd.DataFrame, moment: str, previous: Sequence[str],
                  metric: str = "win_logloss") -> str | None:
    """The previous model with the lowest `metric` at a moment (None if none ran)."""
    rows = summary[(summary["Moment"] == moment) & summary["Model"].isin(previous)]
    if rows.empty:
        return None
    return str(rows.loc[rows[metric].idxmin(), "Model"])


def gate(result: BacktestResult, candidate: str, previous: Sequence[str]) -> GateResult:
    """Compare a candidate with the best previous model at each moment (by win log loss).

    A moment passes only if the 95% intervals of BOTH win and podium log loss
    (candidate minus best previous) lie entirely below 0.
    """
    parts, passed = [], []
    for moment in result.summary["Moment"].unique():
        reference = best_previous(result.summary, str(moment), previous)
        if reference is None:
            continue
        moment_preds = result.predictions[result.predictions["Moment"] == moment]
        comparison = compare_models(moment_preds, candidate, reference)
        if comparison.empty:
            continue
        parts.append(comparison)
        if len(comparison) == 2 and (comparison["High"] < 0).all():
            passed.append(str(moment))
    table = (pd.concat(parts, ignore_index=True) if parts
             else pd.DataFrame(columns=COMPARE_COLUMNS))
    return GateResult(candidate, table, tuple(passed))


# ---------- experiment log ----------
def git_hash() -> str:
    """Short hash of the current git commit, or 'unknown' outside a repository."""
    try:
        done = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=5, check=True)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return done.stdout.strip() or "unknown"


def log_experiments(path: Path, result: BacktestResult, models: Sequence[RaceModel]) -> None:
    """Append one CSV row per (model, moment) to the experiment log."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    params = {m.name: json.dumps(m.params(), sort_keys=True) for m in models}
    stamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_COLUMNS)
        if new_file:
            writer.writeheader()
        for row in result.summary.to_dict("records"):
            writer.writerow({
                "timestamp": stamp, "git": result.git, "model": row["Model"],
                "params": params.get(row["Model"], "{}"), "moment": row["Moment"],
                "seasons": " ".join(map(str, result.seasons)),
                "retrain_every": result.retrain_every,
                **{name: row[name] for name in METRIC_NAMES}})


def _with_labels(prediction: pd.DataFrame, race: pd.DataFrame, model: str) -> pd.DataFrame:
    """Attach the race's real outcome (and context) to a model's prediction."""
    truth = race[["RaceId", "DriverId", *LABEL_CONTEXT, *LABELS]]
    merged = prediction.merge(truth, on=["RaceId", "DriverId"], how="left")
    merged.insert(0, "Model", model)
    return merged
