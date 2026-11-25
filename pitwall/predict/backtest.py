"""Walk-forward backtest: predict every test race using ONLY races dated before it.

For each test race (oldest first) the models are retrained every `retrain_every`
races on rows of the same moment with Date < that race's Date. This is checked in
code: any violation raises LeakageError. The race itself is passed to predict()
WITHOUT its labels or observations, so a model cannot peek even by accident.
Seasons in LOCKED_SEASONS (2026) are only run with final=True: tune on 2025, report
2026 once.

Phase 15: the experiment log v2 (experiments.py: ids, families, per-race loss files) and
gate_v2, the stricter gate every later phase uses (bootstrap CI on 2025, deflated score,
CPCV path median; DM test and block bootstrap as context). gate() is kept unchanged.
"""
from __future__ import annotations

import datetime as dt
import json
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from ..errors import DataNotAvailableError, PitwallError
from .dataset import DIAGNOSTICS, LABELS, MOMENTS, OBSERVATIONS
from .experiments import (
    PHASE, V2_COLUMNS, append_rows, data_hash, experiment_id, losses_dir_for, model_family,
    params_hash, write_race_losses,
)
from .metrics import METRIC_NAMES, paired_bootstrap, per_race_log_loss, race_loss_table, summarise
from .models.base import RaceModel, check_consistency

LOCKED_SEASONS = frozenset({2026})
REPORTS_DIR = Path("reports")
EXPERIMENTS_PATH = REPORTS_DIR / "experiments.csv"
LABEL_CONTEXT = ["RaceKey", "Year", "Date", "EventName", "Moment"]
LOG_COLUMNS_V1 = ["timestamp", "git", "model", "params", "moment", "seasons",
                  "retrain_every", *METRIC_NAMES]
LOG_COLUMNS = [*LOG_COLUMNS_V1, *V2_COLUMNS]  # v2 columns appended: old readers still work
COMPARE_COLUMNS = ["Moment", "Model", "Reference", "Metric", "Mean", "Low", "High", "Races"]


class Configured(Protocol):
    """Anything with a name and its settings (every RaceModel; logged configurations)."""

    name: str

    def params(self) -> dict[str, Any]:
        """Settings worth logging."""
        ...


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
    data_hashes: dict[str, str] = field(default_factory=dict)  # per moment (Phase 15)


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
    hashes = {m: moment_data_hash(data, m) for m in moments}
    result = BacktestResult(predictions, summary, compare_to_baseline(predictions, baseline),
                            seasons, retrain_every, baseline, git_hash(), hashes)
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


def moment_data_hash(data: pd.DataFrame, moment: str,
                     last_year: int = max(LOCKED_SEASONS) - 1) -> str:
    """Fingerprint of a moment's rows up to `last_year` (2025): same data = same hash."""
    rows = data[(data["Moment"] == moment) & (data["Year"] <= last_year)]
    return data_hash(rows)


def log_experiments(path: Path, result: BacktestResult, models: Sequence[Configured],
                    validation: str = "walk_forward", losses: bool = True) -> list[str]:
    """Append one row per (model, moment) to the experiment log (v2 columns included) and
    save each experiment's per-race losses next to it. Returns the experiment ids.
    Models not in `models` (e.g. merged prediction files) are logged with params '{}'."""
    params = {m.name: m.params() for m in models}
    rows, ids = [], []
    seasons = " ".join(map(str, result.seasons))
    stamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    for row in result.summary.to_dict("records"):
        name, moment = str(row["Model"]), str(row["Moment"])
        config = params.get(name, {})
        meta = experiment_meta(name, moment, config, validation, seasons,
                               result.data_hashes.get(moment, ""),
                               retrain_every=result.retrain_every)
        rows.append({"timestamp": stamp, "git": result.git, "model": name,
                     "params": json.dumps(config, sort_keys=True, default=str),
                     "moment": moment, "seasons": seasons,
                     "retrain_every": result.retrain_every,
                     **{n: row[n] for n in METRIC_NAMES}, **meta})
        ids.append(meta["experiment_id"])
        if losses:
            mine = result.predictions[(result.predictions["Model"] == name)
                                      & (result.predictions["Moment"] == moment)]
            write_race_losses(losses_dir_for(path), meta["experiment_id"],
                              race_loss_table(mine))
    append_rows(path, rows, LOG_COLUMNS)
    return ids


def experiment_meta(model: str, moment: str, config: dict[str, Any], validation: str,
                    seasons: str, data: str, embargo: int | str = "", blocks: int | str = "",
                    test_blocks: int | str = "",
                    retrain_every: int | str = "") -> dict[str, str]:
    """The v2 columns of one experiment row. The id is a content hash (no timestamp):
    the same experiment on the same data with the same version gets the same id."""
    from .. import __version__
    phash = params_hash(config)
    spec = {"model": model, "moment": moment, "params_hash": phash, "validation": validation,
            "seasons": seasons, "data": data, "embargo": embargo, "blocks": blocks,
            "test_blocks": test_blocks, "retrain_every": retrain_every,
            "version": __version__}
    return {"experiment_id": experiment_id(**spec), "family": model_family(model),
            "params_hash": phash, "phase": PHASE, "validation": validation,
            "embargo": str(embargo), "cpcv_blocks": str(blocks),
            "cpcv_test_blocks": str(test_blocks), "data_hash": data}


def _with_labels(prediction: pd.DataFrame, race: pd.DataFrame, model: str) -> pd.DataFrame:
    """Attach the race's real outcome (and context) to a model's prediction."""
    truth = race[["RaceId", "DriverId", *LABEL_CONTEXT, *LABELS]]
    merged = prediction.merge(truth, on=["RaceId", "DriverId"], how="left")
    merged.insert(0, "Model", model)
    return merged


# ---------- Phase 15: the gate every later phase uses ----------
GATE_SEASON = 2025
GATE_METRICS = ("win", "podium")
GATE_ALPHA = 0.05
MIN_TRIALS_FOR_DEFLATION = 10


@dataclass(frozen=True)
class Evidence:
    """The logged per-race losses of one model at one moment (no retraining needed)."""

    model: str
    moment: str
    walk_forward: pd.DataFrame               # race_loss_table: RaceId, Date, Year, n_rows, ...
    cpcv: pd.DataFrame | None = None         # per (Split, Block, RaceId) losses
    cpcv_blocks: int = 8
    cpcv_test_blocks: int = 2
    family: str = ""
    experiment_ids: tuple[str, ...] = ()
    data: str = ""

    def season(self, year: int) -> pd.DataFrame:
        """Walk-forward per-race losses of one season."""
        return self.walk_forward[self.walk_forward["Year"] == year]


def load_evidence(model: str, moment: str, log_path: Path | str = EXPERIMENTS_PATH,
                  embargo: int | None = None, season: int = GATE_SEASON,
                  data: str | None = None) -> Evidence:
    """The latest walk-forward run covering `season` (and the latest CPCV run, with the
    given embargo, on the same data) of a model at a moment, from the experiment log."""
    from .experiments import find_experiment, read_log, read_race_losses
    check_lock([season], final=False)
    log = read_log(log_path)
    directory = losses_dir_for(log_path)
    wf = find_experiment(log, model, moment, "walk_forward", data=data, season=season)
    if wf is None:
        raise DataNotAvailableError(
            f"No walk-forward run of '{model}' at '{moment}' covering {season} in the log "
            f"{log_path}. Run: pitwall-f1 validate")
    cv = find_experiment(log, model, moment, "cpcv", embargo=embargo, data=wf["data_hash"])
    cpcv = None if cv is None else read_race_losses(directory, cv["experiment_id"])
    ids = (wf["experiment_id"],) + (() if cv is None else (cv["experiment_id"],))
    return Evidence(model, moment, read_race_losses(directory, wf["experiment_id"]), cpcv,
                    int(cv["cpcv_blocks"]) if cv is not None else 8,
                    int(cv["cpcv_test_blocks"]) if cv is not None else 2,
                    str(wf["family"]), ids, str(wf["data_hash"]))


@dataclass(frozen=True)
class GateCheck:
    """One line of a gate decision."""

    name: str
    required: bool
    passed: bool
    detail: str


@dataclass(frozen=True)
class GateDecision:
    """gate_v2's answer, with the reason for every check."""

    candidate: str
    incumbent: str
    moment: str
    checks: tuple[GateCheck, ...]
    p_value: float      # one-sided paired-bootstrap p on the primary metric (for the FDR)
    n_trials: int

    @property
    def passed(self) -> bool:
        """True only if every REQUIRED check passed."""
        return all(c.passed for c in self.checks if c.required)

    def reasons(self) -> list[str]:
        """Plain-English reason per check ('required' or 'context')."""
        out = []
        for c in self.checks:
            role = "required" if c.required else "context"
            mark = "PASS" if c.passed else "FAIL"
            out.append(f"[{mark}, {role}] {c.name}: {c.detail}")
        return out

    def verdict(self) -> str:
        """The decision in one sentence."""
        if self.passed:
            return (f"{self.candidate} PASSES the gate against {self.incumbent} at "
                    f"'{self.moment}': every required check passed.")
        failed = [c.name for c in self.checks if c.required and not c.passed]
        return (f"{self.candidate} does NOT pass the gate against {self.incumbent} at "
                f"'{self.moment}' (failed: {'; '.join(failed)}). Keep the incumbent; move "
                "the candidate to lab/.")

    def to_markdown(self) -> list[str]:
        """A report section."""
        rows = [[c.name, "required" if c.required else "context",
                 "pass" if c.passed else "fail", c.detail] for c in self.checks]
        table = ["| Check | Role | Result | Detail |", "|---|---|---|---|"]
        table += ["| " + " | ".join(r) + " |" for r in rows]
        return [f"### Gate: {self.candidate} vs {self.incumbent} ({self.moment})", "",
                *table, "", f"**Decision:** {self.verdict()}", ""]


def gate_v2(candidate: Evidence, incumbent: Evidence, n_trials: int = 1,
            season: int = GATE_SEASON, metrics: Sequence[str] = GATE_METRICS,
            alpha: float = GATE_ALPHA, min_trials: int = MIN_TRIALS_FOR_DEFLATION,
            n_boot: int = 10_000, block: int = 3, seed: int = 0) -> GateDecision:
    """Keep a candidate only if, on `season` (2025):

    1. REQUIRED: the paired race-level bootstrap 95% CI of (candidate - incumbent) log
       loss lies entirely below 0, for every metric in `metrics` (win and podium);
    2. REQUIRED once n_trials >= min_trials (10): the deflated p-value of the per-race
       improvement on the first metric is below alpha;
    3. REQUIRED: the candidate's median CPCV path log loss is below the incumbent's;
    and shows the Diebold-Mariano (HLN) test and a block bootstrap (3 races) as context.
    """
    from .significance import (
        block_bootstrap, bootstrap_p_value, deflated_score, diebold_mariano,
    )
    from .validation import path_losses

    check_lock([season], final=False)  # the gate never looks at 2026
    if candidate.moment != incumbent.moment:
        raise ValueError("Candidate and incumbent come from different moments.")
    mine, theirs = candidate.season(season), incumbent.season(season)
    if mine.empty or theirs.empty:
        raise DataNotAvailableError(f"No walk-forward losses for {season} to compare.")
    order = mine.set_index("RaceId")["Date"]
    checks: list[GateCheck] = []
    series = {kind: (mine.set_index("RaceId")[kind], theirs.set_index("RaceId")[kind])
              for kind in metrics}
    for kind, (a, b) in series.items():
        res = paired_bootstrap(a, b, n_boot=n_boot, seed=seed)
        checks.append(GateCheck(
            f"{kind} LL: 95% CI on {season} entirely below 0", True, res.high < 0,
            f"{res.mean:+.4f} [{res.low:+.4f}, {res.high:+.4f}] over {res.n_races} races"))
    primary = metrics[0]
    a, b = series[primary]
    both = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
    improvement = (both["b"] - both["a"]).to_numpy(float)
    deflated = deflated_score(improvement, n_trials)
    needed = n_trials >= min_trials
    note = "" if needed else f" (required once >= {min_trials} trials; now {n_trials})"
    checks.append(GateCheck(f"{primary} LL: deflated p < {alpha}", needed,
                            deflated.p_value < alpha, deflated.explain() + note))
    checks.append(_cpcv_check(candidate, incumbent, primary, path_losses))
    dm = diebold_mariano(a.reindex(both.index) - b.reindex(both.index))
    checks.append(GateCheck(f"{primary} LL: Diebold-Mariano (HLN)", False,
                            dm.p_better < alpha,
                            f"statistic {dm.statistic:+.2f}, one-sided p = {dm.p_better:.3f} "
                            f"(two-sided {dm.p_value:.3f}), {dm.n} races"))
    blocked = block_bootstrap(a, b, block=block, n_boot=n_boot, seed=seed, order=order)
    checks.append(GateCheck(f"{primary} LL: block bootstrap ({block} races)", False,
                            blocked.high < 0,
                            f"{blocked.mean:+.4f} [{blocked.low:+.4f}, {blocked.high:+.4f}]"))
    p_value = bootstrap_p_value(a, b, n_boot=n_boot, seed=seed)
    return GateDecision(candidate.model, incumbent.model, candidate.moment, tuple(checks),
                        p_value, n_trials)


def _cpcv_check(candidate: Evidence, incumbent: Evidence, kind: str,
                path_losses: object) -> GateCheck:
    name = f"{kind} LL: CPCV median path better"
    if candidate.cpcv is None or incumbent.cpcv is None:
        missing = candidate.model if candidate.cpcv is None else incumbent.model
        return GateCheck(name, True, False, f"no CPCV run logged for {missing}: "
                         "run pitwall-f1 validate")
    same = (set(candidate.cpcv["RaceId"]) == set(incumbent.cpcv["RaceId"])
            and (candidate.cpcv_blocks, candidate.cpcv_test_blocks)
            == (incumbent.cpcv_blocks, incumbent.cpcv_test_blocks))
    if not same:
        return GateCheck(name, True, False, "the two CPCV runs used different races or "
                         "blocks: not comparable")
    blocks, tests = candidate.cpcv_blocks, candidate.cpcv_test_blocks
    mine = path_losses(candidate.cpcv, blocks, tests, kind)  # type: ignore[operator]
    theirs = path_losses(incumbent.cpcv, blocks, tests, kind)  # type: ignore[operator]
    wins = int(np.sum(mine < theirs))
    return GateCheck(name, True, float(np.median(mine)) < float(np.median(theirs)),
                     f"median {np.median(mine):.4f} vs {np.median(theirs):.4f}; better on "
                     f"{wins} of {len(mine)} paths")
