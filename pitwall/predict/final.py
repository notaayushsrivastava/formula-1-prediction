"""Phase 12: the ONE final evaluation on the locked 2026 season.

Everything was tuned on 2018-2025. After the last 2026 race (Abu Dhabi, 6 Dec 2026)
`run_final_2026` runs the frozen pipeline (reports/final_config.json) on every 2026
race, walk-forward, with final=True, and writes reports/final_2026.md. A lock file,
reports/FINAL_RUN.json, records that it happened; a second run needs force=True and
is labelled "RE-RUN — not a clean final test", because looking at the test set twice
and changing anything in between would make it a tuning set.
"""
from __future__ import annotations

import datetime as dt
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from typing import cast

import pandas as pd

from ..errors import DataNotAvailableError, PitwallError
from .backtest import (
    REPORTS_DIR, BacktestResult, _with_labels, compare_to_baseline, git_hash, summarise_by_model,
)
from .metrics import BENCHMARKS
from .pipeline import FinalPipeline, MomentChoice, PipelineConfig
from .records import config_hash
from .report import render_report

FINAL_SEASON = 2026
FINAL_REPORT = REPORTS_DIR / "final_2026.md"
LOCK_PATH = REPORTS_DIR / "FINAL_RUN.json"
RERUN_LABEL = "RE-RUN — not a clean final test"
PIPELINE_NAME = "Final pipeline"
BASELINE = "B0 grid slot"


class FinalRunError(PitwallError):
    """The final evaluation may not run (season unfinished, already run, not confirmed)."""


@dataclass(frozen=True)
class FinalOutcome:
    """What the final run wrote."""

    report: Path
    lock: Path
    result: BacktestResult
    rerun: bool


def check_season_finished(races: pd.DataFrame, season: int = FINAL_SEASON) -> None:
    """Raise unless every race of `season` has results. races: Year, Round, EventName,
    HasResults (e.g. ToppedUpHistory.race_table())."""
    season_races = races[races["Year"] == season]
    if season_races.empty:
        raise FinalRunError(f"No {season} races in the data.")
    missing = season_races[~season_races["HasResults"].astype(bool)]
    if not missing.empty:
        names = ", ".join(f"R{r.Round} {r.EventName}" for r in missing.itertuples(index=False))
        raise FinalRunError(
            f"The {season} season is not finished: {len(missing)} race(s) without results "
            f"({names}). The final evaluation runs once, after the last race.")


def read_lock(path: Path | str = LOCK_PATH) -> dict | None:
    """The lock file's contents, or None if the final run never happened."""
    target = Path(path)
    return json.loads(target.read_text(encoding="utf-8")) if target.exists() else None


def run_final_2026(data: pd.DataFrame, races: pd.DataFrame, config: PipelineConfig,
                   history: pd.DataFrame | None = None, confirmed: bool = False,
                   force: bool = False, report_path: Path | str = FINAL_REPORT,
                   lock_path: Path | str = LOCK_PATH, plots: bool = True,
                   moments: Sequence[str] | None = None) -> FinalOutcome:
    """Run the frozen pipeline on 2026 once. Refuses unless confirmed, the season is
    finished, and (for a second run) force=True."""
    if not confirmed:
        raise FinalRunError("Pass --i-understand-this-is-final: this uses up the test set.")
    check_season_finished(races)
    previous = read_lock(lock_path)
    if previous is not None and not force:
        raise FinalRunError(
            f"The final run already happened on {previous.get('timestamp')} (git "
            f"{previous.get('git')}). Use --force only if you must; the report will then "
            f"be labelled '{RERUN_LABEL}'.")
    rerun = previous is not None
    chosen = [m for m in (moments or config.moments) if m in config.moments]
    preds = pd.concat([_moment_predictions(data, m, config, history) for m in chosen],
                      ignore_index=True)
    result = BacktestResult(preds, summarise_by_model(preds), compare_to_baseline(preds, BASELINE),
                            (FINAL_SEASON,), 1, BASELINE, git_hash())
    extra = _final_sections(result, config, rerun, Path(report_path), plots)
    text = render_report(result, extra=extra).replace("# pitwall backtest",
                                                      _title(rerun), 1)
    target = Path(report_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    stamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    lock = {"timestamp": stamp, "git": git_hash(),
            "config_hash": config_hash({m: vars(c) for m, c in config.moments.items()}),
            "season": FINAL_SEASON, "rerun": rerun,
            "previous": previous, "report": str(target)}
    Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
    Path(lock_path).write_text(json.dumps(lock, indent=2, default=str), encoding="utf-8")
    return FinalOutcome(target, Path(lock_path), result, rerun)


def _title(rerun: bool) -> str:
    title = f"# pitwall final evaluation: {FINAL_SEASON}"
    return f"{title} ({RERUN_LABEL})" if rerun else title


def _moment_predictions(data: pd.DataFrame, moment: str, config: PipelineConfig,
                        history: pd.DataFrame | None) -> pd.DataFrame:
    """Walk-forward over 2026: the pipeline, B0, and the pipeline without each model."""
    choice = config.moments[moment]
    variants: dict[str, MomentChoice] = {PIPELINE_NAME: choice, BASELINE: MomentChoice([BASELINE])}
    if len(choice.models) > 1:
        for model in choice.models:
            rest = [m for m in choice.models if m != model]
            method = choice.method if len(rest) > 1 else "single"
            variants[f"{PIPELINE_NAME} without {model}"] = MomentChoice(
                rest, method, choice.calibrate, choice.l2)
    rows = data[(data["Moment"] == moment) & (data["Year"] == FINAL_SEASON)]
    races = rows.drop_duplicates("RaceId").sort_values("Date")
    if races.empty:
        raise DataNotAvailableError(f"No '{moment}' rows for {FINAL_SEASON}.")
    out = []
    for name, variant in variants.items():
        pipeline = FinalPipeline(PipelineConfig({moment: variant}, config.n_seeds), history)
        for race in races.itertuples(index=False):
            pred = pipeline.predict(data, int(race.RaceId), moment, final=True)
            out.append(_with_labels(pred, rows[rows["RaceId"] == race.RaceId], name))
    return pd.concat(out, ignore_index=True)


def _final_sections(result: BacktestResult, config: PipelineConfig, rerun: bool,
                    report_path: Path, plots: bool) -> list[str]:
    lines = ["## Final verdicts", ""]
    if rerun:
        lines += [f"**{RERUN_LABEL}.** The 2026 results had already been seen by an earlier "
                  "final run; treat these numbers as optimistic.", ""]
    summary = result.summary[result.summary["Model"] == PIPELINE_NAME]
    for row in summary.to_dict("records"):
        for key, label in (("win_logloss", "win"), ("podium_logloss", "podium")):
            beaten = row[key] < BENCHMARKS[key]
            lines.append(f"- {row['Moment']}: {label} log loss {row[key]:.4f} "
                         f"{'beats' if beaten else 'does NOT beat'} the {BENCHMARKS[key]} "
                         "target.")
    comp = result.comparisons[result.comparisons["Model"] == PIPELINE_NAME]
    for row in comp.to_dict("records"):
        if row["High"] < 0:
            word = "clearly better than"
        elif row["Low"] > 0:
            word = "clearly worse than"
        else:
            word = "not clearly different from"
        lines.append(f"- {row['Moment']}: {row['Metric']} is {word} B0 "
                     f"({row['Mean']:+.4f}, 95% CI [{row['Low']:+.4f}, {row['High']:+.4f}], "
                     f"{row['Races']} races).")
    lines += ["", "Ablation: 'Final pipeline without X' drops one model; if dropping X makes "
              "things worse, X earned its place.", "",
              "Frozen config: " + json.dumps({m: vars(c) for m, c in config.moments.items()}),
              ""]
    if plots:
        lines += _calibration_plots(result, report_path)
    return lines


def _calibration_plots(result: BacktestResult, report_path: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    from ..plotting.predict_plots import calibration_plot

    lines = ["## Calibration", ""]
    keep = result.predictions["Model"].isin([PIPELINE_NAME, BASELINE])
    for moment, group in result.predictions[keep].groupby("Moment", sort=False):
        path = report_path.parent / f"final_2026_calibration_{moment}.png"
        with plt.style.context("default"):
            ax = calibration_plot(group, bins=5)
            ax.lines[-1].set_color("grey")
            fig = cast(Figure, ax.figure)
            fig.tight_layout()
            fig.savefig(path, dpi=110)
            plt.close(fig)
        lines += [f"![{moment}]({path.name})", ""]
    return lines
