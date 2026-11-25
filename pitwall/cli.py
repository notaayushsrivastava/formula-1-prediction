"""Command line: `pitwall-f1 <command>`. Run `pitwall-f1 --help` for the list.

Exit codes: 0 = done (or nothing to do with --skip-if-*), 1 = error, 2 = bad arguments,
3 = the data for that moment is not available yet.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import __version__
from .auth import token_status
from .cache import DiskCache
from .errors import DataNotAvailableError, PitwallError

EXIT_OK, EXIT_ERROR, EXIT_UNAVAILABLE = 0, 1, 3
MOMENT_CHOICES = ["pre", "practice", "quali"]
MODEL_CHOICES = ["B0 grid slot", "B1 logistic", "B2 LightGBM", "M1 strength",
                 "M1-pure strength", "M2-full", "M3 simulator"]
FAST_MODELS = ["B0 grid slot", "B1 logistic", "M1 strength"]


def make_sources(openf1: bool = True, weather: bool = True) -> Any:
    """The data sources (tests replace this function with fakes)."""
    from .predict.service import default_sources
    return default_sources(openf1=openf1, weather=weather)


def utcnow() -> dt.datetime:
    """The current time (tests replace this)."""
    return dt.datetime.now(dt.timezone.utc)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pitwall-f1",
                                     description="Readable F1 data analysis and prediction.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="show each download")
    sub = parser.add_subparsers(dest="command", required=True)

    results = sub.add_parser("results", help="classification of one session")
    results.add_argument("year", type=int)
    results.add_argument("event", help="round number or name, e.g. 16 or Monza")
    results.add_argument("--kind", choices=["R", "Q", "S"], default="R")

    for name, text in (("title", "who can still win the drivers' title"),
                       ("odds", "Monte Carlo title odds")):
        command = sub.add_parser(name, help=text)
        command.add_argument("year", type=int)
        command.add_argument("--after-round", type=int, default=None)
    sub.choices["odds"].add_argument("--sims", type=int, default=20_000)

    sub.add_parser("token", help="check your F1 TV token (never prints it)")
    sub.add_parser("cache-clear", help="delete every cached download")

    # ----- Phase 12: predictions -----
    sub.add_parser("next", help="the next race and its latest prediction")

    predict = sub.add_parser("predict", help="predict an upcoming race")
    predict.add_argument("year", help="season, or 'next'")
    predict.add_argument("event", nargs="?", default="next",
                         help="round number, name (e.g. Qatar) or 'next'")
    _moment(predict)
    predict.add_argument("--json", type=Path, help="also write the prediction table as JSON")
    predict.add_argument("--no-record", action="store_true",
                         help="do not save predictions/<year>-<round>-<moment>.json")
    predict.add_argument("--skip-if-unavailable", action="store_true",
                         help="exit 0 with a message when the moment's data is not out yet")
    predict.add_argument("--skip-if-exists", action="store_true",
                         help="exit 0 when this prediction was already made")
    predict.add_argument("--fast", action="store_true", help="M2 with 2 seeds (if configured)")
    _within(predict)
    _source_flags(predict)

    backtest = sub.add_parser("backtest", help="walk-forward backtest on a past season")
    backtest.add_argument("--season", type=int, action="append",
                          help="season(s) to test (default 2025); repeat for several")
    backtest.add_argument("--models", nargs="+", choices=MODEL_CHOICES, metavar="MODEL",
                          help=f"models to run (default: all). Choices: {MODEL_CHOICES}")
    backtest.add_argument("--fast", action="store_true",
                          help=f"only {FAST_MODELS}, retrain every 4 races")
    backtest.add_argument("--moments", nargs="+", choices=MOMENT_CHOICES)
    backtest.add_argument("--dataset", type=Path, help="Parquet file (default: the cache)")
    backtest.add_argument("--seed-index", type=int,
                          help="M2 only: train one seed (for the GitHub seed matrix)")
    backtest.add_argument("--save-predictions", type=Path)
    backtest.add_argument("--add-predictions", nargs="+", default=[],
                          help="saved prediction files (globs) to merge; M2 seed files are "
                               "averaged into one 'M2-full'")
    backtest.add_argument("--save-history", action="store_true",
                          help="keep the predictions for ensemble weights in live runs")
    backtest.add_argument("--write-config-if-missing", action="store_true",
                          help="write reports/final_config.json (best model per moment) "
                               "if it does not exist yet")
    backtest.add_argument("--no-report", action="store_true")

    dataset = sub.add_parser("dataset", help="build the prediction dataset")
    dataset_sub = dataset.add_subparsers(dest="dataset_command", required=True)
    build = dataset_sub.add_parser("build", help="build and save it (Parquet, in the cache)")
    build.add_argument("--from", dest="first_year", type=int, default=2018)
    build.add_argument("--out", type=Path)
    _source_flags(build)

    sub.add_parser("score", help="grade finished predictions, update the track record")

    final = sub.add_parser("final-2026", help="the ONE final evaluation on 2026")
    final.add_argument("--i-understand-this-is-final", dest="confirmed", action="store_true")
    final.add_argument("--force", action="store_true",
                       help="run again (the report is labelled as a re-run)")
    final.add_argument("--dataset", type=Path)

    report = sub.add_parser("report", help="LaTeX/PDF race preview (needs LuaLaTeX)")
    report.add_argument("year", help="season, or 'next'")
    report.add_argument("event", nargs="?", default="next")
    _moment(report)
    report.add_argument("--out", type=Path, help="PDF path (default: docs/reports/<stem>.pdf)")
    report.add_argument("--tex-only", action="store_true", help="write the .tex, no PDF")
    report.add_argument("--skip-if-unavailable", action="store_true")
    _within(report)
    _source_flags(report)
    return parser


def _moment(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--moment", choices=MOMENT_CHOICES, default="pre",
                        help="pre (before the weekend), practice (after FP), quali (grid set)")


def _within(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--within-days", type=int, default=None,
                        help="do nothing (exit 0) if the race is further away than this")


def _too_far(sources: Any, year: int, event: str, days: int | None) -> str:
    """A 'nothing to do' message if the race is more than `days` days away, else ''."""
    if days is None:
        return ""
    from .predict.live import find_event
    race = find_event(sources.jolpica.get_schedule(year), event)
    date = dt.date.fromisoformat(str(race["Date"])[:10])
    ahead = (date - utcnow().date()).days
    if ahead > days:
        return (f"Nothing to do: {race['EventName']} is on {date} ({ahead} days away, "
                f"more than {days}).")
    return ""


def _source_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--no-openf1", action="store_true", help="skip OpenF1 (practice data)")
    parser.add_argument("--no-weather", action="store_true", help="skip Open-Meteo")


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the `pitwall-f1` command. Returns the exit code."""
    args = build_parser().parse_args(argv)
    level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(message)s")
    try:
        text, code = _dispatch(args)
    except DataNotAvailableError as exc:
        if getattr(args, "skip_if_unavailable", False):
            print(f"Nothing to do: {exc}")
            return EXIT_OK
        print(f"not available: {exc}")
        return EXIT_UNAVAILABLE
    except (PitwallError, ValueError) as exc:
        print(f"error: {exc}")
        return EXIT_ERROR
    if text:
        print(text)
    return code


def _dispatch(args: argparse.Namespace) -> tuple[str, int]:
    handlers = {"next": _next, "predict": _predict, "backtest": _backtest,
                "dataset": _dataset, "score": _score, "final-2026": _final,
                "report": _report}
    if args.command in handlers:
        return handlers[args.command](args)
    return _run(args), EXIT_OK


# ---------- v1 commands ----------
def _run(args: argparse.Namespace) -> str:
    if args.command == "token":
        return token_status()
    if args.command == "cache-clear":
        return f"Removed {DiskCache().clear()} cached files."
    if args.command == "results":
        return _results(args.year, args.event, args.kind)

    from .season import Season
    from .title_fight import title_fight_summary, who_can_still_win

    season = Season(args.year, after_round=args.after_round).load()
    if args.command == "title":
        table = who_can_still_win(season)
        contenders = table[table["Status"] != "Out"].to_string(index=False)
        return f"{title_fight_summary(season)}\n\n{contenders}"

    from .analysis import simulate_championship  # imported only when needed
    odds = simulate_championship(season, n_sims=args.sims)
    shown = odds[odds["TitleChance"] > 0].copy()
    shown["TitleChance"] = shown["TitleChance"].map("{:.1%}".format)
    return shown.round(1).to_string(index=False)


def _results(year: int, event: str, kind: str) -> str:
    from .core import get_session
    session = get_session(year, int(event) if event.isdigit() else event, kind).load()
    if kind == "Q":
        columns = ["Position", "Driver", "Team", "BestSeconds"]
    else:
        columns = ["Position", "Driver", "Team", "Status", "Points"]
    return f"{session.name}\n{session.results[columns].to_string(index=False)}"


# ---------- Phase 12 commands ----------
def _next(args: argparse.Namespace) -> tuple[str, int]:
    from .predict.live import find_event
    from .predict.records import load_records, record_frame
    from .predict.service import format_prediction, resolve_event

    sources = make_sources(openf1=False, weather=False)
    year, rnd = resolve_event(sources, "next", "next", utcnow())
    race = find_event(sources.jolpica.get_schedule(year), rnd)
    lines = [f"Next race: {year} round {race['Round']}, {race['EventName']} "
             f"({race.get('Locality', '')}, {race.get('Country', '')}) on {race['Date']}"]
    mine = [r for r in load_records() if r["year"] == year and r["round"] == int(race["Round"])]
    if not mine:
        lines.append("No prediction yet. Run: pitwall-f1 predict next --moment pre")
    else:
        latest = mine[-1]
        teams = {d["DriverId"]: d.get("TeamId", "") for d in latest["drivers"]}
        lines += [f"Latest prediction ({latest['moment']}, made {latest['created_utc']}):", "",
                  format_prediction(record_frame(latest), teams, limit=10)]
    return "\n".join(lines), EXIT_OK


def _predict(args: argparse.Namespace) -> tuple[str, int]:
    from .predict.records import RecordExistsError
    from .predict.service import (
        format_prediction, predict_race, resolve_event, save_prediction_plots,
    )

    sources = make_sources(openf1=not args.no_openf1, weather=not args.no_weather)
    year, event = resolve_event(sources, args.year, args.event, utcnow())
    far = _too_far(sources, year, event, args.within_days)
    if far:
        return far, EXIT_OK
    try:
        out = predict_race(year, event, args.moment, sources, save=not args.no_record,
                           now=utcnow, n_seeds=2 if args.fast else None)
    except RecordExistsError as exc:
        if args.skip_if_exists:
            return f"Nothing to do: {exc}", EXIT_OK
        raise
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out.record, indent=1), encoding="utf-8")
    lines = [f"{year} {out.live.event} — moment '{args.moment}' "
             f"({out.record['pipeline']['note'] or 'configured pipeline'})", "",
             format_prediction(out.prediction, out.teams), ""]
    lines += [f"warning: {w}" for w in out.live.warnings]
    lines += [f"note: {n}" for n in out.live.notes]
    if out.path is not None:
        from .predict.records import load_records, write_docs
        save_prediction_plots(out)
        write_docs(load_records())
        lines.append(f"Saved {out.path} (never overwritten).")
    return "\n".join(lines), EXIT_OK


def _load_data(path: Path | None) -> Any:
    from .predict.dataset import DEFAULT_PATH, load_dataset
    target = path or DEFAULT_PATH
    if not Path(target).exists():
        raise DataNotAvailableError(f"No dataset at {target}. Run: pitwall-f1 dataset build")
    return load_dataset(target)


def _backtest(args: argparse.Namespace) -> tuple[str, int]:
    import pandas as pd

    from .predict import gate, run_backtest
    from .predict.ensemble import result_from_predictions
    from .predict.models import NeuralRanker
    from .predict.pipeline import CONFIG_PATH, MomentChoice, PipelineConfig, model_factories
    from .predict.report import verdicts, write_report
    from .predict.service import HISTORY_PREDICTIONS

    seasons = tuple(args.season or [2025])
    names = args.models or (FAST_MODELS if args.fast else MODEL_CHOICES)
    factories = model_factories(2 if args.fast else 10)
    models = []
    for name in names:
        if name == "M2-full" and args.seed_index is not None:
            models.append(NeuralRanker(n_seeds=1, seed=args.seed_index, name="M2-full"))
        elif name == "M2-full" and args.fast:
            models.append(NeuralRanker(n_seeds=2, max_epochs=100, name="M2-full"))
        else:
            models.append(factories[name]())
    data = _load_data(args.dataset)
    moments = args.moments or MOMENT_CHOICES
    result = run_backtest(models, data, seasons=seasons, moments=moments,
                          retrain_every=4 if args.fast else 2)
    preds = result.predictions
    extra_files = sorted({f for pattern in args.add_predictions for f in glob.glob(pattern)})
    if extra_files:
        preds = pd.concat([preds, _merge_seed_files(extra_files)], ignore_index=True)
    baseline = "B0 grid slot" if (preds["Model"] == "B0 grid slot").any() else names[0]
    result = result_from_predictions(preds, baseline, seasons, result.retrain_every)
    if args.save_predictions:
        args.save_predictions.parent.mkdir(parents=True, exist_ok=True)
        preds.to_parquet(args.save_predictions, index=False)
    if args.save_history:
        HISTORY_PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
        preds.to_parquet(HISTORY_PREDICTIONS, index=False)
    lines = [result.summary.round(4).to_string(index=False), ""]
    if not args.no_report and args.seed_index is None:
        others = [m for m in result.summary["Model"].unique() if m != baseline]
        gates = [gate(result, m, [baseline]) for m in others]
        write_report(result, gates=gates, extra=["## Verdicts", "",
                                                 *[f"- {v}" for v in verdicts(result.summary)],
                                                 ""])
        lines.append("Wrote reports/backtest.md and appended reports/experiments.csv.")
    if args.write_config_if_missing and not CONFIG_PATH.exists() and 2026 not in seasons:
        best = {}
        for moment, rows in result.summary.groupby("Moment", sort=False):
            pick = rows.sort_values(["win_logloss", "podium_logloss"]).iloc[0]
            best[str(moment)] = MomentChoice([str(pick["Model"])])
        PipelineConfig(best, note=f"best single model per moment, backtest {seasons}").save()
        lines.append(f"Wrote {CONFIG_PATH} (frozen from now on: delete it to choose again).")
    return "\n".join(lines), EXIT_OK


def _merge_seed_files(files: list[str]) -> Any:
    """Read saved predictions; average the M2-full seed files into one model."""
    import pandas as pd

    from .predict.dataset import LABELS
    frames = [pd.read_parquet(f) for f in files]
    allp = pd.concat(frames, ignore_index=True)
    m2 = allp[allp["Model"] == "M2-full"]
    rest = allp[allp["Model"] != "M2-full"]
    if m2.empty:
        return rest
    keys = ["Model", "Moment", "RaceId", "DriverId"]
    numeric = [c for c in m2.columns if c.startswith("P_") or c == "ExpectedPosition"]
    mean = m2.groupby(keys, sort=False)[numeric].mean().reset_index()
    other = [c for c in m2.columns if c not in numeric and c not in keys]
    first = m2.drop_duplicates(keys)[keys + other]
    merged = mean.merge(first, on=keys)[m2.columns]
    for column in LABELS:
        if column in merged:
            merged[column] = merged[column].astype(m2[column].dtype)
    return pd.concat([rest, merged], ignore_index=True)


def _dataset(args: argparse.Namespace) -> tuple[str, int]:
    from .predict.dataset import build_dataset, save_dataset
    from .predict.live import ToppedUpHistory

    sources = make_sources(openf1=not args.no_openf1, weather=not args.no_weather)
    now = utcnow()
    history = ToppedUpHistory(sources.history, sources.jolpica, [now.year - 1, now.year],
                              now=utcnow)
    data = build_dataset(history, sources.weekend, sources.weather,
                         first_year=args.first_year)
    path = save_dataset(data, args.out)
    races = data["RaceId"].nunique()
    return (f"Saved {len(data)} rows ({races} races, {data['Year'].min()}-"
            f"{data['Year'].max()}) to {path}"), EXIT_OK


def _score(args: argparse.Namespace) -> tuple[str, int]:
    from .predict.records import TRACK_RECORD_PATH
    from .predict.service import score_all

    sources = make_sources(openf1=False, weather=False)
    scores = score_all(sources, now=utcnow)
    if scores.empty:
        return f"No finished races to score yet. Wrote {TRACK_RECORD_PATH}.", EXIT_OK
    shown = scores[["Stem", "win_logloss", "b0_win_logloss", "podium_logloss",
                    "b0_podium_logloss", "Favourite", "Winner"]]
    lines = [shown.round(4).to_string(index=False), ""]
    checks = scores.attrs.get("checks")
    if checks is not None and not checks.empty:
        lines.append("Surprises and data checks:")
        lines += [f"  [{r.Kind}] {r.Race} ({r.Moment}): {r.Driver} {r.Text}"
                  for r in checks.itertuples(index=False)]
        lines.append("")
    lines.append(f"Updated {TRACK_RECORD_PATH}")
    return "\n".join(lines), EXIT_OK


def _final(args: argparse.Namespace) -> tuple[str, int]:
    import pandas as pd

    from .predict.final import run_final_2026
    from .predict.live import ToppedUpHistory
    from .predict.pipeline import CONFIG_PATH, PipelineConfig
    from .predict.service import HISTORY_PREDICTIONS

    if not args.confirmed:
        raise PitwallError("Refusing: add --i-understand-this-is-final. This uses up the "
                           "2026 test set; it should run once, after Abu Dhabi.")
    sources = make_sources(openf1=False, weather=False)
    races = ToppedUpHistory(sources.history, sources.jolpica, [2026], now=utcnow).race_table()
    from .predict.final import check_season_finished
    check_season_finished(races)
    if not CONFIG_PATH.exists():
        raise PitwallError(f"No frozen pipeline at {CONFIG_PATH}; run the 2025 backtest first.")
    history = pd.read_parquet(HISTORY_PREDICTIONS) if HISTORY_PREDICTIONS.exists() else None
    outcome = run_final_2026(_load_data(args.dataset), races, PipelineConfig.load(),
                             history, confirmed=True, force=args.force)
    label = " (RE-RUN)" if outcome.rerun else ""
    return f"Wrote {outcome.report} and {outcome.lock}{label}.", EXIT_OK


# ---------- Phase 13 ----------
def _report(args: argparse.Namespace) -> tuple[str, int]:
    from .report import build_race_report

    sources = make_sources(openf1=not args.no_openf1, weather=not args.no_weather)
    from .predict.service import resolve_event
    year, event = resolve_event(sources, args.year, args.event, utcnow())
    far = _too_far(sources, year, event, args.within_days)
    if far:
        return far, EXIT_OK
    out = build_race_report(year, event, args.moment, sources, out=args.out,
                            tex_only=args.tex_only, now=utcnow)
    return f"Wrote {out}", EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
