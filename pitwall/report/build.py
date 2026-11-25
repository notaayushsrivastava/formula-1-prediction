"""Phase 13: the race-preview PDF.

`build_race_report` gathers everything about one upcoming race at one moment (the
saved prediction, or a new one; weather; circuit history; last year's tyre strategy;
championship; track record), draws the figures, fills the LaTeX template and compiles
it with LuaLaTeX. Each section degrades gracefully: missing data prints "not available"
instead of stopping the report. Maximum 8 pages.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .. import __version__
from ..errors import PitwallError
from ..points import points_system
from ..predict.live import LiveRows, SettledWeekend, build_live_rows, find_event
from ..predict.metrics import BENCHMARKS
from ..predict.records import (
    DOCS_DIR, RECORDS_DIR, load_records, record_frame, record_stem, score_records,
)
from ..predict.weekend import meeting_sessions
from . import figures
from .latex import compile_pdf, render

log = logging.getLogger(__name__)
TEMPLATE = "race_report.tex"
MOMENT_TEXT = {"pre": "before the race weekend", "practice": "after practice",
               "quali": "after qualifying (grid set)"}
FEATURE_TEXT = {
    "grid": "starting position", "quali_gap_pct": "qualifying gap to pole",
    "teammate_quali_gap_pct": "qualifying gap to team-mate",
    "practice_pace_gap_pct": "practice long-run pace gap",
    "driver_avg_finish_5": "average finish, last 5 races",
    "driver_points_5": "points, last 5 races", "team_avg_points_5": "team points, last 5 races",
    "champ_points_before": "championship points so far", "driver_dnf_rate_10": "DNF rate",
    "team_mech_dnf_rate_10": "team mechanical DNF rate", "experience": "races started",
    "age": "age", "driver_avg_finish_circuit": "past finishes at this circuit",
    "team_pit_delta_10": "team pit-stop time", "pit_lane_start": "pit-lane start",
    "circuit_places_moved": "overtaking at this circuit", "circuit_sc_rate": "safety-car rate",
    "air_temp_forecast": "forecast air temperature", "rain_forecast_mm": "forecast rain",
    "track_temp": "track temperature", "wet_practice": "wet practice", "wet_quali": "wet qualifying",
}
LICENCES = [
    ("f1db", "github.com/f1db/f1db", "CC BY 4.0", "race history 1950-today"),
    ("Jolpica F1 API", "jolpi.ca", "free API (Ergast successor)", "newest results, schedule"),
    ("OpenF1", "openf1.org", "CC BY-NC-SA 4.0, historical tier only",
     "practice pace, weather, tyre stints, race control"),
    ("Open-Meteo", "open-meteo.com", "CC BY 4.0, non-commercial", "weather forecasts"),
]


def build_race_report(year: int, event: str, moment: str, sources: Any,
                      out: Path | str | None = None, tex_only: bool = False,
                      now: Callable[[], dt.datetime] | None = None,
                      records_dir: Path | str = RECORDS_DIR, docs_dir: Path | str = DOCS_DIR,
                      first_year: int = 2018) -> Path:
    """Build docs/reports/<stem>.pdf (or `out`) for one race and moment; returns the path.

    Uses the saved prediction record if it exists, otherwise predicts and saves one.
    tex_only=True writes the LaTeX source instead (no TeX installation needed).
    """
    from ..predict.service import predict_race

    clock = now or (lambda: dt.datetime.now(dt.timezone.utc))
    race = find_event(sources.jolpica.get_schedule(year), event)
    stem = record_stem(year, int(race["Round"]), moment)
    live = build_live_rows(year, race["Round"], moment, sources.history, sources.jolpica,
                           sources.weekend, sources.weather, sources.live_forecast,
                           first_year=first_year, now=clock)
    path = Path(records_dir) / f"{stem}.json"
    if path.exists():
        record = json.loads(path.read_text(encoding="utf-8"))
    else:
        record = predict_race(year, str(race["Round"]), moment, sources,
                              records_dir=records_dir, now=clock, live_rows=live).record
    with tempfile.TemporaryDirectory(prefix="pitwall-figures-") as tmp:
        context, figs = report_context(record, live, race, sources, Path(tmp), records_dir,
                                       clock)
        tex = render(TEMPLATE, context)
        target = Path(out) if out else Path(docs_dir) / "reports" / f"{stem}.pdf"
        if tex_only:
            target = target.with_suffix(".tex")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(tex, encoding="utf-8")
            return target
        return compile_pdf(tex, target, figs)


def report_context(record: dict[str, Any], live: LiveRows, race: dict[str, Any], sources: Any,
                   folder: Path, records_dir: Path | str,
                   clock: Callable[[], dt.datetime]) -> tuple[dict[str, Any], dict[str, Path]]:
    """Everything the template needs, plus the figure files (name -> path)."""
    pred = record_frame(record)
    teams = {d["DriverId"]: d.get("TeamId", "") for d in record["drivers"]}
    year = int(record["year"])
    points = points_system(year).race
    figs: dict[str, Path] = {}

    def add(path: Path | None) -> str:
        if path is None:
            return ""
        figs[path.name] = path
        return path.name

    context: dict[str, Any] = {
        "title": f"{year} {record['event']}", "year": year, "round": record["round"],
        "event": record["event"], "race_date": record["race_date"],
        "place": ", ".join(str(race.get(k)) for k in ("Locality", "Country") if race.get(k)),
        "moment": record["moment"], "moment_text": MOMENT_TEXT.get(record["moment"], ""),
        "created": record["created_utc"], "git": record["git"], "version": record["version"],
        "config_hash": record["config_hash"], "built": clock().isoformat(timespec="minutes"),
        "package_version": __version__, "pipeline": _pipeline_text(record),
        "warnings": record.get("warnings", []), "notes": record.get("notes", []),
        "availability": record.get("availability", {}),
        "targets": BENCHMARKS,
    }
    context["summary"] = summary(pred)
    context["weather"] = weather_text(record.get("context") or {}, live)
    context["rows"] = table_rows(pred, record, live)
    context["fig_bars"] = add(figures.win_bars(pred, teams, folder))
    context["fig_heatmap"] = add(figures.heatmap(pred, folder))
    context["fig_teams"] = add(figures.team_comparison(pred, teams, points, folder))
    tyres, tyre_text = _safe(lambda: tyre_figure(live, sources, folder), (None, ""))
    context["fig_tyres"], context["tyre_text"] = add(tyres), tyre_text
    sc_fig, sc_text = _safe(lambda: safety_car_figure(live, folder), (None, ""))
    context["fig_sc"], context["sc_text"] = add(sc_fig), sc_text
    context["why"] = _safe(lambda: explanation(live), [])
    context["championship"] = _safe(lambda: championship(pred, live, race, sources, points),
                                    {"rows": [], "text": "Championship data not available."})
    scores = _safe(lambda: _scores(sources, records_dir, clock), pd.DataFrame())
    context["track"] = track_rows(scores)
    context["fig_track"] = add(_safe(lambda: figures.track_record(scores, folder), None))
    context["market"] = _safe(lambda: market_section(record), {"rows": []})
    context["licences"] = LICENCES
    return context, figs


def market_section(record: dict[str, Any]) -> dict[str, Any]:
    """Engine vs market for this prediction: from its paper ticket if one was made, else
    from hand-typed prices (data/markets). Empty when there are no prices."""
    from ..market import ManualCsvSource, Roster, market_snapshot, race_table
    from ..market.paper import ticket_path

    stem = record.get("_stem") or record_stem(record["year"], record["round"],
                                              record["moment"])
    path = ticket_path(stem)
    if path.exists():
        table = pd.DataFrame(json.loads(path.read_text(encoding="utf-8"))["market_table"])
    else:
        prices = ManualCsvSource().get_prices(int(record["year"]), int(record["round"]),
                                              Roster.from_record(record))
        if prices.empty:
            return {"rows": []}
        table = race_table(record_frame(record),
                           market_snapshot(prices, "win", record["created_utc"]))
    if table.empty:
        return {"rows": []}
    top = table.sort_values("p_engine", ascending=False).head(8)
    rows = [{"driver": r.Driver, "engine": r.p_engine, "market": r.p_market,
             "edge": f"{100 * r.edge:+.1f}", "odds": f"{r.best_odds:.2f}"}
            for r in top.itertuples(index=False)]
    return {"rows": rows, "source": ", ".join(sorted(set(map(str, table["best_source"]))))}


def _safe(make: Callable[[], Any], fallback: Any) -> Any:
    """A section that fails (missing data, a source down) is left out, not fatal."""
    try:
        return make()
    except (PitwallError, KeyError, ValueError, IndexError, TypeError, AttributeError) as exc:
        log.warning("Report section skipped: %s", exc)
        return fallback


def _pipeline_text(record: dict[str, Any]) -> str:
    choice = record.get("pipeline", {}).get("moment", {})
    models = ", ".join(choice.get("models", [])) or "?"
    method = choice.get("method", "single")
    blend = {"single": "a single model", "equal": "an equal-weight blend",
             "linear": "a linear pool with weights fitted on earlier races",
             "log": "a log pool with weights fitted on earlier races"}.get(method, method)
    cal = ", calibrated" if choice.get("calibrate") else ""
    return f"{models} ({blend}{cal})"


def summary(pred: pd.DataFrame) -> dict[str, Any]:
    """The headline numbers for the summary box."""
    by_win = pred.sort_values("P_win", ascending=False)
    by_podium = pred.sort_values("P_podium", ascending=False)
    risky = pred.sort_values("P_dnf", ascending=False).iloc[0]
    top3_win = float(by_win["P_win"].head(3).sum())
    return {"favourite": by_win["Driver"].iloc[0], "favourite_p": by_win["P_win"].iloc[0],
            "second": by_win["Driver"].iloc[1], "second_p": by_win["P_win"].iloc[1],
            "podium": [(r.Driver, r.P_podium) for r in by_podium.head(3).itertuples()],
            "upset": 1 - top3_win, "dnf_driver": risky["Driver"], "dnf_p": risky["P_dnf"],
            "field": len(pred)}


def weather_text(ctx: dict[str, Any], live: LiveRows) -> dict[str, Any]:
    """Forecast for the race window (as the model saw it)."""
    has = bool(ctx.get("has_forecast"))
    if not ctx and not live.race.empty:
        first = live.race.iloc[0]
        has = bool(first.get("has_forecast"))
        ctx = {"air_temp_forecast": first.get("air_temp_forecast"),
               "rain_forecast_mm": first.get("rain_forecast_mm"),
               "track_temp": first.get("track_temp"), "has_track_temp": first.get("has_track_temp")}
    out: dict[str, Any] = {"available": has, "air": None, "rain": None, "track": None}
    if has:
        out["air"] = f"{float(ctx['air_temp_forecast']):.0f}"
        out["rain"] = f"{float(ctx['rain_forecast_mm']):.1f}"
    if ctx.get("has_track_temp") and ctx.get("track_temp") is not None:
        out["track"] = f"{float(ctx['track_temp']):.0f}"
    return out


def table_rows(pred: pd.DataFrame, record: dict[str, Any], live: LiveRows) -> list[dict]:
    """One row per driver for the main table (by expected position)."""
    base = {d["DriverId"]: d["P_win"] for d in record.get("baseline", {}).get("drivers", [])}
    grid = dict(zip(live.race["DriverId"], live.race["grid"]))
    teams = {d["DriverId"]: d.get("TeamId", "") for d in record["drivers"]}
    rows = []
    for i, r in enumerate(pred.sort_values("ExpectedPosition").itertuples(index=False), 1):
        g = grid.get(r.DriverId)
        rows.append({"pos": i, "driver": r.Driver, "team": teams.get(r.DriverId, ""),
                     "grid": "" if g is None or pd.isna(g) else str(int(g)),
                     "win": r.P_win, "podium": r.P_podium, "points": r.P_points,
                     "dnf": r.P_dnf, "exp": f"{r.ExpectedPosition:.1f}",
                     "b0": base.get(r.DriverId)})
    return rows


def tyre_figure(live: LiveRows, sources: Any, folder: Path) -> tuple[Path | None, str]:
    """Last visit's race stints (OpenF1, 2023+)."""
    data = live.data
    past = data[(data["CircuitId"] == live.circuit_id) & (data["Date"] < live.date)
                & (data["Year"] >= 2023)]
    if past.empty or sources.weekend is None:
        return None, "No earlier race at this circuit with OpenF1 tyre data."
    last = past[past["Date"] == past["Date"].max()]
    date = pd.Timestamp(last["Date"].iloc[0])
    weekend = SettledWeekend(sources.weekend)
    sessions = meeting_sessions(weekend.get_sessions(int(date.year)), date)
    if "Race" not in sessions:
        return None, "OpenF1 has no race session for the last visit."
    stints = pd.DataFrame(weekend.get_stints(sessions["Race"]))
    codes = {int(n): str(d) for n, d in zip(last["Number"], last["Driver"]) if pd.notna(n)}
    title = f"Tyre strategy, {int(date.year)} race here (OpenF1)"
    fig = figures.tyre_strategy(stints, codes, title, folder)
    return fig, ("" if fig else "No tyre stints recorded for the last visit.")


def safety_car_figure(live: LiveRows, folder: Path) -> tuple[Path | None, str]:
    """This circuit's neutralisation rate (as the model saw it) vs the field."""
    first = live.race.iloc[0]
    has = bool(first.get("has_sc_history"))
    races = live.data.drop_duplicates("RaceId")
    known = races[races["Neutralised"].notna() & (races["Date"] < live.date)]
    field = float(known["Neutralised"].mean()) if not known.empty else None
    here = known[known["CircuitId"] == live.circuit_id][["Year", "Neutralised"]]
    if not has:
        return None, "No safety-car history for this circuit yet (OpenF1 starts in 2023)."
    rate = float(first["circuit_sc_rate"])
    text = (f"Chance of a safety car, virtual safety car or red flag: about {rate:.0%} "
            f"here, versus {field:.0%} over all circuits." if field is not None else "")
    return figures.safety_car(rate, field, here, folder), text


def explanation(live: LiveRows) -> list[dict[str, Any]]:
    """The biggest factors of a logistic model (B1) fitted on the same training rows."""
    from ..predict.models import LogisticModel

    train = live.data[live.data["Date"] < live.date]
    model = LogisticModel(n_sims=1000).fit(train)
    weights = model.coefficients().head(8)
    from ..predict.models.base import design_matrix
    race = live.race
    z = model.scaler_.transform(design_matrix(race, model.features_))
    favourite = int(np.argmax(z @ model.weights_))
    idx = {f: i for i, f in enumerate(model.features_)}
    out = []
    for feature, weight in weights.items():
        contribution = float(weight * z[favourite, idx[feature]])
        out.append({"feature": FEATURE_TEXT.get(feature, feature.replace("_", " ")),
                    "weight": f"{weight:+.2f}",
                    "direction": "helps" if weight > 0 else "hurts",
                    "fav": f"{contribution:+.2f}"})
    return [{"driver": race["Driver"].iloc[favourite], "rows": out}]


def championship(pred: pd.DataFrame, live: LiveRows, race: dict[str, Any], sources: Any,
                 points: tuple[int, ...]) -> dict[str, Any]:
    """Points now, expected points from this race, and title chances (simple form model)."""
    now = dict(zip(live.race["DriverId"], live.race["champ_points_before"].astype(float)))
    expected = dict(zip(pred["DriverId"], figures.expected_points(pred, points)))
    schedule = sources.jolpica.get_schedule(live.year)
    later = [r for r in schedule if int(r["Round"]) > live.round]
    sprints = sum(bool(r.get("HasSprint")) for r in later) + int(bool(race.get("HasSprint")))
    still = (len(later) + 1) * points[0] + sprints * 8
    odds: dict[str, float] = {}
    try:
        from ..analysis import simulate_championship
        from ..season import Season
        table = simulate_championship(Season(live.year, sources.jolpica).load(), n_sims=5000)
        odds = {str(code): float(p) for code, p in zip(table["Driver"], table["TitleChance"])}
    except (PitwallError, KeyError, ValueError, AttributeError, TypeError) as exc:
        log.warning("Title odds not available: %s", exc)
    leader = max(now.values()) if now else 0.0
    rows = []
    for r in pred.itertuples(index=False):
        pts = now.get(r.DriverId, 0.0)
        rows.append({"driver": r.Driver, "now": f"{pts:.0f}",
                     "exp": f"{expected[r.DriverId]:.1f}",
                     "after": f"{pts + expected[r.DriverId]:.1f}",
                     "alive": "yes" if pts + still >= leader else "no",
                     "odds": odds.get(str(r.Driver)), "sort": pts})
    rows = sorted(rows, key=lambda x: -x["sort"])[:8]
    text = (f"{len(later)} more race(s) after this one; at most {still} points are still "
            "available including this race (sprints counted at 8).")
    return {"rows": rows, "text": text, "has_odds": bool(odds)}


def _scores(sources: Any, records_dir: Path | str,
            clock: Callable[[], dt.datetime]) -> pd.DataFrame:
    from ..predict.service import results_table
    records = load_records(records_dir)
    if not records:
        return pd.DataFrame()
    years = sorted({int(r["year"]) for r in records})
    return score_records(records, results_table(sources, years, clock))


def track_rows(scores: pd.DataFrame) -> dict[str, Any]:
    """Track record summary for the report."""
    if scores.empty:
        return {"n": 0, "rows": []}
    rows = []
    for moment, group in scores.groupby("Moment", sort=False):
        rows.append({"moment": moment, "n": len(group),
                     "win": f"{group['win_logloss'].mean():.3f}",
                     "b0": (f"{group['b0_win_logloss'].mean():.3f}"
                            if group["b0_win_logloss"].notna().any() else "--"),
                     "podium": f"{group['podium_logloss'].mean():.3f}",
                     "hit": f"{group['WinnerHit'].mean():.0%}"})
    return {"n": len(scores), "rows": rows}
