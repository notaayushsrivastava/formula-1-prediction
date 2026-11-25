"""reports/backtest.md: one table per moment, every model against the benchmarks,
gate decisions, an optional ablation table and any extra sections."""
from __future__ import annotations

import datetime as dt
import math
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from .backtest import REPORTS_DIR, BacktestResult, GateResult, compare_models
from .metrics import BENCHMARKS

REPORT_PATH = REPORTS_DIR / "backtest.md"
TABLE_METRICS = [
    ("win_logloss", "Win LL"), ("podium_logloss", "Podium LL"),
    ("points_logloss", "Points LL"), ("dnf_logloss", "DNF LL"),
    ("winner_logloss", "Winner LL"), ("win_brier", "Win Brier"), ("rps", "RPS"),
    ("spearman", "Spearman"), ("winner_hit_rate", "Winner hit"),
    ("podium_overlap", "Podium overlap"), ("win_ece", "Win ECE"), ("n_races", "Races"),
]
Ablation = tuple[str, Sequence[str]]  # (reference model, variant models)


def write_report(result: BacktestResult, path: Path | str = REPORT_PATH,
                 gates: Sequence[GateResult] = (), ablation: Ablation | None = None,
                 extra: Sequence[str] = ()) -> Path:
    """Write the markdown report; returns its path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_report(result, gates, ablation, extra), encoding="utf-8")
    return target


def render_report(result: BacktestResult, gates: Sequence[GateResult] = (),
                  ablation: Ablation | None = None, extra: Sequence[str] = ()) -> str:
    """The whole report as markdown text (extra = ready-made markdown lines, appended)."""
    seasons = ", ".join(map(str, result.seasons))
    lines = [
        "# pitwall backtest",
        "",
        f"Seasons tested: **{seasons}**. Walk-forward: every race is predicted by models "
        f"trained only on earlier races, retrained every {result.retrain_every} races. "
        f"Generated {dt.date.today().isoformat()} (git {result.git}).",
        "",
        "- LL = binary log loss averaged over driver rows (comparable to the 0.075 / 0.191 "
        "targets). Winner LL = mean of -log P_win of the actual winner. Lower is better for "
        "LL, Brier, RPS and ECE; higher for Spearman, Winner hit and Podium overlap.",
        f"- Baseline: **{result.baseline}**. At the 'quali' moment it is the grid baseline.",
        "",
    ]
    for moment in result.summary["Moment"].unique():
        lines += moment_section(result, str(moment))
    for decision in gates:
        lines += gate_section(decision)
    if ablation is not None:
        lines += ablation_section(result, ablation[0], ablation[1])
    lines += list(extra)
    return "\n".join(lines) + "\n"


def moment_section(result: BacktestResult, moment: str) -> list[str]:
    """Metrics table, target row, differences vs baseline and verdicts for one moment."""
    summary = result.summary[result.summary["Moment"] == moment]
    header = ["Model", *[label for _, label in TABLE_METRICS]]
    rows = []
    for row in summary.to_dict("records"):
        name = row["Model"] + (" (baseline)" if row["Model"] == result.baseline else "")
        rows.append([name, *[_fmt(row[key], key) for key, _ in TABLE_METRICS]])
    rows.append(["Target: best open model",
                 *[_fmt(BENCHMARKS.get(key, math.nan), key) for key, _ in TABLE_METRICS]])
    lines = [f"## Moment: {moment}", "", markdown_table(header, rows), ""]
    comparisons = result.comparisons[result.comparisons["Moment"] == moment]
    if not comparisons.empty:
        lines += ["Difference vs baseline (model minus baseline; negative = better), mean per "
                  "race with a 95% paired-bootstrap interval:", "",
                  markdown_table(["Model", "Win LL", "Podium LL"], _difference_rows(comparisons)),
                  ""]
    lines += [f"- {line}" for line in verdicts(summary)] + [""]
    return lines


def gate_section(decision: GateResult) -> list[str]:
    """One gate: candidate minus the best previous model per moment, and the decision."""
    rows = []
    for moment, group in decision.comparisons.groupby("Moment", sort=False):
        cells = [str(moment), str(group["Reference"].iloc[0])]
        for metric in ("win_logloss", "podium_logloss"):
            hit = group[group["Metric"] == metric]
            cells.append(_interval(hit.iloc[0]) if not hit.empty else "—")
        cells.append("yes" if moment in decision.passed_moments else "no")
        rows.append(cells)
    header = ["Moment", "Best previous", "Win LL", "Podium LL", "Both clearly better?"]
    return [f"## Gate: {decision.candidate}", "",
            "Candidate minus the best previous model (chosen per moment by win log loss), "
            "95% paired-bootstrap interval:", "",
            markdown_table(header, rows), "", f"**Decision:** {decision.verdict()}", ""]


def ablation_section(result: BacktestResult, reference: str,
                     variants: Sequence[str]) -> list[str]:
    """What each removed ingredient costs: variant minus reference, per moment."""
    header = ["Moment", "Variant", "Win LL", "Podium LL", "Win LL vs ref", "Podium LL vs ref"]
    rows = []
    for moment in result.summary["Moment"].unique():
        summary = result.summary[result.summary["Moment"] == moment].set_index("Model")
        preds = result.predictions[result.predictions["Moment"] == moment]
        for variant in [reference, *variants]:
            if variant not in summary.index:
                continue
            cells = [str(moment), variant, f"{summary.at[variant, 'win_logloss']:.4f}",
                     f"{summary.at[variant, 'podium_logloss']:.4f}"]
            if variant == reference:
                cells += ["(reference)", "(reference)"]
            else:
                comparison = compare_models(preds, variant, reference)
                for metric in ("win_logloss", "podium_logloss"):
                    hit = comparison[comparison["Metric"] == metric]
                    cells.append(_interval(hit.iloc[0]) if not hit.empty else "—")
            rows.append(cells)
    return [f"## Ablation vs {reference}", "",
            "Each variant minus the reference (negative = the variant is better), 95% "
            "paired-bootstrap interval. A variant that is not worse is simpler and preferred.",
            "", markdown_table(header, rows), ""]


def verdicts(summary: pd.DataFrame) -> list[str]:
    """Plain-language check of every model against the two targets (honest either way)."""
    out = []
    for row in summary.to_dict("records"):
        parts = []
        for key, label in (("win_logloss", "win"), ("podium_logloss", "podium")):
            beaten = row[key] < BENCHMARKS[key]
            parts.append(f"{label} LL {row[key]:.4f} {'beats' if beaten else 'does NOT beat'} "
                         f"the {BENCHMARKS[key]} target")
        out.append(f"{row['Model']}: " + "; ".join(parts) + ".")
    return out


def markdown_table(header: list[str], rows: list[list[str]]) -> str:
    """A GitHub markdown table (no extra library needed)."""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _difference_rows(comparisons: pd.DataFrame) -> list[list[str]]:
    rows = []
    for model, group in comparisons.groupby("Model", sort=False):
        cells = [str(model)]
        for metric in ("win_logloss", "podium_logloss"):
            hit = group[group["Metric"] == metric]
            cells.append(_interval(hit.iloc[0]) if not hit.empty else "—")
        rows.append(cells)
    return rows


def _interval(row: pd.Series) -> str:
    if row["High"] < 0:
        word = "better"
    elif row["Low"] > 0:
        word = "worse"
    else:
        word = "no clear difference"
    return f"{row['Mean']:+.4f} [{row['Low']:+.4f}, {row['High']:+.4f}] ({word})"


def _fmt(value: float, key: str) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{int(value)}" if key == "n_races" else f"{value:.4f}"
