"""reports/market.md: the engine against the market, and the paper-trading diary."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .compare import LABEL_OF_MARKET, fit_benter, score_vs_market
from .paper import INITIAL_BANKROLL, RUIN_FRACTION, paper_stats

MARKET_REPORT = Path("reports") / "market.md"
MOMENTS = ("pre", "practice", "quali")


def graded_market_rows(tickets: list[dict[str, Any]], results: pd.DataFrame) -> pd.DataFrame:
    """Engine and market probabilities saved in each ticket, with the race's outcome.

    results: Year, Round, DriverId and the label columns (Win, Podium, PointsFinish).
    """
    parts = []
    for ticket in tickets:
        race = results[(results["Year"] == ticket["year"]) & (results["Round"] == ticket["round"])]
        table = pd.DataFrame(ticket["market_table"])
        if race.empty or table.empty:
            continue
        label = LABEL_OF_MARKET[ticket["market"]]
        table = table.merge(race[["DriverId", label]].rename(columns={label: "Label"}),
                            on="DriverId")
        table["RaceId"] = f"{ticket['year']}-{ticket['round']:02d}"
        table["Moment"] = ticket["moment"]
        table["Market"] = ticket["market"]
        parts.append(table)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def write_market_report(tickets: list[dict[str, Any]], settlements: list[dict[str, Any]],
                        graded: pd.DataFrame, path: Path | str = MARKET_REPORT,
                        plots: bool = True) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# The engine vs the market (paper trading)", "",
             "**Educational, no real money, not betting advice.** Real-money online betting "
             "is illegal in India; every stake here is pretend. The question is whether the "
             "engine's odds contain information the market's prices do not.", "",
             f"Tickets: {len(tickets)} · settled: {len(settlements)} · each moment starts "
             f"with a pretend bankroll of {INITIAL_BANKROLL:.0f} units.", ""]
    lines += _scoring_section(graded)
    lines += _paper_section(settlements)
    if plots and settlements:
        lines += _plots(settlements, graded, target.parent)
    lines += ["## How to read this", "",
              "- **Log loss vs market**: who forecast better on the same races. Bookmakers "
              "and prediction markets aggregate a lot of information; losing to them is "
              "normal.",
              "- **Benter weight**: in a model combining both, how much the engine's "
              "probabilities still matter once the market is known. Clearly above zero = the "
              "engine adds information (the only real basis for an edge).",
              "- **CLV** (closing line value): getting better odds than the last price before "
              "the start. Consistently positive CLV is the earliest honest sign of skill.",
              "- **ROI interval** and **races needed**: with a few races, profit or loss is "
              "mostly luck. Until the interval excludes zero, there is no evidence either way.",
              ""]
    target.write_text("\n".join(lines), encoding="utf-8")
    return target


def _scoring_section(graded: pd.DataFrame) -> list[str]:
    lines = ["## Who forecast better", ""]
    win = graded[graded["Market"] == "win"] if not graded.empty else graded
    if win.empty:
        return lines + ["_No finished race with market prices yet._", ""]
    lines += ["| Moment | Races | Engine win LL | Market win LL | Engine minus market (95% CI) "
              "| Verdict |", "|---|---|---|---|---|---|"]
    for moment in MOMENTS:
        rows = win[win["Moment"] == moment]
        if rows.empty:
            continue
        score = score_vs_market(rows)
        d = score.difference
        lines.append(f"| {moment} | {score.races} | {score.engine_logloss:.4f} | "
                     f"{score.market_logloss:.4f} | {d.mean:+.4f} [{d.low:+.4f}, {d.high:+.4f}]"
                     f" | {score.verdict} |")
    lines += ["", "### Benter's test: does the engine add information?", ""]
    for moment in MOMENTS:
        rows = win[win["Moment"] == moment]
        if rows["RaceId"].nunique() < 2:
            continue
        try:
            fit = fit_benter(rows)
        except ValueError:
            continue
        verdict = ("yes: the engine adds information beyond the market"
                   if fit.engine_adds_information else
                   "not shown (needs >= 10 races and an engine weight clearly above 0)")
        lines.append(f"- {moment}: engine weight {fit.engine_weight:+.2f} "
                     f"(± {1.96 * fit.engine_se:.2f}), market weight {fit.market_weight:+.2f}, "
                     f"{fit.races} races. {verdict}.")
    return lines + [""]


def _paper_section(settlements: list[dict[str, Any]]) -> list[str]:
    lines = ["## Paper trading", ""]
    stats = [s for s in (paper_stats(settlements, m) for m in MOMENTS) if s is not None]
    if not stats:
        return lines + ["_Nothing settled yet._", ""]
    lines += ["| Moment | Races | Bets | Staked | P&L | ROI (95% CI) | Hit rate | Avg odds | "
              "Mean CLV | Max drawdown | Bankroll | Races needed |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in stats:
        lines.append(
            f"| {s.moment} | {s.races} | {s.bets} | {s.staked:.0f} | {s.pnl:+.0f} | "
            f"{s.roi:+.1%} [{_pct(s.roi_low)}, {_pct(s.roi_high)}] | {_pct(s.hit_rate)} | "
            f"{_num(s.average_odds)} | {_pct(s.mean_clv)} | {s.max_drawdown:.0%} | "
            f"{s.bankroll:.0f} | {_num(s.races_needed)} |")
    lines += ["", f"Risk of ruin (losing {RUIN_FRACTION:.0%} of the bankroll in the next 50 "
              "races, resampling results so far): "
              + ", ".join(f"{s.moment} {_pct(s.risk_of_ruin)}" for s in stats) + ".", ""]
    return lines


def _plots(settlements: list[dict[str, Any]], graded: pd.DataFrame, folder: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lines = []
    with plt.style.context("default"):
        fig = plt.figure(figsize=(7, 3.2))
        ax = fig.add_subplot()
        for moment in MOMENTS:
            rows = [s for s in settlements if s["moment"] == moment]
            if rows:
                curve = INITIAL_BANKROLL + np.cumsum([s["pnl"] for s in rows])
                ax.plot(range(1, len(rows) + 1), curve, marker="o", label=moment)
        ax.axhline(INITIAL_BANKROLL, color="grey", linewidth=0.8)
        ax.set_xlabel("Settled races")
        ax.set_ylabel("Pretend bankroll")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(folder / "market_bankroll.png", dpi=110)
        plt.close(fig)
        lines += ["![Pretend bankroll](market_bankroll.png)", ""]
        win = graded[graded["Market"] == "win"] if not graded.empty else graded
        if len(win) >= 20:
            from ..predict.metrics import calibration_table
            fig = plt.figure(figsize=(4.5, 4.5))
            ax = fig.add_subplot()
            for column, label in (("p_engine", "engine"), ("p_market", "market")):
                table = calibration_table(win[column], win["Label"], bins=5)
                ax.plot(table["Predicted"], table["Observed"], marker="o", label=label)
            ax.plot([0, 1], [0, 1], color="grey", linewidth=0.8)
            ax.set_xlabel("Predicted win probability")
            ax.set_ylabel("Observed win rate")
            ax.legend(frameon=False)
            fig.tight_layout()
            fig.savefig(folder / "market_calibration.png", dpi=110)
            plt.close(fig)
            lines += ["![Calibration: engine vs market](market_calibration.png)", ""]
    return lines


def _pct(x: float) -> str:
    return "–" if x is None or not np.isfinite(x) else f"{x:+.1%}" if x < 0 else f"{x:.1%}"


def _num(x: float) -> str:
    return "–" if x is None or not np.isfinite(x) else f"{x:.1f}"
