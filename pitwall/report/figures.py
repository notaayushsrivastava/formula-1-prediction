"""Figures for the race report (light style, saved as PDF vectors for LaTeX)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from matplotlib.figure import Figure  # noqa: E402

from ..plotting.colours import compound_colour, team_colour  # noqa: E402
from ..plotting.live_plots import position_heatmap, prediction_bars  # noqa: E402
from ..predict.models.base import POSITION_PREFIX  # noqa: E402


def fig_of(ax: Any) -> Figure:
    """The top-level Figure of an Axes (typed, for mypy)."""
    return cast(Figure, ax.figure)


def _save(fig: Any, folder: Path, name: str) -> Path:
    path = folder / name
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def win_bars(pred: pd.DataFrame, teams: dict[str, str], folder: Path) -> Path:
    """Win / podium / points bars."""
    with plt.style.context("default"):
        ax = prediction_bars(pred, teams, title="Win (solid), podium, points (faint)")
        fig_of(ax).set_size_inches(6.8, 0.25 * len(pred) + 1.2)
        return _save(fig_of(ax), folder, "win_bars.pdf")


def heatmap(pred: pd.DataFrame, folder: Path) -> Path:
    """Finishing-position probability heatmap."""
    with plt.style.context("default"):
        ax = position_heatmap(pred, title="P(finishing position)")
        fig_of(ax).set_size_inches(7.0, 0.24 * len(pred) + 1.4)
        return _save(fig_of(ax), folder, "heatmap.pdf")


def expected_points(pred: pd.DataFrame, race_points: tuple[int, ...]) -> np.ndarray:
    """Expected championship points from this race per driver (from P_pos)."""
    columns = [c for c in pred.columns if c.startswith(POSITION_PREFIX)]
    matrix = pred[columns].to_numpy(float)
    paid = np.zeros(matrix.shape[1])
    paid[:min(len(race_points), len(paid))] = race_points[:len(paid)]
    return matrix @ paid


def team_comparison(pred: pd.DataFrame, teams: dict[str, str], race_points: tuple[int, ...],
                    folder: Path) -> Path:
    """Expected points per team from this race, and the team's best win chance."""
    table = pd.DataFrame({"Team": [teams.get(str(d), "?") for d in pred["DriverId"]],
                          "Points": expected_points(pred, race_points),
                          "Win": pred["P_win"].to_numpy(float)})
    by_team = table.groupby("Team").agg(Points=("Points", "sum"), Win=("Win", "sum"))
    by_team = by_team.sort_values("Points")
    with plt.style.context("default"):
        fig, ax = plt.subplots(figsize=(6.8, 0.35 * len(by_team) + 1.2))
        ax.barh(by_team.index, by_team["Points"], color=[team_colour(t) for t in by_team.index])
        for y, (points, win) in enumerate(zip(by_team["Points"], by_team["Win"])):
            ax.text(points + 0.3, y, f"{points:.1f} pts · win {win:.0%}", va="center",
                    fontsize=7)
        ax.set_xlabel("Expected points from this race (both cars)")
        ax.set_title("Teams")
        ax.set_xlim(0, max(1.0, float(by_team["Points"].max())) * 1.35)
        return _save(fig, folder, "teams.pdf")


def tyre_strategy(stints: pd.DataFrame, codes: dict[int, str], title: str,
                  folder: Path) -> Path | None:
    """Last year's race at this circuit: every driver's stints coloured by compound."""
    if stints.empty:
        return None
    stints = stints.dropna(subset=["LapStart", "LapEnd"])
    order = (stints.groupby("Number")["LapEnd"].max().sort_values().index.tolist())
    with plt.style.context("default"):
        fig, ax = plt.subplots(figsize=(6.8, 0.22 * len(order) + 1.3))
        for y, number in enumerate(order):
            for s in stints[stints["Number"] == number].itertuples(index=False):
                ax.barh(y, s.LapEnd - s.LapStart + 1, left=s.LapStart - 1,
                        color=compound_colour(str(s.Compound)), edgecolor="black",
                        linewidth=0.3)
        ax.set_yticks(range(len(order)), [codes.get(int(n), str(n)) for n in order], fontsize=7)
        ax.set_xlabel("Lap")
        ax.set_title(title)
        seen = sorted({str(c) for c in stints["Compound"]})
        handles = [plt.Rectangle((0, 0), 1, 1, color=compound_colour(c)) for c in seen]
        ax.legend(handles, seen, ncols=len(seen), fontsize=7, frameon=False,
                  loc="upper center", bbox_to_anchor=(0.5, -0.15))
        return _save(fig, folder, "tyres.pdf")


def safety_car(rate: float | None, field_rate: float | None, history: pd.DataFrame,
               folder: Path) -> Path | None:
    """The circuit's safety-car / VSC / red-flag rate vs the whole field, and past visits."""
    if rate is None:
        return None
    with plt.style.context("default"):
        fig, (left, right) = plt.subplots(1, 2, figsize=(6.8, 2.4),
                                          gridspec_kw={"width_ratios": [1, 2]})
        values = [rate, field_rate if field_rate is not None else np.nan]
        left.bar(["This circuit", "All circuits"], values, color=["#d62728", "#7f7f7f"])
        left.set_ylim(0, 1)
        left.set_ylabel("P(SC, VSC or red flag)")
        for x, v in enumerate(values):
            if v == v:
                left.text(x, v + 0.03, f"{v:.0%}", ha="center", fontsize=8)
        if history.empty:
            right.text(0.5, 0.5, "No earlier visits with OpenF1 data", ha="center",
                       va="center", transform=right.transAxes, fontsize=8)
            right.set_axis_off()
        else:
            right.bar(history["Year"].astype(str), history["Neutralised"],
                      color=["#d62728" if v else "#c7c7c7" for v in history["Neutralised"]])
            right.set_yticks([0, 1], ["clean", "neutralised"])
            right.set_title("Earlier races here", fontsize=9)
        return _save(fig, folder, "safety_car.pdf")


def track_record(scores: pd.DataFrame, folder: Path) -> Path | None:
    """Running log loss of earlier live predictions."""
    if scores.empty:
        return None
    from ..plotting.live_plots import track_record_plot
    with plt.style.context("default"):
        ax = track_record_plot(scores)
        fig_of(ax).set_size_inches(6.8, 2.8)
        return _save(fig_of(ax), folder, "track_record.pdf")
