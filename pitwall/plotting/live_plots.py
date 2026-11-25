"""Phase 12 plots for one live prediction and for the running track record."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from ..predict.metrics import BENCHMARKS
from ..predict.models.base import POSITION_PREFIX
from .colours import team_colour


def prediction_bars(pred: pd.DataFrame, teams: dict[str, str] | None = None,
                    title: str = "", ax: Axes | None = None) -> Axes:
    """Win, podium and points probability per driver (nested bars, favourite on top)."""
    table = pred.sort_values(["P_win", "P_podium", "P_points"]).reset_index(drop=True)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 0.32 * len(table) + 1.2))
    y = np.arange(len(table))
    colours = [team_colour((teams or {}).get(str(d))) for d in table["DriverId"]]
    ax.barh(y, table["P_points"], color=colours, alpha=0.25, label="Points (top 10)")
    ax.barh(y, table["P_podium"], color=colours, alpha=0.55, label="Podium")
    ax.barh(y, table["P_win"], color=colours, alpha=1.0, label="Win")
    for yi, p in zip(y, table["P_win"]):
        if p >= 0.01:
            ax.text(float(p) + 0.005, float(yi), f"{p:.0%}", va="center", fontsize=7)
    ax.set_yticks(y, table["Driver"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    ax.set_title(title or "Race prediction: win (solid), podium, points (faint)")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    return ax


def position_heatmap(pred: pd.DataFrame, title: str = "", ax: Axes | None = None) -> Axes:
    """P(driver finishes in position k): rows = drivers by expected position."""
    table = pred.sort_values("ExpectedPosition").reset_index(drop=True)
    columns = [c for c in table.columns if c.startswith(POSITION_PREFIX)]
    matrix = table[columns].to_numpy(float)
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 0.3 * len(table) + 1.5))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0,
                      vmax=max(0.05, float(matrix.max())))
    ax.set_yticks(range(len(table)), table["Driver"])
    ax.set_xticks(range(len(columns)), [c.removeprefix(POSITION_PREFIX) for c in columns],
                  fontsize=7)
    ax.set_xlabel("Finishing position")
    ax.set_title(title or "Finishing position probabilities")
    ax.figure.colorbar(image, ax=ax, fraction=0.03, pad=0.02, label="Probability")
    return ax


def track_record_plot(scores: pd.DataFrame, ax: Axes | None = None) -> Axes:
    """Running mean win and podium log loss over scored predictions, vs B0 and targets."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    if scores.empty:
        ax.text(0.5, 0.5, "No scored predictions yet", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    ordered = scores.sort_values(["Year", "Round", "Moment"]).reset_index(drop=True)
    x = np.arange(1, len(ordered) + 1)
    for column, label, style in (("win_logloss", "Win (model)", "-"),
                                 ("b0_win_logloss", "Win (B0)", ":"),
                                 ("podium_logloss", "Podium (model)", "-"),
                                 ("b0_podium_logloss", "Podium (B0)", ":")):
        values = ordered[column].astype(float)
        if values.notna().any():
            ax.plot(x, values.expanding().mean(), style, marker="o", markersize=3, label=label)
    for key, label in (("win_logloss", "win target"), ("podium_logloss", "podium target")):
        ax.axhline(BENCHMARKS[key], linestyle="--", linewidth=0.8, alpha=0.6)
        ax.text(x[-1], BENCHMARKS[key], f" {label}", va="bottom", fontsize=7)
    ax.set_xlabel("Scored predictions (oldest first)")
    ax.set_ylabel("Running mean log loss (lower = better)")
    ax.set_title("Track record")
    ax.legend(frameon=False, fontsize=8)
    return ax
