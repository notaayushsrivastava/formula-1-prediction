"""Phase 11 plots: ensemble weights over a season, M3's start weight vs overtaking."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes

from .colours import STINT_COLOURS

META = {"RaceId", "Date", "Year", "Moment", "Method", "TrainRaces"}


def weights_plot(history: pd.DataFrame, ax: Axes | None = None) -> Axes:
    """Stacked ensemble weights per race (each race's weights sum to 1)."""
    models = [c for c in history.columns if c not in META]
    table = history.sort_values("Date")
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 5))
    colours = [STINT_COLOURS[i % len(STINT_COLOURS)] for i in range(len(models))]
    ax.stackplot(table["Date"], [table[m].to_numpy(float) for m in models], labels=models,
                 colors=colours, alpha=0.85)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Weight")
    method = table["Method"].iloc[0] if "Method" in table else ""
    ax.set_title(f"Ensemble weights over time ({method} pool, fitted on earlier races)")
    ax.legend(bbox_to_anchor=(1.01, 1.0), loc="upper left", fontsize="small", frameon=False)
    return ax


def start_weight_plot(curve: pd.DataFrame, ax: Axes | None = None) -> Axes:
    """M3: share of the result decided by pace (vs grid order) by circuit overtaking."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    ax.plot(curve["places_moved"], curve["a_c"], color=STINT_COLOURS[0], linewidth=2)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Circuit: average places gained/lost per finisher (past races)")
    ax.set_ylabel("a_c: weight of pace (1 - a_c: grid order)")
    ax.set_title("M3: where overtaking is hard, the grid order holds")
    return ax
